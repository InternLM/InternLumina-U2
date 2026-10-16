import copy
import json
import logging
import os
from pathlib import Path
import pickle
from time import sleep
import traceback
import warnings

import h5py
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml
from veomni.utils import helper
from tasks.dataset.item_processor import ItemProcessorBase

logger = helper.create_logger(__name__)


class FinetuneConversationDataset(Dataset):
    def __init__(self, config_path, item_processor: ItemProcessorBase, cache_on_disk=False):

        self.item_processor = item_processor

        logger.info(f"read dataset config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)

        self.cache_on_disk = cache_on_disk
        self._cache_dir = None
        self._cache_h5_path = None
        self._h5_file = None
        self._h5_pid = None
        if self.cache_on_disk:
            cache_dir = self._get_cache_dir(config_path)
            self._cache_dir = cache_dir
            self._cache_h5_path = str(Path(cache_dir) / "data.h5")
            if dist.get_rank() == 0:
                self._collect_annotations_and_save_to_cache(cache_dir)
            dist.barrier()
            self.meta_collection, self.annotations_collection = self._load_annotations_from_cache(cache_dir)
        else:
            cache_dir = None
            self.meta_collection, self.annotations_collection = self._collect_annotations()

        # ---- Length-based filter using cached item_len_list ----
        # Drops samples whose predicted token length exceeds item_processor.max_len.
        # Cache layout is untouched; we only build an in-memory index map.
        self.max_len = getattr(item_processor, "max_len", None)
        self._valid_indices_per_meta = []
        total_dropped = 0
        for meta in self.meta_collection:
            item_lens = meta.get("item_len_list") or []
            orig_len = meta["len"]
            if self.max_len is not None and item_lens:
                # L == 0 means "no image tokens" (text-only or unknown) — keep.
                # L < 0 should not happen but be defensive — keep.
                # Only drop samples whose predicted length strictly exceeds max_len.
                valid = [i for i, L in enumerate(item_lens) if L <= self.max_len]
            else:
                valid = list(range(orig_len))
            dropped = orig_len - len(valid)
            total_dropped += dropped
            if dropped > 0:
                logger.info(
                    f"[length-filter] dropped {dropped}/{orig_len} samples "
                    f"(> max_len={self.max_len}) from {meta.get('path')}"
                )
            self._valid_indices_per_meta.append(valid)
            meta["len"] = len(valid)
        if total_dropped > 0:
            logger.info(f"[length-filter] total dropped: {total_dropped} samples")

    def __len__(self):
        return sum([_["len"] for _ in self.meta_collection])

    def _collect_annotations(self):
        meta_collection = []
        annotations_collection = []

        for meta in self.config["META"]:
            meta, annotations = self._load_meta(meta)
            meta_collection.append(meta)
            annotations_collection.append(annotations)

        return meta_collection, annotations_collection

    def _load_meta(self, meta):
        if "type" not in meta:
            meta["type"] = "default"

        meta_path, meta_type = meta["path"], meta["type"]
        meta_ext = os.path.splitext(meta_path)[-1]
        if meta_ext == ".json":
            with open(meta_path) as f:
                annotations = json.load(f, strict=False)
        elif meta_ext == ".jsonl":
            annotations = []
            with open(meta_path) as f:
                for i, line in enumerate(f):
                    try:
                        annotations.append(json.loads(line))
                    except json.decoder.JSONDecodeError as e:
                        logger.error(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                        raise e
        elif meta_ext == ".pkl":
            with open(meta_path, "rb") as f:
                annotations = pickle.load(f)
            assert isinstance(annotations, list)
        elif meta_ext == ".pth":
            annotations = torch.load(meta_path)
            assert isinstance(annotations, list)
        else:
            raise NotImplementedError(
                f'Unknown meta file extension: "{meta_ext}". '
                f"Currently, .json, .jsonl are supported. "
                "If you are using a supported format, please set the file extension so that the proper parsing "
                "routine can be called."
            )
        logger.info(f"{meta_path}, type{meta_type}: len {len(annotations)}")

        meta["len"] = len(annotations)
        
        meta["item_len_list"] = [self.item_processor.predict_item_token_length(_) for _ in annotations]

        return meta, annotations

    def _collect_annotations_and_save_to_cache(self, cache_dir):
        if (Path(cache_dir) / "data.h5").exists() and (Path(cache_dir) / "ready").exists():
            # off-the-shelf annotation cache exists
            warnings.warn(
                f"Use existing h5 data cache: {Path(cache_dir)}\n"
                f"Note: if the actual data defined by the data config has changed since your last run, "
                f"please delete the cache manually and re-run this experiment, or the data actually used "
                f"will not be updated"
            )
            return

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        meta_collection, annotations_collection = self._collect_annotations()

        # when cache on disk, rank0 saves items to an h5 file
        logger.info(f"start to build data cache to: {Path(cache_dir)}")
        with h5py.File(Path(cache_dir) / "data.h5", "w") as file:
            dt = h5py.vlen_dtype(str)
            for i, annotations in enumerate(annotations_collection):
                serialized_ann = [json.dumps(_) for _ in annotations]
                h5_ann = file.create_dataset(f"ann{i}", (len(serialized_ann),), dtype=dt)
                h5_ann[:] = serialized_ann

            file.create_dataset("meta_collection", data=json.dumps(meta_collection))
        with open(Path(cache_dir) / "ready", "w") as f:
            f.write("ready")
        logger.info(f"data cache built")

    @staticmethod
    def _get_cache_dir(config_path):
        config_identifier = config_path
        disallowed_chars = ["/", "\\", ".", "?", "!"]
        for _ in disallowed_chars:
            config_identifier = config_identifier.replace(_, "-")
        cache_root = os.environ.get("XLLMX_DATA_CACHE_DIR", "./xllmx_data_cache")
        cache_dir = os.path.join(cache_root, config_identifier)
        return cache_dir

    @staticmethod
    def _load_annotations_from_cache(cache_dir):
        while not (Path(cache_dir) / "ready").exists():
            # cache has not yet been completed by rank 0
            assert dist.get_rank() != 0
            sleep(1)
        # Open h5 only long enough to read meta; keep path for per-process reopen.
        # HDF5 is NOT fork-safe: reusing parent-opened datasets in DataLoader workers segfaults.
        with h5py.File(Path(cache_dir) / "data.h5", "r") as cache_file:
            meta_collection = json.loads(cache_file["meta_collection"].asstr()[()])
        # Placeholder list; real datasets are bound in `_ensure_h5_for_current_process`.
        annotations_collection = [None] * len(meta_collection)
        return meta_collection, annotations_collection

    def _ensure_h5_for_current_process(self):
        """Re-open data.h5 in this PID (main or DataLoader worker)."""
        if not self.cache_on_disk:
            return
        pid = os.getpid()
        if self._h5_pid == pid and self._h5_file is not None:
            return
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None
        self._h5_file = h5py.File(self._cache_h5_path, "r")
        self.annotations_collection = [
            self._h5_file[f"ann{i}"] for i in range(len(self.meta_collection))
        ]
        self._h5_pid = pid

    def __getstate__(self):
        """Drop live h5 handles so spawn workers can pickle the dataset."""
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_h5_pid"] = None
        if self.cache_on_disk:
            state["annotations_collection"] = [None] * len(self.meta_collection)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5_file = None
        self._h5_pid = None

    def get_item_func(self, meta_idx, idx_in_meta):
        self._ensure_h5_for_current_process()
        # Map filtered index -> original index in annotations
        real_idx = self._valid_indices_per_meta[meta_idx][idx_in_meta]
        data_item = self.annotations_collection[meta_idx][real_idx]
        if self.cache_on_disk:
            data_item = json.loads(data_item)
        else:
            data_item = copy.deepcopy(data_item)

        # # attach debug info for easier error localization in downstream processors
        try:
            meta = self.meta_collection[meta_idx]
            data_item["_meta_path"] = meta.get("path", None)
        except Exception:
            # best-effort, don't break if meta is unexpected
            data_item["_meta_path"] = None
        data_item["_meta_index"] = meta_idx
        data_item["_annotation_index"] = idx_in_meta

        return self.item_processor.process_item(data_item, training_mode=True)

    def tie_index_to_meta(self, idx: int):
        # Initialize the starting index
        start_idx = 0

        # Iterate through the list of dictionaries
        for i, meta in enumerate(self.meta_collection):
            # Calculate the ending index for the current collection
            end_idx = start_idx + meta["len"]

            # Check if the given index falls within the current collection
            if start_idx <= idx < end_idx:
                # Calculate the new index within the current collection
                new_index = idx - start_idx
                return i, new_index

            # Update the starting index for the next collection
            start_idx = end_idx

        # If the index is out of range of all collections, raise an error
        raise IndexError("Index out of range")

    def __getitem__(self, index):
        # Walk backwards over neighbours on failure. A whole shard can be missing from
        # object storage, so this is bounded: recursing per failure blew the Python stack
        # (RecursionError) instead of surfacing the real error.
        max_retries = 32
        for attempt in range(max_retries):
            meta_idx, idx_in_meta = self.tie_index_to_meta(index)
            try:
                return self.get_item_func(meta_idx, idx_in_meta)
            except Exception as e:
                # Keep this short: a bad shard would otherwise flood the log.
                logger.warning(
                    "Item %d (meta_idx=%d, idx_in_meta=%d) errored: %r",
                    index,
                    meta_idx,
                    idx_in_meta,
                    e,
                )
                last_error = e
                if idx_in_meta != 0:
                    index = index - 1
                else:
                    index = index + self.meta_collection[meta_idx]["len"] - 1
        raise RuntimeError(
            f"{max_retries} consecutive items failed to load around index {index}; "
            f"last error: {last_error!r}"
        )

