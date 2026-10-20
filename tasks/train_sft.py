"""Multitask SFT training entrypoint (block-diffusion, packing).

Objective (matches the BD pretrain contract):
- text mask id 156895 (<|mask|>, LLaDA2's own pretrained mask embedding);
- terminal answer turns are EOA-padded-and-supervised (teaches termination);
- per-block cosine noise (3% fully-masked blocks).

Data comes from ``dataset.xllmx_dataset_sft_multitask_blkpad``; the model is
``models.internluminau2.InternLuminaU2ModelLM``. Pair with
``configs/sft_packing.yaml`` and the matching inference entrypoints.
Checkpoints record ``data_objective`` so an incompatible resume is detected
instead of silently producing garbage.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime

import torch
try:
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
import torch.distributed as dist
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import math


import veomni as _veomni

# The in-tree VeOmni carries patches training depends on -- most importantly the
# torch-2.6 fsdp2 fix, whose absence shows up as silent non-finite values under
# `data_parallel_mode: fsdp2` + `enable_gradient_checkpointing`, not as an error.
# An unrelated `veomni` on sys.path (a pip or editable install) is picked up
# silently, so refuse to train against it. train_sft.sh sets PYTHONPATH already;
# this only fires when the module is invoked directly.
_VEOMNI_EXPECTED = Path(__file__).resolve().parent.parent / "VeOmni" / "veomni"
if Path(_veomni.__file__).resolve().parent != _VEOMNI_EXPECTED:
    raise RuntimeError(
        f"`veomni` resolved to {Path(_veomni.__file__).resolve().parent}, not the "
        f"in-tree {_VEOMNI_EXPECTED}. The bundled VeOmni is patched and training "
        f"needs those patches. Launch via train_sft.sh, or set "
        f"PYTHONPATH=<repo>:<repo>/VeOmni."
    )

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry
ModelRegistry.register_modeling_path("models.internluminau2")
from tasks.dataset.data_transform import process_mdm_tokenized_example, process_mdm_sft_example
from tasks.dataset import (
    build_local_dataset,
    build_xllmx_dataset_sft_multitask_blkpad
)
import xllmx.util as util
import xllmx.util.misc as misc
import shutil

from models.internluminau2.modeling_internluminau2 import load_balancing_loss_func
from veomni.data.data_collator import (
    CollatePipeline,
    DataCollatorWithPacking,
)

logger = helper.create_logger(__name__)

# Marker written into checkpoint extra_state so a resume can verify the data
# objective (text mask id 156895 + EOA-supervised terminal turns) matches.
DATA_OBJECTIVE = "bd_sft"

# The model_type that selects the packed (stacked) MoE expert parameters.
# Inference uses "llada2_moe" (a per-expert ModuleList) with hf_ckpt_split;
# training uses this one with the packed hf_ckpt export.
PACKED_MOE_MODEL_TYPE = "llada2_moe_veomni"

# Per-task loss breakdown reported at each log step. Must stay in sync with the
# task types the dataset accepts -- see TEXT_UNDERSTANDING_TASKS / IMAGE_OUTPUT_TASKS
# in tasks/dataset/xllmx_dataset_sft_multitask_blkpad.py.
TASK_LOSS_NAMES = (
    "text",
    "under",
    "under_think",
    "under_video",
    "under_3d",
    "t2i",
    "t2i_think",
    "single_edit",
)
TASK_TYPE_TO_LOSS_NAME = {name: name for name in TASK_LOSS_NAMES}
TASK_LOSS_NAME_TO_ID = {name: idx for idx, name in enumerate(TASK_LOSS_NAMES)}
_TASK_TYPE_MISMATCH_WARNED = False
_TASK_TYPE_UNKNOWN_WARNED = set()

@dataclass
class InternLuminaU2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )
    starting_point_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the starting point checkpoint."},
    )
    modified_point_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the starting point checkpoint."},
    )
    load_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load the checkpoint."},
    )


@dataclass
class InternLuminaU2DataArguments(DataArguments):
    data_type: Literal["conversation", "tokenid"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: Literal["mapping", "local", "xllmx_sft_multitask"] = field(
        default="xllmx_sft_multitask",
        metadata={"help": "Type of the datasets."},
    )
    text_keys: str = field(
        default="messages",
        metadata={"help": "Key to get text from the training data."},
    )
    noise_range_low: float = field(
        default=0.3,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )
    noise_range_high: float = field(
        default=0.8,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )
    cache_ann_on_disk: bool = field(
        default=True,
        metadata={"help": "Whether to cache ann data on disk."},
    )
    vocab_size: int = field(
        default=161280, # text + image(4096 for 1024 * 1024 resolution) --> token length
        metadata={"help": "Maximum sequence length."},
    )
    num_codebooks: int = field(
        default=8, # text + image(4096 for 1024 * 1024 resolution) --> token length
        metadata={"help": "Maximum sequence length."},
    )
    force_t2i_uncondition: bool = field(
        default=False,
        metadata={
            "help": (
                "Force <uncondition> for supported image-output dropout paths; "
                "no_t2i_think_drop still keeps t2i_think conditional."
            )
        },
    )
    t2i_think_uncondition_mode: Literal[
        "no_t2i_think_drop", "drop_prompt_and_think", "legacy_keep_think"
    ] = field(
        default="no_t2i_think_drop",
        metadata={
            "help": (
                "t2i_think CFG-drop policy: disable dropout (default), drop prompt+think, "
                "or drop only the prompt while keeping the legacy think target."
            )
        },
    )
    think_block_length: int = field(
        default=32,
        metadata={"help": "Legacy block length for t2i_think-only SFT."},
    )
    text_block_length: int = field(
        default=32,
        metadata={"help": "Block length for multitask SFT text targets."},
    )
    max_text_length: int = field(
        default=2048,
        metadata={"help": "Max tokenizer length for each SFT user/assistant text turn."},
    )
    dyn_pack_max_seq_len: int = field(
        default=0,
        metadata={"help": "SFT-only dynamic packing token budget. 0 means use max_seq_len."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) "
                f"cannot be greater than noise_range_high ({self.noise_range_high})."
            )

        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(
                f"noise_range_low must be between 0.0 and 1.0, but got {self.noise_range_low}."
            )

        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(
                f"noise_range_high must be between 0.0 and 1.0, but got {self.noise_range_high}."
            )

        if self.dyn_pack_max_seq_len < 0:
            raise ValueError(f"dyn_pack_max_seq_len must be >= 0, got {self.dyn_pack_max_seq_len}.")
        if self.dyn_pack_max_seq_len and self.dyn_pack_max_seq_len > self.max_seq_len:
            raise ValueError(
                f"dyn_pack_max_seq_len ({self.dyn_pack_max_seq_len}) cannot exceed max_seq_len ({self.max_seq_len})."
            )


@dataclass
class InternLuminaU2TrainingArguments(TrainingArguments):
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW optimizer beta1."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "AdamW optimizer beta2"},
    )
    text_embedding_head_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional absolute LR for the text input embedding and lm_head; "
            "uses the same scheduler multiplier as the global LR."
        },
    )
    image_input_output_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional absolute LR shared by img_embeddings, img_embeddings_proj, "
            "multi_codebook_head, and lm_img_head; uses the global scheduler multiplier."
        },
    )
    no_decay_modules: List[str] = field(
        # These must match real class names in models/internluminau2/modeling_internluminau2.py --
        # a name that matches nothing silently puts every norm back on weight decay.
        default_factory=lambda: ["LLaDA2MoeRMSNorm", "LigerRMSNorm", "LayerNorm"],
        metadata={"help": "Normalization modules whose parameters do not use weight decay."},
    )
    no_decay_params: List[str] = field(
        default_factory=lambda: ["bias"],
        metadata={"help": "Parameter-name fragments that do not use weight decay."},
    )
    block_diffusion_mode: bool = field(
        default=False,
        metadata={"help": "If train MDM in block_diffusion mode. True: use block_diffusion, False: full_attention"}
    )
    block_size: int = field(
        default=32,
        metadata={"help": "The block size for block diffusion block size"}
    )
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "If use same token location labels. True: no shift, False: use next-token prediction shift."}
    )
    ckpt_max_keep: int = field(
        default=2,
        metadata={"help": "Maximum number of checkpoints to keep."},
    )
    prepare_model: bool = field(
        default=False,
        metadata={"help": "If use same token location labels. True: no shift, False: use next-token prediction shift."}
    )
    use_same_datasets: bool = field(
        default=True,
        metadata={"help": "True: restore dataloader state and continue the same sample order; False: restart from the beginning with a new shuffle seed (train.seed + checkpoint global_step)."}
    )
    resume_use_new_lr: bool = field(
        default=False,
        metadata={"help": "When resuming from checkpoint, if True, restore the currently configured per-group LRs and scheduler base_lrs."}
    )
    resume_reset_lr_schedule: bool = field(
        default=False,
        metadata={
            "help": "When resuming from checkpoint, if True, keep model+optimizer state from ckpt but do not restore lr_scheduler state; reset global_step to 0 and start_epoch/start_step to 0 so warmup/decay runs from scratch. Pair with resume_use_new_lr to set base lr from config."
        },
    )
    moe_balance_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for MoE load balancing loss."}
    )
    text_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight multiplier for text loss to balance against image loss."}
    )
    image_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight multiplier for image loss to balance against text loss."}
    )
    tensorboard_log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional TensorBoard log root. Defaults to <save_checkpoint_path>/tensorboard_logs."},
    )
    enable_batch_dump: bool = field(
        default=False,
        metadata={"help": "Rolling-dump each micro_batch to <output_dir>/batch_dump/rank{n}/slot{step%8}.pt for crash forensics. Off by default; ~1% slowdown when on."}
    )

@dataclass
class Arguments:
    model: "InternLuminaU2ModelArguments" = field(default_factory=InternLuminaU2ModelArguments)
    data: "InternLuminaU2DataArguments" = field(default_factory=InternLuminaU2DataArguments)
    train: "InternLuminaU2TrainingArguments" = field(default_factory=InternLuminaU2TrainingArguments)


def _get_sft_param_lr_overrides(
    model: torch.nn.Module,
    text_lr: Optional[float],
    image_lr: Optional[float],
) -> Optional[Dict[str, float]]:
    """Resolve optional text/visual LR targets to exact trainable parameter names."""
    if text_lr is None and image_lr is None:
        return None

    name_by_param_id = {
        id(param): name
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    overrides: Dict[str, float] = {}

    def add_parameter(description: str, param: torch.nn.Parameter, lr: float) -> str:
        name = name_by_param_id.get(id(param))
        if name is None:
            raise ValueError(
                f"Could not resolve the trainable parameter name for {description}."
            )
        if name in overrides:
            raise ValueError(
                f"Dedicated LR target {name!r} belongs to more than one text/visual target."
            )
        overrides[name] = lr
        return name

    if text_lr is not None:
        if text_lr <= 0:
            raise ValueError(f"text_embedding_head_lr must be positive, got {text_lr}.")
        text_targets = (
            ("text input embedding", model.get_input_embeddings().weight),
            ("text lm_head", model.get_output_embeddings().weight),
        )
        text_names = [
            add_parameter(description, param, float(text_lr))
            for description, param in text_targets
        ]
        if len(set(text_names)) != 2:
            raise ValueError(
                "Expected distinct text input-embedding and lm_head parameters, "
                f"but resolved {text_names}."
            )
        logger.info_rank0(
            f"SFT text embedding/head LR: {len(text_names)} parameters use "
            f"lr={float(text_lr):.6e}"
        )

    if image_lr is not None:
        if image_lr <= 0:
            raise ValueError(f"image_input_output_lr must be positive, got {image_lr}.")
        backbone = getattr(model, "model", None)
        visual_targets = (
            ("image input embeddings", getattr(backbone, "img_embeddings", None)),
            ("image input projection", getattr(backbone, "img_embeddings_proj", None)),
            ("multi-codebook output head", getattr(model, "multi_codebook_head", None)),
            ("image logits heads", getattr(model, "lm_img_head", None)),
        )
        for description, module in visual_targets:
            if not isinstance(module, torch.nn.Module):
                raise ValueError(f"Could not resolve trainable module for {description}.")
            names = [
                add_parameter(description, param, float(image_lr))
                for param in module.parameters()
                if param.requires_grad
            ]
            if not names:
                raise ValueError(f"No trainable parameters found for {description}.")
            logger.info_rank0(
                f"SFT visual LR target {description}: {len(names)} parameters use "
                f"lr={float(image_lr):.6e}"
            )

    return overrides or None


def _compute_grad_stats(module):
    """Global grad norm, largest absolute gradient, and which parameter it is in."""
    total_sq = 0.0
    max_abs = 0.0
    worst_name: Optional[str] = None

    for name, p in module.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad.detach()
        # Under FSDP2 / DTensor a local shard can be empty, so take the local
        # tensor first and test numel() on that.
        g_local = g
        to_local = getattr(g_local, "to_local", None)
        if callable(to_local):
            try:
                g_local = to_local()
            except Exception:
                g_local = g
        if g_local.numel() == 0:
            continue
        # Global L2 norm.
        total_sq += float(g_local.float().pow(2).sum().item())
        # Largest absolute value and the parameter it came from.
        cur_max = float(g_local.abs().max().item())
        if cur_max > max_abs:
            max_abs = cur_max
            worst_name = name

    global_norm = math.sqrt(total_sq) if total_sq > 0.0 else 0.0
    return global_norm, max_abs, worst_name



def _extract_task_types(task_type_value):
    if task_type_value is None:
        return []
    if isinstance(task_type_value, str):
        return [task_type_value]
    if isinstance(task_type_value, bytes):
        return [task_type_value.decode("utf-8")]
    if isinstance(task_type_value, (list, tuple)):
        task_types = []
        for x in task_type_value:
            if isinstance(x, bytes):
                task_types.append(x.decode("utf-8"))
            else:
                task_types.append(str(x))
        return task_types
    return [str(task_type_value)]


def _task_group_id(task_type: str) -> int:
    global _TASK_TYPE_UNKNOWN_WARNED
    loss_name = TASK_TYPE_TO_LOSS_NAME.get(task_type, task_type)
    group_id = TASK_LOSS_NAME_TO_ID.get(loss_name, -1)
    if group_id < 0 and task_type not in _TASK_TYPE_UNKNOWN_WARNED:
        logger.warning(
            f"Unknown task type '{task_type}' is excluded from task-loss stats. "
            f"Known task types: {sorted(TASK_TYPE_TO_LOSS_NAME.keys())}"
        )
        _TASK_TYPE_UNKNOWN_WARNED.add(task_type)
    return group_id


def _build_flat_task_group_ids(micro_batch, labels, seq_start: int = 0):
    """Map flattened positions to packed task ids, repeating shared segments for every batch row."""
    global _TASK_TYPE_MISMATCH_WARNED
    device = labels.device
    batch_size, seq_len = labels.shape[:2]
    flat_len = batch_size * seq_len
    task_types = _extract_task_types(micro_batch.get("type"))
    task_group_ids = torch.full((flat_len,), -1, dtype=torch.long, device=device)
    if not task_types:
        return task_group_ids

    cu_seqlens = micro_batch.get("cu_seqlens")
    if cu_seqlens is not None:
        cu = cu_seqlens.detach().cpu().tolist()
        full_seq_len = max(int(cu[-1]) if cu else 0, int(seq_start) + seq_len)
        row_task_ids = torch.full((full_seq_len,), -1, dtype=torch.long, device=device)
        for i in range(max(len(cu) - 1, 0)):
            if i >= len(task_types):
                if not _TASK_TYPE_MISMATCH_WARNED:
                    logger.warning(
                        f"task type count ({len(task_types)}) < packed sample count ({max(len(cu) - 1, 0)}), "
                        "unmatched packed samples will be excluded from task-loss stats."
                    )
                    _TASK_TYPE_MISMATCH_WARNED = True
                continue
            group_id = _task_group_id(task_types[i])
            if group_id < 0:
                continue
            start = max(0, min(int(cu[i]), full_seq_len))
            end = max(0, min(int(cu[i + 1]), full_seq_len))
            if end > start:
                row_task_ids[start:end] = group_id
        local_row = row_task_ids[int(seq_start) : int(seq_start) + seq_len]
        if local_row.numel() < seq_len:
            local_row = torch.nn.functional.pad(local_row, (0, seq_len - local_row.numel()), value=-1)
        return local_row.repeat(batch_size)

    if len(task_types) < batch_size and not _TASK_TYPE_MISMATCH_WARNED:
        logger.warning(
            f"task type count ({len(task_types)}) < batch size ({batch_size}), "
            "unmatched samples will be excluded from task-loss stats."
        )
        _TASK_TYPE_MISMATCH_WARNED = True
    per_sample_ids = [
        _task_group_id(task_types[i]) if i < len(task_types) else -1
        for i in range(batch_size)
    ]
    sample_task_ids = torch.tensor(per_sample_ids, dtype=torch.long, device=device)
    return sample_task_ids.repeat_interleave(seq_len)


def _accumulate_task_loss_stats(task_loss_sums, task_loss_counts, task_ids, losses, counts):
    if losses is None or losses.numel() == 0:
        return
    task_ids = task_ids.detach()
    losses = losses.detach()
    counts = counts.detach()
    for task_idx, task_name in enumerate(TASK_LOSS_NAMES):
        mask = task_ids == task_idx
        if mask.any():
            task_loss_sums[task_name] += float(losses[mask].sum().item())
            task_loss_counts[task_name] += float(counts[mask].sum().item())


def block_diffusion_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                          (q_idx - n) // block_size,
                          q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                           (kv_idx - n) // block_size,
                           kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal

def split_ckpt_str_into_epoch_iter(ckpt_str: str):
    # divide ckpt directory names into epoch and iter parts
    parts = ckpt_str.split("-")
    epoch = int(parts[0].replace("epoch", ""))
    if len(parts) == 2:
        iter_part = int(parts[1].replace("global_step_", ""))
    else:
        iter_part = None
    return epoch, iter_part

def remove_early_ckpts(out_dir, max_keep=2):

    if max_keep <= 0:
        return

    def ckpt_sort_key(s):
        # divide ckpt directory names into epoch and iter parts
        epoch, iteration = split_ckpt_str_into_epoch_iter(s)
        if iteration is None:
            iteration = float("inf")
        return epoch, iteration

    existing_checkpoints = [_ for _ in os.listdir(out_dir) if "epoch" in _]
    existing_checkpoints = sorted(existing_checkpoints, key=ckpt_sort_key, reverse=True)

    for dir_to_remove in existing_checkpoints[max_keep:]:
        dir_to_remove = os.path.join(out_dir, dir_to_remove)
        shutil.rmtree(dir_to_remove)
        logger.info(f"Deleted {dir_to_remove}")


def main():
    util.dist.init_distributed_mode()
    args = parse_args(Arguments)
    # The SFT dataset is a single-source schema; never interpret its YAML as
    # VeOmni's multisource format.
    if args.data.datasets_type == "xllmx_sft_multitask":
        args.data.enable_multisource = False
    
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError(f"No chat template found in the tokenizer.")

        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895, 
        )
    elif args.data.data_type == "tokenid":
        transform = partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895, 
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "local":
            logger.info_rank0("Start building local dataset")
            train_dataset = build_local_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "xllmx_sft_multitask":
            logger.info_rank0("Start building xllmx multitask SFT dataset")
            train_dataset = build_xllmx_dataset_sft_multitask_blkpad(
                args.data.train_path,
                tokenizer=tokenizer,
                cache_on_disk=args.data.cache_ann_on_disk,
                max_len=args.data.max_seq_len,
                block_size=args.data.text_block_length,
                max_text_length=args.data.max_text_length,
                force_t2i_uncondition=args.data.force_t2i_uncondition,
                t2i_think_uncondition_mode=args.data.t2i_think_uncondition_mode,
            )

        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type in ("mapping", "local", "xllmx_sft_multitask"):
            dataset_length = dataset_length / args.train.data_parallel_size
        pack_max_seq_len = args.data.dyn_pack_max_seq_len or args.data.max_seq_len
        if pack_max_seq_len != args.data.max_seq_len:
            logger.info_rank0(
                f"Using dyn_pack_max_seq_len={pack_max_seq_len} for dataloader packing; "
                f"hard max_seq_len remains {args.data.max_seq_len}."
            )
        args.train.compute_train_steps(pack_max_seq_len, args.data.train_size, dataset_length)

        # Use packing-only collator to bypass VeOmni's TextSequenceShardCollator
        # which shifts labels (incompatible with block-diffusion multi-codebook labels).
        collate_fn = CollatePipeline([DataCollatorWithPacking()])
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=pack_max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
            collate_fn=collate_fn,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    logger.info_rank0("Make Init Point")

    logger.info_rank0("Prepare model")
    if args.train.prepare_model:

        if args.train.global_rank == 0:
            model_for_modify = build_foundation_model(
                config_path=args.model.config_path,
                weights_path=args.model.model_path,
                torch_dtype="bfloat16",
                attn_implementation=args.model.attn_implementation,
                moe_implementation=args.model.moe_implementation,
                init_device='cuda', 
                force_use_huggingface=args.model.force_use_huggingface,
            )

            vocab_size = args.data.vocab_size

            model_for_modify.resize_token_embeddings(157184, mean_resizing=False)
            model_for_modify.lm_img_head = torch.nn.ModuleList([torch.nn.Linear(2048, 4096, bias=False) for _ in range(8)])
            model_for_modify.config.vocab_size = vocab_size

            model_for_modify.save_pretrained(args.model.modified_point_path)
            model_for_modify.config.save_pretrained(args.model.modified_point_path)

            config_file_path = os.path.join(args.model.modified_point_path, "config.json")
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            # Keep whatever the config class declares; a different string here
            # produces a directory that AutoConfig cannot load back.
            config_data["model_type"] = type(model_for_modify.config).model_type

            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info_rank0(f"Saved starting point model to {args.model.modified_point_path}")

        dist.barrier()

    config_init_from = str(args.model.config_path)
    reload_model_path = getattr(args.train, "reload_model_checkpoint_path", None)
    if args.model.load_model_path is not None:
        model_init_from = str(args.model.load_model_path)
    elif reload_model_path:
        model_init_from = str(reload_model_path)
    elif args.train.load_checkpoint_path:
        checkpoint_hf_path = os.path.join(args.train.load_checkpoint_path, 'hf_ckpt')
        if os.path.exists(os.path.join(checkpoint_hf_path, 'config.json')):
            model_init_from = checkpoint_hf_path
        else:
            model_init_from = str(args.model.modified_point_path)
            logger.info_rank0(
                f"No hf_ckpt found under {args.train.load_checkpoint_path}; "
                f"initialize from {model_init_from} before distributed checkpoint load."
            )
    else:
        model_init_from = str(args.model.modified_point_path)

    model = build_foundation_model(
        config_path=config_init_from,
        weights_path=model_init_from,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device, 
        force_use_huggingface=args.model.force_use_huggingface,
    )
    dist.barrier()
    model_config = model.config
    # Training requires the PACKED expert layout. model_type is what selects it
    # (LLaDA2MoeSparseMoeBlock._setup_fuse_moe_experts), and getting this wrong
    # does not raise on its own: the loader logs "Unexpected key in state dict"
    # for every MoE layer and then trains from randomly initialised experts,
    # which is most of the parameters. Fail at startup instead.
    if model_config.model_type != PACKED_MOE_MODEL_TYPE:
        raise ValueError(
            f"training needs the packed-expert model config "
            f"(model_type={PACKED_MOE_MODEL_TYPE!r}), but "
            f"{args.model.config_path} declares {model_config.model_type!r}. "
            f"That config builds a per-expert ModuleList, so the packed expert "
            f"weights in the checkpoint match no parameter and every expert "
            f"stays at its random init. Point model.config_path at "
            f"configs/model_configs/internluminau2_packed/."
        )
    helper.print_device_mem_info("VRAM usage after building model")
    
    
    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=model_init_from,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0
    )

    param_lr_overrides = _get_sft_param_lr_overrides(
        model,
        args.train.text_embedding_head_lr,
        args.train.image_input_output_lr,
    )
    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        no_decay_modules=args.train.no_decay_modules,
        no_decay_params=args.train.no_decay_params,
        param_lr_overrides=param_lr_overrides,
    )

    if getattr(optimizer, "_is_multi_optimizer", False):
        configured_optimizer_group_lrs = {
            name: [float(group["lr"]) for group in leaf_optimizer.param_groups]
            for name, leaf_optimizer in optimizer.optimizers_dict.items()
        }
    else:
        configured_optimizer_group_lrs = {
            "default": [float(group["lr"]) for group in optimizer.param_groups]
        }

    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        # save model_assets before training
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)
    
    dist.barrier()

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    # For debug prints when resuming: treat the first step after resume as relative_step=1
    resume_global_step = 0
    
    
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        reload_model_path = getattr(args.train, "reload_model_checkpoint_path", None)
        if reload_model_path:
            import torch.distributed.checkpoint as _dcp
            from torch.distributed.checkpoint import FileSystemReader as _FSReader
            from veomni.checkpoint.checkpointer import OptimizerState as _OptimizerState

            logger.info_rank0(
                f"Optimizer-only DCP load from {args.train.load_checkpoint_path} "
                f"(model already initialized from {reload_model_path})"
            )
            optim_load_state = {"optimizer": _OptimizerState(model=model, optimizer=optimizer)}
            _dcp.load(state_dict=optim_load_state, storage_reader=_FSReader(args.train.load_checkpoint_path))
            logger.info_rank0("Optimizer loaded from DCP successfully!")

            extra_state_dir = os.path.join(args.train.load_checkpoint_path, "extra_state")
            extra_state_path = os.path.join(extra_state_dir, f"extra_state_rank_{dist.get_rank()}.pt")
            state = {"extra_state": torch.load(extra_state_path, weights_only=False)}
        else:
            state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
            dcp_process_group = dist.new_group(backend="gloo")
            Checkpointer.load(
                args.train.load_checkpoint_path,
                state,
                process_group=dcp_process_group,
            )

        saved_data_objective = str(state["extra_state"].get("data_objective") or DATA_OBJECTIVE)
        if saved_data_objective != DATA_OBJECTIVE:
            if args.train.use_same_datasets:
                raise ValueError(
                    "Cannot restore dataloader state across data objectives: "
                    f"checkpoint={saved_data_objective!r}, current={DATA_OBJECTIVE!r}. "
                    "Set --train.use_same_datasets false to rebuild the data stream."
                )
            logger.info_rank0(
                f"Data objective changed {saved_data_objective!r} -> {DATA_OBJECTIVE!r}; "
                "dataloader state will not be restored."
            )

        reset_schedule = getattr(args.train, "resume_reset_lr_schedule", False)
        if reset_schedule:
            logger.info_rank0(
                "resume_reset_lr_schedule=True: keeping optimizer (and model) from checkpoint; "
                "lr_scheduler stays freshly built (warmup/decay from step 0)."
            )
        else:
            lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if getattr(args.train, "resume_use_new_lr", False):
            leaf_optimizers = (
                optimizer.optimizers_dict
                if getattr(optimizer, "_is_multi_optimizer", False)
                else {"default": optimizer}
            )
            for name, leaf_optimizer in leaf_optimizers.items():
                configured_lrs = configured_optimizer_group_lrs[name]
                if len(leaf_optimizer.param_groups) != len(configured_lrs):
                    raise ValueError(
                        f"Optimizer group count changed for {name!r}: "
                        f"loaded={len(leaf_optimizer.param_groups)}, configured={len(configured_lrs)}."
                    )
                for group, configured_lr in zip(leaf_optimizer.param_groups, configured_lrs):
                    group["lr"] = configured_lr

            if getattr(lr_scheduler, "_is_multi_lr_scheduler", False):
                for name, scheduler in lr_scheduler.items():
                    scheduler.base_lrs = list(configured_optimizer_group_lrs[name])
                    scheduler._last_lr = list(configured_optimizer_group_lrs[name])
            else:
                configured_lrs = configured_optimizer_group_lrs["default"]
                lr_scheduler.base_lrs = list(configured_lrs)
                lr_scheduler._last_lr = list(configured_lrs)
            logger.info_rank0(
                "Resume with configured optimizer-group lrs: "
                f"{configured_optimizer_group_lrs} (resume_use_new_lr=True)"
            )
        override_step = getattr(args.train, "resume_override_global_step", None)
        if override_step is not None:
            global_step = override_step
            start_epoch = global_step // args.train.train_steps
            start_step = global_step % args.train.train_steps
            resume_global_step = global_step
            if reset_schedule:
                if getattr(lr_scheduler, "_is_multi_lr_scheduler", False):
                    for sched in lr_scheduler.values():
                        sched.last_epoch = override_step
                        sched._step_count = override_step + 1
                        for i, lr_val in enumerate(sched.get_lr()):
                            sched.optimizer.param_groups[i]["lr"] = lr_val
                        sched._last_lr = [pg["lr"] for pg in sched.optimizer.param_groups]
                else:
                    lr_scheduler.last_epoch = override_step
                    lr_scheduler._step_count = override_step + 1
                    for i, lr_val in enumerate(lr_scheduler.get_lr()):
                        optimizer.param_groups[i]["lr"] = lr_val
                    lr_scheduler._last_lr = [pg["lr"] for pg in optimizer.param_groups]
                logger.info_rank0(
                    f"resume_override_global_step={override_step}: lr_scheduler fast-forwarded to step {override_step}, "
                    f"current lr={optimizer.param_groups[0]['lr']:.6e}"
                )
            logger.info_rank0(f"resume_override_global_step={override_step}: overriding global_step to {global_step}")
        elif reset_schedule:
            global_step = 0
            start_epoch = 0
            start_step = 0
            resume_global_step = 0
            logger.info_rank0(
                "resume_reset_lr_schedule=True: global_step / epoch iteration counters reset to 0 "
                "(new checkpoint filenames may collide if output_dir matches a previous run)."
            )
        else:
            global_step = state["extra_state"]["global_step"]
            start_epoch = global_step // args.train.train_steps
            start_step = global_step % args.train.train_steps
        if args.train.use_same_datasets:
            environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
            train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])

            if start_step == 0:  # resume at the end of epoch
                iter(train_dataloader)  # clear resume state and prefetch data


        else:
            # Restart data from index 0 with a derived shuffle seed so we don't
            # replay the same order as a fresh run (scheme A: seed + ckpt step).
            ckpt_step = state["extra_state"]["global_step"]
            data_seed = args.train.seed + ckpt_step
            sampler = getattr(train_dataloader, "sampler", None)
            if sampler is None and hasattr(train_dataloader, "_dataloader"):
                sampler = getattr(train_dataloader._dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "seed"):
                sampler.seed = data_seed
                if hasattr(sampler, "yielded"):
                    sampler.yielded = 0
                if hasattr(sampler, "next_yielded"):
                    sampler.next_yielded = None
                logger.info_rank0(
                    f"use_same_datasets=False: not restoring dataloader state; "
                    f"reshuffle with data_seed={data_seed} (seed={args.train.seed} + ckpt_step={ckpt_step})"
                )
            else:
                logger.info_rank0(
                    f"use_same_datasets=False: not restoring dataloader state; "
                    f"sampler seed not updated (data_seed={data_seed}). "
                    f"If using an iterable dataset, change train.seed to reshuffle."
                )
        dist.barrier()
        # gc.collect()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")
        resume_global_step = global_step
    # Build block diffusion attention mask
    if args.train.block_diffusion_mode:
        bd_attn_full_len = args.data.max_seq_len * 2
        block_size = args.train.block_size
        # NOTE: Boolean dtype block diffusion attention mask
        block_diffusion_attn_mask_flag = block_diffusion_mask(
            b=None, h=None,
            q_idx=torch.arange(bd_attn_full_len)[:, None],
            kv_idx=torch.arange(bd_attn_full_len)[None, :],
            block_size=block_size,
            n=args.data.max_seq_len
        ).unsqueeze(0).unsqueeze(0)
        
        block_diffusion_attn_mask_prototype = torch.zeros_like(
            block_diffusion_attn_mask_flag, 
            dtype=torch.float32 if args.train.enable_mixed_precision else torch.bfloat16
        )
        block_diffusion_attn_mask_prototype.masked_fill_(block_diffusion_attn_mask_flag.logical_not(), float("-inf"))

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    
    tb_writer = None
    if args.train.global_rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_root = args.train.tensorboard_log_dir or os.path.join(args.train.save_checkpoint_path, "tensorboard_logs")
        tb_log_dir = os.path.join(tb_log_root, timestamp)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info_rank0(f"TensorBoard logs will be saved to {tb_log_dir}")

    # Lazily built inside the epoch loop below. A resume starts its smoothed
    # meters from an empty window rather than restoring the pre-resume history;
    # that affects the displayed averages for a few steps only, not training.
    metric_logger = None
    
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.global_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
            
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            total_loss = 0
            total_loss_img_sum = 0.0
            total_loss_text_sum = 0.0
            # Global token weighting: accumulate raw loss sums and valid-token
            # counts for this rank/step, then divide once after the loop, instead
            # of averaging per-pack means.
            total_img_count = 0.0
            total_text_count = 0.0
            # Online token accuracy (argmax @ supervised positions); same count denom as loss.
            total_img_correct = 0.0
            total_text_correct = 0.0
            total_moe_loss = 0.0
            total_moe_loss_raw = 0.0
            task_loss_sums = {name: 0.0 for name in TASK_LOSS_NAMES}
            task_loss_counts = {name: 0.0 for name in TASK_LOSS_NAMES}
            synchronize()
            start_time = time.time()
            _pack_max = args.data.dyn_pack_max_seq_len or args.data.max_seq_len
            _token_micro_budget = int(args.train.micro_batch_size) * int(_pack_max)

            # ---- Collective skip: drop the entire step for a hard length overflow ----
            # All ranks must agree on the skip, so reduce the max packed length globally.
            _max_packed_len_cfg = max(
                int(args.data.max_seq_len),
                _token_micro_budget,
            )
            _local_max_len = max(
                (
                    int(_mb["input_ids"].shape[0] if _mb["input_ids"].dim() == 2 else _mb["input_ids"].shape[1])
                    for _mb in micro_batches
                    if isinstance(_mb.get("input_ids"), torch.Tensor)
                ),
                default=0,
            )
            _len_state = torch.tensor([_local_max_len], dtype=torch.long, device=get_device_type())
            if dist.is_initialized():
                dist.all_reduce(_len_state, op=dist.ReduceOp.MAX)
            _global_max_len = int(_len_state.item())
            if _global_max_len > _max_packed_len_cfg:
                if args.train.global_rank == 0:
                    logger.warning(
                        f"[BATCH-SKIP step={global_step}] global_max_seq_len={_global_max_len} "
                        f"> max_packed_len={_max_packed_len_cfg} "
                        f"(max_seq_len={args.data.max_seq_len}, micro_batch_size={args.train.micro_batch_size}, "
                        f"pack_max={_pack_max}); types={[mb.get('type') for mb in micro_batches]}; "
                        "skipping whole step"
                    )
                for _mb in micro_batches:
                    for _k in list(_mb.keys()):
                        if isinstance(_mb[_k], torch.Tensor):
                            _mb[_k] = None
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
                continue

            # ================================================================
            # Pre-pass for global token weighting. Compute each micro-batch's
            # valid-token count (count_full) and their sum G_local, then weight
            # each micro-batch loss by count_full/G_local instead of
            # 1/len(micro_batches). Every token then carries the same weight
            # across the step regardless of how one pack happens to be composed.
            # Counting uses labels + type_position_ids + optional loss_weights,
            # matching the main loop.
            # ================================================================
            _num_cb_pp = args.data.num_codebooks
            def _full_token_count(_mb):
                _lbl = _mb.get("labels")
                _tp = _mb.get("type_position_ids")
                if _lbl is None or _tp is None:
                    return 0.0
                if _lbl.dim() == 1:
                    _lbl2 = _lbl.reshape(-1, 1)
                elif _lbl.dim() == 2:
                    _lbl2 = _lbl.reshape(-1, _num_cb_pp) if _lbl.size(-1) == _num_cb_pp else _lbl.reshape(-1, 1)
                else:
                    _lbl2 = _lbl.reshape(-1, _lbl.size(-1))
                _tpf = _tp.reshape(-1)
                _img_all = _lbl2[_tpf == 4]
                if _img_all.numel() > 0:
                    _vpos = (_img_all != -100).any(dim=-1)
                    _n_img = (_img_all[_vpos] != -100).sum().item() / _img_all.size(-1)
                else:
                    _n_img = 0.0
                _txt_all = _lbl2[_tpf == 3]
                if _txt_all.numel() > 0:
                    _txt_valid = _txt_all[:, 0] != -100
                    _weights = _mb.get("loss_weights")
                    if _weights is None:
                        _n_txt = _txt_valid.sum().item()
                    else:
                        _weights_flat = _weights.reshape(-1)
                        if _weights_flat.numel() != _tpf.numel():
                            raise ValueError("loss_weights must match type_position_ids length")
                        _n_txt = _weights_flat[_tpf == 3][_txt_valid].sum().item()
                else:
                    _n_txt = 0.0
                return float(_n_img) + float(_n_txt)

            _mb_token_counts = [_full_token_count(_mb) for _mb in micro_batches]
            _G_local = max(sum(_mb_token_counts), 1.0)

            for _mb_idx, micro_batch in enumerate(micro_batches):
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                # ---- Rolling micro_batch dump for crash forensics (opt-in) ----
                # Enable via train.enable_batch_dump: true in yaml.
                # 64 ranks * 8 slots * ~2MB ≈ 1GB total (≪ 100GB budget); ~1% slowdown when on.
                if args.train.enable_batch_dump:
                    _DUMP_SLOTS = 8
                    _dump_dir = os.path.join(args.train.output_dir, "batch_dump", f"rank{args.train.global_rank}")
                    os.makedirs(_dump_dir, exist_ok=True)
                    _dump_path = os.path.join(_dump_dir, f"slot{global_step % _DUMP_SLOTS}.pt")
                    _dump_payload = {
                        "step": global_step,
                        "rank": args.train.global_rank,
                        "local_rank": args.train.local_rank,
                        "tensors": {k: v.detach().cpu() for k, v in micro_batch.items() if isinstance(v, torch.Tensor)},
                        "non_tensors": {k: v for k, v in micro_batch.items() if not isinstance(v, torch.Tensor)},
                    }
                    torch.save(_dump_payload, _dump_path)
                    del _dump_payload
                # ---- Sanity check: input_ids & labels within valid range ----
                _input_ids = micro_batch["input_ids"]           # [B, S, C] or [S, C] (packing)
                _labels = micro_batch["labels"]                 # [B, S], [B, S, C], [S], or [S, C]
                _type_pos = micro_batch.get("type_position_ids")
                # Normalize to 3D [B, S, C] for uniform checking
                _ids3 = _input_ids if _input_ids.dim() == 3 else _input_ids.unsqueeze(0)
                _tp2 = _type_pos if _type_pos is not None and _type_pos.dim() == 2 else (
                    _type_pos.unsqueeze(0) if _type_pos is not None else None)
                if _tp2 is not None:
                    _img_mask = (_tp2 == 2) | (_tp2 == 4)
                    _text_mask = ~_img_mask
                    if _text_mask.any():
                        _text_ids = _ids3[:, :, 0][_text_mask]
                        _tmax = _text_ids.max().item()
                        _tmin = _text_ids.min().item()
                        assert 0 <= _tmin and _tmax < 157184, \
                            f"[step={global_step}] text input_ids OOB: min={_tmin}, max={_tmax}, expected [0, 157184)"
                    if _img_mask.any():
                        _img_ids = _ids3[_img_mask.unsqueeze(-1).expand_as(_ids3)].view(-1, _ids3.size(-1))
                        _imax = _img_ids.max().item()
                        _imin = _img_ids.min().item()
                        assert 0 <= _imin and _imax < 5010, \
                            f"[step={global_step}] img input_ids OOB: min={_imin}, max={_imax}, expected [0, 5010)"
                # labels check
                _lbl = _labels
                if _lbl.dim() == 1:
                    _lbl = _lbl.unsqueeze(0).unsqueeze(-1)
                elif _lbl.dim() == 2:
                    if _lbl.size(-1) == args.data.num_codebooks:
                        _lbl = _lbl.unsqueeze(0)
                    else:
                        _lbl = _lbl.unsqueeze(-1)
                if _tp2 is not None:
                    _img_label_mask = (_tp2 == 4)
                    if _img_label_mask.any():
                        _img_labs = _lbl[_img_label_mask.unsqueeze(-1).expand_as(_lbl)].view(-1)
                        _valid = _img_labs[_img_labs != -100]
                        if _valid.numel() > 0:
                            _ilmax = _valid.max().item()
                            _ilmin = _valid.min().item()
                            assert 0 <= _ilmin and _ilmax < 4096, \
                                f"[step={global_step}] img labels OOB: min={_ilmin}, max={_ilmax}, expected [0, 4096)"
                    _text_label_mask = (_tp2 == 3)
                    if _text_label_mask.any():
                        _text_labs = _lbl[_text_label_mask.unsqueeze(-1).expand_as(_lbl)].view(-1, _lbl.size(-1))[:, 0]
                        _valid_t = _text_labs[_text_labs != -100]
                        if _valid_t.numel() > 0:
                            _tlmax = _valid_t.max().item()
                            _tlmin = _valid_t.min().item()
                            assert 0 <= _tlmin and _tlmax < 157184, \
                                f"[step={global_step}] text labels OOB: min={_tlmin}, max={_tlmax}, expected [0, 157184)"
                del _input_ids, _labels, _type_pos, _ids3, _tp2, _lbl

                if global_step == 1:
                    # Dump one example to the log so a bad batch shape is visible at step 1.
                    helper.print_example(micro_batch, rank=args.train.local_rank, print_tensor=False)
                if args.train.block_diffusion_mode:
                    noisy_input_ids = micro_batch["noisy_input_ids"]
                    clean_input_ids = micro_batch["input_ids"]
                    batch_size = noisy_input_ids.shape[0]
                    full_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
                    noisy_position_ids = torch.arange(noisy_input_ids.shape[1], device=get_device_type(), dtype=torch.long)
                    clean_position_ids = torch.arange(clean_input_ids.shape[1], device=get_device_type(), dtype=torch.long)
                    position_ids = torch.cat([noisy_position_ids, clean_position_ids], dim=0).unsqueeze(0).expand(batch_size, -1).clone()
                    micro_batch["input_ids"] = full_input_ids
                    micro_batch["position_ids"] = position_ids
                    micro_batch["attention_mask"] = block_diffusion_attn_mask_prototype.expand(batch_size, -1, -1, -1)
                else:
                    # In packing mode, we rely on `attention_mask` + `position_ids` to split packed samples.
                    # Do NOT overwrite it with None.
                    micro_batch["attention_mask"] = micro_batch.get("attention_mask")
                    
                # Avoid blocking in automated runs; enable explicitly via env var.

                loss_weights = micro_batch.pop("loss_weights", None)

                with model_fwd_context:

                    model_outputs = model(
                        **micro_batch,
                        use_cache=False,
                        output_router_logits=True,
                        return_hidden_states_for_loss=True,
                    )
                    logits_img, logits_text, router_outputs, _, head_dummy = model_outputs
                    # ================================================================
                    # 1. prepare labels & type_position_ids
                    # ================================================================
                    labels = micro_batch["labels"]                     # [B, S] or [B, S, C]
                    type_position_ids = micro_batch.get("type_position_ids")

                    # Normalize label / type_position_ids shapes across padding and packing.
                    if args.train.rmpad and type_position_ids is not None:
                        if type_position_ids.dim() == 1:
                            type_position_ids = type_position_ids.unsqueeze(0)  # -> [1, S]
                        if labels is not None and labels.dim() == 2:
                            if labels.size(-1) == args.data.num_codebooks:
                                labels = labels.unsqueeze(0)  # -> [1, S, C]
                            else:
                                labels = labels.unsqueeze(-1)  # -> [1, S, 1]
                    else:
                        if labels is not None and labels.dim() == 2:
                            labels = labels.unsqueeze(-1)  # -> [B, S, 1]

                    if loss_weights is None:
                        loss_weights = torch.ones_like(type_position_ids, dtype=torch.float32)
                    else:
                        if loss_weights.dim() == 1:
                            loss_weights = loss_weights.unsqueeze(0)
                        if loss_weights.dim() != 2 or loss_weights.shape != type_position_ids.shape:
                            raise ValueError(
                                "loss_weights must match normalized type_position_ids shape, "
                                f"got {tuple(loss_weights.shape)} vs {tuple(type_position_ids.shape)}"
                            )
                        loss_weights = loss_weights.to(dtype=torch.float32)

                    labels_flat = labels.reshape(-1, labels.size(-1))
                    type_flat   = type_position_ids.reshape(-1)
                    loss_weights_flat = loss_weights.reshape(-1)
                    flat_task_group_ids = _build_flat_task_group_ids(micro_batch, labels)

                    # ================================================================
                    # 2. Image loss (multi-codebook, FSDP2-safe)
                    # ================================================================
                    img_pos_mask = (type_flat == 4)
                    img_labels_all = labels_flat[img_pos_mask]         # [*, C]
                    img_pos_indices = img_pos_mask.nonzero(as_tuple=False).reshape(-1)

                    # Must match the forward pass exactly.
                    img_token_valid = (img_labels_all != -100).any(dim=-1)
                    img_labels = img_labels_all[img_token_valid]       # [N_img, C]
                    img_valid_pos_indices = img_pos_indices[img_token_valid]

                    if logits_img.shape[0] == 0:
                        loss_img_sum = logits_img.sum() * 0
                        per_img_loss = None
                        num_valid_img = torch.zeros(
                            (), device=logits_img.device, dtype=torch.long
                        )
                    else:
                        assert logits_img.shape[0] == img_labels.shape[0], \
                            f"img logits/labels mismatch: {logits_img.shape[0]} vs {img_labels.shape[0]}"

                        logits_flat = logits_img.flatten(0,1)
                        labels_flat_img = img_labels.flatten()
                        per_img_loss = torch.nn.functional.cross_entropy(
                            # logits_flat,
                            logits_flat.float(),
                            labels_flat_img,
                            reduction="none",
                            ignore_index=-100
                        )

                        per_img_loss = per_img_loss.view_as(img_labels)
                        loss_img_sum = per_img_loss.sum()
                        num_valid_img = (img_labels != -100).sum()


                    num_codebooks = img_labels_all.shape[1]
                    num_valid_img_normalized = num_valid_img / num_codebooks
                    loss_img_sum_normalized = loss_img_sum / num_codebooks

                    text_pos_mask = (type_flat == 3)
                    text_labels_all = labels_flat[text_pos_mask, 0]    # [*]
                    text_weights_all = loss_weights_flat[text_pos_mask]
                    text_pos_indices = text_pos_mask.nonzero(as_tuple=False).reshape(-1)
                    valid_mask_text = text_labels_all != -100
                    text_labels = text_labels_all[valid_mask_text]     # [N_text]
                    text_weights = text_weights_all[valid_mask_text]
                    text_valid_pos_indices = text_pos_indices[valid_mask_text]
                    
                    assert logits_text.shape[0] == text_labels.shape[0], \
                        f"text logits/labels mismatch: {logits_text.shape[0]} vs {text_labels.shape[0]}"

                    if logits_text.shape[0] == 0:
                        loss_text_sum = torch.zeros(
                            (), device=logits_text.device, dtype=logits_text.dtype
                        )
                        per_text_loss = None
                        num_valid_text = torch.zeros(
                            (), device=logits_text.device, dtype=torch.float32
                        )
                    else:
                        per_text_loss = torch.nn.functional.cross_entropy(
                            logits_text.float(), text_labels, reduction="none", ignore_index=-100
                        )
                        loss_text_sum = (per_text_loss * text_weights).sum()
                        num_valid_text = text_weights.sum()

                    if per_img_loss is not None and img_valid_pos_indices.numel() > 0:
                        img_loss_per_position = per_img_loss.sum(dim=-1) / num_codebooks
                        img_count_per_position = (
                            (img_labels != -100).sum(dim=-1).to(img_loss_per_position.dtype)
                            / num_codebooks
                        )
                        _accumulate_task_loss_stats(
                            task_loss_sums,
                            task_loss_counts,
                            flat_task_group_ids[img_valid_pos_indices],
                            img_loss_per_position * args.train.image_loss_weight,
                            img_count_per_position,
                        )
                    if per_text_loss is not None and text_valid_pos_indices.numel() > 0:
                        _accumulate_task_loss_stats(
                            task_loss_sums,
                            task_loss_counts,
                            flat_task_group_ids[text_valid_pos_indices],
                            per_text_loss * text_weights * args.train.text_loss_weight,
                            text_weights,
                        )

                    # Token-level accuracy on the same supervised positions as CE.
                    # Image: per-codebook argmax; Text: lm_head argmax. No grad needed.
                    mb_img_correct = 0.0
                    mb_text_correct = 0.0
                    with torch.no_grad():
                        if logits_img.shape[0] > 0:
                            # logits_img: [N_img, C, V], img_labels: [N_img, C]
                            img_pred = logits_img.argmax(dim=-1)
                            img_valid = img_labels != -100
                            mb_img_correct = float(
                                ((img_pred == img_labels) & img_valid).sum().item()
                            )
                        if logits_text.shape[0] > 0:
                            text_pred = logits_text.argmax(dim=-1)
                            mb_text_correct = float(
                                ((text_pred == text_labels).to(text_weights.dtype) * text_weights).sum().item()
                            )
                    
                    # ---- FSDP2-safe dummy: keep empty local heads attached to the loss graph ----
                    dummy = logits_img.float().sum() * 0.0 + logits_text.float().sum() * 0.0
                    dummy = dummy + head_dummy


                    # ---- MoE balance loss ----
                    moe_loss = 0.0
                    if router_outputs is not None:
                        num_experts = getattr(model.config, "num_experts", 0)
                        if num_experts > 0:
                            # Decide first, so the .item() calls below are skipped
                            # when nothing will be logged.
                            relative_step = global_step - resume_global_step
                            need_debug = (relative_step <= 3 or (global_step % 1000 == 1))
                            
                            if need_debug:
                                # Collect per-layer values only when logging.
                                moe_losses_per_layer = []
                                for router_logits, topk_idx in router_outputs:
                                    layer_moe_loss = load_balancing_loss_func(router_logits, num_experts, topk_idx)
                                    moe_losses_per_layer.append(layer_moe_loss.item())
                                    moe_loss += layer_moe_loss
                                moe_loss = moe_loss / len(router_outputs)
                                
                                moe_loss_raw = moe_loss.item()
                                logger.info_rank0(
                                    f"[rank={args.train.local_rank}] MoE Debug: "
                                    f"num_experts={num_experts}, "
                                    f"moe_layers={len(router_outputs)}, "
                                    f"moe_loss(mean)={moe_loss_raw:.4f}, "
                                    f"per_layer={[f'{x:.4f}' for x in moe_losses_per_layer[:5]]}..."
                                )
                            else:
                                # Steady state: accumulate moe_loss only.
                                for router_logits, topk_idx in router_outputs:
                                    layer_moe_loss = load_balancing_loss_func(router_logits, num_experts, topk_idx)
                                    moe_loss += layer_moe_loss
                                moe_loss = moe_loss / len(router_outputs)
                    image_loss_weighted = loss_img_sum_normalized * args.train.image_loss_weight
                    text_loss_weighted = loss_text_sum * args.train.text_loss_weight
                    total_loss_sum = image_loss_weighted + text_loss_weighted
                    total_count = (num_valid_img_normalized + num_valid_text).to(total_loss_sum.dtype)

                    safe_count = torch.clamp(total_count, min=1)
                    mean_loss_main = total_loss_sum / safe_count

                    # Weight by the step's global token count: mean_loss_main is
                    # this pack's per-token mean, so scaling it by the pack's token
                    # count and dividing by G_local sums to Σloss / Σtokens.
                    _tok_w = _mb_token_counts[_mb_idx] / _G_local
                    loss = (mean_loss_main + moe_loss * args.train.moe_balance_loss_weight) * _tok_w + dummy
                    if global_step <= 100 and not torch.isfinite(loss.detach()).all().item():
                        _moe_loss_val = (
                            moe_loss.detach().float().item()
                            if isinstance(moe_loss, torch.Tensor) else float(moe_loss)
                        )
                        logger.info_rank0(
                            f"[NaN-GUARD step={global_step}] non-finite loss before backward: "
                            f"loss={loss.detach().float().item()}, "
                            f"mean_loss_main={mean_loss_main.detach().float().item()}, "
                            f"moe_loss={_moe_loss_val}, "
                            f"dummy={dummy.detach().float().item()}, "
                            f"total_count={total_count.detach().float().item()}, "
                            f"num_valid_img_norm={num_valid_img_normalized.detach().float().item()}, "
                            f"num_valid_text={num_valid_text.detach().float().item()}"
                        )

                del logits_img, logits_text
                if router_outputs is not None:
                    del router_outputs
                    router_outputs = None

                with model_bwd_context:
                    loss.backward()

                # ================================================================
                # 5. Gradient check: inspect the raw grad norm before clipping.
                # ================================================================
                raw_grad_norm, max_grad_val, worst_param = _compute_grad_stats(model)
                relative_step = global_step - resume_global_step
                need_grad_debug = (
                    relative_step <= 5
                    or (relative_step % 100 == 0)
                    or not math.isfinite(raw_grad_norm)
                    or raw_grad_norm > 1e4
                )
                if need_grad_debug:
                    logger.info_rank0(
                        f"[DEBUG step={global_step}] raw_grad_norm={raw_grad_norm:.4e}, "
                        f"max_grad_val={max_grad_val:.4e}, worst_param={worst_param}"
                    )
                    if not math.isfinite(raw_grad_norm):
                        # Report the first gradient tensor that is not finite.
                        for name, p in model.named_parameters():
                            if p.grad is None:
                                continue
                            g = p.grad.detach()
                            g_local = g
                            to_local = getattr(g_local, "to_local", None)
                            if callable(to_local):
                                try:
                                    g_local = to_local()
                                except Exception:
                                    g_local = g
                            if g_local.numel() == 0:
                                continue
                            if torch.isinf(g_local).any() or torch.isnan(g_local).any():
                                g_abs_max = float(g_local.abs().max().item())
                                logger.info_rank0(
                                    f"[DEBUG step={global_step}] detected NaN/Inf in grad of {name}, "
                                    f"grad_abs_max={g_abs_max:.4e}"
                                )
                                break

                total_loss += loss.item()
                # Accumulate raw loss sums and valid-token counts; the global
                # sum/count reduction happens after the micro-batch loop.
                total_loss_img_sum += loss_img_sum_normalized.item()
                total_loss_text_sum += loss_text_sum.item()
                total_img_count += num_valid_img_normalized.item()
                total_text_count += num_valid_text.item()
                # Match loss_img denom: normalize codebook-token corrects by num_codebooks.
                total_img_correct += mb_img_correct / float(num_codebooks)
                total_text_correct += mb_text_correct
                moe_loss_val = moe_loss.item() if isinstance(moe_loss, torch.Tensor) else float(moe_loss)
                moe_loss_normalized = moe_loss_val * args.train.moe_balance_loss_weight / len(micro_batches)
                total_moe_loss += moe_loss_normalized

                del micro_batch

                # ================================================================
                # 6. sync + FSDP2-safe grad fix
                # ================================================================
            synchronize()   # dist.barrier()


                # ================================================================
                # 7. clip grad
                # ================================================================
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            _gn_val = grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else float(grad_norm)
            if not math.isfinite(_gn_val):
                logger.info_rank0(
                    f"[NaN-GUARD step={global_step}] grad_norm={_gn_val}, "
                    f"skipping optimizer.step() to protect weights"
                )
                optimizer.zero_grad()
            else:
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            task_reduce_values = tuple(task_loss_sums[name] for name in TASK_LOSS_NAMES) + tuple(
                task_loss_counts[name] for name in TASK_LOSS_NAMES
            )
            reduced_values = all_reduce(
                (total_loss, grad_norm, total_loss_img_sum, total_loss_text_sum, total_moe_loss,
                 total_img_count, total_text_count, total_img_correct, total_text_correct,
                 *task_reduce_values),
                group=get_parallel_state().fsdp_group
            )
            (
                total_loss,
                grad_norm,
                total_loss_img_sum,
                total_loss_text_sum,
                total_moe_loss,
                total_img_count,
                total_text_count,
                total_img_correct,
                total_text_correct,
                *task_reduced_values,
            ) = reduced_values
            # Global token weighting: sum/count. all_reduce averages, which divides
            # numerator and denominator by world_size, so the ratio is unchanged.
            image_acc = total_img_correct / total_img_count if total_img_count > 0 else 0.0
            text_acc = total_text_correct / total_text_count if total_text_count > 0 else 0.0
            total_loss_img_sum = total_loss_img_sum / total_img_count if total_img_count > 0 else 0.0
            total_loss_text_sum = total_loss_text_sum / total_text_count if total_text_count > 0 else 0.0
            num_task_names = len(TASK_LOSS_NAMES)
            task_loss_sums = {
                name: task_reduced_values[i]
                for i, name in enumerate(TASK_LOSS_NAMES)
            }
            task_loss_counts = {
                name: task_reduced_values[num_task_names + i]
                for i, name in enumerate(TASK_LOSS_NAMES)
            }
            task_loss_means = {
                name: (task_loss_sums[name] / task_loss_counts[name] if task_loss_counts[name] > 0 else 0.0)
                for name in TASK_LOSS_NAMES
            }
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())

            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            samples_per_second = args.train.global_batch_size / delta_time
            tokens_per_second_m = train_metrics.get("tokens_per_second(M)", 0.0)
            task_loss_postfix = ", ".join(
                f"{name}: {task_loss_means[name]:.2f}"
                for name in TASK_LOSS_NAMES
                if task_loss_counts[name] > 0
            )
            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}, samples/s: {samples_per_second:.2f}, "
                f"tokens/s: {tokens_per_second_m:.2f}M, "
                f"loss_img: {total_loss_img_sum:.2f}, loss_text: {total_loss_text_sum:.2f}, "
                f"image_acc: {image_acc:.4f}, text_acc: {text_acc:.4f}, "
                f"moe_loss: {total_moe_loss:.4f}, "
                f"task_loss: {task_loss_postfix}"
            )
            data_loader_tqdm.update()
            if args.train.global_rank == 0:
                # Record to tensorboard
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", total_loss, global_step)
                    tb_writer.add_scalar("train/loss_img", total_loss_img_sum, global_step)
                    tb_writer.add_scalar("train/loss_text", total_loss_text_sum, global_step)
                    tb_writer.add_scalar("train/image_acc", image_acc, global_step)
                    tb_writer.add_scalar("train/text_acc", text_acc, global_step)
                    tb_writer.add_scalar("train/moe_loss", total_moe_loss, global_step)
                    for task_name in TASK_LOSS_NAMES:
                        if task_loss_counts[task_name] > 0:
                            tb_writer.add_scalar(f"train/task_loss/{task_name}", task_loss_means[task_name], global_step)
                            tb_writer.add_scalar(f"train/task_loss_count/{task_name}", task_loss_counts[task_name], global_step)
                    tb_writer.add_scalar("train/lr", lr, global_step)
                    tb_writer.add_scalar("train/grad_norm", grad_norm, global_step)
                

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"epoch{epoch}-global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                        "data_objective": DATA_OBJECTIVE,
                    },
                }
                Checkpointer.save(save_checkpoint_path, state)
                dist.barrier()
                if args.train.global_rank == 0 and save_checkpoint_path is not None:
                    if args.train.save_hf_weights:
                        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                        model_state_dict = ckpt_to_state_dict(
                            save_checkpoint_path=save_checkpoint_path,
                            output_dir=args.train.output_dir,
                            ckpt_manager=args.train.ckpt_manager,
                        )
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
                    remove_early_ckpts(args.train.save_checkpoint_path, max_keep=args.train.ckpt_max_keep)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"epoch{epoch}-global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "data_objective": DATA_OBJECTIVE,
                },
            }
            Checkpointer.save(save_checkpoint_path, state)
            dist.barrier()
            if args.train.global_rank == 0 and save_checkpoint_path is not None:
                if args.train.save_hf_weights:
                    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                    model_state_dict = ckpt_to_state_dict(
                        save_checkpoint_path=save_checkpoint_path,
                        output_dir=args.train.output_dir,
                        ckpt_manager=args.train.ckpt_manager,
                    )
                    save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                    logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
                remove_early_ckpts(args.train.save_checkpoint_path, max_keep=args.train.ckpt_max_keep)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    if tb_writer is not None:
        tb_writer.close()
        logger.info_rank0("TensorBoard writer closed")
    helper.empty_cache()
    # HF weights are exported at each save point when train.save_hf_weights is
    # set (see the periodic and epoch-end branches above), so there is nothing
    # left to write here.
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
