"""Muon entrypoint for the multitask SFT task.

The training loop stays in ``train_sft``.  This module only supplies Muon-specific
arguments, optimizer construction, and checkpoint policy through process-local
injection before calling that task's ``main``.

Launch via ``train_1024_sft_muon.sh`` (which routes the base launcher to this
entry), or directly with torchrun using the same config/CLI as ``train_sft.py``
plus the ``--train.muon_*`` options below.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from tasks import train_sft as base
from veomni.optim.muon import Muon, build_muon_param_groups
from veomni.optim.optimizer import _make_param_groups_for_subset
from veomni.utils.arguments import parse_args as _veomni_parse_args


_ACTIVE_ARGS: Optional["Arguments"] = None
_ORIGINAL_BUILD_CHECKPOINTER = base.build_checkpointer
_MUON_OPTIMIZER_SCHEMA_VERSION = 1
_MUON_MOMENTUM_BUFFER_FORMAT = "additive_v1"


@dataclass
class MuonTrainingArguments(base.InternLuminaU2TrainingArguments):
    optimizer: Literal["muon"] = field(
        default="muon",
        metadata={"help": "Use the Per-Head Muon optimizer."},
    )
    muon_checkpoint_mode: Literal["model_only", "resume"] = field(
        default="model_only",
        metadata={
            "help": "model_only loads model+extra state from any DCP with a fresh Muon; "
            "resume requires a Muon-marked DCP and restores optimizer state."
        },
    )
    muon_momentum: float = field(default=0.95, metadata={"help": "Muon momentum."})
    muon_nesterov: bool = field(default=True, metadata={"help": "Use Nesterov Muon momentum."})
    muon_ns_steps: int = field(default=5, metadata={"help": "Newton-Schulz iteration count."})
    muon_rms_scale: float = field(default=0.2, metadata={"help": "Muon RMS matching scale."})
    muon_per_head: bool = field(default=True, metadata={"help": "Orthogonalize Q/K/V per head."})
    muon_per_head_o_proj: bool = field(
        default=False,
        metadata={"help": "Also orthogonalize attention output projections per head."},
    )
    muon_gather_dtype: str = field(
        default="",
        metadata={"help": "Matrix gather dtype: empty, bf16/bfloat16, or fp32."},
    )
    muon_owner_alltoall: bool = field(
        default=False,
        metadata={
            "help": "Experimental owner-based all-to-all path. Keep false unless the target NCCL/FSDP2 GPU smoke passes."
        },
    )
    muon_allow_legacy_resume: bool = field(
        default=False,
        metadata={
            "help": "Allow one-time optimizer resume from an old Muon DCP that has only optimizer_type=muon."
        },
    )
    muon_adamw_patterns: str = field(
        default="",
        metadata={"help": "Comma-separated parameter-name patterns forced to AdamW."},
    )
    muon_muon_patterns: str = field(
        default="",
        metadata={"help": "Comma-separated parameter-name patterns forced to Muon."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.expert_parallel_size != 1:
            raise ValueError("Muon requires train.expert_parallel_size=1; EP is not supported.")
        if not 0.0 <= self.muon_momentum < 1.0:
            raise ValueError(f"muon_momentum must be in [0, 1), got {self.muon_momentum}.")
        if self.muon_ns_steps <= 0:
            raise ValueError(f"muon_ns_steps must be positive, got {self.muon_ns_steps}.")
        if self.muon_rms_scale <= 0:
            raise ValueError(f"muon_rms_scale must be positive, got {self.muon_rms_scale}.")

        gather_dtype = self.muon_gather_dtype.strip().lower()
        if gather_dtype not in ("", "bf16", "bfloat16", "fp32"):
            raise ValueError(
                "muon_gather_dtype must be empty, bf16, bfloat16, or fp32; "
                f"got {self.muon_gather_dtype!r}."
            )

        if self.load_checkpoint_path:
            if self.ckpt_manager != "dcp":
                raise ValueError("Muon checkpoint loading requires train.ckpt_manager=dcp.")
            if self.reload_model_checkpoint_path:
                raise ValueError(
                    "reload_model_checkpoint_path bypasses the Muon checkpoint policy and cannot be combined "
                    "with load_checkpoint_path in this entrypoint."
                )
            if self.muon_checkpoint_mode == "model_only" and not self.resume_reset_lr_schedule:
                raise ValueError(
                    "muon_checkpoint_mode=model_only requires resume_reset_lr_schedule=true so an AdamW "
                    "scheduler is never restored into the fresh Muon optimizer."
                )


@dataclass
class Arguments:
    model: base.InternLuminaU2ModelArguments = field(default_factory=base.InternLuminaU2ModelArguments)
    data: base.InternLuminaU2DataArguments = field(default_factory=base.InternLuminaU2DataArguments)
    train: MuonTrainingArguments = field(default_factory=MuonTrainingArguments)


def _parse_args(_rootclass) -> Arguments:
    global _ACTIVE_ARGS
    args = _veomni_parse_args(Arguments)
    _ACTIVE_ARGS = args
    return args


def _patterns(value: str) -> Optional[List[str]]:
    patterns = [item.strip() for item in value.split(",") if item.strip()]
    return patterns or None


def _gather_dtype(value: str) -> Optional[torch.dtype]:
    return {
        "": None,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[value.strip().lower()]


def _validate_lr_overrides(
    model: torch.nn.Module,
    param_lr_overrides: Optional[Mapping[str, float]],
) -> None:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    for name, override_lr in (param_lr_overrides or {}).items():
        if float(override_lr) <= 0:
            raise ValueError(f"LR override for {name!r} must be positive, got {override_lr}.")
        if name not in trainable_names:
            raise ValueError(f"LR override parameter {name!r} is not trainable or does not exist.")
        base.logger.info_rank0(f"Parameter {name!r} uses lr={float(override_lr):.6e}")


def _validate_groups(
    model: torch.nn.Module,
    groups: Sequence[Dict[str, Any]],
    require_per_head: bool,
) -> None:
    expected = [param for param in model.parameters() if param.requires_grad]
    actual = [param for group in groups for param in group.get("params", [])]
    actual_ids = [id(param) for param in actual]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Muon optimizer parameter groups contain duplicate parameters.")

    expected_ids = {id(param) for param in expected}
    actual_id_set = set(actual_ids)
    if expected_ids != actual_id_set:
        missing = len(expected_ids - actual_id_set)
        unexpected = len(actual_id_set - expected_ids)
        raise ValueError(
            "Muon optimizer parameter groups do not exactly cover trainable parameters: "
            f"missing={missing}, unexpected={unexpected}."
        )

    if require_per_head and not any(
        group.get("use_muon", False) and int(group.get("head_rows", 0)) > 0 for group in groups
    ):
        raise ValueError(
            "muon_per_head=true, but no per-head Muon group was found. Check the model attention "
            "head_dim and Q/K/V projection module names."
        )


def _build_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "muon",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    param_lr_overrides: Optional[Mapping[str, float]] = None,
) -> torch.optim.Optimizer:
    del fused
    if _ACTIVE_ARGS is None:
        raise RuntimeError("Muon arguments must be parsed before building the optimizer.")
    if optimizer_type != "muon":
        raise ValueError(f"This entrypoint only supports optimizer_type='muon', got {optimizer_type!r}.")
    if param_groups is not None:
        raise ValueError("Explicit param_groups are not supported by this thin Muon entrypoint.")

    _validate_lr_overrides(model, param_lr_overrides)
    base_groups = _make_param_groups_for_subset(
        model,
        model.parameters(),
        lr,
        weight_decay,
        no_decay_modules,
        no_decay_params,
        param_lr_overrides,
    )
    train_args = _ACTIVE_ARGS.train
    muon_groups, group_lines = build_muon_param_groups(
        model,
        base_groups,
        per_head=train_args.muon_per_head,
        per_head_o_proj=train_args.muon_per_head_o_proj,
        extra_adamw_patterns=_patterns(train_args.muon_adamw_patterns),
        extra_muon_patterns=_patterns(train_args.muon_muon_patterns),
    )
    _validate_groups(model, muon_groups, require_per_head=train_args.muon_per_head)

    muon_param_count = sum(
        param.numel()
        for group in muon_groups
        if group.get("use_muon", False)
        for param in group["params"]
    )
    adamw_param_count = sum(
        param.numel()
        for group in muon_groups
        if not group.get("use_muon", False)
        for param in group["params"]
    )
    base.logger.info_rank0("Muon optimizer parameter groups:\n" + "\n".join(group_lines))
    base.logger.info_rank0(
        f"Muon parameter coverage: muon={muon_param_count:,}, adamw={adamw_param_count:,}, "
        f"groups={len(muon_groups)}"
    )

    return Muon(
        muon_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        momentum=train_args.muon_momentum,
        nesterov=train_args.muon_nesterov,
        ns_steps=train_args.muon_ns_steps,
        rms_scale=train_args.muon_rms_scale,
        gather_dtype=_gather_dtype(train_args.muon_gather_dtype),
        owner_alltoall=train_args.muon_owner_alltoall,
    )


def _optimizer_fingerprint(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> str:
    """Hash the optimizer-state schema by parameter FQN, global shape and state family.

    LR, weight decay, momentum, NS settings and per-head block size deliberately stay out:
    they do not change the state tensors and may be changed explicitly on resume.  A
    Muon/AdamW routing change does change the tensor family and must never be accepted
    silently (``momentum_buffer`` versus ``step/exp_avg/exp_avg_sq``).
    """
    if not isinstance(optimizer, Muon):
        raise TypeError(f"Muon checkpoint metadata requires a Muon optimizer, got {type(optimizer).__name__}.")

    name_by_id = {id(param): name for name, param in model.named_parameters()}
    entries: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        family = "muon_additive" if group.get("use_muon", True) else "adamw"
        for param in group["params"]:
            param_id = id(param)
            if param_id in seen:
                raise ValueError("Muon optimizer fingerprint found a parameter in more than one group.")
            seen.add(param_id)
            name = name_by_id.get(param_id)
            if name is None:
                raise ValueError("Muon optimizer fingerprint found a parameter absent from model.named_parameters().")
            entries.append(
                {
                    "fqn": name,
                    "global_shape": [int(size) for size in param.shape],
                    "state_family": family,
                }
            )

    entries.sort(key=lambda item: item["fqn"])
    payload = {
        "schema_version": _MUON_OPTIMIZER_SCHEMA_VERSION,
        "parameters": entries,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extra_state_path(checkpoint_path: str, rank: int) -> str:
    return os.path.join(checkpoint_path, "extra_state", f"extra_state_rank_{rank}.pt")


def _validate_resume_marker(checkpoint_path: str, state: Dict[str, Any]) -> None:
    if _ACTIVE_ARGS is None:
        raise RuntimeError("Muon arguments must be parsed before validating a checkpoint.")
    expected_fingerprint = _optimizer_fingerprint(state["model"], state["optimizer"])
    allow_legacy = _ACTIVE_ARGS.train.muon_allow_legacy_resume
    rank = dist.get_rank() if dist.is_initialized() else 0
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    try:
        extra_state = torch.load(
            _extra_state_path(checkpoint_path, rank),
            map_location="cpu",
            weights_only=False,
        )
        metadata = {
            "optimizer_type": extra_state.get("optimizer_type"),
            "schema_version": extra_state.get("muon_optimizer_schema_version"),
            "momentum_format": extra_state.get("muon_momentum_buffer_format"),
            "fingerprint": extra_state.get("muon_optimizer_fingerprint"),
        }
    except Exception as exc:  # report the same pre-load failure on every rank
        error = f"{type(exc).__name__}: {exc}"

    statuses = [(rank, metadata, error)]
    if dist.is_initialized():
        statuses = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, (rank, metadata, error))

    invalid: List[Tuple[int, str]] = []
    legacy_ranks: List[int] = []
    for status_rank, saved, status_error in statuses:
        if status_error is not None:
            invalid.append((status_rank, status_error))
            continue
        if saved.get("optimizer_type") != "muon":
            invalid.append((status_rank, f"optimizer_type={saved.get('optimizer_type')!r}"))
            continue

        schema_values = (
            saved.get("schema_version"),
            saved.get("momentum_format"),
            saved.get("fingerprint"),
        )
        if schema_values == (None, None, None):
            if allow_legacy:
                legacy_ranks.append(status_rank)
            else:
                invalid.append(
                    (
                        status_rank,
                        "legacy marker-only Muon DCP; use muon_checkpoint_mode=model_only or set "
                        "muon_allow_legacy_resume=true once",
                    )
                )
            continue
        if saved.get("schema_version") != _MUON_OPTIMIZER_SCHEMA_VERSION:
            invalid.append((status_rank, f"schema_version={saved.get('schema_version')!r}"))
        elif saved.get("momentum_format") != _MUON_MOMENTUM_BUFFER_FORMAT:
            invalid.append((status_rank, f"momentum_format={saved.get('momentum_format')!r}"))
        elif saved.get("fingerprint") != expected_fingerprint:
            invalid.append((status_rank, "optimizer routing/shape fingerprint mismatch"))

    if invalid:
        details = "; ".join(f"rank={item[0]} {item[1]}" for item in invalid[:8])
        raise ValueError(
            "muon_checkpoint_mode=resume requires a compatible Muon DCP "
            f"(schema={_MUON_OPTIMIZER_SCHEMA_VERSION}, momentum={_MUON_MOMENTUM_BUFFER_FORMAT}); {details}"
        )
    if legacy_ranks:
        base.logger.info_rank0(
            "WARNING: resuming a legacy marker-only Muon DCP because "
            "muon_allow_legacy_resume=true. The next save will add schema/fingerprint metadata; "
            "disable the legacy flag after that save."
        )


def _build_checkpointer(dist_backend: str = "fsdp1", ckpt_manager: str = "omnistore"):
    backing = _ORIGINAL_BUILD_CHECKPOINTER(dist_backend=dist_backend, ckpt_manager=ckpt_manager)

    class MuonCheckpointer:
        @classmethod
        def save(cls, path: str, state: Dict[str, Any], *args, **kwargs):
            save_state = dict(state)
            extra_state = dict(state.get("extra_state") or {})
            extra_state["optimizer_type"] = "muon"
            extra_state["muon_optimizer_schema_version"] = _MUON_OPTIMIZER_SCHEMA_VERSION
            extra_state["muon_momentum_buffer_format"] = _MUON_MOMENTUM_BUFFER_FORMAT
            extra_state["muon_optimizer_fingerprint"] = _optimizer_fingerprint(
                state["model"], state["optimizer"]
            )
            save_state["extra_state"] = extra_state
            return backing.save(path, save_state, *args, **kwargs)

        @classmethod
        def load(cls, path: str, state: Dict[str, Any], *args, **kwargs):
            if _ACTIVE_ARGS is None:
                raise RuntimeError("Muon arguments must be parsed before loading a checkpoint.")
            mode = _ACTIVE_ARGS.train.muon_checkpoint_mode
            if mode == "resume":
                _validate_resume_marker(path, state)
                return backing.load(path, state, *args, **kwargs)

            load_state = dict(state)
            load_state.pop("optimizer", None)
            result = backing.load(path, load_state, *args, **kwargs)
            loaded_state = result if isinstance(result, dict) else load_state
            state["extra_state"] = loaded_state.get("extra_state", {})
            base.logger.info_rank0(
                f"Loaded model and extra state from {path}; optimizer state was intentionally skipped "
                "(muon_checkpoint_mode=model_only)."
            )
            return state

    return MuonCheckpointer


def main() -> None:
    global _ACTIVE_ARGS
    originals = (base.Arguments, base.parse_args, base.build_optimizer, base.build_checkpointer, _ACTIVE_ARGS)
    base.Arguments = Arguments
    base.parse_args = _parse_args
    base.build_optimizer = _build_optimizer
    base.build_checkpointer = _build_checkpointer
    try:
        base.main()
    finally:
        base.Arguments, base.parse_args, base.build_optimizer, base.build_checkpointer, _ACTIVE_ARGS = originals


if __name__ == "__main__":
    main()
