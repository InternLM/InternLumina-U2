"""FSDP2 all-gather copy-out version-counter fix, backported to torch 2.6.0.

torch 2.6.0's ``foreach_all_gather_copy_out`` lets ``split_with_sizes_copy`` bump
the autograd version counter of every unsharded (all-gathered) parameter tensor
on each re-gather. With activation recompute / repeated unshard-reshard cycles,
autograd's saved-tensor version bookkeeping then goes out of sync and backward
can silently route reads/writes through stale storage. Observed symptom in this
repo (2026-09-20, flat-vis ablation): after the first backward, the unsharded
embedding weights on one rank contain sparse non-finite garbage at
deterministic, data-dependent positions -- forward-only runs are clean.

Upstream fixed this in torch 2.7 by wrapping the copy-out in
``torch.autograd._unsafe_preserve_version_counter`` (which in 2.7 accepts a
tuple). 2.6's context manager only takes a single tensor, so this backport
saves and restores ``_version`` manually via
``torch._C._autograd._unsafe_set_version_counter``.

Applied automatically on ``veomni.distributed.fsdp2`` import (see __init__).
"""

from typing import List, Tuple, cast

import torch
import torch.distributed as dist

from ...utils import logging


logger = logging.get_logger(__name__)

_PATCHED_FLAG = "_veomni_vc_patch_applied"


def _patched_foreach_all_gather_copy_out(
    all_gather_result,
    fsdp_params,
    group: dist.ProcessGroup,
) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import _get_device_handle
    from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled
    from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState

    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_input_split_sizes,
    ) = all_gather_result
    _dtype, device = all_gather_output.dtype, all_gather_output.device
    device_handle = _get_device_handle(device.type)
    if all_gather_event is not None:  # sync op
        device_handle.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        all_gather_work.wait()
    world_size, device = group.size(), all_gather_output.device

    split_with_sizes_out: List[torch.Tensor] = []
    shard_i_copy_infos: List[Tuple[object, List[torch.Tensor]]] = []
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
    ):
        force_recreate = compiled_autograd_enabled()
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels,
            all_gather_input_dtypes,
            world_size,
            device,
            force_recreate=force_recreate,
        )
        if not force_recreate:
            fsdp_param.alloc_all_gather_outputs()
        param_all_gather_outputs = fsdp_param.all_gather_outputs
        if fsdp_param.fsdp_placement.dim != 0:
            param_all_gather_outputs = [
                torch.empty_like(t) for t in param_all_gather_outputs
            ]
            shard_i_copy_infos.append((fsdp_param, param_all_gather_outputs))
        split_with_sizes_out.extend(param_all_gather_outputs)

    all_gather_output = all_gather_output.view(world_size, -1)
    if all_gather_output.dtype == torch.uint8:
        out = [t.view(world_size, -1).view(torch.uint8) for t in split_with_sizes_out]
    else:
        out = [t.view(world_size, -1) for t in split_with_sizes_out]

    # ---- backport of the 2.7 fix: keep version counters unchanged ----
    vc_targets = [t for t in out if not t.is_inference()]
    vc_saved = [t._version for t in vc_targets]
    torch.ops.fsdp.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=1, out=out
    )
    for t, v in zip(vc_targets, vc_saved):
        torch._C._autograd._unsafe_set_version_counter(t, v)

    for fsdp_param, param_all_gather_outputs in shard_i_copy_infos:
        shard_dim = fsdp_param.fsdp_placement.dim
        for param_all_gather_output, target_all_gather_output in zip(
            param_all_gather_outputs, fsdp_param.all_gather_outputs
        ):
            padded_sharded_size = (
                fsdp_param.padded_sharded_param_size
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(
                    torch.Tensor, fsdp_param._sharded_post_forward_param_data
                ).size()
            )
            pre_param_size = list(padded_sharded_size)
            pre_param_size[0] *= world_size
            chunks = torch.chunk(
                param_all_gather_output.view(pre_param_size), world_size, dim=0
            )
            post_param_size = list(padded_sharded_size)
            post_param_size[shard_dim] *= world_size
            cat_out = target_all_gather_output.view(post_param_size)
            vc = (
                None
                if target_all_gather_output.is_inference()
                else target_all_gather_output._version
            )
            torch.cat(chunks, dim=shard_dim, out=cat_out)
            if vc is not None:
                torch._C._autograd._unsafe_set_version_counter(
                    target_all_gather_output, vc
                )


def apply_patch() -> None:
    """Monkeypatch foreach_all_gather_copy_out at both its definition and use sites."""
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group

    if getattr(_fsdp_collectives, _PATCHED_FLAG, False):
        return
    if not torch.__version__.startswith("2.6."):
        logger.info_rank0(
            f"fsdp2 VC patch: torch {torch.__version__} != 2.6.x, skipping "
            "(upstream already carries the fix from 2.7 on)"
        )
        return
    _fsdp_collectives.foreach_all_gather_copy_out = _patched_foreach_all_gather_copy_out
    # _fsdp_param_group imported the symbol by name; patch its reference too.
    _fsdp_param_group.foreach_all_gather_copy_out = _patched_foreach_all_gather_copy_out
    setattr(_fsdp_collectives, _PATCHED_FLAG, True)
    logger.info_rank0(
        "✅ fsdp2 VC patch applied: foreach_all_gather_copy_out now preserves "
        "all-gather output version counters (torch 2.6 backport of the 2.7 fix)"
    )
