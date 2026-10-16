"""FSQ-distance biased masking for image-edit answers.

AToken recodes globally, so ``(clean != ref).any(-1)`` is almost always True and
the old biased branch collapsed to "mask the whole output image".  FSQ codes are
metric: a one-level jitter across all 48 dims is L2≈3.46, while a real content
change is larger.  This module keeps the cosine mask *count* identical to t2i
and only changes *which* tokens are drawn.

Env:
  EDIT_BIASED_MASK_PROB   default 0.5   (0 disables)
  EDIT_CHANGE_THRESHOLD   default 3.5   (gate on changed_frac only)
  EDIT_FRAC_LO            default 0.005
  EDIT_FRAC_HI            default 0.50  (global restyles fall back to uniform)
  ATOKEN_FSQ_CODEBOOK     optional .pt with ``{'codebook': [4096,6], 'levels': ...}``
"""

from __future__ import annotations

import math
import os
import random
from functools import lru_cache
from typing import List, Optional, Sequence

import torch

FSQ_LEVELS = (4, 4, 4, 4, 4, 4)
FSQ_CODEBOOK_SIZE = 4096


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def biased_mask_prob() -> float:
    return _env_float("EDIT_BIASED_MASK_PROB", 0.5)


def change_threshold() -> float:
    return _env_float("EDIT_CHANGE_THRESHOLD", 3.5)


def frac_lo() -> float:
    return _env_float("EDIT_FRAC_LO", 0.005)


def frac_hi() -> float:
    return _env_float("EDIT_FRAC_HI", 0.50)


def biased_mask_requested(task_type: Optional[str], reference_tokens) -> bool:
    del task_type  # every edit task in this release is eligible
    if reference_tokens is None:
        return False
    return random.random() < biased_mask_prob()


def cosine_num_to_mask(token_len: int) -> int:
    if token_len <= 0:
        return 0
    r = random.uniform(0, 1)
    mask_ratio = 1.0 if token_len <= 5 else math.cos(r * math.pi / 2)
    return max(1, int(token_len * mask_ratio))


def _default_fsq_codebook() -> torch.Tensor:
    levels = torch.tensor(FSQ_LEVELS, dtype=torch.int32)
    basis = torch.cumprod(torch.tensor([1, *FSQ_LEVELS[:-1]], dtype=torch.int32), dim=0)
    half = levels // 2
    idx = torch.arange(FSQ_CODEBOOK_SIZE)
    codes_nc = (idx.unsqueeze(-1) // basis) % levels
    return (codes_nc - half).float() / half.float()


@lru_cache(maxsize=1)
def fsq_codebook() -> Optional[torch.Tensor]:
    path = (os.environ.get("ATOKEN_FSQ_CODEBOOK") or "").strip()
    try:
        if path:
            obj = torch.load(path, map_location="cpu")
            cb = obj["codebook"] if isinstance(obj, dict) else obj
            cb = cb.detach().float().cpu()
            if cb.ndim != 2 or cb.shape[0] != FSQ_CODEBOOK_SIZE or cb.shape[1] != len(FSQ_LEVELS):
                return None
            return cb
        return _default_fsq_codebook()
    except Exception:
        return None


def _codebook_indices(tokens: torch.Tensor) -> Optional[torch.Tensor]:
    idx = tokens.long()
    if idx.numel() == 0:
        return idx
    if int(idx.min()) < 0:
        return None
    if int(idx.max()) >= FSQ_CODEBOOK_SIZE:
        return None
    return idx


def fsq_embedding_l2(clean_tokens: torch.Tensor, reference_tokens: torch.Tensor) -> Optional[torch.Tensor]:
    """Per-position L2 in concatenated FSQ embedding space, shape [N]."""
    cb = fsq_codebook()
    if cb is None:
        return None
    clean = _codebook_indices(clean_tokens)
    ref = _codebook_indices(reference_tokens)
    if clean is None or ref is None:
        return None
    if clean.shape != ref.shape:
        return None
    if clean.ndim == 1:
        clean = clean.unsqueeze(-1)
        ref = ref.unsqueeze(-1)
    e_clean = cb[clean]
    e_ref = cb[ref]
    return (e_clean - e_ref).reshape(clean.shape[0], -1).norm(dim=-1)


def _try_biased_indices(clean_tokens: torch.Tensor, reference_tokens: torch.Tensor, num_to_mask: int) -> Optional[List[int]]:
    try:
        if tuple(clean_tokens.shape) != tuple(reference_tokens.shape):
            return None
        dist = fsq_embedding_l2(clean_tokens, reference_tokens)
        if dist is None or dist.numel() != clean_tokens.shape[0]:
            return None
        changed_frac = float((dist > change_threshold()).float().mean())
        if not (frac_lo() <= changed_frac <= frac_hi()):
            return None
        token_len = int(dist.numel())
        if num_to_mask >= token_len:
            return list(range(token_len))
        weights = dist.clamp(min=0).to(dtype=torch.float64).pow(2).add(1e-6)
        chosen = torch.multinomial(weights, num_to_mask, replacement=False)
        return chosen.tolist()
    except Exception:
        return None


def select_mask_indices(
    clean_tokens: torch.Tensor,
    reference_tokens: Optional[torch.Tensor] = None,
    use_biased_mask: bool = False,
) -> List[int]:
    """Cosine-scheduled k indices. Biased draw falls back to uniform silently."""
    if not isinstance(clean_tokens, torch.Tensor):
        clean_tokens = torch.tensor(clean_tokens, dtype=torch.int64)
    token_len = int(clean_tokens.shape[0])
    if token_len <= 0:
        return []
    num_to_mask = cosine_num_to_mask(token_len)
    if use_biased_mask and reference_tokens is not None:
        if not isinstance(reference_tokens, torch.Tensor):
            reference_tokens = torch.tensor(reference_tokens, dtype=torch.int64)
        biased = _try_biased_indices(clean_tokens, reference_tokens, num_to_mask)
        if biased is not None:
            return biased
    return random.sample(range(token_len), num_to_mask)


def apply_image_mask(
    clean_tokens: torch.Tensor,
    mask_id: int,
    reference_tokens: Optional[torch.Tensor] = None,
    use_biased_mask: bool = False,
    indices_to_mask: Optional[Sequence[int]] = None,
):
    """Return (noisy, labels) for a [N, C] image-token tensor."""
    if not isinstance(clean_tokens, torch.Tensor):
        clean_tokens = torch.tensor(clean_tokens, dtype=torch.int64)
    if clean_tokens.shape[0] <= 0:
        return clean_tokens.clone(), torch.full_like(clean_tokens, -100)
    if indices_to_mask is None:
        indices_to_mask = select_mask_indices(clean_tokens, reference_tokens, use_biased_mask)
    labels = torch.full_like(clean_tokens, -100)
    noisy = clean_tokens.clone()
    for idx in indices_to_mask:
        labels[idx] = clean_tokens[idx]
        noisy[idx] = mask_id
    return noisy, labels
