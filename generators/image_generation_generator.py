# -*- coding: utf-8 -*-
"""Image generation generator.

One public entry point, ``generate_image_ar``. Optional training-free
accelerations are selected with ``accel_preset``; see
``generators/accel_presets.py``.
"""
import logging
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk
from .accel_presets import resolve_accel_preset
from .text_understanding_generator import build_t2i_attention_mask_4d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acceleration primitives (see generators/accel_presets.py for the tiering)
# ---------------------------------------------------------------------------


def _pad_additive_mask_4d(mask: torch.Tensor, real_len: int, padded_len: int) -> torch.Tensor:
    """Grow a [1,1,L,L] additive mask to [1,1,P,P] keeping padding invisible.

    Real queries cannot attend pad keys (-inf); pad queries attend real keys (0)
    purely so SDPA never sees an all-masked row. Pad outputs are never consumed:
    pad positions carry type id 0 and only type-4 hidden states are extracted.
    """
    if mask.dim() != 4 or mask.shape[-1] != real_len or mask.shape[-2] != real_len:
        raise ValueError(f"expected [1,1,{real_len},{real_len}] mask, got {tuple(mask.shape)}")
    if real_len == padded_len:
        return mask
    neg_inf = torch.finfo(mask.dtype).min
    padded = torch.full((1, 1, padded_len, padded_len), neg_inf, dtype=mask.dtype, device=mask.device)
    padded[:, :, :real_len, :real_len] = mask
    padded[:, :, real_len:, :real_len] = 0
    return padded


def _pack_additive_masks_4d(cond_mask, cond_len, uncond_mask, uncond_len, padded_len):
    """Pack two [1,1,L,L] additive masks into one [2,1,P,P] buffer, filled in place.

    Written as a single allocation because these masks dominate the generator's
    memory at 1024x1024 (O(P^2) each, and the packed decode needs two of them);
    a naive pad-then-cat would transiently hold five of them.

    Padding stays invisible exactly as in ``_pad_additive_mask_4d``: real queries
    cannot attend pad keys, pad queries may attend real keys only so SDPA never
    sees an all-masked row, and pad outputs are never consumed.
    """
    if cond_mask.dim() != 4 or uncond_mask.dim() != 4:
        raise ValueError("expected 4-D additive masks")
    dtype, device = cond_mask.dtype, cond_mask.device
    neg_inf = torch.finfo(dtype).min
    out = torch.full((2, 1, padded_len, padded_len), neg_inf, dtype=dtype, device=device)
    for row, (mask, real_len) in enumerate(((cond_mask, cond_len), (uncond_mask, uncond_len))):
        if mask.shape[-1] != real_len or mask.shape[-2] != real_len:
            raise ValueError(
                f"row {row}: expected [1,1,{real_len},{real_len}] mask, got {tuple(mask.shape)}"
            )
        out[row, :, :real_len, :real_len] = mask[0]
        if real_len < padded_len:
            out[row, :, real_len:, :real_len] = 0
    return out


def _right_pad(t: torch.Tensor, padded_len: int, dim: int = 1) -> torch.Tensor:
    if t.shape[dim] == padded_len:
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = padded_len - t.shape[dim]
    return torch.cat([t, t.new_zeros(pad_shape)], dim=dim)


def _branch_position_ids(explicit, real_len: int, padded_len: int, device) -> torch.Tensor:
    """Explicit (possibly gapped) positions for the real region, arange-continued pads."""
    if explicit is None:
        return torch.arange(padded_len, dtype=torch.long, device=device).unsqueeze(0)
    explicit = explicit.to(device)
    if explicit.dim() == 1:
        explicit = explicit.unsqueeze(0)
    if explicit.shape[-1] != real_len:
        raise ValueError(f"position_ids length {explicit.shape[-1]} != seq_len {real_len}")
    if real_len == padded_len:
        return explicit.long()
    tail_start = int(explicit[0, -1].item()) + 1
    tail = torch.arange(tail_start, tail_start + padded_len - real_len, dtype=torch.long, device=device)
    return torch.cat([explicit.long(), tail.unsqueeze(0)], dim=-1)


def _ar_head_sample(model, base_cond, temperature, generator, use_kv_cache):
    """Sample all codebooks for every still-masked spatial position.

    ``base_cond``: [N, D] backbone hidden (post CFG + condition_proj) for the
    masked positions. Returns (next_tokens [N, C], position_conf [N]).

    With ``use_kv_cache=True`` the channel decoder runs incrementally (C token
    forwards instead of C*(C+1)/2). ``CausalAttention`` implements exact KV
    caching and the head's norms/adaLN are per-token, so the incremental
    last-token logits are algebraically the full-prefix recomputation. They are
    not bit-identical in bf16, because the reduction order differs, which is why
    the cache is opt-in.
    """
    head = model.multi_codebook_head
    num_codebooks = len(model.lm_img_head)
    num_valid = base_cond.shape[0]
    cond = base_cond.unsqueeze(1) + head.timesteps_embeddings  # [N, C, D]

    if not use_kv_cache:
        channel_sequence = [base_cond.unsqueeze(1)]
        next_tokens, all_conf = [], []
        for c in range(num_codebooks):
            x = torch.cat(channel_sequence, dim=1)
            x = x + head.channel_embed[:, :x.shape[1]]
            curr_len = x.size(1)
            attn_mask = head.channel_mask[:curr_len, :curr_len]
            curr_cond = cond[:, :curr_len]
            for block in head.channel_blocks:
                x = block(x, attn_mask=attn_mask, c=curr_cond)
            x = head.channel_norm(x)
            x = head.channel_final(x, curr_cond)
            logits = model.lm_img_head[c](x[:, -1:]).view(num_valid, -1)
            token = gumbel_max_sample(logits, temperature, generator=generator)
            next_tokens.append(token)
            probs = F.softmax(logits / temperature, dim=-1) if temperature > 0 else F.softmax(logits, dim=-1)
            all_conf.append(probs.gather(-1, token.unsqueeze(-1)).squeeze(-1))
            if c < num_codebooks - 1:
                channel_sequence.append(model.model.img_embeddings[c](token).unsqueeze(1))
        return torch.stack(next_tokens, dim=1), torch.stack(all_conf, dim=1).mean(dim=-1)

    blocks = list(head.channel_blocks)
    for block in blocks:
        if block.attn.use_cache or block.attn.k_cache is not None:
            raise RuntimeError("AR-head KV cache already active before accelerated decode")
        block.attn.use_cache = True
        block.attn.reset_cache()
    try:
        x_token = base_cond.unsqueeze(1)
        next_tokens, all_conf = [], []
        for c in range(num_codebooks):
            x = x_token + head.channel_embed[:, c:c + 1]
            cond_c = cond[:, c:c + 1]
            # After the cached K/V are appended the key length is c+1; the single
            # (last) query sees everything, matching the causal prefix mask.
            attn_zero = x.new_zeros((1, c + 1))
            for block in blocks:
                x = block(x, attn_mask=attn_zero, c=cond_c)
            x = head.channel_norm(x)
            x = head.channel_final(x, cond_c)
            logits = model.lm_img_head[c](x[:, -1:]).view(num_valid, -1)
            token = gumbel_max_sample(logits, temperature, generator=generator)
            next_tokens.append(token)
            probs = F.softmax(logits / temperature, dim=-1) if temperature > 0 else F.softmax(logits, dim=-1)
            all_conf.append(probs.gather(-1, token.unsqueeze(-1)).squeeze(-1))
            if c < num_codebooks - 1:
                x_token = model.model.img_embeddings[c](token).unsqueeze(1)
        return torch.stack(next_tokens, dim=1), torch.stack(all_conf, dim=1).mean(dim=-1)
    finally:
        for block in blocks:
            block.attn.use_cache = False
            block.attn.reset_cache()


def _halton_1d(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _halton_rank(grid_h: int, grid_w: int, num_img: int, device) -> torch.Tensor:
    """Reveal-order rank per image position; lower rank = revealed earlier.

    A low-discrepancy (base 2 / base 3) sweep spreads commits uniformly over the
    grid instead of letting them clump where confidence happens to be high.
    Structural NEW_LINE columns get an out-of-range rank -- they are never masked.
    """
    order, seen = [], set()
    i = 1
    limit = 200 * grid_h * grid_w
    while len(order) < grid_h * grid_w and i < limit:
        x = min(grid_w - 1, int(_halton_1d(i, 2) * grid_w))
        y = min(grid_h - 1, int(_halton_1d(i, 3) * grid_h))
        if (y, x) not in seen:
            seen.add((y, x))
            order.append((y, x))
        i += 1
    for y in range(grid_h):
        for x in range(grid_w):
            if (y, x) not in seen:
                order.append((y, x))
    rank = torch.full((num_img,), float(num_img + 10), device=device)
    for r, (y, x) in enumerate(order):
        rank[y * (grid_w + 1) + x] = float(r)
    return rank


def _cache_layers(cache):
    """Version-portable DynamicCache accessors (key_cache lists vs .layers)."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache, cache.value_cache
    return [layer.keys for layer in cache.layers], [layer.values for layer in cache.layers]


def _make_compat_cache():
    """DynamicCache carrying the legacy ``get_usable_length`` the modeling code calls.

    Removed in transformers>=4.5x; re-adding it here keeps the model files
    untouched.
    """
    from transformers.cache_utils import DynamicCache

    class _CompatDynamicCache(DynamicCache):
        def get_usable_length(self, new_seq_length, layer_idx=0):
            return self.get_seq_length(layer_idx)

    return _CompatDynamicCache()


class _MLCacheState:
    """Per-layer K/V + hidden-state cache over the packed CFG batch.

    On a refresh step a full forward runs with ``use_cache=True`` so the model's
    own cache plumbing captures post-RoPE K/V for every position. On a cached
    step only a small compute set of image positions is forwarded: those whose
    input ids changed since the snapshot, plus the lowest-confidence
    ``1 - cache_ratio`` fraction of the still-masked ones. Everything else
    contributes cached K/V through ``past_key_values`` (attention columns
    ordered [cached, computed]) and reuses its hidden state.

    Computed K/V are scattered back into the master cache and the id snapshot is
    updated, so staleness does not accumulate between refreshes.
    """

    def __init__(self, model, num_layers):
        self.model = model
        self.num_layers = num_layers
        self.master_k = None      # list of [R, kvh, P, hd]
        self.master_v = None
        self.snapshot = None      # [R, P] input ids (codebook 0) at last sync
        self.cond_hidden = None   # [1, num_img, H]
        self.uncond_hidden = None

    def refresh(self, batch_ids, batch_types, batch_mask, batch_pos, num_img, rows):
        ids = batch_ids[rows]
        cache = _make_compat_cache()
        hidden = self.model(
            input_ids=ids,
            type_position_ids=batch_types[rows],
            attention_mask=batch_mask[rows],
            position_ids=batch_pos[rows],
            infer_t2i=True,
            use_cache=True,
            past_key_values=cache,
        )
        keys, values = _cache_layers(cache)
        self.master_k = list(keys)
        self.master_v = list(values)
        self.snapshot = ids[:, :, 0].clone()
        if len(rows) == 2:
            self.cond_hidden = hidden[:, :num_img].clone()
            self.uncond_hidden = hidden[:, num_img:].clone()
        else:
            self.cond_hidden = hidden.clone()
            self.uncond_hidden = None
        return self.cond_hidden, self.uncond_hidden

    def cached_step(self, batch_ids, batch_mask, batch_pos, branch_img_idx, s_img, num_img, rows):
        """Partial forward for the image positions ``s_img`` (indices in num_img space)."""
        device = batch_ids.device
        n_s = int(s_img.numel())
        padded_len = batch_ids.shape[1]

        s_seq = [branch_img_idx[b][s_img] for b in rows]
        keep_idx = []
        for b_pos in range(len(rows)):
            keep_mask = torch.ones(padded_len, dtype=torch.bool, device=device)
            keep_mask[s_seq[b_pos]] = False
            keep_idx.append(keep_mask.nonzero(as_tuple=False).squeeze(-1))
        n_keep = int(keep_idx[0].numel())

        part_cache = _make_compat_cache()
        for layer in range(self.num_layers):
            k = torch.stack([self.master_k[layer][b_pos, :, keep_idx[b_pos]]
                             for b_pos in range(len(rows))], dim=0)
            v = torch.stack([self.master_v[layer][b_pos, :, keep_idx[b_pos]]
                             for b_pos in range(len(rows))], dim=0)
            part_cache.update(k, v, layer)

        part_ids = torch.stack([batch_ids[b, s_seq[i]] for i, b in enumerate(rows)], dim=0)
        part_types = part_ids.new_full(part_ids.shape[:2], 4)
        part_pos = torch.stack([batch_pos[b, s_seq[i]] for i, b in enumerate(rows)], dim=0)
        part_mask = torch.stack([
            batch_mask[b, 0][s_seq[i]][:, torch.cat([keep_idx[i], s_seq[i]])]
            for i, b in enumerate(rows)
        ], dim=0).unsqueeze(1)

        hidden = self.model(
            input_ids=part_ids,
            type_position_ids=part_types,
            attention_mask=part_mask,
            position_ids=part_pos,
            infer_t2i=True,
            use_cache=True,
            past_key_values=part_cache,
        )

        keys, values = _cache_layers(part_cache)
        for layer in range(self.num_layers):
            new_k = keys[layer][:, :, n_keep:]
            new_v = values[layer][:, :, n_keep:]
            for b_pos in range(len(rows)):
                self.master_k[layer][b_pos, :, s_seq[b_pos]] = new_k[b_pos]
                self.master_v[layer][b_pos, :, s_seq[b_pos]] = new_v[b_pos]
        for b_pos, b in enumerate(rows):
            self.snapshot[b_pos, s_seq[b_pos]] = batch_ids[b, s_seq[b_pos], 0]

        if len(rows) == 2:
            self.cond_hidden[0, s_img] = hidden[0, :n_s]
            self.uncond_hidden[0, s_img] = hidden[0, n_s:]
        else:
            self.cond_hidden[0, s_img] = hidden[0]
        return self.cond_hidden, self.uncond_hidden


# No-grad is load-bearing, not hygiene: the reference loop wraps every step in
# torch.inference_mode(), and without the equivalent here each round's
# activations are retained for autograd across all `timesteps` rounds, which
# exhausts an 80 GB device even at 512x512.
@torch.no_grad()
def _generate_image_ar_accelerated(
    model,
    input_ids,
    uncon_input_ids,
    type_position_ids,
    uncon_type_position_ids,
    *,
    timesteps,
    mask_token_id,
    newline_id,
    temperature,
    cfg_scale,
    code_start,
    noise_schedule,
    generator,
    user_prefix_len,
    media_region_start,
    uncon_user_prefix_len,
    uncon_media_region_start,
    position_ids,
    uncon_position_ids,
    grid_hw,
    ar_kv_cache,
    ml_cache,
    cache_ratio,
    refresh_interval,
    warmup_ratio,
    vverify_delta,
    plateau_delta,
    plateau_floor,
    plateau_patience,
    plateau_min_samples,
    conf_ema_beta,
    halton,
    return_stats,
):
    """Packed-CFG decode loop with the optional cache / plateau accelerations.

    Only reached for the simple-CFG layout; the caller falls back to the
    reference loop for dual-CFG and negative ``cfg_scale``.
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    input_ids = input_ids.to(device)
    uncon_input_ids = uncon_input_ids.to(device)
    type_position_ids = type_position_ids.to(device)
    uncon_type_position_ids = uncon_type_position_ids.to(device)

    if halton and grid_hw is None:
        raise ValueError(
            "the plateau/halton tier needs grid_hw=(grid_h, grid_w) to build the "
            "spatial reveal order; pass it from the caller"
        )

    cond_len = int(input_ids.shape[1])
    uncond_len = int(uncon_input_ids.shape[1])
    cond_mask = build_t2i_attention_mask_4d(
        type_position_ids, int(user_prefix_len), int(media_region_start), device, model_dtype,
    )
    uncond_mask = build_t2i_attention_mask_4d(
        uncon_type_position_ids, int(uncon_user_prefix_len), int(uncon_media_region_start),
        device, model_dtype,
    )
    cond_img_idx = torch.nonzero((type_position_ids == 4)[0], as_tuple=False).squeeze(-1)
    uncond_img_idx = torch.nonzero((uncon_type_position_ids == 4)[0], as_tuple=False).squeeze(-1)
    num_img = int(cond_img_idx.numel())
    if int(uncond_img_idx.numel()) != num_img:
        raise ValueError(
            f"uncond type-4 count {int(uncond_img_idx.numel())} does not match cond {num_img}"
        )

    padded_len = max(cond_len, uncond_len)
    batch_ids = torch.cat(
        [_right_pad(input_ids, padded_len), _right_pad(uncon_input_ids, padded_len)], dim=0
    )
    batch_types = torch.cat(
        [_right_pad(type_position_ids, padded_len), _right_pad(uncon_type_position_ids, padded_len)],
        dim=0,
    )
    # The 4-D masks are the largest tensors here (O(padded_len^2) each). Build the
    # packed [2,1,P,P] buffer once and fill it in place, then drop the per-branch
    # masks, instead of holding five S^2 tensors at once through a cat().
    batch_mask = _pack_additive_masks_4d(cond_mask, cond_len, uncond_mask, uncond_len, padded_len)
    del cond_mask, uncond_mask
    batch_pos = torch.cat(
        [_branch_position_ids(position_ids, cond_len, padded_len, device),
         _branch_position_ids(uncon_position_ids, uncond_len, padded_len, device)], dim=0
    )
    cond_ids_view = batch_ids[0:1, :cond_len]
    uncond_ids_view = batch_ids[1:2, :uncond_len]

    if not torch.equal(
        uncond_ids_view[0, uncond_img_idx, 0] == mask_token_id,
        cond_ids_view[0, cond_img_idx, 0] == mask_token_id,
    ):
        raise ValueError("uncond initial image MASK pattern does not match cond")

    total_to_generate = int((cond_ids_view[0, cond_img_idx, 0] == mask_token_id).sum().item())
    halton_rank = None
    if halton:
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        # The type-4 region is the grid plus one NEW_LINE column per row, and the
        # rank table is indexed as y*(grid_w+1)+x. If that identity does not hold
        # the reveal order would be silently scrambled rather than wrong-looking.
        if num_img != grid_h * (grid_w + 1):
            raise ValueError(
                f"grid_hw={(grid_h, grid_w)} implies {grid_h * (grid_w + 1)} type-4 positions "
                f"(grid + one NEW_LINE per row) but the sequence has {num_img}"
            )
        halton_rank = _halton_rank(grid_h, grid_w, num_img, device)

    # plateau-commit state. A position may commit early when its confidence EMA
    # has been flat for `plateau_patience` consecutive rounds AND sits above
    # `plateau_floor` -- i.e. it converged at a ceiling, rather than still being
    # parked at the pre-context floor.
    ema_conf = torch.zeros(num_img, dtype=torch.float32, device=device)
    plateau_run = torch.zeros(num_img, dtype=torch.long, device=device)
    sample_cnt = torch.zeros(num_img, dtype=torch.long, device=device)
    plateaued = torch.zeros(num_img, dtype=torch.bool, device=device)

    ml_state = None
    conf_map = None
    branch_img_idx = None
    warmup_step = 0
    force_refresh = False
    if ml_cache:
        ml_state = _MLCacheState(model, len(model.model.layers))
        conf_map = torch.zeros(num_img, dtype=torch.float32, device=device)
        branch_img_idx = [cond_img_idx, uncond_img_idx]
        warmup_step = int(timesteps * warmup_ratio)

    rows = [0, 1]
    # "backbone_forwards" counts packed B=2 forwards (cond+uncond together),
    # not one per branch -- that is the whole point of packing.
    stats = dict(rounds=0, backbone_forwards=0, cached_steps=0,
                 forced_refreshes=0, plateau_commits=0)

    for step in range(timesteps):
        mask_flags = cond_ids_view[0, cond_img_idx, 0] == mask_token_id
        masked_positions = mask_flags.nonzero(as_tuple=False).squeeze(-1)
        if masked_positions.numel() == 0:
            break
        stats["rounds"] += 1

        # keep_n: how many positions should still be MASKed after this round.
        # The cosine schedule is a pacing floor -- plateau can only commit more.
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = int((total_to_generate * frac).floor().clamp_min(1).item())
        else:
            keep_n = 0

        if ml_cache:
            refresh = force_refresh or step <= warmup_step or (step - warmup_step) % refresh_interval == 0
            force_refresh = False
            if refresh:
                cond_hidden, uncond_hidden = ml_state.refresh(
                    batch_ids, batch_types, batch_mask, batch_pos, num_img, rows,
                )
            else:
                stats["cached_steps"] += 1
                changed = batch_ids[0, cond_img_idx, 0] != ml_state.snapshot[0, cond_img_idx]
                n_low = max(1, int(masked_positions.numel() * (1.0 - cache_ratio)))
                low = masked_positions[torch.argsort(conf_map[masked_positions])[:n_low]]
                s_img = torch.unique(
                    torch.cat([changed.nonzero(as_tuple=False).squeeze(-1), low])
                )
                old = ml_state.cond_hidden[0, s_img].clone() if vverify_delta > 0 else None
                cond_hidden, uncond_hidden = ml_state.cached_step(
                    batch_ids, batch_mask, batch_pos, branch_img_idx, s_img, num_img, rows,
                )
                if vverify_delta > 0:
                    new = ml_state.cond_hidden[0, s_img]
                    drift = (new.float() - old.float()).abs().mean() / old.float().abs().mean().clamp_min(1e-6)
                    if float(drift) > vverify_delta:
                        # V-verify: the recomputed positions moved enough that the
                        # cached ones are probably stale too. Force a full refresh.
                        force_refresh = True
                        stats["forced_refreshes"] += 1
            stats["backbone_forwards"] += 1
        else:
            hidden = model(
                input_ids=batch_ids,
                type_position_ids=batch_types,
                attention_mask=batch_mask,
                position_ids=batch_pos,
                infer_t2i=True,
                use_cache=False,
            )
            cond_hidden, uncond_hidden = hidden[:, :num_img], hidden[:, num_img:]
            stats["backbone_forwards"] += 1

        # Hidden-state CFG: cfg_scale=4 computes 5*cond - 4*uncond.
        if cfg_scale == 0:
            hidden_states = cond_hidden
        else:
            hidden_states = (1 + cfg_scale) * cond_hidden - cfg_scale * uncond_hidden

        base_cond = model.multi_codebook_head.condition_proj(hidden_states)
        if base_cond.dim() == 3:
            base_cond = base_cond[0]
        base_cond = base_cond[mask_flags]

        next_tokens, position_conf = _ar_head_sample(
            model, base_cond, temperature, generator, ar_kv_cache,
        )
        if conf_map is not None:
            conf_map[masked_positions] = position_conf.float()
        # Cond and uncond must hold identical sampled image state before the next
        # CFG round; only their prefixes differ.
        cond_ids_view[0, cond_img_idx[masked_positions]] = next_tokens
        uncond_ids_view[0, uncond_img_idx[masked_positions]] = next_tokens

        if plateau_delta > 0:
            old_ema = ema_conf[masked_positions]
            first = sample_cnt[masked_positions] == 0
            new_ema = torch.where(
                first,
                position_conf.float(),
                conf_ema_beta * old_ema + (1 - conf_ema_beta) * position_conf.float(),
            )
            rel = (new_ema - old_ema).abs() / old_ema.clamp_min(1e-3)
            flat = (rel < plateau_delta) & (sample_cnt[masked_positions] >= plateau_min_samples)
            plateau_run[masked_positions] = torch.where(
                flat,
                plateau_run[masked_positions] + 1,
                torch.zeros_like(plateau_run[masked_positions]),
            )
            ema_conf[masked_positions] = new_ema
            sample_cnt[masked_positions] += 1
            plateaued[masked_positions] = (
                (plateau_run[masked_positions] >= plateau_patience) & (new_ema > plateau_floor)
            )

        # ---- which sampled positions stay masked for the next round ----
        # Note the split of responsibilities: plateau decides HOW MANY commit,
        # the gumbel-noised topk (or Halton) decides WHICH stay masked. Replacing
        # the noisy selection with a deterministic one degrades quality even when
        # the commit count is unchanged -- MaskGIT's noisy remask is load-bearing.
        if plateau_delta > 0:
            cand = (~plateaued[masked_positions]).nonzero(as_tuple=False).squeeze(-1)
            stats["plateau_commits"] += int(masked_positions.numel() - cand.numel())
            keep_target = max(0, min(keep_n, int(cand.numel()), masked_positions.numel() - 1))
            re_mask = None
            if keep_target > 0:
                if halton:
                    order = torch.argsort(halton_rank[masked_positions[cand]], descending=True)
                    re_mask = masked_positions[cand[order[:keep_target]]]
                else:
                    sel = mask_by_random_topk(
                        torch.tensor([keep_target], device=device),
                        position_conf[cand].unsqueeze(0),
                        temperature=temperature,
                        generator=generator,
                    )[0]
                    re_mask = masked_positions[cand[sel]] if sel.any() else None
        else:
            # Exactly the reference generator's semantics, so cache-only presets
            # stay directly comparable to an unaccelerated run.
            mask_sel = mask_by_random_topk(
                torch.tensor([keep_n], device=device),
                position_conf.unsqueeze(0),
                temperature=temperature,
                generator=generator,
            )[0]
            re_mask = masked_positions[mask_sel] if mask_sel.any() else None

        if re_mask is not None and re_mask.numel() > 0:
            cond_ids_view[0, cond_img_idx[re_mask]] = mask_token_id
            uncond_ids_view[0, uncond_img_idx[re_mask]] = mask_token_id

    img_only = cond_ids_view[:, code_start + 1:-2, :]
    valid_rows = img_only[0, :, 0] != newline_id
    vq_ids = img_only[:, valid_rows, :].contiguous()
    if return_stats:
        return vq_ids, stats
    return vq_ids


def generate_image_ar(
    model,
    input_ids: torch.LongTensor,
    uncon_input_ids: torch.LongTensor,
    type_position_ids: torch.LongTensor,
    uncon_type_position_ids: torch.LongTensor,
    *,
    timesteps: int = 64,
    mask_token_id: int = 5000,
    newline_id: int = 5001,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    cfg_text: Optional[float] = None,
    cfg_img: Optional[float] = None,
    uncon_text_input_ids: Optional[torch.LongTensor] = None,
    uncon_image_input_ids: Optional[torch.LongTensor] = None,
    uncon_text_type_position_ids: Optional[torch.LongTensor] = None,
    uncon_image_type_position_ids: Optional[torch.LongTensor] = None,
    uncon_text_user_prefix_len: Optional[int] = None,
    uncon_image_user_prefix_len: Optional[int] = None,
    code_start: int,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    generator: Optional[torch.Generator] = None,
    debug: bool = False,
    pad_len: Optional[int] = None,
    user_prefix_len: Optional[int] = None,
    media_region_start: Optional[int] = None,
    uncon_user_prefix_len: Optional[int] = None,
    uncon_media_region_start: Optional[int] = None,
    position_ids: Optional[torch.LongTensor] = None,
    uncon_position_ids: Optional[torch.LongTensor] = None,
    uncon_text_position_ids: Optional[torch.LongTensor] = None,
    uncon_image_position_ids: Optional[torch.LongTensor] = None,
    # ---- acceleration (see generators/accel_presets.py) ----
    accel_preset: str = "off",
    grid_hw: Optional[tuple] = None,
    return_accel_stats: bool = False,
    **accel_overrides,
) -> torch.LongTensor:
    """Iteratively generate all output-image VQ positions.

    Spatial positions use MaskGIT-style parallel refinement. Inside one
    spatial position, codebooks 0..7 are predicted sequentially, so "AR" in
    this function name refers to codebook channels rather than left-to-right
    image generation.

    Args:
        input_ids: Conditional sequence, shape [1, cond_S, 8].
        uncon_input_ids: CFG-negative sequence, shape [1, uncond_S, 8].
        type_position_ids: Cond role labels; type 4 identifies image answers.
        uncon_type_position_ids: Uncond branch role labels. Image-answer count and
            MASK pattern must match cond, but the surrounding sequence may be shorter.
        timesteps: Number of spatial MaskGIT refinement rounds.
        mask_token_id: Value marking an unknown image code.
        newline_id: Structural row separator removed from the final result.
        temperature: Gumbel sampling and confidence-remasking temperature.
        cfg_scale: Hidden-state CFG scale. The implemented formula is
            (1 + scale) * cond - scale * uncond.
        cfg_text: Edit-only text CFG scale. Must be paired with cfg_img.
        cfg_img: Edit-only image CFG scale. Must be paired with cfg_text.
        code_start: BOI sequence index; image tokens begin at code_start + 1.
        noise_schedule: Maps progress to the number of positions kept masked.
        generator: Seeded CUDA RNG used by sampling and remasking.
        user_prefix_len: End of conditional clean SFT turn 0.
        media_region_start: Conditional BOI / noisy image-turn start.
        uncon_user_prefix_len: End of unconditional clean turn 0. Together with
            uncon_media_region_start, opts into a branch-local asymmetric layout.
        uncon_media_region_start: BOI index in the uncond sequence. When omitted,
            it is derived from the conditional answer suffix.
        position_ids: Optional explicit cond RoPE positions.
            to add a virtual gap without inserting physical padding tokens.
        uncon_position_ids: Corresponding uncond RoPE positions.

        The number of generated positions is derived from type-4 IMAGE_MASK entries.
        accel_preset: One of "off", "cache", "fast", "quality". "off" still uses the
            nothing, reproducing the unaccelerated output. Every other preset
            differs at least at the bf16-rounding level.
            Tier-1/tier-2 flags may be overridden individually via keyword; the
            plateau tier is preset-only (see generators/accel_presets.py).
        grid_hw: (grid_h, grid_w) of the image token grid. Required by the
            presets that enable the plateau/Halton tier.
        return_accel_stats: Return (vq_ids, stats) with round and forward counts.

    Returns:
        Tensor [1, number_of_spatial_positions, 8], with NEW_LINE removed.
    """

    accel = resolve_accel_preset(accel_preset)
    unknown = set(accel_overrides) - set(accel)
    if unknown:
        raise TypeError(
            f"unknown acceleration override(s) {sorted(unknown)}; "
            f"tunable for preset {accel_preset!r}: {sorted(accel)}"
        )
    accel.update(accel_overrides)
    # halton is bound to plateau and is not independently selectable.
    accel["halton"] = accel.get("plateau_delta", 0.0) > 0

    ar_kv_cache = bool(accel.get("ar_kv_cache", True))
    wants_packed = bool(accel.get("cfg_batch") or accel.get("ml_cache") or accel.get("plateau_delta", 0.0) > 0)
    if return_accel_stats and not wants_packed:
        # Only the accelerated loop produces stats. Silently returning a bare
        # tensor here would blow up at the caller's `vq, stats = ...` unpack,
        # far from the cause. The text decoder raises in the same situation.
        raise ValueError(
            f"return_accel_stats=True needs an accelerating preset, but "
            f"accel_preset={accel_preset!r} enables none of cfg_batch/ml_cache/plateau"
        )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    uncon_input_ids = uncon_input_ids.to(device)
    type_position_ids = type_position_ids.to(device)
    uncon_type_position_ids = uncon_type_position_ids.to(device)
    dual_cfg = cfg_text is not None or cfg_img is not None
    if (cfg_text is None) != (cfg_img is None):
        raise ValueError("cfg_text and cfg_img must be provided together")
    if dual_cfg and cfg_scale != 0:
        raise ValueError("non-zero cfg_scale cannot be combined with cfg_text/cfg_img")
    if dual_cfg:
        dual_inputs = (
            uncon_text_input_ids,
            uncon_image_input_ids,
            uncon_text_type_position_ids,
            uncon_image_type_position_ids,
        )
        if any(value is None for value in dual_inputs):
            raise ValueError("dual-CFG requires both negative input/type-id branches")
        uncon_text_input_ids = uncon_text_input_ids.to(device)
        uncon_image_input_ids = uncon_image_input_ids.to(device)
        uncon_text_type_position_ids = uncon_text_type_position_ids.to(device)
        uncon_image_type_position_ids = uncon_image_type_position_ids.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)
    if uncon_position_ids is not None:
        uncon_position_ids = uncon_position_ids.to(device)
    if uncon_text_position_ids is not None:
        uncon_text_position_ids = uncon_text_position_ids.to(device)
    if uncon_image_position_ids is not None:
        uncon_image_position_ids = uncon_image_position_ids.to(device)
    if input_ids.dim() != 3:
        raise ValueError(f"img_prompt should be [B, seq_len, num_codebooks], got {input_ids.shape}")
    # This local seq_len is the complete model sequence length and overwrites
    # the legacy keyword argument. Actual image length is measured below.
    B, seq_len, num_codebooks = input_ids.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    position_branches = (("cond", input_ids, type_position_ids, position_ids),)
    if dual_cfg:
        position_branches += (
            ("uncond_text", uncon_text_input_ids, uncon_text_type_position_ids, uncon_text_position_ids),
            ("uncond_image", uncon_image_input_ids, uncon_image_type_position_ids, uncon_image_position_ids),
        )
    else:
        position_branches += (("uncond", uncon_input_ids, uncon_type_position_ids, uncon_position_ids),)
    for branch_name, branch_input_ids, branch_type_ids, branch_position_ids in position_branches:
        if branch_input_ids.dim() != 3:
            raise ValueError(f"{branch_name} input_ids must be [B, S, C], got {branch_input_ids.shape}")
        if branch_type_ids.dim() != 2 or branch_type_ids.shape != branch_input_ids.shape[:2]:
            raise ValueError(
                f"{branch_name} type_position_ids shape {branch_type_ids.shape} "
                f"does not match input_ids batch/sequence shape {branch_input_ids.shape[:2]}"
            )
        if branch_position_ids is not None and branch_position_ids.shape[-1] != branch_input_ids.shape[1]:
            raise ValueError(
                f"{branch_name} position_ids length {branch_position_ids.shape[-1]} "
                f"does not match sequence length {branch_input_ids.shape[1]}"
            )

    cond_t2i_attn_mask = None
    uncond_t2i_attn_mask = None
    uncon_text_t2i_attn_mask = None
    uncon_image_t2i_attn_mask = None
    # Metadata-aware masking separates the clean user prefix from the noisy image turn.
    # A negative branch may use its own sequence length and media boundary when both
    # values are explicit; otherwise derive the boundary from the conditional suffix.
    if (user_prefix_len is None) != (media_region_start is None):
        raise ValueError("user_prefix_len and media_region_start must be provided together")
    if user_prefix_len is not None:
        model_dtype = next(model.parameters()).dtype

        def validate_attention_layout(
            branch_name,
            branch_input_ids,
            branch_prefix_len,
            branch_media_region_start,
        ):
            branch_seq_len = int(branch_input_ids.shape[1])
            branch_prefix_len = int(branch_prefix_len)
            branch_media_region_start = int(branch_media_region_start)
            if not 0 <= branch_prefix_len <= branch_media_region_start < branch_seq_len:
                raise ValueError(
                    f"{branch_name} boundaries must satisfy 0 <= user_prefix_len <= "
                    f"media_region_start < seq_len, got user_prefix_len={branch_prefix_len}, "
                    f"media_region_start={branch_media_region_start}, seq_len={branch_seq_len}"
                )
            return branch_prefix_len, branch_media_region_start

        user_prefix_len, media_region_start = validate_attention_layout(
            "cond", input_ids, user_prefix_len, media_region_start
        )
        cond_t2i_attn_mask = build_t2i_attention_mask_4d(
            type_position_ids,
            user_prefix_len,
            media_region_start,
            device,
            model_dtype,
        )
        cond_seq_len = int(input_ids.shape[1])
        answer_suffix_len = cond_seq_len - user_prefix_len
        answer_offset = media_region_start - user_prefix_len

        def build_negative_attention_mask(
            branch_name,
            branch_input_ids,
            branch_type_ids,
            branch_prefix_len,
            branch_media_region_start=None,
        ):
            branch_seq_len = int(branch_input_ids.shape[1])
            has_explicit_layout = (
                branch_prefix_len is not None and branch_media_region_start is not None
            )
            if has_explicit_layout:
                branch_prefix_len, branch_media_region_start = validate_attention_layout(
                    branch_name,
                    branch_input_ids,
                    branch_prefix_len,
                    branch_media_region_start,
                )
            else:
                if branch_prefix_len is None:
                    branch_prefix_len = branch_seq_len - answer_suffix_len
                branch_prefix_len = int(branch_prefix_len)
                if branch_media_region_start is None:
                    branch_media_region_start = branch_prefix_len + answer_offset
                if branch_seq_len != branch_prefix_len + answer_suffix_len:
                    raise ValueError(
                        f"{branch_name} seq_len mismatch: got {branch_seq_len}, expected "
                        f"{branch_prefix_len + answer_suffix_len} "
                        f"(prefix_len={branch_prefix_len}, answer_suffix_len={answer_suffix_len}); "
                        f"provide both branch user/media boundaries for an asymmetric layout"
                    )
                branch_prefix_len, branch_media_region_start = validate_attention_layout(
                    branch_name,
                    branch_input_ids,
                    branch_prefix_len,
                    branch_media_region_start,
                )
            return (
                build_t2i_attention_mask_4d(
                    branch_type_ids,
                    branch_prefix_len,
                    branch_media_region_start,
                    device,
                    model_dtype,
                ),
                branch_prefix_len,
                branch_media_region_start,
            )

        if dual_cfg:
            (
                uncon_text_t2i_attn_mask,
                uncon_text_user_prefix_len,
                _,
            ) = build_negative_attention_mask(
                "uncond_text",
                uncon_text_input_ids,
                uncon_text_type_position_ids,
                uncon_text_user_prefix_len,
            )
            (
                uncon_image_t2i_attn_mask,
                uncon_image_user_prefix_len,
                _,
            ) = build_negative_attention_mask(
                "uncond_image",
                uncon_image_input_ids,
                uncon_image_type_position_ids,
                uncon_image_user_prefix_len,
            )
        else:
            (
                uncond_t2i_attn_mask,
                uncon_user_prefix_len,
                uncon_media_region_start,
            ) = build_negative_attention_mask(
                "uncond",
                uncon_input_ids,
                uncon_type_position_ids,
                uncon_user_prefix_len,
                uncon_media_region_start,
            )

    same_positions = (
        (position_ids is None and uncon_position_ids is None)
        or (
            position_ids is not None
            and uncon_position_ids is not None
            and torch.equal(position_ids, uncon_position_ids)
        )
    )
    same_condition_branch = (
        not dual_cfg
        and torch.equal(input_ids, uncon_input_ids)
        and torch.equal(type_position_ids, uncon_type_position_ids)
        and same_positions
        and user_prefix_len == uncon_user_prefix_len
        and media_region_start == uncon_media_region_start
    )
    # CFG is numerically the conditional branch when scale is zero or both
    # branches are identical. Avoid an otherwise redundant backbone forward.
    single_branch_cfg = not dual_cfg and (cfg_scale == 0 or same_condition_branch)

    # ---- acceleration dispatch ----------------------------------------------
    # The packed-CFG tiers assume the simple-CFG layout. Anything else falls back
    # to the reference loop rather than raising: `single_edit` legitimately uses
    # dual-CFG, and a caller asking for speed should not get an exception.
    if wants_packed:
        blockers = []
        if dual_cfg:
            blockers.append("dual-CFG (cfg_text/cfg_img)")
        if cfg_scale < 0:
            blockers.append("negative cfg_scale")
        if user_prefix_len is None:
            blockers.append("no user_prefix_len/media_region_start metadata")
        if pad_len is not None:
            blockers.append("legacy physical pad_len")
        if debug:
            blockers.append("debug=True")
        if blockers:
            logger.warning(
                "accel_preset=%r is not supported for this request (%s); falling back to "
                "the reference decode loop. The exact AR-head KV cache stays enabled.",
                accel_preset, ", ".join(blockers),
            )
        else:
            try:
                return _generate_image_ar_accelerated(
                        model,
                    input_ids,
                    uncon_input_ids,
                    type_position_ids,
                    uncon_type_position_ids,
                    timesteps=timesteps,
                    mask_token_id=mask_token_id,
                    newline_id=newline_id,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    code_start=code_start,
                    noise_schedule=noise_schedule,
                    generator=generator,
                    user_prefix_len=user_prefix_len,
                    media_region_start=media_region_start,
                    uncon_user_prefix_len=uncon_user_prefix_len,
                    uncon_media_region_start=uncon_media_region_start,
                    position_ids=position_ids,
                    uncon_position_ids=uncon_position_ids,
                    grid_hw=grid_hw,
                    ar_kv_cache=ar_kv_cache,
                    ml_cache=bool(accel.get("ml_cache", False)),
                    cache_ratio=float(accel.get("cache_ratio", 0.9)),
                    refresh_interval=int(accel.get("refresh_interval", 12)),
                    warmup_ratio=float(accel.get("warmup_ratio", 0.3)),
                    vverify_delta=float(accel.get("vverify_delta", 0.0)),
                    plateau_delta=float(accel.get("plateau_delta", 0.0)),
                    plateau_floor=float(accel.get("plateau_floor", 0.015)),
                    plateau_patience=int(accel.get("plateau_patience", 5)),
                    plateau_min_samples=int(accel.get("plateau_min_samples", 4)),
                    conf_ema_beta=float(accel.get("conf_ema_beta", 0.7)),
                    halton=bool(accel["halton"]),
                    return_stats=return_accel_stats,
                )
            except torch.cuda.OutOfMemoryError as exc:
                # The packed decode runs cond and uncond as one B=2 forward, so its
                # peak activation memory is roughly twice the reference loop's. At
                # high resolution that can exceed the device even though ACCEL=off
                # fits comfortably. Say so, instead of surfacing a bare allocator
                # error that looks like a bug in the model.
                raise torch.cuda.OutOfMemoryError(
                    f"Ran out of memory in the accelerated decode path "
                    f"(accel_preset={accel_preset!r}, sequence length {int(input_ids.shape[1])}). "
                    "It packs the two CFG branches into one B=2 forward, so it needs "
                    "roughly twice the activation memory of accel_preset='off'. Either "
                    "lower the output resolution, run with accel_preset='off', or use a "
                    "device with more memory. Original error: " + str(exc)
                ) from exc

    img_ans_mask = (type_position_ids == 4)  # [1, full_cond_sequence]
    img_ans_global_indices = torch.nonzero(img_ans_mask[0], as_tuple=False).squeeze(-1)
    if dual_cfg:
        uncon_text_img_ans_global_indices = torch.nonzero(
            (uncon_text_type_position_ids == 4)[0], as_tuple=False
        ).squeeze(-1)
        uncon_image_img_ans_global_indices = torch.nonzero(
            (uncon_image_type_position_ids == 4)[0], as_tuple=False
        ).squeeze(-1)
        branch_image_indices = [
            ("uncond_text", uncon_text_input_ids, uncon_text_img_ans_global_indices),
            ("uncond_image", uncon_image_input_ids, uncon_image_img_ans_global_indices),
        ]
    else:
        uncon_img_ans_global_indices = torch.nonzero(
            (uncon_type_position_ids == 4)[0], as_tuple=False
        ).squeeze(-1)
        branch_image_indices = [
            ("uncond", uncon_input_ids, uncon_img_ans_global_indices)
        ]
    cond_image_state = input_ids[0, img_ans_global_indices]
    for branch_name, branch_input_ids, branch_indices in branch_image_indices:
        if branch_indices.numel() != img_ans_global_indices.numel():
            raise ValueError(
                f"{branch_name} type-4 count {branch_indices.numel()} does not match "
                f"cond count {img_ans_global_indices.numel()}"
            )
        if not torch.equal(
            branch_input_ids[0, branch_indices] == mask_token_id,
            cond_image_state == mask_token_id,
        ):
            raise ValueError(f"{branch_name} initial image MASK pattern does not match cond")

    vq_mask = input_ids[img_ans_mask] == mask_token_id  # [image_turn_tokens, 8]
    # All eight codebooks at one spatial position are masked/committed together,
    # so codebook 0 is sufficient to decide whether the position is unknown.
    position_mask = vq_mask[..., 0]
    unknown_cnt = position_mask.sum(dim=0, keepdim=True)  # Dynamic MASK count.
    vq_len = unknown_cnt.clone()                         # Fixed initial count.

    debug_steps = [] if debug else None

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # keep_n is the target number of spatial positions that remain MASKed
        # after this round. The cosine schedule decreases it toward zero; the
        # final round commits every remaining position.
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)
        with torch.inference_mode():
            # infer_t2i returns hidden states only for type-4 image-answer
            # positions. Explicit position_ids, when supplied by blkpad, are
            # consumed by model RoPE; they do not alter physical tensor length.
            cond_hidden_states = model(
                input_ids=input_ids,
                type_position_ids=type_position_ids,
                attention_mask=cond_t2i_attn_mask,
                position_ids=position_ids,
                infer_t2i=True,
                cat='cond',
                use_cache=False,
            )

            if dual_cfg:
                uncon_text_hidden_states = model(
                    input_ids=uncon_text_input_ids,
                    type_position_ids=uncon_text_type_position_ids,
                    attention_mask=uncon_text_t2i_attn_mask,
                    position_ids=uncon_text_position_ids,
                    infer_t2i=True,
                    cat='uncond_text',
                    use_cache=False,
                )
                uncon_image_hidden_states = model(
                    input_ids=uncon_image_input_ids,
                    type_position_ids=uncon_image_type_position_ids,
                    attention_mask=uncon_image_t2i_attn_mask,
                    position_ids=uncon_image_position_ids,
                    infer_t2i=True,
                    cat='uncond_image',
                    use_cache=False,
                )
                if (
                    uncon_text_hidden_states.shape != cond_hidden_states.shape
                    or uncon_image_hidden_states.shape != cond_hidden_states.shape
                ):
                    raise ValueError(
                        "dual-CFG hidden-state shapes must match: "
                        f"cond={cond_hidden_states.shape}, "
                        f"text={uncon_text_hidden_states.shape}, "
                        f"image={uncon_image_hidden_states.shape}"
                    )
                hidden_states = (
                    cond_hidden_states
                    + cfg_text * (cond_hidden_states - uncon_text_hidden_states)
                    + cfg_img * (cond_hidden_states - uncon_image_hidden_states)
                )
            elif not single_branch_cfg:
                uncond_hidden_states = model(
                    input_ids=uncon_input_ids,
                    type_position_ids=uncon_type_position_ids,
                    attention_mask=uncond_t2i_attn_mask,
                    position_ids=uncon_position_ids,
                    infer_t2i=True,
                    cat='uncond',
                    use_cache=False,
                )
            # Hidden-state CFG convention: cfg_scale=4 computes
            # 5*cond - 4*uncond. Identical branches reduce exactly to cond.
            if not dual_cfg:
                if single_branch_cfg:
                    hidden_states = cond_hidden_states
                elif cfg_scale >= 0:
                    hidden_states = (
                        (1 + cfg_scale) * cond_hidden_states
                        - cfg_scale * uncond_hidden_states
                    )
                else:
                    hidden_states = uncond_hidden_states
            
            img_ans_mask = (type_position_ids == 4)
            # type 4 includes both real VQ positions and fixed NEW_LINE tokens.
            # vq_mask is false for NEW_LINE, so position_mask selects only the
            # still-unknown real spatial positions in this round.
            if vq_mask.dim() == 3:
                img_answer_vq_mask = vq_mask[img_ans_mask]  # [num_img_answer_tokens, num_codebooks]
            else:
                img_answer_vq_mask = vq_mask
            position_mask = img_answer_vq_mask[:, 0]  # [num_img_answer_tokens]
            
            # Project the backbone state into the eight-codebook channel
            # decoder's conditioning space, then retain current MASKs.
            cond = model.multi_codebook_head.condition_proj(hidden_states)

            if cond.dim() == 3:
                cond = cond[0]


            base_cond = cond[position_mask]
            masked_positions = position_mask.nonzero(as_tuple=False).squeeze(-1)  # [num_valid]
            num_valid = masked_positions.shape[0]

            # Spatial positions are batched in parallel; the loop inside
            # _ar_head_sample is the AR part, where codebook c conditions on the
            # sampled embeddings of codebooks 0..c-1 at the same position. With
            # ar_kv_cache it decodes incrementally; that is algebraically the same
            # but not bf16-identical, which is why it is opt-in. It is wired here too
            # so the dual-CFG edit path can use it when asked.
            # position_conf is the mean of the per-codebook sampled probabilities;
            # lower-confidence positions get selected for remasking.
            next_tokens, position_conf = _ar_head_sample(
                model, base_cond, temperature, generator, ar_kv_cache,
            )

            # Cond and uncond must contain identical sampled image states before
            # the next CFG round; only their user/text prefixes differ.
            input_ids[0, img_ans_global_indices[masked_positions]] = next_tokens
            for _, branch_input_ids, branch_indices in branch_image_indices:
                branch_input_ids[0, branch_indices[masked_positions]] = next_tokens

            if masked_positions.numel() == 0:
                break
            
            # Ensure keep_n is 1D (B,) for mask_by_random_topk
            if keep_n.dim() == 0:
                keep_n_1d = keep_n.unsqueeze(0)
            elif keep_n.dim() > 1:
                keep_n_1d = keep_n.squeeze(1)
            else:
                keep_n_1d = keep_n
            mask_sel = mask_by_random_topk(
                keep_n_1d, position_conf.unsqueeze(0), temperature=temperature, generator=generator
            )[0]
            # mask_sel is over positions sampled in this round; keep_n controls
            # how many remain unknown for the next round.
            
            re_mask_count = 0
            if mask_sel.any():
                re_mask_pos = masked_positions[mask_sel]
                re_mask_count = re_mask_pos.numel()
                input_ids[0, img_ans_global_indices[re_mask_pos]] = mask_token_id
                for _, branch_input_ids, branch_indices in branch_image_indices:
                    branch_input_ids[0, branch_indices[re_mask_pos]] = mask_token_id

            vq_mask = input_ids[img_ans_mask] == mask_token_id
            unknown_cnt = vq_mask[:, 0].sum().view(1, 1)

            if debug:
                keep_n_val = int(keep_n.item()) if keep_n.numel() == 1 else int(keep_n[0].item())
                frac_val = float(noise_schedule(torch.tensor([(step + 1) / timesteps], device=device)).item()) if step < timesteps - 1 else 0.0
                debug_steps.append({
                    "step": step,
                    "masked": num_valid,
                    "keep_n": keep_n_val,
                    "frac": frac_val,
                    "enter_re_mask_block": True,  # ori always runs re-mask when keep_n>0
                    "re_mask_count": re_mask_count,
                    "next_tokens_sample": next_tokens[0, :4].cpu().tolist() if num_valid > 0 else [],
                })

    # Slice between BOI and [EOI, EOA]. pad_len is a legacy physical-padding
    # path; current blkpad inference leaves it None and instead passes explicit
    # position_ids. Finally remove structural NEW_LINE entries.
    if pad_len is not None:
        img_only = input_ids[:, code_start+1:-(2+pad_len), :]
    else:
        img_only = input_ids[:, code_start+1:-2, :]
    valid_rows = (img_only[0, :, 0] != newline_id)
    vq_ids = img_only[:, valid_rows, :].contiguous()
    if debug:
        vq_len_val = int(vq_len.item()) if vq_len.numel() == 1 else int(vq_len[0].item())
        debug_info = {"name": "generate_image_ar", "total_to_generate": vq_len_val, "steps": debug_steps}
        return (vq_ids, debug_info)
    return vq_ids
