# -*- coding: utf-8 -*-
"""
Text understanding generator
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import re
from pathlib import Path
from typing import Optional, List
from utils.generation_utils import add_gumbel_noise, get_num_transfer_tokens


_SPLIT_EXPERT_KEY = re.compile(
    r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)(?:\.weight)?$"
)
_PACKED_EXPERT_KEY = re.compile(
    r"\.mlp\.experts\.(?:gate_proj|up_proj|down_proj)(?:\.weight)?$"
)


def _checkpoint_weight_keys(checkpoint: Path):
    """Read weight names without loading tensor data, when a local index exists."""
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint / index_name
        if index_path.is_file():
            try:
                with index_path.open("r", encoding="utf-8") as handle:
                    index = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read checkpoint index {index_path}: {exc}") from exc
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Checkpoint index has no valid weight_map: {index_path}")
            return tuple(weight_map)

    single_file = checkpoint / "model.safetensors"
    if single_file.is_file():
        try:
            from safetensors import safe_open

            with safe_open(str(single_file), framework="pt", device="cpu") as handle:
                return tuple(handle.keys())
        except Exception:
            # Metadata inspection is optional; the normal loader reports the
            # underlying error if the file itself is invalid.
            return None
    return None


def _validate_checkpoint_layout(checkpoint: str) -> None:
    """Reject packed MoE weights that this inference model cannot consume."""
    path = Path(checkpoint)
    if not path.is_dir():
        return
    keys = _checkpoint_weight_keys(path)
    if keys is None:
        return
    packed = [key for key in keys if _PACKED_EXPERT_KEY.search(key)]
    split = any(_SPLIT_EXPERT_KEY.search(key) for key in keys)
    if packed:
        raise ValueError(
            "Incompatible packed MoE checkpoint: found keys such as "
            f"{packed[0]!r}. Use the per-expert hf_ckpt_split export instead."
        )
    if not split:
        raise ValueError(
            "Checkpoint does not contain per-expert MoE weights "
            "(mlp.experts.<index>.*). Use a compatible hf_ckpt_split export."
        )


def load_inference_model(checkpoint: str, device: str = "cuda"):
    """Load an SFT checkpoint with the same LM class used in training."""
    _validate_checkpoint_layout(checkpoint)
    from models.internluminau2.modeling_internluminau2 import LLaDA2MoeModelLM

    model = LLaDA2MoeModelLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    return model.to(device).eval()


def build_bd_infer_attention_mask_4d(
    seq_len: int,
    prefix_len: int,
    code_start: int,
    cur_block_len: int,
    block_length: int,
    answer_block_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """BD inference mask: answer blocks numbered from prefix end, not seq 0.

    Block ids:
      - prefix [0, prefix_len): 0
      - prev decoded [prefix_len, code_start): 1 .. answer_block_idx
      - current noisy [code_start, code_start + cur_block_len): answer_block_idx + 1

    Query block Q attends to key block K iff Q >= K (full bidirectional inside a block).
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    block_idx = torch.zeros(seq_len, device=device, dtype=torch.long)
    if code_start > prefix_len:
        prev_len = code_start - prefix_len
        prev_offsets = torch.arange(prev_len, device=device, dtype=torch.long)
        block_idx[prefix_len:code_start] = prev_offsets // int(block_length) + 1
    cur_block_id = int(answer_block_idx) + 1
    block_idx[code_start:code_start + cur_block_len] = cur_block_id
    allow = block_idx.unsqueeze(1) >= block_idx.unsqueeze(0)
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(~allow.unsqueeze(0).unsqueeze(0), neg_inf)
    return mask


def type_ids_for_text_answer_tokens(
    token_ids,
    *,
    special_type1_ids,
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Assign type ids matching SFT training (TYPE_TEXT_OR_SPECIAL=1 for core specials)."""
    special = {int(x) for x in special_type1_ids}
    out = [1 if int(t) in special else 3 for t in token_ids]
    return torch.tensor(out, dtype=dtype, device=device)


def build_t2i_attention_mask_4d(
    type_position_ids: torch.Tensor,
    user_prefix_len: int,
    media_region_start: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """SFT T2I infer mask: noisy image sees clean prefix; clean rows do not see noisy image.

    Layout (matches infer_t2i_sft.py):
      - [0, user_prefix_len): user turn, clean
      - [user_prefix_len, S): image generation turn (BOI + image + EOI + EOA), noisy

    Uses the same m_visual_* / m_clean_* rules as packing_sft training.
    """
    tp = type_position_ids[0] if type_position_ids.dim() == 2 else type_position_ids
    seq_len = int(tp.shape[0])
    user_prefix_len = int(user_prefix_len)
    media_region_start = int(media_region_start)

    turn_ids = torch.zeros(seq_len, device=device, dtype=torch.long)
    is_clean = torch.zeros(seq_len, device=device, dtype=torch.long)

    is_clean[:user_prefix_len] = 1
    turn_ids[:user_prefix_len] = 0

    turn_ids[user_prefix_len:] = 1
    is_clean[user_prefix_len:] = 0

    block_ids = torch.zeros(seq_len, device=device, dtype=torch.long)

    clean_q = is_clean.unsqueeze(-1).bool()
    clean_kv = is_clean.unsqueeze(-2).bool()
    turn_q = turn_ids.unsqueeze(-1)
    turn_kv = turn_ids.unsqueeze(-2)
    type_q = tp.unsqueeze(-1).to(device)
    type_kv = tp.unsqueeze(-2).to(device)
    block_q = block_ids.unsqueeze(-1)
    block_kv = block_ids.unsqueeze(-2)

    same_turn = turn_q == turn_kv
    prev_turn = turn_kv < turn_q
    is_text_q = type_q == 3
    is_text_kv = type_kv == 3
    is_visual_q = type_q == 4
    is_special_kv = type_kv == 1
    is_non_text_kv = type_kv != 3

    m_text_nn = (~clean_q) & is_text_q & (~clean_kv) & same_turn & (
        (is_text_kv & (block_q == block_kv))
        | (is_special_kv & (block_kv <= block_q))
    )
    m_text_nc_prev = (~clean_q) & is_text_q & clean_kv & prev_turn
    m_text_nc_same_prior = (~clean_q) & is_text_q & clean_kv & same_turn & (
        is_text_kv & (block_kv < block_q)
    )
    m_visual_nn = (~clean_q) & is_visual_q & (~clean_kv) & same_turn
    m_visual_nc = (~clean_q) & is_visual_q & clean_kv & prev_turn
    m_other_noisy = (~clean_q) & (~is_text_q) & (~is_visual_q) & (
        ((~clean_kv) & same_turn & is_non_text_kv)
        | (clean_kv & prev_turn)
    )
    m_clean_text = clean_q & is_text_q & clean_kv & (
        prev_turn
        | (same_turn & (
            (is_text_kv & (block_kv <= block_q))
            | (is_special_kv & (block_kv <= block_q))
        ))
    )
    m_clean_other = clean_q & (~is_text_q) & clean_kv & (
        prev_turn | (same_turn & is_non_text_kv)
    )

    bd_allowed = (
        m_text_nn | m_text_nc_prev | m_text_nc_same_prior
        | m_visual_nn | m_visual_nc
        | m_other_noisy
        | m_clean_text | m_clean_other
    )

    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(~bd_allowed.unsqueeze(0).unsqueeze(0), neg_inf)
    return mask


def build_text_attention_mask_4d_from_meta(
    type_ids: torch.Tensor,
    turn_ids: torch.Tensor,
    is_clean: torch.Tensor,
    block_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build text SFT mask from explicit per-token metadata."""
    tp = type_ids[0] if type_ids.dim() == 2 else type_ids
    turn = turn_ids[0] if turn_ids.dim() == 2 else turn_ids
    clean = is_clean[0] if is_clean.dim() == 2 else is_clean
    block = block_ids[0] if block_ids.dim() == 2 else block_ids

    type_q = tp.unsqueeze(-1).to(device)
    type_kv = tp.unsqueeze(-2).to(device)
    turn_q = turn.unsqueeze(-1).to(device)
    turn_kv = turn.unsqueeze(-2).to(device)
    clean_q = clean.unsqueeze(-1).to(device).bool()
    clean_kv = clean.unsqueeze(-2).to(device).bool()
    block_q = block.unsqueeze(-1).to(device)
    block_kv = block.unsqueeze(-2).to(device)

    same_turn = turn_q == turn_kv
    prev_turn = turn_kv < turn_q
    is_text_q = type_q == 3
    is_text_kv = type_kv == 3
    is_visual_q = type_q == 4
    is_special_kv = type_kv == 1
    is_non_text_kv = type_kv != 3

    m_text_nn = (~clean_q) & is_text_q & (~clean_kv) & same_turn & (
        (is_text_kv & (block_q == block_kv))
        | (is_special_kv & (block_kv <= block_q))
    )
    m_text_nc_prev = (~clean_q) & is_text_q & clean_kv & prev_turn
    m_text_nc_same_prior = (~clean_q) & is_text_q & clean_kv & same_turn & (
        is_text_kv & (block_kv < block_q)
    )
    m_visual_nn = (~clean_q) & is_visual_q & (~clean_kv) & same_turn
    m_visual_nc = (~clean_q) & is_visual_q & clean_kv & prev_turn
    m_other_noisy = (~clean_q) & (~is_text_q) & (~is_visual_q) & (
        ((~clean_kv) & same_turn & is_non_text_kv)
        | (clean_kv & prev_turn)
    )
    m_clean_text = clean_q & is_text_q & clean_kv & (
        prev_turn
        | (same_turn & (
            (is_text_kv & (block_kv <= block_q))
            | (is_special_kv & (block_kv <= block_q))
        ))
    )
    m_clean_other = clean_q & (~is_text_q) & clean_kv & (
        prev_turn | (same_turn & is_non_text_kv)
    )

    bd_allowed = (
        m_text_nn | m_text_nc_prev | m_text_nc_same_prior
        | m_visual_nn | m_visual_nc
        | m_other_noisy
        | m_clean_text | m_clean_other
    )
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, tp.shape[0], tp.shape[0], device=device, dtype=dtype)
    mask.masked_fill_(~bd_allowed.unsqueeze(0).unsqueeze(0), neg_inf)
    return mask


@torch.no_grad()
def generate_text_bd(
    model,
    prefix_tokens: torch.Tensor,
    prefix_type_ids: torch.Tensor,
    gen_length: int,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 156895,
    answer_type_id: int = 3,
    early_stop_tokens: Optional[List[int]] = None,
    tokenizer=None,
    prefix_turn_ids: Optional[torch.Tensor] = None,
    prefix_block_ids: Optional[torch.Tensor] = None,
    prefix_is_clean: Optional[torch.Tensor] = None,
    prefix_position_ids: Optional[torch.Tensor] = None,
    generated_turn_id: Optional[int] = None,
    generated_position_start: Optional[int] = None,
    require_meta_mask: bool = False,
    steps_per_block: Optional[int] = None,
):
    """
    Block-diffusion text generation with LLaDA2-Uni-style layout:
      Each block sees only [prefix + clean_prev_blocks + mask_current_block].
      No future masked blocks are visible.

    Supports early-stop: if any token in early_stop_tokens appears in a
    decoded block, generation stops and output is truncated at that token.

    Args:
        prefix_tokens: (prefix_len, num_codebooks) - clean input prefix
        prefix_type_ids: (prefix_len,) - type ids for prefix
        gen_length: max number of answer tokens to generate
        block_length: block size for block diffusion
        steps: denoising steps per generated block when steps_per_block is omitted
        temperature: sampling temperature (0 = greedy)
        cfg_scale: CFG scale for text (typically 0)
        remasking: 'low_confidence' or 'random'
        mask_id: mask token id
        answer_type_id: type id for answer tokens (default 3)
        early_stop_tokens: list of token ids that signal generation end
        tokenizer: optional, for debug printing
        steps_per_block: denoising steps independently applied to every block.
            When omitted, the value of steps is used for each block.

    Returns:
        generated_tokens: 1D tensor of generated token ids (first codebook),
                          length <= gen_length, truncated at early-stop token.
    """
    device = next(model.parameters()).device
    prefix_tokens = prefix_tokens.to(device)
    prefix_type_ids = prefix_type_ids.to(device)
    num_codebooks = prefix_tokens.shape[1]
    if prefix_turn_ids is not None:
        prefix_turn_ids = prefix_turn_ids.to(device).long()
    if prefix_block_ids is not None:
        prefix_block_ids = prefix_block_ids.to(device).long()
    if prefix_is_clean is not None:
        prefix_is_clean = prefix_is_clean.to(device).long()
    if prefix_position_ids is not None:
        prefix_position_ids = prefix_position_ids.to(device).long()

    use_meta_mask = (
        prefix_turn_ids is not None
        and prefix_block_ids is not None
        and prefix_is_clean is not None
    )
    if require_meta_mask and not use_meta_mask:
        raise ValueError("require_meta_mask=True but prefix metadata is incomplete")
    if require_meta_mask and prefix_position_ids is None:
        raise ValueError("require_meta_mask=True but prefix_position_ids is missing")
    if require_meta_mask:
        pn = int(prefix_tokens.shape[0])
        for name, t in [
            ("prefix_type_ids", prefix_type_ids),
            ("prefix_turn_ids", prefix_turn_ids),
            ("prefix_block_ids", prefix_block_ids),
            ("prefix_is_clean", prefix_is_clean),
            ("prefix_position_ids", prefix_position_ids),
        ]:
            if t is not None and int(t.shape[0]) != pn:
                raise ValueError(
                    f"{name} length {int(t.shape[0])} != prefix_tokens length {pn}"
                )
    if use_meta_mask and generated_turn_id is None:
        generated_turn_id = int(prefix_turn_ids.max().item()) + 1 if prefix_turn_ids.numel() > 0 else 1 # 将generate的turn id
    if generated_position_start is None:
        generated_position_start = int(prefix_tokens.shape[0])

    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    if steps_per_block is None:
        steps_per_block = max(int(steps), 1)
    else:
        steps_per_block = max(int(steps_per_block), 1)

    if early_stop_tokens is None:
        early_stop_tokens = []
    early_stop_set = set(early_stop_tokens)

    all_decoded = []  # list of 1D tensors, each of length block_length

    for blk_idx in range(num_blocks):
        # Remaining tokens to generate
        remaining = gen_length - blk_idx * block_length
        cur_block_len = min(block_length, remaining)

        # Build input: prefix + prev_decoded + MASK*cur_block_len
        prev_decoded_2d = None
        if all_decoded:
            prev_cat = torch.cat(all_decoded, dim=0)  # (prev_len,)
            prev_decoded_2d = prev_cat.unsqueeze(1).expand(-1, num_codebooks).to(device)

        mask_block = torch.full(
            (cur_block_len, num_codebooks), mask_id, dtype=torch.long, device=device
        )

        if prev_decoded_2d is not None:
            x = torch.cat([prefix_tokens, prev_decoded_2d, mask_block], dim=0)
        else:
            x = torch.cat([prefix_tokens, mask_block], dim=0)

        # Build type_ids
        prev_len = sum(d.shape[0] for d in all_decoded) if all_decoded else 0 # 已经generate的文本token长度
        answer_part_type = torch.full(
            (prev_len + cur_block_len,), answer_type_id, dtype=torch.long, device=device
        )
        type_ids = torch.cat([prefix_type_ids, answer_part_type], dim=0) # 构造answer部分的type（包括之前已经generate的文本和当前block的文本）

        x = x.unsqueeze(0)  # (1, seq_len, codebooks)
        type_ids_batch = type_ids.unsqueeze(0)
        seq_len = x.shape[1]
        prefix_len = prefix_tokens.shape[0]
        code_start = prefix_len + prev_len
        if use_meta_mask:
            prev_block_ids = torch.arange(prev_len, device=device, dtype=torch.long) // int(block_length)
            cur_block_ids = torch.full((cur_block_len,), blk_idx, dtype=torch.long, device=device)
            block_ids = torch.cat([prefix_block_ids, prev_block_ids, cur_block_ids], dim=0) # instruction & user prefix全为-1，answer的blk从0开始

            turn_ids = torch.cat([
                prefix_turn_ids,
                torch.full((prev_len + cur_block_len,), int(generated_turn_id), dtype=torch.long, device=device),  # instruction从0开始
            ], dim=0)
            is_clean = torch.cat([
                prefix_is_clean,
                torch.ones(prev_len, dtype=torch.long, device=device),
                torch.zeros(cur_block_len, dtype=torch.long, device=device), # instruction+clean prefix都是1，要generate的blk是0，只有is_clean是随着每个blk推理变化的
            ], dim=0)
            if prefix_position_ids is not None:
                gen_pos = torch.arange(
                    int(generated_position_start),
                    int(generated_position_start) + prev_len + cur_block_len,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = torch.cat([prefix_position_ids, gen_pos], dim=0)
            else:
                position_ids = None
            is_clean_batch = is_clean.unsqueeze(0)
            attn_mask = build_text_attention_mask_4d_from_meta(
                type_ids,
                turn_ids,
                is_clean,
                block_ids,
                device,
                next(model.parameters()).dtype,
            )
        else:
            # Prefix and previously decoded blocks are clean; only current mask block is noisy.
            is_clean = torch.zeros(seq_len, dtype=torch.long, device=device)
            is_clean[:code_start] = 1
            position_ids = None
            is_clean_batch = is_clean.unsqueeze(0)
            attn_mask = build_bd_infer_attention_mask_4d(
                seq_len,
                prefix_len,
                code_start,
                cur_block_len,
                block_length,
                blk_idx,
                device,
                next(model.parameters()).dtype,
            )
        block_mask_index = (x[0, code_start:code_start + cur_block_len, 0] == mask_id).unsqueeze(0)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block) # [[steps_block0, steps_block1, ...]]

        prompt_index = (x[:, :, 0] != mask_id)

        for step_i in range(steps_per_block):
            mask_index = (x[:, :, 0] == mask_id)

            logits = model(
                x, type_ids_batch, attention_mask=attn_mask, infer_mmu=True,
                is_clean=is_clean_batch,
                position_ids=position_ids.unsqueeze(0) if position_ids is not None else None,
            )

            # Map logits back to sequence positions (model returns only noisy type=3 positions)
            type_flat = type_ids_batch.reshape(-1).to(logits.device)
            is_clean_flat = is_clean_batch.reshape(-1).to(logits.device)
            text_mask_flat = (type_flat == answer_type_id) & (is_clean_flat == 0)
            text_positions_flat = torch.nonzero(text_mask_flat, as_tuple=False).view(-1)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0_mask = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p_mask = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0_mask, -1)), -1)
            elif remasking == 'random':
                x0_p_mask = torch.rand((x0_mask.shape[0],), device=x0_mask.device)
            else:
                raise NotImplementedError(remasking)

            # Scatter predictions back to full sequence
            x0_full = x[:, :, 0].reshape(-1).clone()
            x0_full[text_mask_flat] = x0_mask
            x0_full = x0_full.view(x.shape[0], x.shape[1])

            confidence_full = torch.full_like(x0_full, -np.inf, dtype=x0_p_mask.dtype)
            confidence_full_flat = confidence_full.reshape(-1)
            confidence_full_flat[text_mask_flat] = x0_p_mask
            confidence_full = confidence_full_flat.view_as(x0_full)

            x0 = torch.where(mask_index, x0_full, x[:, :, 0])
            confidence = torch.where(mask_index, confidence_full, -np.inf)
            
            # Select tokens to commit
            k = int(num_transfer_tokens[0, step_i].item()) # number of tokens to unmask
            if k > 0:
                _, select_index = torch.topk(confidence[0, code_start:code_start + cur_block_len], k=k)
                select_index = select_index + code_start
                x[0, select_index, 0] = x0[0, select_index]
                # Replicate across codebooks for answer positions
                x[0, select_index, 1:] = x[0, select_index, 0:1].expand(-1, num_codebooks - 1)

        # Extract decoded block
        decoded_block = x[0, code_start:code_start + cur_block_len, 0]
        all_decoded.append(decoded_block)
        # Early-stop check
        if early_stop_set:
            for pos, tok_id in enumerate(decoded_block.tolist()):
                if tok_id in early_stop_set:
                    # Truncate at this position (exclusive of stop token)
                    final_decoded = torch.cat(all_decoded[:-1], dim=0) if all_decoded[:-1] else torch.tensor([], dtype=torch.long, device=device)
                    final_decoded = torch.cat([final_decoded, decoded_block[:pos]], dim=0)
                    if tokenizer:
                        print(f"  [early-stop] Found token {tok_id} at block {blk_idx} pos {pos}, total len={final_decoded.shape[0]}")
                    return final_decoded

    return torch.cat(all_decoded, dim=0)
