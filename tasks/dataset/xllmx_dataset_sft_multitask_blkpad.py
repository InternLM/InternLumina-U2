"""Multitask SFT dataset for LLaDA2 mode-8 block-diffusion training .

Layout is [clean prefix/user turns + noisy generation turns | clean target memory].
- User/media condition turns are clean-only.
- Text answer turns are noisy + clean, with type=3 and fixed-size block ids.
- Final image answer turns are noisy-only; image tokens stay fully bidirectional.
Appended clean target memory reuses the corresponding noisy target position_ids,
matching the two-stream BD training layout.

The text objective matches BD pretrain:
1. Text mask id 156895 (<|mask|>, LLaDA2's own pretrained mask embedding).
2. Terminal answer turns pad the final block with real EOA BEFORE masking, so the pad
   takes part in its block's noise draw and is supervised with EOA (teaches
   termination).
3. Per-block noise: each text block independently draws a mask ratio (3% fully
   masked, otherwise cosine).

Split think tasks (t2i_think / under_think) pad the non-terminal think turn with
absorbing </think> targets whose total loss weight is capped at one natural token.
By default t2i_think stays fully conditional (no CFG dropout); drop_prompt_and_think
and legacy_keep_think reproduce the two earlier dropout layouts for lineage
experiments. Pair with the train task/config and the matching inference entrypoints.
"""

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tasks.dataset.edit_biased_mask import apply_image_mask, biased_mask_requested
from tasks.dataset.item_processor import ItemProcessorBase
from tasks.dataset.xllmx_dataset_base import FinetuneConversationDataset
from utils.media_layout import (
    add_break_line_multi_codebook,
    add_break_line_multi_codebook_3d,
    add_break_line_multi_codebook_video,
    make_special_2d_with_label,
    normalize_message_text_content,
)
from utils.media_token_io import (
    load_3d_tokens_from_rclone,
    load_image_tokens_from_rclone,
    load_video_tokens_from_rclone,
)

IMG_CONTEXT_TOKEN_ID = 157160
BOI_TOKEN_ID = 157167
EOI_TOKEN_ID = 157168
IMAGE_MASK_ID = 5000
NEW_LINE_ID = 5001
LINE_3D_ID = 5002
LAYER_3D_ID = 5003
LINE_VIDEO_ID = 5004
FRAME_VIDEO_ID = 5005

SYSTEM_START_TOKEN_ID = 157153
SYSTEM_END_TOKEN_ID = 157154
USER_START_TOKEN_ID = 157155
UNCONDITION_TOKEN_ID = 157156
USER_END_TOKEN_ID = 157157
CONTEXT_3D_TOKEN_ID = 157158
VIDEO_CONTEXT_TOKEN_ID = 157161
BOA_TOKEN_ID = 157165
EOA_TOKEN_ID = 157166
BO3D_TOKEN_ID = 157169
EO3D_TOKEN_ID = 157170
BOV_TOKEN_ID = 157171
EOV_TOKEN_ID = 157172
THINK_OPEN_TOKEN_ID = 157175
THINK_CLOSE_TOKEN_ID = 157176

TYPE_TEXT_OR_SPECIAL = 1
TYPE_MEDIA_PROMPT = 2
TYPE_TEXT_ANSWER = 3
TYPE_MEDIA_ANSWER = 4

PLACEHOLDER_TO_MEDIA = {
    IMG_CONTEXT_TOKEN_ID: "image_url",
    CONTEXT_3D_TOKEN_ID: "3d_url",
    VIDEO_CONTEXT_TOKEN_ID: "video_url",
}
MEDIA_TO_PLACEHOLDER_TEXT = {
    "image_url": "<IMG_CONTEXT>",
    "3d_url": "<3D_CONTEXT>",
    "video_url": "<VIDEO_CONTEXT>",
}
MEDIA_TO_BOUNDARY = {
    "image_url": (BOI_TOKEN_ID, EOI_TOKEN_ID),
    "3d_url": (BO3D_TOKEN_ID, EO3D_TOKEN_ID),
    "video_url": (BOV_TOKEN_ID, EOV_TOKEN_ID),
}

SPECIAL_TOKEN_IDS = {
    "<system>": SYSTEM_START_TOKEN_ID,
    "</system>": SYSTEM_END_TOKEN_ID,
    "<user>": USER_START_TOKEN_ID,
    "<uncondition>": UNCONDITION_TOKEN_ID,
    "</user>": USER_END_TOKEN_ID,
    "<3D_CONTEXT>": CONTEXT_3D_TOKEN_ID,
    "<IMG_CONTEXT>": IMG_CONTEXT_TOKEN_ID,
    "<VIDEO_CONTEXT>": VIDEO_CONTEXT_TOKEN_ID,
    "<answer>": BOA_TOKEN_ID,
    "</answer>": EOA_TOKEN_ID,
    "<BOI>": BOI_TOKEN_ID,
    "<EOI>": EOI_TOKEN_ID,
    "<BO3d>": BO3D_TOKEN_ID,
    "<EO3d>": EO3D_TOKEN_ID,
    "<BOV>": BOV_TOKEN_ID,
    "<EOV>": EOV_TOKEN_ID,
    "<think>": THINK_OPEN_TOKEN_ID,
    "</think>": THINK_CLOSE_TOKEN_ID,
}

# Supported task types.  Each one has a corresponding layout in `process_item`.
#
# Note the deliberate asymmetry on the `*_think` pair: `t2i_think` and
# `under_think` are trainable here (reasoning is split into its own turn), but
# the released inference entrypoints do not expose think decoding.  See the
# README's Training section.
TEXT_UNDERSTANDING_TASKS = {"text", "under", "under_think", "under_video", "under_3d"}
IMAGE_OUTPUT_TASKS = {"t2i", "t2i_think", "single_edit"}
# Tasks whose conditioning may be dropped to train the unconditional prior used
# by classifier-free guidance at inference time.
UNCONDITION_TASKS = {"t2i", "t2i_think", "single_edit"}
LENGTH_ESTIMATE_MARGIN = 1024
MAX_SAMPLE_ALL_LEN = 18432


@dataclass
class Turn:
    noisy_tokens: torch.Tensor
    clean_tokens: torch.Tensor
    noisy_labels: torch.Tensor
    type_ids: torch.Tensor
    block_ids: torch.Tensor
    turn_id: int
    include_noisy: bool
    include_clean: bool
    raw_media_tokens: Optional[List[torch.Tensor]] = None
    loss_weights: Optional[torch.Tensor] = None


@dataclass
class MediaItem:
    media_type: str
    placeholder_id: int
    start_id: int
    end_id: int
    tokens: torch.Tensor
    labels: torch.Tensor
    raw_tokens: Optional[torch.Tensor] = None


def verify_sft_special_token_ids(tokenizer) -> None:
    errors = []
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        enc = tokenizer.encode(token, add_special_tokens=False)
        dec = tokenizer.convert_ids_to_tokens(expected_id)
        if enc != [expected_id]:
            errors.append(f"{token!r} encodes as {enc}, expected [{expected_id}]")
        if dec != token:
            errors.append(f"id {expected_id} decodes as {dec!r}, expected {token!r}")
    if errors:
        raise ValueError("SFT tokenizer special-token mismatch:\n" + "\n".join(f"  - {e}" for e in errors))


def _as_2d(ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.int64).unsqueeze(1).repeat(1, 8)


def _tokenize(tokenizer, text: str, max_length: int) -> List[int]:
    return tokenizer(
        text or "",
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    ).input_ids[0].tolist()


def _block_ids(n: int, block_size: int) -> torch.Tensor:
    if n <= 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.arange(n, dtype=torch.long) // int(block_size)


TEXT_CORE_SPECIAL_TOKEN_IDS = {THINK_OPEN_TOKEN_ID}


def _text_never_mask_ids(tokenizer) -> set:
    ids = set(int(x) for x in SPECIAL_TOKEN_IDS.values())
    ids.update(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    ids.discard(EOA_TOKEN_ID)
    ids.discard(THINK_CLOSE_TOKEN_ID)
    return ids


def _mask_media_tokens(
    clean_tokens: torch.Tensor,
    reference_tokens: Optional[torch.Tensor] = None,
    use_biased_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return apply_image_mask(
        clean_tokens,
        IMAGE_MASK_ID,
        reference_tokens=reference_tokens,
        use_biased_mask=use_biased_mask,
    )


def _iter_content_parts(content):
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, str):
                yield {"type": "text", "text": item}
    elif isinstance(content, str):
        yield {"type": "text", "text": content}


def _extract_user_content(content) -> Tuple[str, List[dict]]:
    text_parts: List[str] = []
    media_parts: List[dict] = []
    for part in _iter_content_parts(content):
        ptype = part.get("type")
        if ptype == "text":
            text_parts.append(part.get("text", ""))
        elif ptype in MEDIA_TO_BOUNDARY:
            media_parts.append(part)
    return "\n".join([x for x in text_parts if x]).strip(), media_parts


def _assistant_text(content) -> str:
    return normalize_message_text_content(content).strip()


def _assistant_media_parts(content) -> List[dict]:
    return [part for part in _iter_content_parts(content) if part.get("type") in MEDIA_TO_BOUNDARY]


def _strip_placeholders(text: str) -> str:
    for placeholder in MEDIA_TO_PLACEHOLDER_TEXT.values():
        text = text.replace(placeholder, "")
    return text.strip()


def _split_think_and_answer(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    m = re.search(r"</think>", text)
    if not m:
        return text.strip(), ""
    think = text[: m.end()].strip()
    answer = text[m.end() :].strip()
    return think, answer


def _metadata_expected_answer(data_item: dict) -> str:
    metadata = data_item.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    if not isinstance(metadata, dict):
        return ""
    for key in ("expected_answer", "answer", "final_answer"):
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return ""


def _format_user_turn_text(system_prompt: str, user_text: str, first_user: bool) -> str:
    if first_user:
        return f"<system>{system_prompt or ''}</system><user>{user_text or ''}</user>"
    return f"<user>{user_text or ''}</user>"


def _load_media_prompt(part: dict) -> MediaItem:
    media_type = part.get("type")
    token_path = (part.get("token") or {}).get("token_path")
    if not token_path:
        raise ValueError(f"{media_type} item missing token.token_path")

    raw_tokens = None
    if media_type == "image_url":
        tokens, h, w = load_image_tokens_from_rclone(token_path)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64)
        raw_tokens = tokens.clone()
        labels = torch.full_like(tokens, -100)
        tokens = add_break_line_multi_codebook(tokens, h, w, new_number=NEW_LINE_ID)
        labels = add_break_line_multi_codebook(labels, h, w, new_number=-100)
    elif media_type == "3d_url":
        tokens, coords, h, w, d = load_3d_tokens_from_rclone(token_path)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64)
        labels = torch.full_like(tokens, -100)
        tokens = add_break_line_multi_codebook_3d(tokens, coords, h, w, d, new_number=LINE_3D_ID, layer_end_number=LAYER_3D_ID)
        labels = add_break_line_multi_codebook_3d(labels, coords, h, w, d, new_number=-100, layer_end_number=-100)
    elif media_type == "video_url":
        tokens, coords, t, h, w = load_video_tokens_from_rclone(token_path)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64)
        labels = torch.full_like(tokens, -100)
        tokens = add_break_line_multi_codebook_video(tokens, coords, t, h, w, new_number=LINE_VIDEO_ID, frame_end_number=FRAME_VIDEO_ID)
        labels = add_break_line_multi_codebook_video(labels, coords, t, h, w, new_number=-100, frame_end_number=-100)
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

    start_id, end_id = MEDIA_TO_BOUNDARY[media_type]
    placeholder_id = {
        "image_url": IMG_CONTEXT_TOKEN_ID,
        "3d_url": CONTEXT_3D_TOKEN_ID,
        "video_url": VIDEO_CONTEXT_TOKEN_ID,
    }[media_type]
    return MediaItem(media_type, placeholder_id, start_id, end_id, tokens, labels, raw_tokens)


def _load_media_answer(
    part: dict,
    reference_tokens: Optional[torch.Tensor] = None,
    use_biased_mask: bool = False,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    media_type = part.get("type")
    token_path = (part.get("token") or {}).get("token_path")
    if media_type != "image_url":
        raise ValueError(f"SFT media generation currently supports image_url outputs, got {media_type}")
    if not token_path:
        raise ValueError("image output missing token.token_path")
    clean_tokens, h, w = load_image_tokens_from_rclone(token_path)
    if not isinstance(clean_tokens, torch.Tensor):
        clean_tokens = torch.tensor(clean_tokens, dtype=torch.int64)
    noisy_tokens, labels = _mask_media_tokens(
        clean_tokens,
        reference_tokens=reference_tokens,
        use_biased_mask=use_biased_mask,
    )
    clean_tokens = add_break_line_multi_codebook(clean_tokens, h, w, new_number=NEW_LINE_ID)
    noisy_tokens = add_break_line_multi_codebook(noisy_tokens, h, w, new_number=NEW_LINE_ID)
    labels = add_break_line_multi_codebook(labels, h, w, new_number=-100)
    return media_type, noisy_tokens, clean_tokens, labels


def _replace_media_placeholders(token_ids: List[int], media_items: List[MediaItem]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_2d = _as_2d(token_ids)
    text_labels = torch.full_like(text_2d, -100)
    media_by_placeholder: Dict[int, List[MediaItem]] = {}
    for item in media_items:
        media_by_placeholder.setdefault(item.placeholder_id, []).append(item)

    chunks_t: List[torch.Tensor] = []
    chunks_l: List[torch.Tensor] = []
    chunks_type: List[torch.Tensor] = []
    start = 0
    used_media = 0
    for i, tok in enumerate(token_ids):
        if tok not in PLACEHOLDER_TO_MEDIA or not media_by_placeholder.get(tok):
            continue
        item = media_by_placeholder[tok].pop(0)
        if i > start:
            chunks_t.append(text_2d[start:i])
            chunks_l.append(text_labels[start:i])
            chunks_type.append(torch.full((i - start,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long))
        bo_t, bo_l = make_special_2d_with_label(item.start_id)
        eo_t, eo_l = make_special_2d_with_label(item.end_id)
        chunks_t.extend([bo_t, item.tokens, eo_t])
        chunks_l.extend([bo_l, item.labels, eo_l])
        chunks_type.extend([
            torch.ones(1, dtype=torch.long),
            torch.full((item.tokens.shape[0],), TYPE_MEDIA_PROMPT, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
        ])
        start = i + 1
        used_media += 1

    if start < len(token_ids):
        chunks_t.append(text_2d[start:])
        chunks_l.append(text_labels[start:])
        chunks_type.append(torch.full((len(token_ids) - start,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long))

    leftovers: List[MediaItem] = []
    for queue in media_by_placeholder.values():
        leftovers.extend(queue)
    if leftovers:
        prefix_t: List[torch.Tensor] = []
        prefix_l: List[torch.Tensor] = []
        prefix_type: List[torch.Tensor] = []
        for item in leftovers:
            bo_t, bo_l = make_special_2d_with_label(item.start_id)
            eo_t, eo_l = make_special_2d_with_label(item.end_id)
            prefix_t.extend([bo_t, item.tokens, eo_t])
            prefix_l.extend([bo_l, item.labels, eo_l])
            prefix_type.extend([
                torch.ones(1, dtype=torch.long),
                torch.full((item.tokens.shape[0],), TYPE_MEDIA_PROMPT, dtype=torch.long),
                torch.ones(1, dtype=torch.long),
            ])
        chunks_t = prefix_t + chunks_t
        chunks_l = prefix_l + chunks_l
        chunks_type = prefix_type + chunks_type

    if not chunks_t:
        return text_2d, text_labels, torch.ones(text_2d.shape[0], dtype=torch.long)
    return torch.cat(chunks_t, dim=0), torch.cat(chunks_l, dim=0), torch.cat(chunks_type, dim=0)


def _make_user_turn(
    tokenizer,
    system_prompt: str,
    content,
    first_user: bool,
    turn_id: int,
    max_text_length: int,
    force_uncondition: bool = False,
) -> Turn:
    user_text, media_parts = _extract_user_content(content)
    media_items = [] if force_uncondition else [_load_media_prompt(part) for part in media_parts]
    if force_uncondition:
        user_text = "<uncondition>"
    elif media_items:
        missing_placeholders = [
            MEDIA_TO_PLACEHOLDER_TEXT[item.media_type]
            for item in media_items
            if MEDIA_TO_PLACEHOLDER_TEXT[item.media_type] not in user_text
        ]
        if len(missing_placeholders) == len(media_items):
            user_text = "\n".join(missing_placeholders + ([user_text] if user_text else []))

    wrapped = _format_user_turn_text(system_prompt, user_text, first_user)
    token_ids = _tokenize(tokenizer, wrapped, max_text_length)
    tokens, labels, type_ids = _replace_media_placeholders(token_ids, media_items)
    block_ids = torch.full((tokens.shape[0],), -1, dtype=torch.long)
    raw_media_tokens = [item.raw_tokens for item in media_items if item.raw_tokens is not None]
    return Turn(
        noisy_tokens=tokens,
        clean_tokens=tokens.clone(),
        noisy_labels=labels,
        type_ids=type_ids,
        block_ids=block_ids,
        turn_id=turn_id,
        include_noisy=False,
        include_clean=True,
        raw_media_tokens=raw_media_tokens,
    )


def _make_media_answer_turn(
    media_part: dict,
    turn_id: int,
    add_boa: bool,
    add_eoa: bool,
    include_clean: bool,
    reference_tokens: Optional[torch.Tensor] = None,
    use_biased_mask: bool = False,
) -> Turn:
    media_type, noisy_media, clean_media, media_labels = _load_media_answer(
        media_part,
        reference_tokens=reference_tokens,
        use_biased_mask=use_biased_mask,
    )
    start_id, end_id = MEDIA_TO_BOUNDARY[media_type]
    chunks_noisy: List[torch.Tensor] = []
    chunks_clean: List[torch.Tensor] = []
    chunks_label: List[torch.Tensor] = []
    chunks_type: List[torch.Tensor] = []

    if add_boa:
        boa_t, boa_l = make_special_2d_with_label(BOA_TOKEN_ID)
        chunks_noisy.append(boa_t)
        chunks_clean.append(boa_t.clone())
        chunks_label.append(boa_l)
        chunks_type.append(torch.ones(1, dtype=torch.long))

    bo_t, bo_l = make_special_2d_with_label(start_id)
    eo_t, eo_l = make_special_2d_with_label(end_id)
    chunks_noisy.extend([bo_t, noisy_media, eo_t])
    chunks_clean.extend([bo_t.clone(), clean_media, eo_t.clone()])
    chunks_label.extend([bo_l, media_labels, eo_l])
    chunks_type.extend([
        torch.ones(1, dtype=torch.long),
        torch.full((noisy_media.shape[0],), TYPE_MEDIA_ANSWER, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
    ])

    if add_eoa:
        eoa_t, eoa_l = make_special_2d_with_label(EOA_TOKEN_ID)
        chunks_noisy.append(eoa_t)
        chunks_clean.append(eoa_t.clone())
        chunks_label.append(eoa_l)
        chunks_type.append(torch.ones(1, dtype=torch.long))

    tokens = torch.cat(chunks_noisy, dim=0)
    clean = torch.cat(chunks_clean, dim=0)
    labels = torch.cat(chunks_label, dim=0)
    type_ids = torch.cat(chunks_type, dim=0)
    block_ids = torch.zeros(tokens.shape[0], dtype=torch.long)
    return Turn(tokens, clean, labels, type_ids, block_ids, turn_id, True, include_clean)


def _pack_turns(turns: List[Turn], task_type: str) -> Tuple[dict]:
    # Pack as [clean prefix/user turns + noisy answer turns | clean answer memory].
    # Clean-only turns stay in dialogue position; assistant answer x0 copies are
    # appended as clean memory and only exposed by the SFT block mask when valid.
    if not any(t.include_noisy for t in turns):
        raise ValueError(f"No generation turn found for SFT task_type={task_type}")
    if not any(t.include_clean for t in turns):
        raise ValueError(f"No clean context turn found for SFT task_type={task_type}")

    token_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    type_parts: List[torch.Tensor] = []
    block_parts: List[torch.Tensor] = []
    turn_parts: List[torch.Tensor] = []
    clean_flag_parts: List[torch.Tensor] = []
    position_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []
    noisy_position_by_turn: Dict[int, torch.Tensor] = {}
    has_loss_weights = any(t.loss_weights is not None for t in turns)

    def append_part(
        tokens: torch.Tensor,
        labels: torch.Tensor,
        type_ids: torch.Tensor,
        block_ids: torch.Tensor,
        turn_id: int,
        clean_flag: int,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> None:
        start_pos = sum(x.shape[0] for x in token_parts)
        if clean_flag == 1 and turn_id in noisy_position_by_turn:
            pos = noisy_position_by_turn[turn_id][:tokens.shape[0]]
        else:
            pos = torch.arange(start_pos, start_pos + tokens.shape[0], dtype=torch.long)
        token_parts.append(tokens)
        label_parts.append(labels)
        type_parts.append(type_ids)
        block_parts.append(block_ids)
        turn_parts.append(torch.full((tokens.shape[0],), turn_id, dtype=torch.long))
        clean_flag_parts.append(torch.full((tokens.shape[0],), clean_flag, dtype=torch.long))
        position_parts.append(pos)
        if has_loss_weights:
            if loss_weights is None:
                loss_weights = torch.ones(tokens.shape[0], dtype=torch.float32)
            if loss_weights.ndim != 1 or loss_weights.shape[0] != tokens.shape[0]:
                raise ValueError("Turn loss_weights must be a sequence-length 1D tensor")
            weight_parts.append(loss_weights.to(dtype=torch.float32))
        if clean_flag == 0:
            noisy_position_by_turn[turn_id] = pos.clone()

    for t in turns:
        if t.include_clean and not t.include_noisy:
            append_part(
                t.clean_tokens,
                torch.full_like(t.clean_tokens, -100),
                t.type_ids,
                t.block_ids,
                t.turn_id,
                1,
            )
        if t.include_noisy:
            append_part(
                t.noisy_tokens,
                t.noisy_labels,
                t.type_ids,
                t.block_ids,
                t.turn_id,
                0,
                t.loss_weights,
            )

    for t in turns:
        if t.include_noisy and t.include_clean:
            clean_len = t.clean_tokens.shape[0]
            append_part(
                t.clean_tokens,
                torch.full_like(t.clean_tokens, -100),
                t.type_ids[:clean_len],
                t.block_ids[:clean_len],
                t.turn_id,
                1,
            )

    all_tokens = torch.cat(token_parts, dim=0)
    all_labels = torch.cat(label_parts, dim=0)
    type_ids = torch.cat(type_parts, dim=0)
    block_ids = torch.cat(block_parts, dim=0)
    turn_ids = torch.cat(turn_parts, dim=0)
    is_clean = torch.cat(clean_flag_parts, dim=0)

    position_ids = torch.cat(position_parts, dim=0)

    data_dict = {
        "input_ids": all_tokens,
        "labels": all_labels,
        "attention_mask": torch.ones(all_tokens.shape[0], dtype=torch.int64),
        "type_position_ids": type_ids,
        "turn_ids": turn_ids,
        "is_clean": is_clean,
        "block_ids": block_ids,
        "position_ids": position_ids,
        "type": task_type,
        "all_len": all_tokens.shape[0],
    }
    if has_loss_weights:
        data_dict["loss_weights"] = torch.cat(weight_parts, dim=0)
    return (data_dict,)



# ---- text mask token -------------------------------------------------------
# LLaDA2's own `<|mask|>`, whose input embedding is pretrained. Must stay in sync
# with infer_pretrain_bd_common_v1.MASK -- train and inference have to agree on
# which id means "predict me".
TEXT_MASK_ID = 156895


# ---- per-block noise (classic block-diffusion training) --------------------
# The SFT engine's `_make_text_turn` samples ONE cosine mask ratio for the whole
# answer turn and only afterwards splits the (already masked) sequence into
# blocks by position. Real block-diffusion training (e.g. BD3LM) instead samples
# an independent noise level per block, so the model sees every combination of
# "how noisy is this block" across the sequence rather than one shared level.
# The helpers below reproduce `_make_text_turn`'s exact segment/offset layout
# (so block_ids/type_ids/blkpad stay identical) but replace the single
# whole-turn `mask_codes` call with independent per-block noise sampling.


def _text_core_block_ids(clean_core: List[int], block_size: int) -> List[int]:
    """Position -> block id for a text-answer core sequence. Mirrors
    `_append_text_core_chunks`'s segment/offset logic exactly (a `<think>`
    token is its own block-0 unit; running offset continues across segments)."""
    block_ids = [0] * len(clean_core)
    i = 0
    n = len(clean_core)
    text_block_offset = 0
    while i < n:
        tok = clean_core[i]
        if tok in TEXT_CORE_SPECIAL_TOKEN_IDS:
            block_ids[i] = 0
            i += 1
            continue
        j = i
        while j < n and clean_core[j] not in TEXT_CORE_SPECIAL_TOKEN_IDS:
            j += 1
        seg_len = j - i
        for k in range(seg_len):
            b = k // block_size
            if text_block_offset > 0:
                b += text_block_offset
            block_ids[i + k] = b
        if seg_len > 0:
            text_block_offset = block_ids[i + seg_len - 1] + 1
        i = j
    return block_ids


def _mask_text_ids_per_block(
    clean_ids: List[int],
    never_mask_ids: Sequence[int],
    block_ids: Sequence[int],
    *,
    force_full_for_short: bool = True,
) -> Tuple[List[int], List[int]]:
    """Like the SFT engine's `_mask_text_ids`, but samples independently per
    block: 3% fully masked, otherwise cosine-scheduled, instead of one ratio for
    all of `clean_ids`. ``force_full_for_short`` preserves the legacy <=5-token
    full-mask rule by default; SFT may disable it for an artificial padding-only draw.
    """
    if not clean_ids:
        return [], []
    never = set(int(x) for x in never_mask_ids)
    noisy_ids = list(clean_ids)
    labels = [-100] * len(clean_ids)

    positions_by_block: Dict[int, List[int]] = {}
    for pos, b in enumerate(block_ids):
        positions_by_block.setdefault(int(b), []).append(pos)

    p_full_mask = 0.03
    for positions in positions_by_block.values():
        candidates = [p for p in positions if int(clean_ids[p]) not in never]
        if not candidates:
            continue
        r = random.uniform(0, 1)
        if (force_full_for_short and len(candidates) <= 5) or r < p_full_mask:
            mask_ratio = 1.0
        else:
            cosine_r = (r - p_full_mask) / (1.0 - p_full_mask)
            mask_ratio = math.cos(cosine_r * math.pi / 2)
        num_to_mask = max(1, int(len(candidates) * mask_ratio))
        for pos in random.sample(candidates, num_to_mask):
            labels[pos] = clean_ids[pos]
            noisy_ids[pos] = TEXT_MASK_ID
    return noisy_ids, labels


def _append_text_core_chunks_bd(
    noisy_core: List[int],
    clean_core: List[int],
    label_core: List[int],
    block_core: List[int],
    chunks_noisy: List[torch.Tensor],
    chunks_clean: List[torch.Tensor],
    chunks_label: List[torch.Tensor],
    chunks_type: List[torch.Tensor],
    chunks_block: List[torch.Tensor],
) -> None:
    """Same chunking as `_append_text_core_chunks`, but block ids are taken
    from an already-computed per-position array instead of recomputed."""
    i = 0
    n = len(clean_core)
    while i < n:
        tok = clean_core[i]
        if tok in TEXT_CORE_SPECIAL_TOKEN_IDS:
            chunks_noisy.append(_as_2d([noisy_core[i]]))
            chunks_clean.append(_as_2d([clean_core[i]]))
            chunks_label.append(_as_2d([label_core[i]]))
            chunks_type.append(torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long))
            chunks_block.append(torch.zeros(1, dtype=torch.long))
            i += 1
            continue
        j = i
        while j < n and clean_core[j] not in TEXT_CORE_SPECIAL_TOKEN_IDS:
            j += 1
        seg_len = j - i
        chunks_noisy.append(_as_2d(noisy_core[i:j]))
        chunks_clean.append(_as_2d(clean_core[i:j]))
        chunks_label.append(_as_2d(label_core[i:j]))
        chunks_type.append(torch.full((seg_len,), TYPE_TEXT_ANSWER, dtype=torch.long))
        chunks_block.append(torch.tensor(block_core[i:j], dtype=torch.long))
        i = j


T2I_THINK_UNCONDITION_MODES = {
    "no_t2i_think_drop",
    "drop_prompt_and_think",
    "legacy_keep_think",
}

# Probability that a terminal answer whose natural length is < 5 tokens has its whole
# natural region (answer + natural EOA) masked at once. The remaining draws fall back to
# the ordinary per-block cosine schedule.
SHORT_ANSWER_FULL_MASK_PROB = 0.6


# Which media part type a flat record's `token_path` becomes, per task type.
FLAT_MEDIA_TYPE = {
    "t2i": "image_url",
    "t2i_think": "image_url",
    "single_edit": "image_url",
    "under": "image_url",
    "under_think": "image_url",
    "under_video": "video_url",
    "under_3d": "3d_url",
}

# Accepted spellings for the two text fields of a flat record. `prompt` and
# `answer` are canonical; the rest are tolerated because real pretrain-era
# metadata uses them.
_FLAT_PROMPT_KEYS = ("prompt", "question", "user_prompt", "instruction")
_FLAT_ANSWER_KEYS = ("answer", "answer_text", "response", "caption")


def _flat_text(data_item: dict, keys) -> str:
    for key in keys:
        value = data_item.get(key)
        if value:
            return value
    return ""


def _flat_media_part(data_item: dict, path_key: str, media_type: str) -> dict:
    token = {"token_path": data_item[path_key]}
    for key in ("token_length", "image_width", "image_height"):
        if key in data_item:
            token[key] = data_item[key]
    return {"type": media_type, "token": token}


def _normalize_flat_item(data_item: dict) -> dict:
    """Map a flat (non-``messages``) record to the canonical messages schema.

    A flat record carries the sample's parts as top-level keys instead of a
    conversation. The shape is fixed by ``task_type``:

    ===================== ========================= =======================
    task_type             user                      assistant
    ===================== ========================= =======================
    t2i, t2i_think        prompt                    token_path (image)
    single_edit           input_token_path + prompt token_path (image)
    text                  prompt                    answer
    under, under_think    token_path + prompt       answer
    under_video           token_path (video)        answer
    under_3d              token_path (3D)           answer
    ===================== ========================= =======================

    ``token_path`` therefore means "this sample's own media": the generated
    image for image-output tasks, the media being asked about for
    understanding tasks. Editing needs both, so its source is
    ``input_token_path``.

    Records that already carry ``messages`` are returned untouched -- the
    messages schema stays the source of truth, and this is only a convenience
    for metadata that predates it.
    """
    if data_item.get("messages"):
        return data_item

    task_type = data_item.get("task_type")
    if task_type not in FLAT_MEDIA_TYPE and task_type != "text":
        return data_item  # let process_item raise with the task-type message

    prompt = _flat_text(data_item, _FLAT_PROMPT_KEYS)
    normalized = dict(data_item)

    if task_type in IMAGE_OUTPUT_TASKS:
        if not data_item.get("token_path"):
            raise ValueError(f"flat {task_type} item missing token_path (the generated image)")
        user_content = [{"type": "text", "text": prompt}]
        if task_type == "single_edit":
            if not data_item.get("input_token_path"):
                raise ValueError(
                    "flat single_edit item missing input_token_path (the image being edited); "
                    "token_path is the edit result, not the source"
                )
            user_content.insert(0, _flat_media_part(data_item, "input_token_path", "image_url"))
        normalized["messages"] = [
            {"role": "user", "content": user_content},
            {"role": "assistant",
             "content": [_flat_media_part(data_item, "token_path", "image_url")]},
        ]
        return normalized

    # Understanding and text: the assistant answers in text.
    answer = _flat_text(data_item, _FLAT_ANSWER_KEYS)
    if not answer:
        raise ValueError(
            f"flat {task_type} item missing an answer; expected one of "
            f"{list(_FLAT_ANSWER_KEYS)}"
        )
    user_content = [{"type": "text", "text": prompt}]
    if task_type != "text":
        if not data_item.get("token_path"):
            raise ValueError(f"flat {task_type} item missing token_path (the media being asked about)")
        # The placeholder is injected by _make_user_turn when absent, so the
        # media part alone is enough here.
        user_content.insert(0, _flat_media_part(data_item, "token_path", FLAT_MEDIA_TYPE[task_type]))
    normalized["messages"] = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]
    return normalized


def _make_text_turn_bd(
    tokenizer,
    text: str,
    turn_id: int,
    block_size: int,
    max_text_length: int,
    add_boa: bool,
    add_eoa: bool,
    ensure_think_close: bool = False,
) -> Turn:
    """Text-answer turn: TEXT_MASK_ID mask, per-block noise, and last-block padding.

    ``add_eoa=True``  (terminal answer): pad the final block with real EOA before
        masking. Natural EOA keeps weight 1; the artificial tail has total weight 1.
        All blocks use the same cosine mask draw, with a 3% full-block-mask probability.
        Exception: when the natural answer is shorter than 5 tokens (EOA and padding
        excluded) it would otherwise be drowned by the ~30 artificial pads sharing its
        block, so the natural region and the artificial tail are drawn separately and
        the natural region is fully masked with probability SHORT_ANSWER_FULL_MASK_PROB.
    ``add_eoa=False`` (a think turn that continues into the answer): NEVER inject EOA --
        that would end the response mid-reasoning. The final block is padded with
        repeated ``</think>`` absorbing targets, which take part in the block's noise
        draw and are supervised when masked.

    Mirrors the BD pretrain text-turn construction except for the ``add_eoa``-gated pad
    token; the clean copy is trimmed to the natural length in both cases so pad
    positions never leak into later turns' clean memory.
    """
    clean_core = _tokenize(tokenizer, text, max_text_length)
    if (
        ensure_think_close
        and clean_core
        and clean_core[0] == THINK_OPEN_TOKEN_ID
        and clean_core[-1] != THINK_CLOSE_TOKEN_ID
    ):
        clean_core.append(THINK_CLOSE_TOKEN_ID)
    if add_eoa and (not clean_core or clean_core[-1] != EOA_TOKEN_ID):
        clean_core.append(EOA_TOKEN_ID)
    if not clean_core:
        clean_core = [EOA_TOKEN_ID]

    natural_core_len = len(clean_core)
    # The answer's own length, excluding the terminal EOA (whether it was already present
    # in the input or appended just above) and any artificial padding added below.
    natural_answer_len = natural_core_len - 1 if clean_core[-1] == EOA_TOKEN_ID else natural_core_len
    short_terminal = add_eoa and natural_answer_len < 5
    core_block_ids = _text_core_block_ids(clean_core, block_size)

    # Fill the final block to block_size: terminal turns pad with EOA, non-terminal think
    # turns pad with absorbing </think>. Count only non-special positions of the last
    # block, so a mid-sequence <think> (which restarts the segment offset) pads to "final
    # block is full" rather than "total is a multiple of block_size".
    pad_token = EOA_TOKEN_ID if add_eoa else THINK_CLOSE_TOKEN_ID
    last_block = core_block_ids[-1]
    last_block_len = sum(
        1
        for tok, blk in zip(clean_core, core_block_ids)
        if blk == last_block and tok not in TEXT_CORE_SPECIAL_TOKEN_IDS
    )
    pad_len = (-last_block_len) % block_size
    if pad_len:
        clean_core = clean_core + [pad_token] * pad_len
        core_block_ids = core_block_ids + [last_block] * pad_len

    prefix_ids = [BOA_TOKEN_ID] if add_boa else []
    full_clean = prefix_ids + clean_core
    full_block_ids = ([0] * len(prefix_ids)) + core_block_ids

    never_mask_ids = set(_text_never_mask_ids(tokenizer))
    if short_terminal:
        # A one- or two-token answer shares its block with ~30 artificial EOA pads, so a
        # single draw dilutes it away. Sample the natural region and the artificial tail
        # independently: the natural side is fully masked with probability
        # SHORT_ANSWER_FULL_MASK_PROB (its candidate count is <= 5, so force_full_for_short
        # is exactly the "mask everything" branch), otherwise it takes the ordinary cosine
        # draw. BOA / <think> stay never-mask in both branches.
        natural_end = len(prefix_ids) + natural_core_len
        full_noisy, full_label = _mask_text_ids_per_block(
            full_clean[:natural_end],
            never_mask_ids,
            full_block_ids[:natural_end],
            force_full_for_short=random.random() < SHORT_ANSWER_FULL_MASK_PROB,
        )
        if pad_len:
            pad_noisy, pad_label = _mask_text_ids_per_block(
                full_clean[natural_end:],
                never_mask_ids,
                full_block_ids[natural_end:],
                force_full_for_short=False,
            )
            full_noisy = full_noisy + pad_noisy
            full_label = full_label + pad_label
    else:
        full_noisy, full_label = _mask_text_ids_per_block(
            full_clean,
            never_mask_ids,
            full_block_ids,
            force_full_for_short=False,
        )

    prefix_len = len(prefix_ids)
    noisy_prefix = full_noisy[:prefix_len]
    label_prefix = full_label[:prefix_len]
    noisy_core = full_noisy[prefix_len:]
    label_core = full_label[prefix_len:]

    chunks_noisy: List[torch.Tensor] = []
    chunks_clean: List[torch.Tensor] = []
    chunks_label: List[torch.Tensor] = []
    chunks_type: List[torch.Tensor] = []
    chunks_block: List[torch.Tensor] = []

    if add_boa:
        chunks_noisy.append(_as_2d(noisy_prefix))
        chunks_clean.append(_as_2d(prefix_ids))
        chunks_label.append(_as_2d(label_prefix))
        boa_type = TYPE_TEXT_ANSWER if label_prefix[0] != -100 else TYPE_TEXT_OR_SPECIAL
        chunks_type.append(torch.full((1,), boa_type, dtype=torch.long))
        chunks_block.append(torch.zeros(1, dtype=torch.long))

    _append_text_core_chunks_bd(
        noisy_core,
        clean_core,
        label_core,
        core_block_ids,
        chunks_noisy,
        chunks_clean,
        chunks_label,
        chunks_type,
        chunks_block,
    )

    tokens = torch.cat(chunks_noisy, dim=0)
    clean = torch.cat(chunks_clean, dim=0)
    labels = torch.cat(chunks_label, dim=0)
    type_ids = torch.cat(chunks_type, dim=0)
    block_ids = torch.cat(chunks_block, dim=0)
    # Pads sit at the tail (noisy side); the clean copy is read only by later turns, so
    # trim it back to the natural length (prefix + real answer) -- matching inference,
    # which truncates at EOA before re-encoding.
    clean = clean[: len(prefix_ids) + natural_core_len]
    # Both pad flavours (EOA and absorbing </think>) are supervised, so every turn
    # carries weights.
    loss_weights = torch.ones(tokens.shape[0], dtype=torch.float32)
    if pad_len:
        # Keep the natural EOA/</think> at weight 1, while treating the entire
        # artificial absorbing tail like one additional natural target.
        loss_weights[-pad_len:] = 1.0 / float(pad_len)
    return Turn(
        tokens,
        clean,
        labels,
        type_ids,
        block_ids,
        turn_id,
        True,
        True,
        loss_weights=loss_weights,
    )


class ItemProcessorBlkPad(ItemProcessorBase):
    def __init__(
        self,
        tokenizer,
        max_len,
        block_size: Optional[int] = 32,
        max_text_length: Optional[int] = 2048,
        force_t2i_uncondition: Optional[bool] = False,
        t2i_think_uncondition_mode: str = "no_t2i_think_drop",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.block_size = int(block_size or 32)
        self.max_text_length = int(max_text_length or 2048)
        self.force_t2i_uncondition = bool(force_t2i_uncondition)
        if t2i_think_uncondition_mode not in T2I_THINK_UNCONDITION_MODES:
            raise ValueError(
                "t2i_think_uncondition_mode must be one of "
                f"{sorted(T2I_THINK_UNCONDITION_MODES)}, got {t2i_think_uncondition_mode!r}"
            )
        self.t2i_think_uncondition_mode = t2i_think_uncondition_mode
        verify_sft_special_token_ids(tokenizer)


    def process_item(self, data_item: dict, training_mode=False) -> Tuple[dict]:
        data_item = _normalize_flat_item(data_item)
        task_type = data_item.get("task_type")
        if task_type not in TEXT_UNDERSTANDING_TASKS and task_type not in IMAGE_OUTPUT_TASKS:
            raise ValueError(
                f"Unsupported SFT task_type: {task_type!r}. Supported: "
                f"{sorted(TEXT_UNDERSTANDING_TASKS | IMAGE_OUTPUT_TASKS)}"
            )

        messages = data_item.get("messages") or []
        if not messages:
            raise ValueError(f"SFT item missing messages for task_type={task_type}")
        system_prompt = data_item.get("system_prompt", "")

        turns: List[Turn] = []
        turn_id = 0
        first_user = True
        pending_boa_open = False
        latest_input_image_tokens: List[torch.Tensor] = []
        sample_forced_uncondition = (
            task_type == "t2i_think"
            and self.t2i_think_uncondition_mode != "no_t2i_think_drop"
            and (self.force_t2i_uncondition or random.random() < 0.1)
        )
        assistant_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]

        for msg_idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                if task_type == "t2i_think":
                    force_uncond = sample_forced_uncondition
                else:
                    force_uncond = (
                        task_type in UNCONDITION_TASKS
                        and (self.force_t2i_uncondition or random.random() < 0.1)
                    )
                user_turn = _make_user_turn(
                    self.tokenizer,
                    system_prompt,
                    content,
                    first_user,
                    turn_id,
                    self.max_text_length,
                    force_uncondition=force_uncond,
                )
                turns.append(user_turn)
                if task_type in IMAGE_OUTPUT_TASKS and user_turn.raw_media_tokens:
                    latest_input_image_tokens.extend(user_turn.raw_media_tokens)
                first_user = False
                turn_id += 1
                continue

            if role != "assistant":
                continue

            media_parts = _assistant_media_parts(content)
            text = _assistant_text(content)
            meaningful_text = _strip_placeholders(text)
            is_last_assistant = msg_idx == assistant_indices[-1] if assistant_indices else True

            split_text_think = task_type in {"t2i_think", "under_think"}
            if split_text_think and meaningful_text:
                think_text, answer_text = _split_think_and_answer(text)
                drop_think = (
                    task_type == "t2i_think"
                    and sample_forced_uncondition
                    and self.t2i_think_uncondition_mode == "drop_prompt_and_think"
                )
                if think_text and not drop_think:
                    turns.append(
                        _make_text_turn_bd(
                            self.tokenizer,
                            think_text,
                            turn_id,
                            self.block_size,
                            self.max_text_length,
                            add_boa=not pending_boa_open,
                            add_eoa=False,
                            ensure_think_close=True,
                        )
                    )
                    pending_boa_open = True
                    turn_id += 1
                if task_type == "under_think":
                    turns.append(
                        _make_text_turn_bd(
                            self.tokenizer,
                            answer_text,
                            turn_id,
                            self.block_size,
                            self.max_text_length,
                            add_boa=not pending_boa_open,
                            add_eoa=True,
                        )
                    )
                    pending_boa_open = False
                    turn_id += 1

            if media_parts:
                output_media = next((p for p in media_parts if p.get("type") == "image_url"), None)
                if output_media is None:
                    raise ValueError(f"Assistant media output for task_type={task_type} has no image_url")
                reference_tokens = latest_input_image_tokens[-1] if latest_input_image_tokens else None
                use_biased_mask = biased_mask_requested(task_type, reference_tokens)
                turns.append(
                    _make_media_answer_turn(
                        output_media,
                        turn_id,
                        add_boa=not pending_boa_open,
                        add_eoa=True,
                        include_clean=not is_last_assistant,
                        reference_tokens=reference_tokens,
                        use_biased_mask=use_biased_mask,
                    )
                )
                pending_boa_open = False
                turn_id += 1
            elif not split_text_think and task_type not in {"t2i_think", "under_think"}:
                turns.append(
                    _make_text_turn_bd(
                        self.tokenizer,
                        text,
                        turn_id,
                        self.block_size,
                        self.max_text_length,
                        add_boa=True,
                        add_eoa=True,
                    )
                )
                turn_id += 1

        result = _pack_turns(turns, task_type)
        sample_len = result[0]["input_ids"].shape[0]
        limit = int(self.max_len) if self.max_len is not None else sample_len
        if sample_len > limit:
            raise ValueError(
                f"Sample too long ({sample_len} > max_len={limit}), "
                f"task_type={task_type}, skipping"
            )
        return result

    def predict_item_token_length(self, data_item: dict) -> int:
        data_item = _normalize_flat_item(data_item)
        messages = data_item.get("messages") or []
        assistant_indices = [
            i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        last_assistant_idx = assistant_indices[-1] if assistant_indices else -1

        def media_est(part: dict) -> int:
            token_len = (part.get("token") or {}).get("token_length", 0)
            if isinstance(token_len, list):
                token_len = token_len[0] if token_len else 0
            token_len = int(token_len or 0)
            if token_len <= 0:
                return int(self.max_len or 0) + 1
            if part.get("type") in ("3d_url", "video_url"):
                # 3D/video are clean-only user turns and appear once in a
                # single packed view. Account for inserted row/layer/frame
                # separators with the same 10% margin used by pretrain BD.
                return token_len + max(token_len // 10, 1) + 8
            return token_len + max(token_len // 64, 0) + 4

        def text_est(text: str) -> int:
            return min(len(text or "")*0.7, self.max_text_length) + 32

        total = 128
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                text, media_parts = _extract_user_content(content)
                total += text_est(text)
                total += sum(media_est(part) for part in media_parts)
            elif role == "assistant":
                text = _strip_placeholders(_assistant_text(content))
                media_parts = _assistant_media_parts(content)
                if text:
                    # Text targets are present on both noisy and clean sides for block diffusion.
                    total += 2 * text_est(text)
                for part in media_parts:
                    # Final image targets are noisy-only; earlier image targets may become clean context.
                    total += media_est(part) * (1 if idx == last_assistant_idx else 2)
        if "len" in data_item:
            try:
                total += int(data_item["len"])
            except Exception:
                pass
        est = total + LENGTH_ESTIMATE_MARGIN
        limit = int(self.max_len) if self.max_len is not None else MAX_SAMPLE_ALL_LEN
        if est > limit:
            return limit + 1
        return est


def build_xllmx_dataset_sft_multitask_blkpad(
    config_path,
    tokenizer,
    cache_on_disk,
    max_len,
    block_size: Optional[int] = 32,
    max_text_length: Optional[int] = 2048,
    force_t2i_uncondition: Optional[bool] = False,
    t2i_think_uncondition_mode: str = "no_t2i_think_drop",
):
    item_processor = ItemProcessorBlkPad(
        tokenizer=tokenizer,
        max_len=max_len,
        block_size=block_size,
        max_text_length=max_text_length,
        force_t2i_uncondition=force_t2i_uncondition,
        t2i_think_uncondition_mode=t2i_think_uncondition_mode,
    )
    return FinetuneConversationDataset(config_path, item_processor=item_processor, cache_on_disk=cache_on_disk)
