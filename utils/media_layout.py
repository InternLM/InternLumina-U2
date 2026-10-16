"""Shared media-token layout helpers.

These lay precomputed AToken codes out into the 1-D sequence the model expects:
row separators for 2-D image grids, row/layer separators for sparse 3-D grids,
row/frame separators for video, and the mask draw used for image answers.

Both the training data pipeline (``tasks/dataset/``) and the inference
entrypoints (``infer_*_sft.py``) import from here, so the layout is defined
exactly once -- a divergence between the two would be a silent train/infer
mismatch. Loading the code files themselves lives next door in
``utils/media_token_io.py``.
"""

import math
import random

import numpy as np
import torch

# Text mask id: LLaDA2's own pretrained ``<|mask|>`` embedding.
TEXT_MASK_ID = 156895

# Codebook depth of the visual tokenizer.
NUM_CODEBOOKS = 8


def as_2d(token_ids, device=None, num_codebooks: int = NUM_CODEBOOKS) -> torch.Tensor:
    """Expand text/special ids to the same N-column shape as image tokens.

    Text has one semantic token id. Repeating it here only satisfies the shared
    storage layout; text/special types still use the model's text embedding path
    rather than independent image codebooks.
    """
    t = torch.as_tensor(token_ids, dtype=torch.int64, device=device)
    return t.unsqueeze(1).repeat(1, num_codebooks)


def make_special_2d(token_id: int, device=None, num_codebooks: int = NUM_CODEBOOKS) -> torch.Tensor:
    """One special token as a [1, num_codebooks] row."""
    return torch.full((1, num_codebooks), token_id, dtype=torch.long, device=device)


def make_special_2d_with_label(token_id: int, device=None, num_codebooks: int = NUM_CODEBOOKS):
    """``make_special_2d`` plus a matching all -100 (unsupervised) label row."""
    t = make_special_2d(token_id, device=device, num_codebooks=num_codebooks)
    return t, torch.full_like(t, -100)


def add_break_line_multi_codebook(tokens, H: int, W: int, new_number: int = 0) -> torch.Tensor:
    """Append one structural token after each row of an H x W VQ grid.

    Input and output are [sequence, num_codebooks]. The output length is
    H * (W + 1); the extra H entries are removed before AToken decoding and are
    therefore not image pixels.
    """
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.int64)
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens shape [seq_len, num_codebooks], got {tokens.shape}")
    seq_len, num_codebooks = tokens.shape
    if seq_len != H * W:
        raise ValueError(f"Token length {seq_len} does not match image size {H}x{W}")
    tokens = tokens.view(H, W, num_codebooks)
    pad_row = torch.full(
        (H, 1, num_codebooks), fill_value=new_number, dtype=tokens.dtype, device=tokens.device
    )
    return torch.cat([tokens, pad_row], dim=1).view(-1, num_codebooks)


def add_break_line_multi_codebook_3d(
    tokens, coords, H: int, W: int, D: int, new_number: int = 5002, layer_end_number: int = 5003
) -> torch.Tensor:
    """Sort sparse 3-D tokens by AToken coords, then close each row and layer.

    Matches the AToken convention: ``coords`` is ``[batch, t, x, y, z]`` (cols
    2/3/4 are x/y/z) or plain ``[x, y, z]``. The scan order is (z, y, x) -- layer
    first, then row, then column. H/W/D are accepted for signature compatibility;
    the row/layer boundaries come from ``coords``, not from them.

    Every row is closed with ``new_number`` and every layer with
    ``layer_end_number``, including the last of each, so the structure is
    explicitly terminated. At a layer change the current row is closed first.
    """
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.int64)
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.int64)

    device = tokens.device
    dtype = tokens.dtype
    token_len, num_codebooks = tokens.shape
    if token_len == 0:
        return tokens
    if coords.shape[0] != token_len:
        raise ValueError(
            f"tokens length {token_len} does not match coords length {coords.shape[0]}"
        )

    coords_np = coords.cpu().numpy() if coords.device.type != "cpu" else coords.numpy()
    if coords_np.ndim != 2 or coords_np.shape[1] not in (3, 5):
        raise ValueError(f"Unexpected coords shape {coords_np.shape}")
    if coords_np.shape[1] == 5:
        x, y, z = coords_np[:, 2], coords_np[:, 3], coords_np[:, 4]
    else:
        x, y, z = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]

    order = np.lexsort((x, y, z))
    tokens = tokens[torch.tensor(order, device=device, dtype=torch.int64)]
    z_arr = z[order]
    y_arr = y[order]

    pad_row = torch.full((1, num_codebooks), new_number, dtype=dtype, device=device)
    pad_layer = torch.full((1, num_codebooks), layer_end_number, dtype=dtype, device=device)
    out = []
    for i in range(token_len):
        out.append(tokens[i:i + 1])
        is_last = i == token_len - 1
        cur_z, cur_y = z_arr[i], y_arr[i]
        next_z = z_arr[i + 1] if not is_last else None
        next_y = y_arr[i + 1] if not is_last else None
        is_row_end = is_last or next_y != cur_y or next_z != cur_z
        is_layer_end = is_last or next_z != cur_z
        if is_row_end:
            out.append(pad_row)
        if is_layer_end:
            out.append(pad_layer)
    return torch.cat(out, dim=0)


def add_break_line_multi_codebook_video(
    tokens, coords, T: int, H: int, W: int, new_number: int = 5004, frame_end_number: int = 5005
) -> torch.Tensor:
    """Video layout: frames play the role 3-D layers play, so reuse the 3-D path."""
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.int64)
    remapped = coords[:, [0, 4, 2, 3, 1]]  # [batch, t, x, y, z] -> [batch, z, x, y, t]
    return add_break_line_multi_codebook_3d(
        tokens, remapped, H, W, T, new_number=new_number, layer_end_number=frame_end_number
    )


def mask_codes(codes, mask_id, sch: str = "cosine", mask: bool = False):
    """Draw a mask ratio and apply it to ``codes``, returning (masked, labels).

    ``mask_id`` is required on purpose. It used to default to the pre-release
    text mask id, and feeding that to a checkpoint trained against
    ``TEXT_MASK_ID`` produces garbage without raising -- the model simply never
    saw that row as "predict me". Making it explicit keeps the mistake
    impossible rather than silent.
    """
    if sch not in ("cosine", "linear"):
        raise ValueError(f"unknown mask schedule {sch!r}; expected 'cosine' or 'linear'")

    r = random.uniform(0, 1)
    if len(codes) <= 5 or mask:
        mask_ratio = 1.0
    elif sch == "cosine":
        mask_ratio = math.cos(r * math.pi / 2)
    else:
        mask_ratio = max(r, 0.05)

    num_to_mask = max(1, int(len(codes) * mask_ratio))
    indices_to_mask = random.sample(range(len(codes)), num_to_mask)
    masked_codes = codes[:]
    labels = [-100] * len(codes)
    for index in indices_to_mask:
        labels[index] = codes[index]
        masked_codes[index] = mask_id
    return masked_codes, labels


def normalize_message_text_content(content) -> str:
    """Accept a plain string, a list of characters, or an OpenAI-style content list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if all(isinstance(x, str) for x in content):
            return "".join(content)
        text_parts = []
        for x in content:
            if isinstance(x, str):
                text_parts.append(x)
            elif isinstance(x, dict) and x.get("type") == "text":
                text_parts.append(x.get("text", ""))
        return "".join(text_parts)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return str(content.get("text", ""))
    return str(content)
