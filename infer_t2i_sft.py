# -*- coding: utf-8 -*-
"""
SFT text-to-image inference script.

Supports text-to-image generation and single-image editing.  The main path is:

1. Build the SFT user prefix and optionally encode input images.
2. Append an all-MASK image grid and iteratively predict its VQ tokens.
3. Decode the eight-codebook VQ tokens to pixels with AToken.

The comments below deliberately distinguish three often-confused tensors:
``input_ids`` contains token values, ``type_position_ids`` contains modality /
answer-role labels, and ``position_ids`` contains the actual RoPE positions.
"""
import os
import hashlib
import argparse
import time
import torch
try:
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
import numpy as np
from transformers import AutoTokenizer
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.generation_utils import setup_seed
from utils.tokenizer_contract import validate_tokenizer_contract
from generators.image_generation_generator import generate_image_ar
from generators.text_understanding_generator import load_inference_model
from utils.media_token_io import load_image_tokens_from_rclone
# Special-token ids. These values are part of the train/infer contract and
# must stay identical to the tokenizer and dataset implementation.
IMAGE_MASK_ID = 5000
NEW_LINE = 5001       # Structural separator appended after every VQ row.
BOA = 157165          # Begin of answer.
EOA = 157166          # End of answer.
BOI = 157167          # Begin of image.
EOI = 157168          # End of image.
IMG_CONTEXT = 157160  # must match tokenizer.encode("<IMG_CONTEXT>") and the training IMG_CONTEXT id
NUM_CODEBOOKS = 8     # AToken represents each spatial position with 8 ids.

# type_position_ids uses these values. Despite the variable name, these are
# role/modality labels rather than positional indices.
TYPE_TEXT_OR_SPECIAL = 1
TYPE_MEDIA_PROMPT = 2   # Input/reference image token.
TYPE_MEDIA_ANSWER = 4   # Output image token; infer_t2i selects this type.

DEFAULT_SYSTEM_PROMPTS = {
    "t2i": "Generate an image according to the text prompt.",
    "single_edit": "Generate an image applying the following editing instruction based on the original image.",
}
DEFAULT_CFG_SCALES = {
    "t2i": 3.0,
    "single_edit": 1.0,
}
DEFAULT_TIMESTEPS = {
    "t2i": 96,
    "single_edit": 64,
}


def _build_uncondition_branch(
    uncon_prefix_tokens,
    uncon_prefix_type_ids,
    answer_part,
    answer_type_part,
    media_offset,
):
    """Build the negative branch with the same answer suffix and no user condition."""
    prefix_len = int(uncon_prefix_tokens.shape[0])
    input_ids = torch.cat([uncon_prefix_tokens, answer_part], dim=0)
    type_ids = torch.cat([uncon_prefix_type_ids, answer_type_part], dim=0)
    return input_ids, type_ids, prefix_len, prefix_len + int(media_offset)


def prompt_to_seed(prompt_text: str, base_seed: int = 0) -> int:
    hash_obj = hashlib.md5(prompt_text.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)
    return (base_seed + hash_int) % (2**31 - 1)


def text_to_filename_slug(text: str, fallback: str = "sample") -> str:
    text = (text or "").strip()
    for prefix in ("<IMG_CONTEXT>\n", "<IMG_CONTEXT> ", "<IMG_CONTEXT>"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    ascii_slug = "_".join(text.split())
    ascii_slug = "".join(
        c for c in ascii_slug if c.isascii() and (c.isalnum() or c in ("_", "-"))
    )
    base = (ascii_slug[:80].strip("_-") or fallback)[:80]
    return f"{base}_{digest}"


def concat_images_grid(images, cols=4, spacing=0):
    if not images:
        return None
    images = [img for img in images if img is not None]
    if not images:
        return None
    img_width, img_height = images[0].size
    rows = (len(images) + cols - 1) // cols
    total_width = cols * img_width + (cols - 1) * spacing
    total_height = rows * img_height + (rows - 1) * spacing
    grid_image = Image.new('RGB', (total_width, total_height), color='white')
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * (img_width + spacing)
        y = row * (img_height + spacing)
        grid_image.paste(img, (x, y))
    return grid_image


def add_break_line_multi_codebook(tokens: torch.Tensor, H: int, W: int, new_number: int = 0) -> torch.Tensor:
    """Append one structural token after each row of an H x W VQ grid.

    Input and output are [sequence, num_codebooks]. The output length is
    H * (W + 1); the extra H entries are removed before AToken decoding and
    therefore are not image pixels.
    """
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens shape [seq_len, num_codebooks], got {tokens.shape}")
    seq_len, num_codebooks = tokens.shape
    if seq_len != H * W:
        raise ValueError(f"Token length {seq_len} does not match image size {H}x{W}")
    tokens = tokens.view(H, W, num_codebooks)
    pad_row = torch.full((H, 1, num_codebooks), fill_value=new_number, dtype=tokens.dtype, device=tokens.device)
    tokens = torch.cat([tokens, pad_row], dim=1)
    return tokens.view(-1, num_codebooks)


def calculate_vq_params(image_height: int, image_width: int, vae_scale: int = 16):
    """Convert pixel resolution to the spatial VQ grid used by AToken."""
    token_grid_height = image_height // vae_scale
    token_grid_width = image_width // vae_scale
    seq_len = token_grid_height * token_grid_width
    return seq_len, token_grid_height, token_grid_width


def unnormalize(img):
    img = img.permute(0, 2, 3, 1)
    img = (img + 1) / 2
    img = (img * 255).clamp(0, 255).to(torch.uint8)
    img = img.cpu().numpy()
    return img


def prepare_image_for_encode(img: Image.Image, max_edge: int = 0) -> Image.Image:
    if max_edge > 0:
        scale = max_edge / max(img.size)
        new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
        if new_size != img.size:
            img = img.resize(new_size, Image.LANCZOS)
    if min(img.size) < 16:
        raise ValueError(f"Image too small for encode: {img.size}")
    return img


def _as_2d(token_ids, device='cuda'):
    """Expand text/special ids to the same 8-column shape as image tokens.

    Text has one semantic token id. Repeating it here only satisfies the shared
    storage layout; type 1 and type 3 still use the model's text embedding
    path rather than eight independent image codebooks.
    """
    t = torch.tensor(token_ids, dtype=torch.int64, device=device)
    return t.unsqueeze(1).repeat(1, NUM_CODEBOOKS)


def make_special_2d(token_id: int, device='cuda') -> torch.Tensor:
    return torch.full((1, NUM_CODEBOOKS), token_id, dtype=torch.long, device=device)


def build_sft_user_prefix(tokenizer, system_prompt, user_text, media_token_list=None, device='cuda'):
    """Build turn-0 condition tokens and their type labels.

    Text-only layout:
        <system>system_prompt</system><user>user_text</user>

    For edit tasks each IMG_CONTEXT placeholder is replaced by
    BOI + input VQ grid + EOI. The VQ grid is type 2; text and image
    boundaries are type 1.

    Returns tensors with shapes [P, 8] and [P], where P is the complete
    condition-prefix length after input media expansion.
    """
    if media_token_list is None:
        media_token_list = []

    # Match dataset _make_user_turn: inject placeholders inside <user> when all are missing.
    if media_token_list:
        placeholder_ids = [pid for pid, _ in media_token_list]
        missing = [
            "<IMG_CONTEXT>" for pid in placeholder_ids
            if pid == IMG_CONTEXT and "<IMG_CONTEXT>" not in user_text
        ]
        if missing and len(missing) == len(placeholder_ids):
            user_text = "\n".join(missing + ([user_text] if user_text else []))

    wrapped = f"<system>{system_prompt}</system><user>{user_text}</user>"
    token_ids = tokenizer(wrapped, truncation=True, max_length=4096, padding=False).input_ids

    if media_token_list is None:
        media_token_list = []

    media_by_placeholder = {}
    for placeholder_id, media_tokens in media_token_list:
        media_by_placeholder.setdefault(placeholder_id, []).append(media_tokens)

    chunks_t = []
    chunks_type = []
    start = 0
    for i, tok in enumerate(token_ids):
        if tok in media_by_placeholder and media_by_placeholder[tok]:
            media_tokens = media_by_placeholder[tok].pop(0)
            if i > start:
                text_chunk = _as_2d(token_ids[start:i], device)
                chunks_t.append(text_chunk)
                chunks_type.append(torch.full((text_chunk.shape[0],), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device))
            bo = make_special_2d(BOI, device)
            eo = make_special_2d(EOI, device)
            chunks_t.extend([bo, media_tokens.to(device), eo])
            chunks_type.extend([
                torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
                torch.full((media_tokens.shape[0],), TYPE_MEDIA_PROMPT, dtype=torch.long, device=device),
                torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
            ])
            start = i + 1

    if start < len(token_ids):
        text_chunk = _as_2d(token_ids[start:], device)
        chunks_t.append(text_chunk)
        chunks_type.append(torch.full((text_chunk.shape[0],), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device))

    leftover = sum(len(queue) for queue in media_by_placeholder.values())
    if leftover:
        raise ValueError(
            "An input-image placeholder was missing or truncated from the user prompt; "
            "increase the prompt budget or include <IMG_CONTEXT> once per input image."
        )

    prefix_tokens = torch.cat(chunks_t, dim=0)
    prefix_type_ids = torch.cat(chunks_type, dim=0).to(device)
    return prefix_tokens, prefix_type_ids


def build_sft_uncondition_prefix(tokenizer, system_prompt, device='cuda'):
    """Keep system text but replace the user condition with <uncondition>."""
    wrapped = f"<system>{system_prompt or ''}</system><user><uncondition></user>"
    token_ids = tokenizer(wrapped, truncation=True, max_length=4096, padding=False).input_ids
    tokens = _as_2d(token_ids, device)
    type_ids = torch.full((tokens.shape[0],), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device)
    return tokens, type_ids


def build_sft_edit_dual_cfg_prefixes(
    tokenizer,
    system_prompt,
    user_text,
    input_media_list,
    device='cuda',
):
    """Build the two negative prefixes used by edit dual-CFG.

    Text-uncondition keeps every input image and replaces the edit instruction
    with ``<uncondition>``. Image-uncondition keeps the edit instruction and
    removes every input image together with its ``<IMG_CONTEXT>`` placeholder.
    """
    placeholders = "\n".join("<IMG_CONTEXT>" for _ in input_media_list)
    text_uncondition_user = "\n".join(
        part for part in (placeholders, "<uncondition>") if part
    )
    uncon_text_prefix = build_sft_user_prefix(
        tokenizer,
        system_prompt,
        text_uncondition_user,
        input_media_list,
        device,
    )

    image_uncondition_user = user_text.replace("<IMG_CONTEXT>", "").strip()
    uncon_image_prefix = build_sft_user_prefix(
        tokenizer,
        system_prompt,
        image_uncondition_user,
        media_token_list=[],
        device=device,
    )
    return uncon_text_prefix, uncon_image_prefix


def peek_input_grid_sizes(input_token_paths, input_image_paths, input_max_edge=0):
    """Return token-grid (h, w) for each input without loading the full model."""
    sizes = []
    for tp in input_token_paths:
        _, h, w = load_image_tokens_from_rclone(tp)
        sizes.append((h, w))
    for ip in input_image_paths:
        img = Image.open(ip).convert('RGB')
        img = prepare_image_for_encode(img, input_max_edge)
        width, height = img.size
        _, th, tw = calculate_vq_params(height, width)
        sizes.append((th, tw))
    return sizes


def resolve_output_image_size(image_height, image_width, input_grid_sizes, task_type):
    """<=0 dimensions follow the first input image for edit tasks, else default to 1024."""
    default_size = 1024
    if image_height > 0 and image_width > 0:
        return image_height, image_width

    if task_type == "single_edit":
        if not input_grid_sizes:
            raise ValueError(
                f"{task_type} requires --input_token_paths/--input_image_paths when "
                "--image_height/--image_width are not set"
            )
        ref_h, ref_w = input_grid_sizes[0]
        if image_height <= 0:
            image_height = ref_h * 16
        if image_width <= 0:
            image_width = ref_w * 16
        return image_height, image_width

    if image_height <= 0:
        image_height = default_size
    if image_width <= 0:
        image_width = default_size
    return image_height, image_width


def load_input_media_list(input_token_paths, input_image_paths, wrapper, input_max_edge, device='cuda'):
    """Load precomputed VQ grids or encode raw images for edit-task prefixes.

    Each result is (IMG_CONTEXT, tokens_with_row_separators). Raw images are
    normalized to [-1, 1] before AToken encoding. The returned tokens become
    type-2 prompt media in build_sft_user_prefix.
    """
    input_media_list = []
    for tp in input_token_paths:
        tokens, h, w = load_image_tokens_from_rclone(tp)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64)
        tokens = add_break_line_multi_codebook(tokens, h, w, new_number=NEW_LINE)
        input_media_list.append((IMG_CONTEXT, tokens))

    for ip in input_image_paths:
        img = Image.open(ip).convert('RGB')
        img = prepare_image_for_encode(img, input_max_edge)
        width, height = img.size
        _, th, tw = calculate_vq_params(height, width)
        img_tensor = torch.from_numpy(np.array(img)).cuda()
        img_tensor = (img_tensor.float() / 255.0) * 2 - 1
        img_sparse = wrapper.image_video_to_sparse_tensor([img_tensor])
        indices, _ = wrapper.extract_encode(img_sparse)
        indices = indices.cpu()
        tokens = add_break_line_multi_codebook(indices, th, tw, new_number=NEW_LINE)
        input_media_list.append((IMG_CONTEXT, tokens))
    return input_media_list


def generate_single_image_sft(
    model,
    tokenizer,
    wrapper,
    device,
    task_type,
    system_prompt,
    user_text,
    input_media_list,
    image_height,
    image_width,
    timesteps,
    cfg_scale,
    temperature,
    seed,
    cfg_text=None,
    cfg_img=None,
):
    """Run one complete SFT image inference request.

    timesteps, cfg_scale and temperature control image decoding.
    Edit requests may instead provide cfg_text and cfg_img together to enable
    inference-only dual-CFG.

    Returns (PIL image, image-stage time).
    """
    dual_cfg = cfg_text is not None or cfg_img is not None
    if (cfg_text is None) != (cfg_img is None):
        raise ValueError("cfg_text and cfg_img must be provided together")
    if dual_cfg and task_type != "single_edit":
        raise ValueError("cfg_text/cfg_img are only supported for edit tasks")
    if dual_cfg and cfg_scale != 0:
        raise ValueError("non-zero cfg_scale cannot be combined with cfg_text/cfg_img")

    # Pixel H/W are divided by 16. seq_len counts real spatial positions only;
    # img_mask_token later contains one additional NEW_LINE per VQ row.
    seq_len, token_grid_height, token_grid_width = calculate_vq_params(image_height, image_width)

    prefix_tokens, prefix_type_ids = build_sft_user_prefix(
        tokenizer, system_prompt, user_text, input_media_list, device
    )
    boa = make_special_2d(BOA, device)
    boi = make_special_2d(BOI, device)
    eoi = make_special_2d(EOI, device)
    eoa = make_special_2d(EOA, device)
    base_mask = torch.full(
        (token_grid_height * token_grid_width, NUM_CODEBOOKS),
        IMAGE_MASK_ID,
        dtype=torch.long,
    )
    img_mask_token = add_break_line_multi_codebook(
        base_mask, token_grid_height, token_grid_width, new_number=NEW_LINE
    ).to(device)
    # Layout: [clean user prefix | BOA | BOI | image MASK grid | EOI | EOA].
    input_ids = torch.cat([prefix_tokens, boa, boi, img_mask_token, eoi, eoa], dim=0)
    type_ids = torch.cat([
        prefix_type_ids,
        torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
        torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
        torch.full((img_mask_token.shape[0],), TYPE_MEDIA_ANSWER, dtype=torch.long, device=device),
        torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
        torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
    ], dim=0)
    code_start = prefix_tokens.shape[0] + 1

    uncon_prefix_tokens, uncon_prefix_type_ids = build_sft_uncondition_prefix(
        tokenizer, system_prompt, device
    )
    answer_part = input_ids[prefix_tokens.shape[0]:]
    answer_type_part = type_ids[prefix_tokens.shape[0]:]

    (
        uncon_input_ids,
        uncon_type_ids,
        uncon_user_prefix_len,
        uncon_media_region_start,
    ) = _build_uncondition_branch(
        uncon_prefix_tokens,
        uncon_prefix_type_ids,
        answer_part,
        answer_type_part,
        code_start - prefix_tokens.shape[0],
    )

    uncon_text_input_ids = None
    uncon_text_type_ids = None
    uncon_image_input_ids = None
    uncon_image_type_ids = None
    uncon_text_prefix_len = None
    uncon_image_prefix_len = None
    if dual_cfg:
        (
            (uncon_text_prefix_tokens, uncon_text_prefix_type_ids),
            (uncon_image_prefix_tokens, uncon_image_prefix_type_ids),
        ) = build_sft_edit_dual_cfg_prefixes(
            tokenizer,
            system_prompt,
            user_text,
            input_media_list,
            device,
        )
        uncon_text_input_ids = torch.cat(
            [uncon_text_prefix_tokens, answer_part], dim=0
        )
        uncon_text_type_ids = torch.cat(
            [uncon_text_prefix_type_ids, answer_type_part], dim=0
        )
        uncon_image_input_ids = torch.cat(
            [uncon_image_prefix_tokens, answer_part], dim=0
        )
        uncon_image_type_ids = torch.cat(
            [uncon_image_prefix_type_ids, answer_type_part], dim=0
        )
        uncon_text_prefix_len = uncon_text_prefix_tokens.shape[0]
        uncon_image_prefix_len = uncon_image_prefix_tokens.shape[0]
    # setup_seed covers global RNGs. This explicit generator is also passed to
    # image token sampling and confidence-based remasking.
    setup_seed(seed)
    start_time = time.time()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # The image generator builds the conditional branches and attention masks,
    # applies hidden-state CFG, and iteratively fills type-4 MASKs.
    vq_tokens = generate_image_ar(
        model,
        input_ids=input_ids.unsqueeze(0),
        uncon_input_ids=uncon_input_ids.unsqueeze(0),
        type_position_ids=type_ids.unsqueeze(0),
        uncon_type_position_ids=uncon_type_ids.unsqueeze(0),
        timesteps=timesteps,
        mask_token_id=IMAGE_MASK_ID,
        newline_id=NEW_LINE,
        temperature=temperature,
        cfg_scale=cfg_scale,
        cfg_text=cfg_text,
        cfg_img=cfg_img,
        uncon_text_input_ids=(
            uncon_text_input_ids.unsqueeze(0)
            if uncon_text_input_ids is not None else None
        ),
        uncon_image_input_ids=(
            uncon_image_input_ids.unsqueeze(0)
            if uncon_image_input_ids is not None else None
        ),
        uncon_text_type_position_ids=(
            uncon_text_type_ids.unsqueeze(0)
            if uncon_text_type_ids is not None else None
        ),
        uncon_image_type_position_ids=(
            uncon_image_type_ids.unsqueeze(0)
            if uncon_image_type_ids is not None else None
        ),
        uncon_text_user_prefix_len=uncon_text_prefix_len,
        uncon_image_user_prefix_len=uncon_image_prefix_len,
        code_start=code_start,
        user_prefix_len=prefix_tokens.shape[0],  # End of clean SFT turn 0.
        media_region_start=code_start,           # BOI starts the noisy image turn.
        uncon_user_prefix_len=uncon_user_prefix_len,
        uncon_media_region_start=uncon_media_region_start,
        generator=generator,
    )

    # Output is [1, H*W, 8] with NEW_LINE positions already removed. AToken
    # additionally needs one [batch, time, h, w, reserved] coordinate for each
    # spatial VQ position.
    vq_tokens = vq_tokens.squeeze(0)
    coords = torch.zeros(seq_len, 5, dtype=torch.long).cuda()
    idx = 0
    for h in range(token_grid_height):
        for w in range(token_grid_width):
            coords[idx, 0] = 0
            coords[idx, 1] = 0
            coords[idx, 2] = h
            coords[idx, 3] = w
            coords[idx, 4] = 0
            idx += 1

    vq_tokens = vq_tokens.cuda()
    with torch.inference_mode():
        rec_load = wrapper.extract_decode(vq_tokens, coords)
    from atoken_inference.model.utils import sparse_to_img_list

    rec_load_list = sparse_to_img_list(rec_load.cpu(), [4, 16, 16], task_types=["image"])
    rec_load_img = rec_load_list[0]
    rec_load_img = unnormalize(rec_load_img)[0]
    rec_load_pil = Image.fromarray(rec_load_img)
    elapsed_time = time.time() - start_time

    return rec_load_pil, elapsed_time


def build_image_cfg_param_suffix(args) -> str:
    cfg_text = getattr(args, "cfg_text", None)
    cfg_img = getattr(args, "cfg_img", None)
    if cfg_text is not None and cfg_img is not None:
        return f"cfgtext{cfg_text}_cfgimg{cfg_img}"
    return f"cfg{args.cfg_scale}"


def build_t2i_output_path(args, prompt_text=None, grid_idx=None, seed_override=None) -> str:
    text = prompt_text if prompt_text is not None else args.user_text
    slug = text_to_filename_slug(text, fallback=args.task_type)
    idx_part = f"_idx{grid_idx:03d}" if grid_idx is not None else ""
    seed_slug = f"seed{seed_override}" if seed_override is not None else f"seed{args.seed}"
    cfg_suffix = build_image_cfg_param_suffix(args)
    filename = (
        f"{args.task_type}_{slug}{idx_part}_{args.image_height}x{args.image_width}_"
        f"t{args.timesteps}_{cfg_suffix}_{seed_slug}_"
        f"temperature{args.temperature}.png"
    )
    return os.path.join(args.output_dir, filename)


def validate_cfg_args(parser, args) -> None:
    has_cfg_text = args.cfg_text is not None
    has_cfg_img = args.cfg_img is not None
    if has_cfg_text != has_cfg_img:
        parser.error("--cfg_text and --cfg_img must be provided together")
    if not has_cfg_text:
        return
    if args.task_type != "single_edit":
        parser.error("--cfg_text/--cfg_img are only supported for edit tasks")
    if args.cfg_scale != 0:
        parser.error(
            "non-zero --cfg_scale cannot be combined with --cfg_text/--cfg_img"
        )


def build_t2i_grid_paths(args, prompts) -> list:
    want_fixed = args.seed_mode in ("fixed", "both")
    want_hash = args.seed_mode in ("hash", "both")
    paths = []
    for idx in range(len(prompts)):
        if want_fixed:
            paths.append(build_t2i_output_path(args, prompt_text=prompts[idx], grid_idx=idx))
        if not want_hash:
            continue
        hash_seed = prompt_to_seed(prompts[idx], args.seed)
        slug = text_to_filename_slug(prompts[idx], fallback="t2i")
        paths.append(os.path.join(
            args.output_dir,
            f"{slug}_idx{idx:03d}_{args.image_height}x{args.image_width}_"
            f"t{args.timesteps}_cfg{args.cfg_scale}_hashseed{hash_seed}_"
            f"temperature{args.temperature}.png",
        ))
    if want_fixed:
        paths.append(os.path.join(
            args.output_dir,
            f"grid_all_prompts_{args.image_height}x{args.image_width}_t{args.timesteps}_"
            f"cfg{args.cfg_scale}_seed{args.seed}_temperature{args.temperature}.png",
        ))
    if want_hash:
        paths.append(os.path.join(
            args.output_dir,
            f"grid_all_prompts_{args.image_height}x{args.image_width}_t{args.timesteps}_"
            f"cfg{args.cfg_scale}_hash_seed_temperature{args.temperature}.png",
        ))
    return paths


def run_t2i_grid(args, model, tokenizer, wrapper, device, prompts):
    want_fixed = args.seed_mode in ("fixed", "both")
    want_hash = args.seed_mode in ("hash", "both")
    images_fixed = []
    images_hash = []
    for idx, prompt_text in enumerate(prompts) if want_fixed else []:
        single_path = build_t2i_output_path(args, prompt_text=prompt_text, grid_idx=idx)
        if args.skip_existing and os.path.isfile(single_path):
            print(f"  [Skip] {single_path}")
            images_fixed.append(Image.open(single_path).convert('RGB'))
            continue
        print(f"  [grid fixed seed] prompt {idx + 1}/{len(prompts)}...")
        img, _ = generate_single_image_sft(
            model, tokenizer, wrapper, device,
            task_type="t2i",
            system_prompt=args.system_prompt,
            user_text=prompt_text,
            input_media_list=[],
            image_height=args.image_height,
            image_width=args.image_width,
            timesteps=args.timesteps,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            seed=args.seed,
        )
        images_fixed.append(img)
        img.save(single_path)

    grid_fixed = concat_images_grid(images_fixed, cols=args.grid_cols, spacing=args.grid_spacing)
    if grid_fixed:
        grid_path = os.path.join(
            args.output_dir,
            f"grid_all_prompts_{args.image_height}x{args.image_width}_t{args.timesteps}_"
            f"cfg{args.cfg_scale}_seed{args.seed}_temperature{args.temperature}.png",
        )
        if args.skip_existing and os.path.isfile(grid_path):
            print(f"  [Skip] {grid_path}")
        else:
            grid_fixed.save(grid_path)
            print(f"  Saved grid (fixed seed): {grid_path}")

    for idx, prompt_text in enumerate(prompts) if want_hash else []:
        hash_seed = prompt_to_seed(prompt_text, args.seed)
        slug = text_to_filename_slug(prompt_text, fallback="t2i")
        single_path = os.path.join(
            args.output_dir,
            f"{slug}_idx{idx:03d}_{args.image_height}x{args.image_width}_"
            f"t{args.timesteps}_cfg{args.cfg_scale}_hashseed{hash_seed}_"
            f"temperature{args.temperature}.png",
        )
        if args.skip_existing and os.path.isfile(single_path):
            print(f"  [Skip] {single_path}")
            images_hash.append(Image.open(single_path).convert('RGB'))
            continue
        print(f"  [grid hash seed] prompt {idx + 1}/{len(prompts)} (seed={hash_seed})...")
        img, _ = generate_single_image_sft(
            model, tokenizer, wrapper, device,
            task_type="t2i",
            system_prompt=args.system_prompt,
            user_text=prompt_text,
            input_media_list=[],
            image_height=args.image_height,
            image_width=args.image_width,
            timesteps=args.timesteps,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            seed=hash_seed,
        )
        images_hash.append(img)
        img.save(single_path)

    grid_hash = concat_images_grid(images_hash, cols=args.grid_cols, spacing=args.grid_spacing)
    if grid_hash:
        grid_path = os.path.join(
            args.output_dir,
            f"grid_all_prompts_{args.image_height}x{args.image_width}_t{args.timesteps}_"
            f"cfg{args.cfg_scale}_hash_seed_temperature{args.temperature}.png",
        )
        if args.skip_existing and os.path.isfile(grid_path):
            print(f"  [Skip] {grid_path}")
        else:
            grid_hash.save(grid_path)
            print(f"  Saved grid (hash seed): {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="SFT text-to-image inference")
    # Request identity and SFT condition text.
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=["t2i", "single_edit"])
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--user_text", type=str, default="")
    parser.add_argument("--prompt_file", type=str, default="", help="One prompt per line (t2i only, batch grid mode)")
    # Image-stage controls. timesteps is the number of MaskGIT refinement rounds.
    parser.add_argument("--image_height", type=int, default=0, help="Output height. <=0 follows input image for edit tasks, else 1024.")
    parser.add_argument("--image_width", type=int, default=0, help="Output width. <=0 follows input image for edit tasks, else 1024.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument(
        "--cfg_text", type=float, default=None,
        help="Edit-only text CFG; must be provided together with --cfg_img.",
    )
    parser.add_argument(
        "--cfg_img", type=float, default=None,
        help="Edit-only image CFG; must be provided together with --cfg_text.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=65513)
    parser.add_argument(
        "--seed_mode", type=str, default="fixed", choices=["fixed", "hash", "both"],
        help="--prompt_file only. fixed uses --seed for every prompt; hash derives a "
             "deterministic seed per prompt; both renders both variants.",
    )
    parser.add_argument("--output_dir", type=str, default="output_sft_t2i")
    parser.add_argument("--input_token_paths", type=str, nargs='*', default=[], help="Input image token paths (for edit tasks)")
    parser.add_argument("--input_image_paths", type=str, nargs='*', default=[], help="Input image file paths (for edit tasks, alternative to token_paths)")
    parser.add_argument("--input_max_edge", type=int, default=0, help="Max long-edge for input image encode (0 = no resize; 16-align via center crop at encode)")
    parser.add_argument("--grid_cols", type=int, default=4, help="Columns when concatenating prompt grid")
    parser.add_argument("--grid_spacing", type=int, default=2, help="Spacing when concatenating prompt grid")
    parser.add_argument(
        "--skip_existing", action=argparse.BooleanOptionalAction, default=False,
        help="Skip inference when the output file already exists (default: false)",
    )
    parser.add_argument(
        "--sweep_resolutions", type=int, nargs="*", default=None,
        help="If set (e.g. 512 1024), load model once and sweep square resolutions.",
    )
    parser.add_argument(
        "--sweep_cfg_temp", type=str, nargs="*", default=None,
        help="Pairs cfg:temp (e.g. 2.0:0.6 4.0:0.8). Requires --sweep_resolutions.",
    )
    args = parser.parse_args()
    if args.system_prompt is None:
        args.system_prompt = DEFAULT_SYSTEM_PROMPTS[args.task_type]
    if args.cfg_scale is None:
        args.cfg_scale = DEFAULT_CFG_SCALES[args.task_type]
    if args.timesteps is None:
        args.timesteps = DEFAULT_TIMESTEPS[args.task_type]
    validate_cfg_args(parser, args)

    input_count = len(args.input_token_paths) + len(args.input_image_paths)
    if args.task_type == "single_edit" and input_count != 1:
        parser.error("single_edit requires exactly one input image or token path")
    if args.task_type == "t2i" and input_count:
        parser.error("input image/token paths are only supported for edit tasks")

    if args.timesteps <= 0 or args.temperature < 0:
        parser.error("--timesteps must be positive and --temperature must be non-negative")
    if args.input_max_edge < 0:
        parser.error("--input_max_edge must be non-negative")
    if args.grid_cols <= 0 or args.grid_spacing < 0:
        parser.error("--grid_cols must be positive and --grid_spacing must be non-negative")

    if args.prompt_file and args.task_type != "t2i":
        parser.error("--prompt_file is only supported for task_type=t2i")
    if args.prompt_file and args.user_text:
        parser.error("provide either --prompt_file or --user_text, not both")
    if args.sweep_cfg_temp and not args.sweep_resolutions:
        parser.error("--sweep_cfg_temp requires --sweep_resolutions")
    if args.sweep_resolutions and args.prompt_file:
        parser.error("--sweep_resolutions cannot be combined with --prompt_file")
    if args.sweep_resolutions and args.task_type != "t2i":
        parser.error("--sweep_resolutions is only supported for task_type=t2i")

    sweep_pairs = []
    if args.sweep_resolutions:
        if args.sweep_cfg_temp:
            for item in args.sweep_cfg_temp:
                if ":" not in item:
                    parser.error(f"invalid --sweep_cfg_temp entry (want cfg:temp): {item}")
                cfg_s, temp_s = item.split(":", 1)
                try:
                    sweep_pairs.append((float(cfg_s), float(temp_s)))
                except ValueError:
                    parser.error(f"invalid --sweep_cfg_temp entry (want numeric cfg:temp): {item}")
        else:
            sweep_pairs.append((float(args.cfg_scale), float(args.temperature)))
        if any(r <= 0 or r % 16 != 0 for r in args.sweep_resolutions):
            parser.error("--sweep_resolutions values must be positive multiples of 16")
        if any(temperature < 0 for _, temperature in sweep_pairs):
            parser.error("--sweep_cfg_temp temperatures must be non-negative")

    prompts = []
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompt_file}")
    elif not args.user_text:
        parser.error("Provide either --user_text or --prompt_file")

    os.makedirs(args.output_dir, exist_ok=True)

    input_grid_sizes = peek_input_grid_sizes(
        args.input_token_paths, args.input_image_paths, args.input_max_edge,
    )
    if not args.sweep_resolutions:
        args.image_height, args.image_width = resolve_output_image_size(
            args.image_height, args.image_width, input_grid_sizes, args.task_type,
        )
        if args.image_height <= 0 or args.image_width <= 0:
            parser.error("output image dimensions must be positive")
        if args.image_height % 16 != 0 or args.image_width % 16 != 0:
            parser.error("output image dimensions must be multiples of 16")

    # Skip happens before either model is loaded. Output names encode selected
    # sampling arguments but not checkpoint/system_prompt, so use a new
    # output_dir or --no-skip_existing when either condition changes.
    if args.sweep_resolutions:
        pending = []
        for res in args.sweep_resolutions:
            for cfg_scale, temperature in sweep_pairs:
                args.image_height = int(res)
                args.image_width = int(res)
                args.cfg_scale = float(cfg_scale)
                args.temperature = float(temperature)
                save_path = build_t2i_output_path(args)
                if args.skip_existing and os.path.isfile(save_path):
                    print(f"[Skip] Output already exists: {save_path}")
                else:
                    pending.append((int(res), float(cfg_scale), float(temperature), save_path))
        if not pending:
            print(f"[Skip] All sweep outputs already exist under {args.output_dir}")
            return
        print(f"[sweep] {len(pending)} configs to run "
              f"(resolutions={args.sweep_resolutions}, cfg_temp={len(sweep_pairs)} pairs)")
    elif args.prompt_file:
        if args.skip_existing and all(os.path.isfile(p) for p in build_t2i_grid_paths(args, prompts)):
            print(f"[Skip] All grid outputs already exist under {args.output_dir}")
            return
    else:
        save_path = build_t2i_output_path(args)
        if args.skip_existing and os.path.isfile(save_path):
            print(f"[Skip] Output already exists: {save_path}")
            return

    if not torch.cuda.is_available():
        parser.error("CUDA is required for image generation and AToken decoding")
    device = "cuda"

    # Validate AToken assets before loading the large language model.
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.getenv(
        "ATOKEN_MODEL_PATH", os.path.join(_repo_root, "atoken_inference/checkpoints/atoken-sod.pt")
    )
    config_path = os.getenv(
        "ATOKEN_CONFIG_PATH", os.path.join(_repo_root, "atoken_inference/configs/atoken-sod.yaml")
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"AToken config not found: {config_path}. Set ATOKEN_CONFIG_PATH."
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"AToken checkpoint not found: {model_path}. Set ATOKEN_MODEL_PATH (see README for weight release)."
        )

    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    validate_tokenizer_contract(tokenizer)
    model = load_inference_model(args.checkpoint, device=device)

    from atoken_inference.atoken_wrapper import ATokenWrapper

    wrapper = ATokenWrapper(config_path, model_path).cuda().to(torch.bfloat16).eval()

    input_media_list = load_input_media_list(
        args.input_token_paths, args.input_image_paths, wrapper, args.input_max_edge, device
    )

    if args.prompt_file:
        seq_len, token_grid_height, token_grid_width = calculate_vq_params(
            args.image_height, args.image_width
        )
        print(
            f"[INFO] Output image: {args.image_height}x{args.image_width}, "
            f"token grid: {token_grid_height}x{token_grid_width}, seq_len: {seq_len}"
        )
        print(f"[t2i grid] Loaded {len(prompts)} prompts from {args.prompt_file}")
        run_t2i_grid(args, model, tokenizer, wrapper, device, prompts)
        return

    if args.sweep_resolutions:
        for i, (res, cfg_scale, temperature, save_path) in enumerate(pending, 1):
            args.image_height = res
            args.image_width = res
            args.cfg_scale = cfg_scale
            args.temperature = temperature
            seq_len, token_grid_height, token_grid_width = calculate_vq_params(
                args.image_height, args.image_width
            )
            print(
                f"[sweep {i}/{len(pending)}] res={res} cfg={cfg_scale} temp={temperature} "
                f"token_grid={token_grid_height}x{token_grid_width} seq_len={seq_len}"
            )
            setup_seed(args.seed)
            img, elapsed_time = generate_single_image_sft(
                model, tokenizer, wrapper, device,
                task_type=args.task_type,
                system_prompt=args.system_prompt,
                user_text=args.user_text,
                input_media_list=input_media_list,
                image_height=args.image_height,
                image_width=args.image_width,
                timesteps=args.timesteps,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                seed=args.seed,
                cfg_text=args.cfg_text,
                cfg_img=args.cfg_img,
            )
            img.save(save_path)
            print(f"[Done] Saved {save_path} (Time: {elapsed_time:.2f}s)")
        print(f"[sweep] finished {len(pending)} configs")
        return

    seq_len, token_grid_height, token_grid_width = calculate_vq_params(
        args.image_height, args.image_width
    )
    print(
        f"[INFO] Output image: {args.image_height}x{args.image_width}, "
        f"token grid: {token_grid_height}x{token_grid_width}, seq_len: {seq_len}"
    )

    setup_seed(args.seed)
    img, elapsed_time = generate_single_image_sft(
        model, tokenizer, wrapper, device,
        task_type=args.task_type,
        system_prompt=args.system_prompt,
        user_text=args.user_text,
        input_media_list=input_media_list,
        image_height=args.image_height,
        image_width=args.image_width,
        timesteps=args.timesteps,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        seed=args.seed,
        cfg_text=args.cfg_text,
        cfg_img=args.cfg_img,
    )

    save_path = build_t2i_output_path(args)
    img.save(save_path)
    print(f"[Done] Saved {save_path} (Time: {elapsed_time:.2f}s)")


if __name__ == '__main__':
    main()
