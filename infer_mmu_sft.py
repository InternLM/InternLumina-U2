# -*- coding: utf-8 -*-
"""
SFT multimodal understanding inference script.
Supports: text (pure text) and under (single-turn multimodal understanding).
"""
import os
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
from generators.text_understanding_generator import (
    generate_text_bd,
    load_inference_model,
)
from utils.media_token_io import load_image_tokens_from_rclone

# ─── Special Token IDs (must match training) ───
# Text mask id 156895 (<|mask|>), matching training.
# A legacy checkpoint (mask 157147) is NOT compatible.
MASK = 156895
NEW_LINE = 5001
BOA = 157165
EOA = 157166
BOI = 157167
EOI = 157168
IMG_CONTEXT = 157160  # must match tokenizer.encode("<IMG_CONTEXT>") and the training IMG_CONTEXT id
VIDEO_CONTEXT = 157161
CONTEXT_3D = 157158
NUM_CODEBOOKS = 8

TYPE_TEXT_OR_SPECIAL = 1
TYPE_MEDIA_PROMPT = 2
TYPE_TEXT_ANSWER = 3
TYPE_MEDIA_ANSWER = 4

DEFAULT_SYSTEM_PROMPTS = {
    "text": "You are a helpful assistant.",
    "under": "You are a multimodal assistant. Answer directly using the provided image(s), text, and conversation context; if there is no question, describe the image(s) clearly and accurately.",
}


def resolve_output_path(args) -> str:
    param_suffix = build_infer_param_suffix(args)
    name = f"{args.task_type}_result_{param_suffix}.txt"
    return os.path.join(args.output_dir, name)


def build_infer_param_suffix(args) -> str:
    return "_".join([
        f"t{args.timesteps}",
        f"blk{args.block_length}",
        f"seed{args.seed}",
        f"gen{args.gen_length}",
    ])


def prepare_image_for_encode(img: Image.Image, max_edge: int = 0, no_upscale: bool = False) -> Image.Image:
    """max_edge resizes the long edge to exactly max_edge -- including scaling small
    images UP, which is what every caller relied on before no_upscale existed. Pass
    no_upscale=True to treat max_edge as a ceiling instead, keeping native pixels for
    anything already smaller (benchmark diagrams are often 200-900px)."""
    if max_edge > 0 and not (no_upscale and max(img.size) <= max_edge):
        scale = max_edge / max(img.size)
        new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
        if new_size != img.size:
            img = img.resize(new_size, Image.LANCZOS)
    if min(img.size) < 16:
        raise ValueError(f"Image too small for encode: {img.size}")
    return img


def add_break_line_multi_codebook(tokens: torch.Tensor, H: int, W: int, new_number: int = 0) -> torch.Tensor:
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens shape [seq_len, num_codebooks], got {tokens.shape}")
    seq_len, num_codebooks = tokens.shape
    if seq_len != H * W:
        raise ValueError(f"Token length {seq_len} does not match image size {H}x{W}")
    tokens = tokens.view(H, W, num_codebooks)
    pad_row = torch.full((H, 1, num_codebooks), fill_value=new_number, dtype=tokens.dtype, device=tokens.device)
    tokens = torch.cat([tokens, pad_row], dim=1)
    return tokens.view(-1, num_codebooks)


def _as_2d(token_ids, device='cuda'):
    t = torch.tensor(token_ids, dtype=torch.int64, device=device)
    return t.unsqueeze(1).repeat(1, NUM_CODEBOOKS)


def make_special_2d(token_id: int, device='cuda') -> torch.Tensor:
    return torch.full((1, NUM_CODEBOOKS), token_id, dtype=torch.long, device=device)


def _append_boa_with_meta(
    prefix_tokens: torch.Tensor,
    prefix_type_ids: torch.Tensor,
    prefix_turn_ids: torch.Tensor,
    prefix_block_ids: torch.Tensor,
    prefix_is_clean: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    boa_turn_id: int,
    boa_position_id: int,
    device: str,
) -> tuple:
    boa = make_special_2d(BOA, device)
    return (
        torch.cat([prefix_tokens, boa], dim=0),
        torch.cat([prefix_type_ids, torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device)], dim=0),
        torch.cat([prefix_turn_ids, torch.full((1,), boa_turn_id, dtype=torch.long, device=device)], dim=0),
        torch.cat([prefix_block_ids, torch.zeros(1, dtype=torch.long, device=device)], dim=0),
        torch.cat([prefix_is_clean, torch.zeros(1, dtype=torch.long, device=device)], dim=0),
        torch.cat([prefix_position_ids, torch.full((1,), boa_position_id, dtype=torch.long, device=device)], dim=0),
    )


def build_sft_user_prefix(tokenizer, system_prompt, user_text, media_token_list=None, first_user=True, device='cuda'):
    """Build SFT prefix with template wrapping.

    first_user=True:  <system>sys</system><user>text</user>
    first_user=False: <user>text</user>

    media_token_list: list of (placeholder_id, media_2d_tokens) for IMG/VIDEO/3D.
    """
    if media_token_list is None:
        media_token_list = []

    # Match dataset _make_user_turn: inject placeholders inside <user> when all are missing.
    if media_token_list:
        _PH_TEXT = {
            IMG_CONTEXT: "<IMG_CONTEXT>",
            VIDEO_CONTEXT: "<VIDEO_CONTEXT>",
            CONTEXT_3D: "<3D_CONTEXT>",
        }
        placeholder_ids = [pid for pid, _ in media_token_list]
        missing = [_PH_TEXT[pid] for pid in placeholder_ids if pid in _PH_TEXT and _PH_TEXT[pid] not in user_text]
        if missing and len(missing) == len(placeholder_ids):
            user_text = "\n".join(missing + ([user_text] if user_text else []))

    if first_user:
        wrapped = f"<system>{system_prompt}</system><user>{user_text}</user>"
    else:
        wrapped = f"<user>{user_text}</user>"
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
            "An image placeholder was missing or truncated from the user prompt; "
            "increase the prompt budget or include <IMG_CONTEXT> explicitly."
        )

    if not chunks_t:
        return torch.zeros(0, NUM_CODEBOOKS, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.long, device=device)
    prefix_tokens = torch.cat(chunks_t, dim=0)
    prefix_type_ids = torch.cat(chunks_type, dim=0).to(device)
    return prefix_tokens, prefix_type_ids


def infer_single_turn_text(model, tokenizer, prefix_tokens, prefix_type_ids,
                           gen_length, block_length, timesteps, temperature, device,
                           prefix_turn_ids=None, prefix_block_ids=None, prefix_is_clean=None,
                           prefix_position_ids=None, generated_turn_id=None, generated_position_start=None):
    """Single-turn BD text generation using block-wise layout (prefix+prev_clean+mask_block).
    Generates up to gen_length tokens, with early-stop on EOA.

    In training, BOA is type=1 (never-masked special token).
    Generated content (including EOA) is type=3.
    """
    if prefix_turn_ids is None or prefix_block_ids is None or prefix_is_clean is None or prefix_position_ids is None:
        if prefix_type_ids is not None and bool(
            ((prefix_type_ids == TYPE_TEXT_ANSWER) | (prefix_type_ids == TYPE_MEDIA_ANSWER)).any()
        ):
            raise ValueError(
                "prefix_type_ids contains answer-history tokens (type 3/4) but no explicit metadata was passed. "
                "Default metadata assumes a fresh user/context prefix. Pass explicit turn_ids/block_ids/is_clean/position_ids."
            )
        prefix_len = int(prefix_tokens.shape[0])
        prefix_turn_ids = torch.zeros(prefix_len, dtype=torch.long, device=device)
        prefix_block_ids = torch.full((prefix_len,), -1, dtype=torch.long, device=device)
        prefix_is_clean = torch.ones(prefix_len, dtype=torch.long, device=device)
        prefix_position_ids = torch.arange(prefix_len, dtype=torch.long, device=device)
        if generated_turn_id is None:
            generated_turn_id = 1
        if generated_position_start is None:
            generated_position_start = prefix_len + 1
    elif generated_turn_id is None:
        generated_turn_id = int(prefix_turn_ids.max().item()) + 1
    if generated_position_start is None:
        generated_position_start = int(prefix_position_ids[-1].item()) + 2

    full_prefix, full_prefix_type, full_turn_ids, full_block_ids, full_is_clean, full_pos_ids = _append_boa_with_meta(
        prefix_tokens,
        prefix_type_ids,
        prefix_turn_ids,
        prefix_block_ids,
        prefix_is_clean,
        prefix_position_ids,
        boa_turn_id=generated_turn_id,
        boa_position_id=generated_position_start - 1,
        device=device,
    )
    generated = generate_text_bd(
        model,
        prefix_tokens=full_prefix,
        prefix_type_ids=full_prefix_type,
        gen_length=gen_length,
        block_length=block_length,
        steps_per_block=timesteps,
        temperature=temperature,
        cfg_scale=0.0,
        remasking='low_confidence',
        mask_id=MASK,
        answer_type_id=TYPE_TEXT_ANSWER,
        early_stop_tokens=[EOA],
        tokenizer=tokenizer,
        prefix_turn_ids=full_turn_ids,
        prefix_block_ids=full_block_ids,
        prefix_is_clean=full_is_clean,
        prefix_position_ids=full_pos_ids,
        generated_turn_id=generated_turn_id,
        generated_position_start=generated_position_start,
        require_meta_mask=True,
    )
    return generated


def main():
    parser = argparse.ArgumentParser(description="SFT multimodal understanding inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=["text", "under"])
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--user_text", type=str, default="")
    parser.add_argument("--token_path", type=str, default="", help="Image token path for understanding tasks")
    parser.add_argument("--image_path", type=str, default="", help="Image file path (alternative to token_path)")
    parser.add_argument("--input_max_edge", type=int, default=0, help="Max long-edge for image encode (0 = no resize; 16-align via center crop at encode)")
    parser.add_argument(
        "--input_no_upscale", action="store_true",
        help="Treat --input_max_edge as a ceiling: images already smaller keep their "
             "native resolution instead of being scaled up to it.",
    )
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=32, help="BD denoising steps per generated block")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=65513)
    parser.add_argument("--output_dir", type=str, default="output_sft_mmu")
    parser.add_argument("--text_gt", type=str, default="", help="Ground truth text for comparison")
    parser.add_argument("--text_gt_file", type=str, default="", help="File containing ground truth text")
    parser.add_argument(
        "--skip_existing", action=argparse.BooleanOptionalAction, default=False,
        help="Skip inference when the output file already exists (default: false)",
    )
    args = parser.parse_args()
    if args.system_prompt is None:
        args.system_prompt = DEFAULT_SYSTEM_PROMPTS[args.task_type]
    if args.gen_length is None:
        args.gen_length = 1024 if args.task_type == "text" else 4096
    has_token = bool(args.token_path.strip())
    has_image = bool(args.image_path.strip())
    if has_token and has_image:
        parser.error("provide only one of --token_path or --image_path")
    if args.task_type == "text" and (has_token or has_image):
        parser.error("task_type=text does not accept image media; use under")
    if args.task_type == "text" and not args.user_text.strip():
        parser.error("task_type=text requires non-empty --user_text")
    if args.task_type == "under" and not (has_token or has_image):
        parser.error("task_type=under requires --token_path or --image_path")
    if args.text_gt and args.text_gt_file:
        parser.error("provide only one of --text_gt or --text_gt_file")
    if args.gen_length <= 0 or args.timesteps <= 0:
        parser.error("--gen_length and --timesteps must be positive")
    if args.block_length <= 0:
        parser.error("--block_length must be positive")
    if args.input_max_edge < 0:
        parser.error("--input_max_edge must be non-negative")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = resolve_output_path(args)
    if args.skip_existing and os.path.isfile(output_path):
        print(f"[Skip] Output already exists: {output_path}")
        return

    setup_seed(args.seed)
    if not torch.cuda.is_available():
        parser.error("CUDA is required for Intern Lumina U2 inference")
    device = "cuda"

    # Load image tokens before the language model so raw AToken encoding cannot
    # transiently compete with the LLaDA weights for GPU memory.
    # Load image tokens if provided
    image_tokens_2d = None
    if args.token_path:
        tokens, h, w = load_image_tokens_from_rclone(args.token_path)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64)
        image_tokens_2d = add_break_line_multi_codebook(tokens, h, w, new_number=NEW_LINE).to(device)
        print(f"[INFO] Loaded image tokens: {args.token_path}, grid {h}x{w}, seq_len={image_tokens_2d.shape[0]}")
    elif args.image_path:
        from atoken_inference.atoken_wrapper import ATokenWrapper
        img = Image.open(args.image_path).convert('RGB')
        img = prepare_image_for_encode(img, args.input_max_edge, args.input_no_upscale)
        width, height = img.size
        h, w = height // 16, width // 16
        img_tensor = torch.from_numpy(np.array(img)).cuda()
        img_tensor = (img_tensor.float() / 255.0) * 2 - 1
        _repo_root = os.path.dirname(os.path.abspath(__file__))
        model_path = os.getenv("ATOKEN_MODEL_PATH", os.path.join(_repo_root, "atoken_inference/checkpoints/atoken-sod.pt"))
        config_path = os.getenv("ATOKEN_CONFIG_PATH", os.path.join(_repo_root, "atoken_inference/configs/atoken-sod.yaml"))
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"AToken config not found: {config_path}. Set ATOKEN_CONFIG_PATH."
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"AToken checkpoint not found: {model_path}. Set ATOKEN_MODEL_PATH (see README for weight release)."
            )
        atoken_wrapper = ATokenWrapper(config_path, model_path).cuda().to(torch.bfloat16).eval()
        img_sparse = atoken_wrapper.image_video_to_sparse_tensor([img_tensor])
        indices, _ = atoken_wrapper.extract_encode(img_sparse)
        indices = indices.cpu()
        image_tokens_2d = add_break_line_multi_codebook(indices, h, w, new_number=NEW_LINE).to(device)
        print(f"[INFO] Encoded image: {args.image_path}, grid {h}x{w}, seq_len={image_tokens_2d.shape[0]}")
        del atoken_wrapper, img_sparse, img_tensor
        torch.cuda.empty_cache()

    # Load the language model only after any raw image encoding has released
    # the AToken encoder.
    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    validate_tokenizer_contract(tokenizer)
    model = load_inference_model(args.checkpoint, device=device)

    # Load text GT
    text_gt = None
    if args.text_gt_file:
        with open(args.text_gt_file, 'r', encoding='utf-8') as f:
            text_gt = f.read()
    elif args.text_gt:
        text_gt = args.text_gt

    # ─── Single-turn text/under inference ───
    media_list = [(IMG_CONTEXT, image_tokens_2d)] if image_tokens_2d is not None else None
    prefix_tokens, prefix_type_ids = build_sft_user_prefix(
        tokenizer, args.system_prompt, args.user_text, media_list, True, device
    )

    print(f"[INFO] prefix shape: {prefix_tokens.shape}, gen_length: {args.gen_length}")
    start_time = time.time()
    single_block_length = args.block_length
    single_prefix_len = int(prefix_tokens.shape[0])
    single_turn_ids = torch.zeros(single_prefix_len, dtype=torch.long, device=device)
    single_block_ids = torch.full((single_prefix_len,), -1, dtype=torch.long, device=device)
    single_is_clean = torch.ones(single_prefix_len, dtype=torch.long, device=device)
    single_pos = torch.arange(single_prefix_len, dtype=torch.long, device=device)

    text_tokens = infer_single_turn_text(
        model, tokenizer, prefix_tokens, prefix_type_ids,
        args.gen_length, single_block_length, args.timesteps, args.temperature, device,
        prefix_turn_ids=single_turn_ids,
        prefix_block_ids=single_block_ids,
        prefix_is_clean=single_is_clean,
        prefix_position_ids=single_pos,
        generated_turn_id=1,
        generated_position_start=single_prefix_len + 1,
    )
    generated_ids = text_tokens.tolist()
    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    elapsed_time = time.time() - start_time

    print(f"\n[Generated] {result_text}")
    if text_gt:
        print(f"[GT]        {text_gt}")

    # Token-level comparison
    if text_gt:
        gt_ids = tokenizer(text_gt, truncation=True, max_length=4096, padding=False).input_ids
        pred_ids = generated_ids
        min_len = min(len(pred_ids), len(gt_ids))
        match_count = sum(1 for i in range(min_len) if pred_ids[i] == gt_ids[i])
        print(f"[Match] {match_count}/{min_len} tokens match ({100*match_count/max(min_len,1):.1f}%)")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"task_type: {args.task_type}\n")
        f.write(f"gen_length: {args.gen_length}, block_length: {args.block_length}\n")
        f.write(f"steps_per_block: {args.timesteps}, seed: {args.seed}, temperature: {args.temperature}\n")
        f.write(f"time: {elapsed_time:.2f}s\n\n")
        f.write(f"=== Generated ===\n{result_text}\n")
        if text_gt:
            f.write(f"\n=== GT ===\n{text_gt}\n")
    print(f"[Done] Saved: {output_path} (Time: {elapsed_time:.2f}s)")


if __name__ == '__main__':
    main()
