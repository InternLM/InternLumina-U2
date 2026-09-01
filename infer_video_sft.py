# -*- coding: utf-8 -*-
"""
SFT video understanding inference script.
"""
import io
import os
import argparse
import time
import torch
import numpy as np
from transformers import AutoTokenizer
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.generation_utils import setup_seed
from utils.tokenizer_contract import validate_tokenizer_contract
from generators.text_understanding_generator import generate_text_bd, load_inference_model
from utils.media_token_io import load_video_tokens_from_rclone

# ─── Special Token IDs ───
# Text mask id 156895 (<|mask|>). A legacy checkpoint (mask 157147) is NOT compatible.
MASK = 156895
NEW_LINE = 5001
LINE_VIDEO = 5004
FRAME_VIDEO = 5005
BOA = 157165
EOA = 157166
BOV = 157171
EOV = 157172
VIDEO_CONTEXT = 157161
NUM_CODEBOOKS = 8

TYPE_TEXT_OR_SPECIAL = 1
TYPE_MEDIA_PROMPT = 2
TYPE_TEXT_ANSWER = 3
DEFAULT_SYSTEM_PROMPT = (
    "You are a multimodal model that can process text and video. Answer the following "
    "question based on the provided video frames or clips. Analyze temporal dynamics, "
    "motion, interactions, and scene changes across frames. Track relevant objects and "
    "events over time, and integrate information to produce a consistent and accurate answer."
)


def build_output_path(args) -> str:
    param_suffix = build_infer_param_suffix(args)
    return os.path.join(args.output_dir, f"under_video_result_{param_suffix}.txt")


def build_infer_param_suffix(args) -> str:
    suffix = f"t{args.timesteps}_blk{args.block_length}_seed{args.seed}_gen{args.gen_length}"
    if args.video_path.strip():
        suffix += f"_nf{args.num_frames}_vms{args.video_max_size}"
    return suffix


_ATOKEN_TEMPORAL_PATCH = 4


def _read_video_bytes(path: str) -> bytes:
    if ":s3://" in path or path.startswith("s3://"):
        raise ValueError(
            "Raw --video_path must be a local file; use --token_path for object-storage inputs."
        )
    with open(path, "rb") as f:
        return f.read()


def _video_resize(frames: torch.Tensor, max_size: int, size_factor: int = 16) -> torch.Tensor:
    _T, H, W, _C = frames.shape
    long_edge = max(H, W)
    scale = max_size / long_edge if long_edge > max_size else 1.0
    new_H = max(size_factor, int(round(H * scale)) // size_factor * size_factor)
    new_W = max(size_factor, int(round(W * scale)) // size_factor * size_factor)
    if (new_H, new_W) == (H, W):
        return frames
    f = frames.permute(0, 3, 1, 2).float()
    f = torch.nn.functional.interpolate(f, size=(new_H, new_W), mode="bilinear", align_corners=False)
    return f.permute(0, 2, 3, 1)


def uniform_sample_frames(video_bytes: bytes, num_frames: int):
    import decord
    reader = decord.VideoReader(io.BytesIO(video_bytes), num_threads=1)
    vlen = len(reader)
    if vlen <= 0:
        raise ValueError("Video has 0 frames")

    acc_samples = min(num_frames, vlen)
    boundaries = np.linspace(0, vlen, acc_samples + 1).astype(np.int64)
    frame_indices = [int((boundaries[i] + boundaries[i + 1]) // 2) for i in range(acc_samples)]
    frame_indices = [min(idx, vlen - 1) for idx in frame_indices]

    if len(frame_indices) < num_frames:
        frame_indices += [frame_indices[-1]] * (num_frames - len(frame_indices))

    decord.bridge.set_bridge("torch")
    frames = reader.get_batch(frame_indices).to("cpu").float()
    return frames, frame_indices


def encode_video_to_tokens(video_path: str, num_frames: int = 32, max_size: int = 448):
    from atoken_inference.atoken_wrapper import ATokenWrapper

    video_bytes = _read_video_bytes(video_path)
    frames, frame_indices = uniform_sample_frames(video_bytes, num_frames)
    print(f"[INFO] Sampled {len(frame_indices)} frames (uniform-middle) from {video_path}")
    print(f"[INFO] Frame indices: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")

    frames = _video_resize(frames, max_size=max_size)
    v = frames.cuda()
    v = (v.float() / 255.0) * 2 - 1

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
    wrapper = ATokenWrapper(config_path, model_path).cuda().to(torch.bfloat16).eval()

    img_sparse = wrapper.image_video_to_sparse_tensor([v])
    indices, coords = wrapper.extract_encode(img_sparse)
    indices = indices.cpu()
    coords = coords.cpu()

    T = frames.shape[0] // _ATOKEN_TEMPORAL_PATCH
    H = int(frames.shape[1]) // 16
    W = int(frames.shape[2]) // 16
    print(f"[INFO] AToken encode done: indices={tuple(indices.shape)}, grid T={T} H={H} W={W}")
    del wrapper, img_sparse, v
    torch.cuda.empty_cache()
    return indices, coords, T, H, W


def add_break_line_multi_codebook_3d(tokens, coords, H, W, D, new_number=5002, layer_end_number=5003):
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.int64)
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.int64)
    device = tokens.device
    dtype = tokens.dtype
    token_len, num_codebooks = tokens.shape
    if token_len == 0:
        return tokens
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
        out.append(tokens[i:i+1])
        is_last = i == token_len - 1
        cur_z, cur_y = z_arr[i], y_arr[i]
        next_z = z_arr[i+1] if not is_last else None
        next_y = y_arr[i+1] if not is_last else None
        is_row_end = is_last or next_y != cur_y or next_z != cur_z
        is_layer_end = is_last or next_z != cur_z
        if is_row_end:
            out.append(pad_row)
        if is_layer_end:
            out.append(pad_layer)
    return torch.cat(out, dim=0)


def add_break_line_multi_codebook_video(tokens, coords, T, H, W, new_number=5004, frame_end_number=5005):
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.int64)
    remapped = coords[:, [0, 4, 2, 3, 1]]  # [batch, t, x, y, z] -> [batch, z, x, y, t]
    return add_break_line_multi_codebook_3d(
        tokens, remapped, H, W, T,
        new_number=new_number, layer_end_number=frame_end_number,
    )


def _as_2d(token_ids, device='cuda'):
    t = torch.tensor(token_ids, dtype=torch.int64, device=device)
    return t.unsqueeze(1).repeat(1, NUM_CODEBOOKS)


def make_special_2d(token_id: int, device='cuda') -> torch.Tensor:
    return torch.full((1, NUM_CODEBOOKS), token_id, dtype=torch.long, device=device)


def build_sft_video_prefix(tokenizer, system_prompt, user_text, video_tokens_2d, device='cuda'):
    """Build SFT prefix with video: <system>sys</system><user>text_with_VIDEO_CONTEXT</user>
    VIDEO_CONTEXT placeholder gets replaced with [BOV]+video+[EOV].
    """
    if video_tokens_2d is not None and "<VIDEO_CONTEXT>" not in user_text:
        user_text = "\n".join(["<VIDEO_CONTEXT>"] + ([user_text] if user_text else []))
    wrapped = f"<system>{system_prompt}</system><user>{user_text}</user>"
    token_ids = tokenizer(wrapped, truncation=True, max_length=4096, padding=False).input_ids

    chunks_t = []
    chunks_type = []
    start = 0
    video_used = False
    for i, tok in enumerate(token_ids):
        if tok == VIDEO_CONTEXT and not video_used:
            video_used = True
            if i > start:
                text_chunk = _as_2d(token_ids[start:i], device)
                chunks_t.append(text_chunk)
                chunks_type.append(torch.full((text_chunk.shape[0],), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device))
            bo = make_special_2d(BOV, device)
            eo = make_special_2d(EOV, device)
            chunks_t.extend([bo, video_tokens_2d.to(device), eo])
            chunks_type.extend([
                torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
                torch.full((video_tokens_2d.shape[0],), TYPE_MEDIA_PROMPT, dtype=torch.long, device=device),
                torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
            ])
            start = i + 1

    if video_tokens_2d is not None and not video_used:
        raise ValueError(
            "<VIDEO_CONTEXT> was missing or truncated from the user prompt; "
            "increase the prompt budget or include the placeholder explicitly."
        )
    if start < len(token_ids):
        text_chunk = _as_2d(token_ids[start:], device)
        chunks_t.append(text_chunk)
        chunks_type.append(torch.full((text_chunk.shape[0],), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device))

    prefix_tokens = torch.cat(chunks_t, dim=0)
    prefix_type_ids = torch.cat(chunks_type, dim=0).to(device)
    return prefix_tokens, prefix_type_ids


def main():
    parser = argparse.ArgumentParser(description="SFT video understanding inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user_text", type=str, default="")
    parser.add_argument("--token_path", type=str, default="", help="Video token path (pkl)")
    parser.add_argument("--video_path", type=str, default="", help="Direct video file path; encoded on-the-fly via AToken")
    parser.add_argument("--num_frames", type=int, default=64, help="Frames to sample when using --video_path (multiple of 4)")
    parser.add_argument("--video_max_size", type=int, default=448, help="Max long-edge for video resize before encode")
    parser.add_argument("--gen_length", type=int, default=1024)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=32, help="BD denoising steps per generated block")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=65513)
    parser.add_argument("--output_dir", type=str, default="output_sft_video")
    parser.add_argument("--text_gt", type=str, default="")
    parser.add_argument(
        "--skip_existing", action=argparse.BooleanOptionalAction, default=False,
        help="Skip inference when the output file already exists (default: false)",
    )
    args = parser.parse_args()

    use_pkl = bool(args.token_path.strip())
    use_video = bool(args.video_path.strip())
    if not use_pkl and not use_video:
        parser.error("Must provide either --token_path (pkl) or --video_path (video file).")
    if use_pkl and use_video:
        parser.error("Provide only one of --token_path or --video_path, not both.")
    if use_video:
        if args.num_frames <= 0 or args.num_frames % _ATOKEN_TEMPORAL_PATCH != 0:
            parser.error(
                f"--num_frames must be a positive multiple of {_ATOKEN_TEMPORAL_PATCH}, "
                f"got {args.num_frames}"
            )
        if args.video_max_size <= 0:
            parser.error("--video_max_size must be positive")
    if args.gen_length <= 0 or args.block_length <= 0 or args.timesteps <= 0:
        parser.error("--gen_length, --block_length, and --timesteps must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")

    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = build_output_path(args)
    if args.skip_existing and os.path.isfile(output_path):
        print(f"[Skip] Output already exists: {output_path}")
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is required for Intern Lumina U2 inference")
    device = "cuda"

    # Encode/load video before the language model so raw AToken encoding cannot
    # transiently compete with the LLaDA weights for GPU memory.
    # Load video tokens
    if use_pkl:
        print(f"Loading video tokens: {args.token_path}")
        video_tokens, video_coords, video_T, video_H, video_W = load_video_tokens_from_rclone(args.token_path)
        if not isinstance(video_tokens, torch.Tensor):
            video_tokens = torch.tensor(video_tokens, dtype=torch.int64)
    else:
        print(f"Encoding video on-the-fly: {args.video_path}")
        video_tokens, video_coords, video_T, video_H, video_W = encode_video_to_tokens(
            args.video_path.strip(), num_frames=args.num_frames, max_size=args.video_max_size
        )
        if not isinstance(video_tokens, torch.Tensor):
            video_tokens = torch.tensor(video_tokens, dtype=torch.int64)
    print(f"[INFO] Video grid: T={video_T}, H={video_H}, W={video_W}, raw_len={video_tokens.shape[0]}")

    video_tokens_2d = add_break_line_multi_codebook_video(
        video_tokens, video_coords, video_T, video_H, video_W,
        new_number=LINE_VIDEO, frame_end_number=FRAME_VIDEO,
    ).to(device)
    print(f"[INFO] Video tokens with separators: {video_tokens_2d.shape[0]}")

    # Load the language model only after raw video encoding has released the
    # AToken encoder.
    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    validate_tokenizer_contract(tokenizer)
    model = load_inference_model(args.checkpoint, device=device)

    # Build SFT input
    prefix_tokens, prefix_type_ids = build_sft_video_prefix(
        tokenizer, args.system_prompt, args.user_text, video_tokens_2d, device
    )

    # Build prefix for block-wise generation: prefix + BOA(type=1)
    boa = make_special_2d(BOA, device)
    full_prefix = torch.cat([prefix_tokens, boa], dim=0)
    full_prefix_type = torch.cat([
        prefix_type_ids,
        torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
    ], dim=0)

    print(f"[INFO] prefix shape: {full_prefix.shape}, gen_length: {args.gen_length}")

    # Build SFT metadata: user prefix = turn 0 (clean), BOA = turn 1 (noisy)
    plen = int(prefix_tokens.shape[0])
    fp_len = int(full_prefix.shape[0])  # plen + 1 (BOA)
    fp_turn = torch.cat([
        torch.zeros(plen, dtype=torch.long, device=device),
        torch.ones(1, dtype=torch.long, device=device),
    ], dim=0)
    fp_block = torch.cat([
        torch.full((plen,), -1, dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
    ], dim=0)
    fp_clean = torch.cat([
        torch.ones(plen, dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
    ], dim=0)
    fp_pos = torch.arange(fp_len, dtype=torch.long, device=device)

    # Generate with block-wise layout + early-stop on EOA
    start_time = time.time()
    text_tokens = generate_text_bd(
        model,
        prefix_tokens=full_prefix,
        prefix_type_ids=full_prefix_type,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps_per_block=args.timesteps,
        temperature=args.temperature,
        cfg_scale=0.0,
        remasking='low_confidence',
        mask_id=MASK,
        answer_type_id=TYPE_TEXT_ANSWER,
        early_stop_tokens=[EOA],
        prefix_turn_ids=fp_turn,
        prefix_block_ids=fp_block,
        prefix_is_clean=fp_clean,
        prefix_position_ids=fp_pos,
        generated_turn_id=1,
        generated_position_start=fp_len,
        require_meta_mask=True,
    )
    result_text = tokenizer.decode(text_tokens.tolist(), skip_special_tokens=True)
    elapsed_time = time.time() - start_time

    print(f"\n[Generated] {result_text}")
    if args.text_gt:
        print(f"[GT]        {args.text_gt}")

    with open(output_path, 'w', encoding='utf-8') as f:
        if use_pkl:
            f.write(f"source: token_path={args.token_path}\n")
        else:
            f.write(f"source: video_path={args.video_path}\n")
            f.write(f"num_frames: {args.num_frames}, video_max_size: {args.video_max_size}\n")
        f.write(f"gen_length: {args.gen_length}, block_length: {args.block_length}\n")
        f.write(f"steps_per_block: {args.timesteps}, seed: {args.seed}, temperature: {args.temperature}\n")
        f.write(f"time: {elapsed_time:.2f}s\n\n")
        f.write(f"=== Generated ===\n{result_text}\n")
        if args.text_gt:
            f.write(f"\n=== GT ===\n{args.text_gt}\n")
    print(f"[Done] Saved: {output_path} (Time: {elapsed_time:.2f}s)")


if __name__ == '__main__':
    main()
