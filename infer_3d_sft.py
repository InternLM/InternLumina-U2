# -*- coding: utf-8 -*-
"""
SFT 3D understanding inference script.
"""
import os
import shutil
import argparse
import time
import re
import torch
try:
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
import numpy as np
from transformers import AutoTokenizer
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.generation_utils import setup_seed
from utils.tokenizer_contract import validate_tokenizer_contract
from generators.text_understanding_generator import generate_text_bd, load_inference_model
from utils.media_token_io import load_3d_tokens_from_rclone

# ─── Special Token IDs ───
# Text mask id 156895 (<|mask|>). A legacy checkpoint (mask 157147) is NOT compatible.
MASK = 156895
LINE_3D = 5002
LAYER_3D = 5003
BOA = 157165
EOA = 157166
BO3D = 157169
EO3D = 157170
CONTEXT_3D = 157158
NUM_CODEBOOKS = 8
VOXEL_RESOLUTION = 48
MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}
POINT_SUFFIXES = {".npy", ".npz", ".pt", ".pth"}

TYPE_TEXT_OR_SPECIAL = 1
TYPE_MEDIA_PROMPT = 2
TYPE_TEXT_ANSWER = 3
DEFAULT_SYSTEM_PROMPT = (
    "You are a multimodal model that can process text and 3D content. Answer the following "
    "question based on the provided 3D representations. Carefully analyze the object's "
    "geometry, spatial structure, viewpoint consistency, and fine-grained details across "
    "views. Combine relevant information to produce a coherent and accurate answer."
)


def build_output_path(args) -> str:
    param_suffix = build_infer_param_suffix(args)
    return os.path.join(args.output_dir, f"under_3d_result_{param_suffix}.txt")


def build_infer_param_suffix(args) -> str:
    asset_prefix = ""
    source_path = args.asset_path or args.token_path
    if source_path:
        stem = os.path.splitext(os.path.basename(source_path))[0]
        asset_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") + "_"
    return f"{asset_prefix}t{args.timesteps}_blk{args.block_length}_seed{args.seed}_gen{args.gen_length}"


def _mesh_to_points(mesh, count, rng):
    import trimesh

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("3D asset has no triangular faces")
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        xyz, face_ids = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    try:
        rgb = np.asarray(mesh.visual.to_color().face_colors)[face_ids, :3]
    except Exception:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.float32)


def _load_3d_asset_points(path, mesh_samples, seed):
    suffix = os.path.splitext(path)[1].lower()
    rng = np.random.default_rng(seed)
    if suffix in MESH_SUFFIXES:
        import trimesh

        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.points.PointCloud):
            xyz = np.asarray(loaded.vertices, dtype=np.float32)
            colors = loaded.colors
            rgb = (
                np.asarray(colors, dtype=np.float32)[:, :3]
                if colors is not None
                else np.full_like(xyz, 0.5)
            )
            return xyz, rgb
        return _mesh_to_points(loaded, mesh_samples, rng)
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        array = data[data.files[0]]
    elif suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            array = data.get("points", data.get("point_cloud", data.get("xyzrgb")))
            if array is None:
                raise KeyError(f"No points/point_cloud/xyzrgb field in {path}")
        else:
            array = data
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
    else:
        supported = sorted(MESH_SUFFIXES | POINT_SUFFIXES)
        raise ValueError(f"Unsupported 3D asset format {suffix!r}; expected one of {supported}")
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"Expected [N,3+] point array, got {array.shape} from {path}")
    xyz = np.asarray(array[:, :3], dtype=np.float32)
    rgb = (
        np.asarray(array[:, 3:6], dtype=np.float32)
        if array.shape[1] >= 6
        else np.full_like(xyz, 0.5)
    )
    return xyz, rgb


def _voxelize_3d_asset(xyz, rgb, resolution):
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
    xyz, rgb = xyz[valid], rgb[valid]
    if len(xyz) == 0:
        raise ValueError("3D asset contains no finite points")
    bounds_min, bounds_max = xyz.min(0), xyz.max(0)
    extent = float((bounds_max - bounds_min).max())
    if extent <= 0:
        raise ValueError("3D asset has degenerate bounds")
    xyz = (xyz - (bounds_min + bounds_max) / 2.0) / extent
    coords = np.floor((xyz + 0.5) * resolution).astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)
    if rgb.max(initial=0) > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse, rgb)
    counts = np.bincount(inverse, minlength=len(unique))[:, None]
    return (sums / counts).astype(np.float32), unique.astype(np.int32)


def encode_3d_asset(args, device):
    if args.voxel_resolution != VOXEL_RESOLUTION:
        raise ValueError(
            f"This inference path is fixed to voxel resolution {VOXEL_RESOLUTION}, "
            f"got {args.voxel_resolution}"
        )
    if device != "cuda":
        raise RuntimeError("CUDA is required to encode a raw 3D asset with AToken-SOD")
    if not os.path.isfile(args.asset_path):
        raise FileNotFoundError(args.asset_path)

    suffix = Path(args.asset_path).suffix.lower()
    use_glb_multiview = False
    blender_bin = args.blender_bin or shutil.which("blender") or ""
    if blender_bin and not os.path.isfile(blender_bin):
        blender_bin = shutil.which(blender_bin) or ""
    if not blender_bin:
        # tools/setup_blender.sh installs into third_party/.
        _repo = Path(__file__).resolve().parent
        _candidates = sorted(_repo.glob("third_party/blender-*/blender"))
        if _candidates:
            blender_bin = str(_candidates[-1])
    if suffix in {".glb", ".gltf"}:
        atoken_3d_root = Path(args.atoken_3d_root).expanduser().resolve()
        if not atoken_3d_root.is_dir():
            raise FileNotFoundError(f"AToken GLB encoder directory not found: {atoken_3d_root}")
        atoken_3d_root_str = str(atoken_3d_root)
        if atoken_3d_root_str not in sys.path:
            sys.path.insert(0, atoken_3d_root_str)
        from encode_glb_tokens import encode_glb_to_sod_tokens

        atoken_root = Path(args.atoken_root).expanduser().resolve()
        if not atoken_root.is_dir():
            raise FileNotFoundError(f"AToken root not found: {atoken_root}")

        if blender_bin and os.path.isfile(blender_bin):
            use_glb_multiview = True
        else:
            raise RuntimeError(
                "Blender is required for GLB/GLTF 3D understanding (training-consistent "
                "multiview encoding). Install it with `bash tools/setup_blender.sh`, or "
                "point BLENDER_BIN / --blender_bin at an existing Blender binary."
            )
    if use_glb_multiview:
        indices, latent_coords = encode_glb_to_sod_tokens(
            asset_path=Path(args.asset_path),
            atoken_root=atoken_root,
            blender_bin=Path(blender_bin),
            blender_lib_dir=(Path(args.blender_lib_dir) if args.blender_lib_dir else None),
            num_views=args.render_num_views,
            image_size=args.render_image_size,
            render_samples=args.render_samples,
            resolution=args.voxel_resolution,
            render_dir=Path(args.render_dir) if args.render_dir else None,
            seed=args.seed,
            atoken_config=Path(args.atoken_config),
            atoken_model=Path(args.atoken_model),
        )
        return indices, latent_coords, args.voxel_resolution, args.voxel_resolution, args.voxel_resolution

    # Point-cloud/surface-sampling path for non-GLB assets.
    xyz, rgb = _load_3d_asset_points(args.asset_path, args.mesh_samples, args.seed)
    feats, coords = _voxelize_3d_asset(xyz, rgb, args.voxel_resolution)
    atoken_root = Path(args.atoken_root).expanduser().resolve()
    if not atoken_root.is_dir():
        raise FileNotFoundError(f"AToken root not found: {atoken_root}")
    atoken_root_str = str(atoken_root)
    if atoken_root_str not in sys.path:
        sys.path.insert(0, atoken_root_str)
    from atoken_inference.atoken_wrapper import ATokenWrapper
    from atoken_inference.model.basic import SparseTensor

    atoken_config = Path(args.atoken_config).expanduser().resolve()
    atoken_model = Path(args.atoken_model).expanduser().resolve()
    if not atoken_config.is_file():
        raise FileNotFoundError(f"AToken config not found: {atoken_config}")
    if not atoken_model.is_file():
        raise FileNotFoundError(f"AToken checkpoint not found: {atoken_model}")
    wrapper = ATokenWrapper(str(atoken_config), str(atoken_model)).cuda().to(torch.bfloat16).eval()
    feats_t = torch.as_tensor(feats, device=device, dtype=torch.bfloat16) * 2 - 1
    padded = torch.zeros((len(feats_t), 3072), device=device, dtype=torch.bfloat16)
    padded[:, : feats_t.shape[1]] = feats_t
    coords_t = torch.as_tensor(coords, device=device, dtype=torch.int32)
    coords_5d = torch.cat(
        (torch.zeros_like(coords_t[:, :1]), torch.ones_like(coords_t[:, :1]), coords_t), dim=1
    )
    with torch.inference_mode():
        indices, latent_coords = wrapper.extract_encode(SparseTensor(feats=padded, coords=coords_5d))
    indices = indices.to(torch.int64).cpu()
    latent_coords = latent_coords[:, 2:5].to(torch.int64).cpu()
    del wrapper, padded, feats_t, coords_t, coords_5d
    torch.cuda.empty_cache()
    return indices, latent_coords, args.voxel_resolution, args.voxel_resolution, args.voxel_resolution


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


def _as_2d(token_ids, device='cuda'):
    t = torch.tensor(token_ids, dtype=torch.int64, device=device)
    return t.unsqueeze(1).repeat(1, NUM_CODEBOOKS)


def make_special_2d(token_id: int, device='cuda') -> torch.Tensor:
    return torch.full((1, NUM_CODEBOOKS), token_id, dtype=torch.long, device=device)


def build_sft_3d_prefix(tokenizer, system_prompt, user_text, d3_tokens_2d, device='cuda'):
    """Build SFT prefix with 3D: <system>sys</system><user>text_with_3D_CONTEXT</user>
    3D_CONTEXT placeholder gets replaced with [BO3D]+3d_tokens+[EO3D].
    """
    if d3_tokens_2d is not None and "<3D_CONTEXT>" not in user_text:
        user_text = "\n".join(["<3D_CONTEXT>"] + ([user_text] if user_text else []))
    wrapped = f"<system>{system_prompt}</system><user>{user_text}</user>"
    token_ids = tokenizer(wrapped, truncation=True, max_length=4096, padding=False).input_ids

    chunks_t = []
    chunks_type = []
    start = 0
    d3_used = False
    for i, tok in enumerate(token_ids):
        if tok == CONTEXT_3D and not d3_used:
            d3_used = True
            if i > start:
                text_chunk = _as_2d(token_ids[start:i], device)
                chunks_t.append(text_chunk)
                chunks_type.append(torch.full((text_chunk.shape[0],), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device))
            bo = make_special_2d(BO3D, device)
            eo = make_special_2d(EO3D, device)
            chunks_t.extend([bo, d3_tokens_2d.to(device), eo])
            chunks_type.extend([
                torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
                torch.full((d3_tokens_2d.shape[0],), TYPE_MEDIA_PROMPT, dtype=torch.long, device=device),
                torch.full((1,), TYPE_TEXT_OR_SPECIAL, dtype=torch.long, device=device),
            ])
            start = i + 1

    if d3_tokens_2d is not None and not d3_used:
        raise ValueError(
            "<3D_CONTEXT> was missing or truncated from the user prompt; "
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
    parser = argparse.ArgumentParser(description="SFT 3D understanding inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user_text", type=str, default="")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--asset_path", type=str, default="",
        help="Raw 3D asset; GLB/GLTF uses Blender multiview encoding, other formats use the legacy point path",
    )
    source.add_argument(
        "--token_path", type=str, default="",
        help="Backward-compatible precomputed 3D token path (.pt)",
    )
    parser.add_argument("--voxel_resolution", type=int, default=48)
    parser.add_argument("--mesh_samples", type=int, default=100000)
    parser.add_argument(
        "--atoken_3d_root", type=str,
        default=os.getenv(
            "ATOKEN_3D_ROOT",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoken_inference/glb_encode"),
        ),
        help="Directory containing the GLB multiview encode module (encode_glb_tokens)",
    )
    parser.add_argument(
        "--atoken_root", type=str,
        default=os.getenv("ATOKEN_ROOT", os.path.dirname(os.path.abspath(__file__))),
    )
    parser.add_argument(
        "--blender_bin", type=str, default=os.getenv("BLENDER_BIN", ""),
        help="Blender binary; only needed for GLB/GLTF multiview encoding",
    )
    parser.add_argument(
        "--blender_lib_dir", type=str, default=os.getenv("BLENDER_LIB_DIR", ""),
    )
    parser.add_argument("--render_num_views", type=int, default=64)
    parser.add_argument("--render_image_size", type=int, default=256)
    parser.add_argument("--render_samples", type=int, default=8)
    parser.add_argument("--render_dir", type=str, default="")
    parser.add_argument(
        "--atoken_config", type=str,
        default=os.getenv(
            "ATOKEN_CONFIG_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoken_inference/configs/atoken-sod.yaml"),
        ),
    )
    parser.add_argument(
        "--atoken_model", type=str,
        default=os.getenv(
            "ATOKEN_MODEL_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoken_inference/checkpoints/atoken-sod.pt"),
        ),
    )
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=32, help="BD denoising steps per generated block")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=65513)
    parser.add_argument("--output_dir", type=str, default="output_sft_3d")
    parser.add_argument("--text_gt", type=str, default="")
    parser.add_argument(
        "--skip_existing", action=argparse.BooleanOptionalAction, default=False,
        help="Skip inference when the output file already exists (default: false)",
    )
    args = parser.parse_args()
    if args.gen_length <= 0 or args.block_length <= 0 or args.timesteps <= 0:
        parser.error("--gen_length, --block_length, and --timesteps must be positive")
    if args.asset_path and args.voxel_resolution != VOXEL_RESOLUTION:
        parser.error(f"--voxel_resolution must be {VOXEL_RESOLUTION} for raw asset encoding")
    if args.asset_path and Path(args.asset_path).suffix.lower() in {".glb", ".gltf"} and args.render_image_size != 256:
        parser.error("--render_image_size must be 256 for GLB/GLTF encoding")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.mesh_samples <= 0:
        parser.error("--mesh_samples must be positive")
    if args.render_num_views <= 0 or args.render_samples <= 0 or args.render_image_size <= 0:
        parser.error("render view count, samples, and image size must be positive")

    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = build_output_path(args)
    if args.skip_existing and os.path.isfile(output_path):
        print(f"[Skip] Output already exists: {output_path}")
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is required for Intern Lumina U2 inference")
    device = "cuda"

    # Encode a raw asset before loading the language model so the AToken encoder
    # and LLaDA weights never need to coexist on the GPU.
    if args.asset_path:
        print(f"Encoding 3D asset at voxel resolution 48: {args.asset_path}")
        d3_tokens, d3_coords, d3_H, d3_W, d3_D = encode_3d_asset(args, device)
    else:
        print(f"Loading precomputed 3D tokens: {args.token_path}")
        d3_tokens, d3_coords, d3_H, d3_W, d3_D = load_3d_tokens_from_rclone(args.token_path)
    if not isinstance(d3_tokens, torch.Tensor):
        d3_tokens = torch.tensor(d3_tokens, dtype=torch.int64)
    print(f"[INFO] 3D grid: H={d3_H}, W={d3_W}, D={d3_D}, raw_len={d3_tokens.shape[0]}")

    d3_tokens_2d = add_break_line_multi_codebook_3d(
        d3_tokens, d3_coords, d3_H, d3_W, d3_D,
        new_number=LINE_3D, layer_end_number=LAYER_3D,
    ).to(device)
    print(f"[INFO] 3D tokens with separators: {d3_tokens_2d.shape[0]}")

    # Load model after raw-asset encoding has released the AToken encoder.
    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    validate_tokenizer_contract(tokenizer)
    model = load_inference_model(args.checkpoint, device=device)

    # Build SFT input
    prefix_tokens, prefix_type_ids = build_sft_3d_prefix(
        tokenizer, args.system_prompt, args.user_text, d3_tokens_2d, device
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
        f.write(f"source: {args.asset_path or args.token_path}\n")
        f.write(f"voxel_resolution: {d3_H}x{d3_W}x{d3_D}, raw_token_length: {d3_tokens.shape[0]}\n")
        f.write(f"gen_length: {args.gen_length}, block_length: {args.block_length}\n")
        f.write(f"steps_per_block: {args.timesteps}, seed: {args.seed}, temperature: {args.temperature}\n")
        f.write(f"time: {elapsed_time:.2f}s\n\n")
        f.write(f"=== Generated ===\n{result_text}\n")
        if args.text_gt:
            f.write(f"\n=== GT ===\n{args.text_gt}\n")
    print(f"[Done] Saved: {output_path} (Time: {elapsed_time:.2f}s)")


if __name__ == '__main__':
    main()
