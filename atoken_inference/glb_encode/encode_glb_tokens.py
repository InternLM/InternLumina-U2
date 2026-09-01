#!/usr/bin/env python3
"""Encode a GLB/GLTF asset into v48 AToken-SOD indices."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
# AToken package root (`.../atoken_inference`). The resolver below also accepts
# the repository root for callers that pass `--atoken-root /path/to/repo`.
DEFAULT_ATOKEN_ROOT = SCRIPT_DIR.parent
DEFAULT_BLENDER = Path(os.environ.get("BLENDER_BIN", "blender"))
DEFAULT_BLENDER_LIBS = os.environ.get("BLENDER_LIB_DIR") or None


def _resolve_atoken_root(root: Path | str) -> Path:
    """Resolve either the repo root or its ``atoken_inference`` directory."""
    requested = Path(root).expanduser().resolve()
    candidates = [requested]
    if requested.name != "atoken_inference":
        candidates.append(requested / "atoken_inference")
    for candidate in candidates:
        if (candidate / "configs/atoken-sod.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"AToken root not found: {requested}; expected <root>/configs and "
        "<root>/checkpoints (root may be the repository or atoken_inference directory)"
    )


def _add_atoken_import_paths(atoken_root: Path) -> None:
    """Make the local AToken package and GLB helpers importable."""
    for path in (atoken_root.parent, atoken_root, atoken_root / "glb_encode"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _check_file(path: Path, label: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def render_glb(
    asset_path: Path,
    render_dir: Path,
    blender_bin: Path = DEFAULT_BLENDER,
    blender_lib_dir: Path | str | None = DEFAULT_BLENDER_LIBS,
    num_views: int = 64,
    image_size: int = 256,
    render_samples: int = 8,
) -> None:
    try:
        from .flatten_glb_scene import flatten_glb
    except ImportError:
        from flatten_glb_scene import flatten_glb

    asset_path = _check_file(asset_path, "3D asset")
    blender_bin = Path(blender_bin).expanduser()
    if not blender_bin.is_file():
        resolved_blender = shutil.which(str(blender_bin))
        if resolved_blender:
            blender_bin = Path(resolved_blender)
    blender_bin = _check_file(blender_bin, "Blender binary")
    if asset_path.suffix.lower() not in {".glb", ".gltf"}:
        raise ValueError("The Blender multiview encoder accepts .glb or .gltf assets")
    if image_size != 256:
        raise ValueError("The current AToken RGB patch encoder requires image_size=256")

    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    flattened = render_dir / "flattened.glb"
    flatten_glb(asset_path, flattened)

    env = os.environ.copy()
    if blender_lib_dir:
        lib_dir = str(Path(blender_lib_dir).resolve())
        env["LD_LIBRARY_PATH"] = lib_dir + (
            f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
        )
    command = [
        str(blender_bin), "--background", "--factory-startup",
        "--python", str(SCRIPT_DIR / "render_glb_blender.py"), "--",
        "--asset", str(flattened), "--output-dir", str(render_dir),
        "--num-views", str(num_views), "--image-size", str(image_size),
        "--fov-deg", "40", "--camera-radius", "2",
        "--render-samples", str(render_samples),
    ]
    subprocess.run(command, check=True, env=env)


def encode_rendered_sod(
    render_dir: Path,
    atoken_root: Path = DEFAULT_ATOKEN_ROOT,
    resolution: int = 48,
    batch_size: int = 512,
    seed: int = 0,
    config_path: Path | str | None = None,
    model_path: Path | str | None = None,
):
    if resolution != 48:
        raise ValueError(f"AToken-SOD inference is fixed to resolution 48, got {resolution}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AToken-SOD encoding")

    atoken_root = _resolve_atoken_root(atoken_root)
    # Same overrides the image/video entrypoints honor.
    config_path = _check_file(
        Path(config_path or os.environ.get("ATOKEN_CONFIG_PATH", atoken_root / "configs/atoken-sod.yaml")),
        "AToken config",
    )
    model_path = _check_file(
        Path(model_path or os.environ.get("ATOKEN_MODEL_PATH", atoken_root / "checkpoints/atoken-sod.pt")),
        "AToken checkpoint",
    )
    _add_atoken_import_paths(atoken_root)
    from atoken_inference.atoken_wrapper import ATokenWrapper
    from atoken_inference.model.basic import SparseTensor
    from multiview_features import compute_multiview_features

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    features, coords = compute_multiview_features(
        render_dir=render_dir,
        resolution=resolution,
        bounding_radius=0.5,
        batch_size=batch_size,
        device="cuda",
    )
    wrapper = ATokenWrapper(str(config_path), str(model_path)).cuda().to(torch.bfloat16).eval()
    features = features.to(dtype=torch.bfloat16) * 2 - 1
    if features.shape[1] > 3072:
        raise ValueError(f"Unexpected multiview feature width: {features.shape[1]}")
    padded = torch.zeros((len(features), 3072), device="cuda", dtype=torch.bfloat16)
    padded[:, :features.shape[1]] = features
    coords = coords.to(device="cuda", dtype=torch.int32)
    coords_5d = torch.cat(
        (torch.zeros_like(coords[:, :1]), torch.ones_like(coords[:, :1]), coords), dim=1
    )
    with torch.inference_mode():
        indices, latent_coords = wrapper.extract_encode(
            SparseTensor(feats=padded, coords=coords_5d)
        )
    indices = indices.to(torch.int64).cpu()
    latent_coords = latent_coords[:, 2:5].to(torch.int64).cpu()
    if indices.ndim != 2 or indices.shape[1] != 8:
        raise ValueError(f"Unexpected AToken indices shape: {tuple(indices.shape)}")
    if len(indices) == 0 or int(indices.min()) < 0 or int(indices.max()) >= 4096:
        raise ValueError("AToken indices are empty or outside [0, 4095]")
    del wrapper, padded, features, coords, coords_5d
    torch.cuda.empty_cache()
    return indices, latent_coords


def encode_glb_to_sod_tokens(
    asset_path: Path,
    atoken_root: Path = DEFAULT_ATOKEN_ROOT,
    blender_bin: Path = DEFAULT_BLENDER,
    blender_lib_dir: Path | str | None = DEFAULT_BLENDER_LIBS,
    num_views: int = 64,
    image_size: int = 256,
    render_samples: int = 8,
    resolution: int = 48,
    render_dir: Path | None = None,
    seed: int = 0,
    atoken_config: Path | str | None = None,
    atoken_model: Path | str | None = None,
):
    if render_dir is not None:
        render_dir = Path(render_dir)
        render_glb(
            asset_path, render_dir, blender_bin, blender_lib_dir,
            num_views, image_size, render_samples,
        )
        return encode_rendered_sod(
            render_dir, atoken_root, resolution, seed=seed,
            config_path=atoken_config, model_path=atoken_model,
        )

    with tempfile.TemporaryDirectory(prefix="atoken3d_") as temporary:
        temporary_path = Path(temporary)
        render_glb(
            asset_path, temporary_path, blender_bin, blender_lib_dir,
            num_views, image_size, render_samples,
        )
        return encode_rendered_sod(
            temporary_path, atoken_root, resolution, seed=seed,
            config_path=atoken_config, model_path=atoken_model,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--atoken-root", type=Path, default=DEFAULT_ATOKEN_ROOT)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--blender-lib-dir", type=Path, default=DEFAULT_BLENDER_LIBS)
    parser.add_argument("--num-views", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--render-samples", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    indices, coords = encode_glb_to_sod_tokens(
        asset_path=args.glb,
        atoken_root=args.atoken_root,
        blender_bin=args.blender_bin,
        blender_lib_dir=args.blender_lib_dir,
        num_views=args.num_views,
        image_size=args.image_size,
        render_samples=args.render_samples,
        resolution=args.resolution,
        render_dir=args.render_dir,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "object_id": args.glb.stem,
            "voxel_resolution": args.resolution,
            "indices": indices.to(torch.int32),
            "coords": coords.to(torch.int32),
            "source": str(args.glb.resolve()),
        },
        args.output,
    )
    print(f"saved={args.output} token_length={len(indices)}")


if __name__ == "__main__":
    main()
