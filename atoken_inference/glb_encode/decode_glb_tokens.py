#!/usr/bin/env python -u
import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Repo root: atoken_inference/ lives directly under it.
DEFAULT_ATOKEN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEFAULT_ATOKEN_ROOT))

from atoken_inference.atoken_wrapper import ATokenWrapper
from atoken_inference.model.basic import SparseTensor
from atoken_inference.model.utils import DiagonalGaussianDistribution
from atoken_inference.model.gs.utils import render_utils


sys.stdout.reconfigure(line_buffering=True)


def load_tokens(pkl_path: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    if pkl_path.suffix.lower() in {".pkl", ".pickle"}:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    else:
        data = torch.load(pkl_path, weights_only=False)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict payload in {pkl_path}, got {type(data)!r}")
    if "coords" not in data:
        raise KeyError(f"{pkl_path} does not contain 'coords'")

    coords = data["coords"]
    coords = coords.int() if isinstance(coords, torch.Tensor) else torch.tensor(coords).int()

    if "indices" in data:
        feats = data["indices"]
        # indices 需要为 long 类型，供 quantizer.indices_to_codes 查表
        feats = feats.long() if isinstance(feats, torch.Tensor) else torch.tensor(feats).long()
        token_type = "indices"
    elif "tokens" in data:
        feats = data["tokens"]
        feats = feats.float() if isinstance(feats, torch.Tensor) else torch.tensor(feats).float()
        token_type = "tokens"
    else:
        raise KeyError(f"{pkl_path} must contain either 'indices' or 'tokens'")

    return coords, feats, token_type


def build_wrapper(
    token_type: str,
    base_dir: Path,
    gs_config_path: Path | None = None,
    gs_model_path: Path | None = None,
) -> ATokenWrapper:
    if token_type == "indices":
        config_path = base_dir / "configs/atoken-sod.yaml"
        model_path = base_dir / "checkpoints/atoken-sod.pt"
    elif token_type == "tokens":
        config_path = base_dir / "configs/atoken-soc.yaml"
        model_path = base_dir / "checkpoints/atoken-soc.pt"
    else:
        raise ValueError(f"Unsupported token_type: {token_type}")

    gs_config_path = gs_config_path or base_dir / "configs/3d_decode_gs.yaml"
    gs_model_path = gs_model_path or base_dir / "checkpoints/3d_decode_gs.pt"
    for path, label in (
        (config_path, "AToken config"), (model_path, "AToken checkpoint"),
        (gs_config_path, "3D Gaussian decoder config"),
        (gs_model_path, "3D Gaussian decoder checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    return (
        ATokenWrapper(
            config_path=str(config_path),
            model_path=str(model_path),
            gs_config_path=str(gs_config_path),
            gs_model_path=str(gs_model_path),
        )
        .cuda()
        .to(torch.bfloat16)
    )


def build_latent(
    wrapper: ATokenWrapper,
    coords: torch.Tensor,
    feats: torch.Tensor,
    token_type: str,
    deterministic: bool,
) -> SparseTensor:
    if token_type == "indices":
        if not hasattr(wrapper.model, "quantizer"):
            raise RuntimeError("Token payload contains indices, but the loaded model has no quantizer.")
        return SparseTensor(
            feats=wrapper.model.quantizer.indices_to_codes(feats),
            coords=coords,
        )

    dtype = next(wrapper.model.parameters()).dtype
    z_params = SparseTensor(feats=feats.to(dtype), coords=coords)

    latent_channels = getattr(getattr(wrapper.model, "config", None), "latent_channels", None)
    if latent_channels is None:
        latent_channels = int(wrapper.model.latent_channels)
    else:
        latent_channels = int(latent_channels)

    if z_params.feats.shape[1] == 2 * latent_channels:
        pos = DiagonalGaussianDistribution(z_params, deterministic=deterministic)
        latent_feats = pos.mode() if deterministic else pos.sample()
        return z_params.replace(latent_feats)
    if z_params.feats.shape[1] == latent_channels:
        return z_params

    raise ValueError(
        f"Unexpected token dim {z_params.feats.shape[1]}, "
        f"expected {latent_channels} or {2 * latent_channels}."
    )


def build_orbit_cameras(
    render_views: int,
    pitch: float = 0.2,
    radius: float = 2.0,
    fov_deg: float = 40.0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    extrinsics = []
    intrinsics = []
    device = torch.device("cuda")

    for yaw in torch.linspace(0, 2 * np.pi, render_views).tolist():
        yaw_t = torch.tensor(float(yaw), device=device)
        pitch_t = torch.tensor(float(pitch), device=device)
        fov_rad = torch.deg2rad(torch.tensor(float(fov_deg), device=device))

        camera_origin = torch.stack(
            [
                torch.sin(yaw_t) * torch.cos(pitch_t),
                torch.cos(yaw_t) * torch.cos(pitch_t),
                torch.sin(pitch_t),
            ],
            dim=0,
        ) * radius
        target = torch.zeros(3, device=device)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        forward = target - camera_origin
        forward = forward / (forward.norm() + 1e-8)
        right = torch.linalg.cross(forward, world_up)
        right = right / (right.norm() + 1e-8)
        up = torch.linalg.cross(right, forward)

        extr = torch.eye(4, device=device)
        # The Gaussian renderer expects camera-space +z to point forward.
        extr[0, :3] = right
        extr[1, :3] = up
        extr[2, :3] = forward
        extr[0, 3] = -torch.dot(right, camera_origin)
        extr[1, 3] = -torch.dot(up, camera_origin)
        extr[2, 3] = -torch.dot(forward, camera_origin)

        intr = torch.eye(3, device=device)
        focal = 0.5 / torch.tan(fov_rad / 2)
        intr[0, 0] = focal
        intr[1, 1] = focal
        intr[0, 2] = 0.5
        intr[1, 2] = 0.5

        extrinsics.append(extr)
        intrinsics.append(intr)

    return extrinsics, intrinsics


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode tokens+coords and render recon views.")
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=DEFAULT_ATOKEN_ROOT,
        help="ml-atoken repo root (contains configs/ and checkpoints/).",
    )
    parser.add_argument("--gs_config", type=Path)
    parser.add_argument("--gs_model", type=Path)
    parser.add_argument("--render_views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--data_channels", type=int, default=768)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decoding.")

    coords, feats, token_type = load_tokens(args.pkl)
    wrapper = build_wrapper(
        token_type,
        args.base_dir.resolve(),
        args.gs_config.resolve() if args.gs_config else None,
        args.gs_model.resolve() if args.gs_model else None,
    )
    coords = coords.cuda()
    feats = feats.cuda()

    if coords.dim() == 2 and coords.shape[1] == 3:
        coords = torch.cat(
            [
                torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device),
                torch.ones(coords.shape[0], 1, dtype=coords.dtype, device=coords.device),
                coords,
            ],
            dim=1,
        )

    latent = build_latent(wrapper, coords, feats, token_type, args.deterministic)
    rec = wrapper.decode(latent, training=False).sample

    rec_x_no_t = SparseTensor(
        feats=rec.feats[:, : args.data_channels],
        coords=coords[:, [0, 2, 3, 4]],
    ).to(device="cuda", dtype=torch.bfloat16)
    wrapper.gs_model.resolution = 48
    gaussians = wrapper.gs_model.forward_decoder(rec_x_no_t)
    sample_gaussians = gaussians[0] if isinstance(gaussians, (list, tuple)) else gaussians

    extrinsics, intrinsics = build_orbit_cameras(args.render_views)

    g = sample_gaussians
    xyz = g.get_xyz
    scl = g.get_scaling
    opa = g.get_opacity
    rot = g.get_rotation
    print("N gaussians:", xyz.shape[0])
    print("xyz nan/inf:", torch.isnan(xyz).any().item(), torch.isinf(xyz).any().item(),
        "min/max:", xyz.min().item(), xyz.max().item())
    print("scale nan/inf:", torch.isnan(scl).any().item(), torch.isinf(scl).any().item(),
        "min/max:", scl.min().item(), scl.max().item())
    print("opacity nan/inf:", torch.isnan(opa).any().item(), torch.isinf(opa).any().item(),
        "min/max:", opa.min().item(), opa.max().item())
    print("rot nan/inf:", torch.isnan(rot).any().item(), torch.isinf(rot).any().item())

    frames = render_utils.render_frames(
        sample_gaussians,
        extrinsics,
        intrinsics,
        {"resolution": args.resolution, "bg_color": (1, 1, 1), "near": 0.1, "far": 6.0},
        verbose=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    per_view_images: list[Image.Image] = []
    for i, color in enumerate(frames["color"]):
        img = Image.fromarray(color)
        img.save(args.out_dir / f"recon_{i:03d}.png")
        per_view_images.append(img)

    if per_view_images:
        cols = 4
        rows = math.ceil(len(per_view_images) / cols)
        w, h = per_view_images[0].size
        grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
        for idx, img in enumerate(per_view_images):
            r = idx // cols
            c = idx % cols
            grid.paste(img, (c * w, r * h))
        grid.save(args.out_dir / "recon_grid.png")

    print(f"Decoded token type: {token_type}")
    print(f"Saved {len(frames['color'])} views to {args.out_dir}")


if __name__ == "__main__":
    main()
