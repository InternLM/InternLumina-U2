#!/usr/bin/env python3
"""Dependency-light RGB/depth packing used by the 3D AToken encoder."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_SIZE = 256
PATCH_SIZE = 16


def _image_patches(image_paths, device):
    images = []
    for path in image_paths:
        image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        background = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if image.mode == "RGBA":
            background.paste(image, (0, 0), image.getchannel("A"))
        else:
            background.paste(image.convert("RGB"), (0, 0))
        images.append(torch.from_numpy(np.asarray(background, dtype=np.float32) / 255.0).permute(2, 0, 1))
    tensor = torch.stack(images).to(device)
    height = IMAGE_SIZE // PATCH_SIZE
    width = IMAGE_SIZE // PATCH_SIZE
    return tensor.reshape(-1, 3, height, PATCH_SIZE, width, PATCH_SIZE).permute(
        0, 3, 5, 1, 2, 4
    ).reshape(-1, PATCH_SIZE * PATCH_SIZE * 3, height, width).contiguous()


def _surface_points(depths, intrinsics, w2c, device, max_points=500_000):
    points = []
    for depth_np, intr_np, w2c_np in zip(depths, intrinsics, w2c):
        depth = torch.as_tensor(depth_np, dtype=torch.float32, device=device)
        valid = depth >= 0
        rows, cols = torch.nonzero(valid, as_tuple=True)
        values = depth[rows, cols]
        pixels = torch.stack(
            (cols.float() / depth.shape[1], rows.float() / depth.shape[0], torch.ones_like(values)), dim=1
        ) * values[:, None]
        intr_inv = torch.linalg.inv(torch.as_tensor(intr_np, dtype=torch.float32, device=device))
        cam = (intr_inv @ pixels.T).T
        cam_h = torch.cat((cam, torch.ones((len(cam), 1), device=device)), dim=1)
        c2w = torch.linalg.inv(torch.as_tensor(w2c_np, dtype=torch.float32, device=device))
        points.append((c2w @ cam_h.T).T[:, :3])
    points = torch.cat(points)
    if len(points) > max_points:
        points = points[torch.randperm(len(points), device=device)[:max_points]]
    return points


def _occupied_coords(points, resolution, bounding_radius):
    coords = torch.floor(
        (points + bounding_radius) * (resolution / (2.0 * bounding_radius))
    ).to(torch.int32)
    valid = ((coords >= 0) & (coords < resolution)).all(dim=1)
    if not valid.any():
        finite = torch.isfinite(points).all(dim=1)
        points = points[finite]
        if len(points) == 0:
            return coords[:0]
        minimum, maximum = points.amin(dim=0), points.amax(dim=0)
        extent = (maximum - minimum).amax()
        if not torch.isfinite(extent) or extent <= 0:
            return coords[:0]
        points = (points - (minimum + maximum) * 0.5) / extent
        coords = torch.floor((points + 0.5) * (resolution - 1)).to(torch.int32)
        valid = ((coords >= 0) & (coords < resolution)).all(dim=1)
    return torch.unique(coords[valid], dim=0, sorted=True)


@torch.inference_mode()
def compute_multiview_features(
    render_dir: Path,
    resolution: int = 48,
    bounding_radius: float = 0.5,
    batch_size: int = 512,
    device: str = "cuda",
):
    render_dir = Path(render_dir)
    image_paths = sorted((render_dir / "color").glob("color_*.png"))
    depths = np.load(render_dir / "all_depth.npy")
    w2c = np.load(render_dir / "all_w2c.npy")
    intrinsics = np.load(render_dir / "all_intrinsics.npy")
    if not (len(image_paths) == len(depths) == len(w2c) == len(intrinsics)):
        raise ValueError("Rendered image/depth/camera counts do not match")

    device = torch.device(device)
    view_features = _image_patches(image_paths, device)
    points = _surface_points(depths, intrinsics, w2c, device)
    coords = _occupied_coords(points, resolution, bounding_radius)
    if len(coords) == 0:
        raise RuntimeError("No occupied voxels after multiview projection")

    interval = 2.0 * bounding_radius / resolution
    centers = -bounding_radius + (coords.float() + 0.5) * interval
    intr = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    extr = torch.as_tensor(w2c, dtype=torch.float32, device=device)
    output = []
    for start in range(0, len(centers), batch_size):
        world = centers[start:start + batch_size]
        world_h = torch.cat((world, torch.ones((len(world), 1), device=device)), dim=1)
        cam = torch.einsum("vij,bj->vbi", extr, world_h)[..., :3]
        projected = torch.einsum("vij,vbj->vbi", intr, cam)
        uv = projected[..., :2] / projected[..., 2:3]
        nearest = projected[..., 2].argmin(dim=0)
        grid = (uv * 2 - 1).unsqueeze(1)
        sampled = torch.nn.functional.grid_sample(
            view_features, grid, align_corners=True, padding_mode="zeros", mode="nearest"
        ).squeeze(2)
        batch_index = torch.arange(len(world), device=device)
        output.append(sampled[nearest, :, batch_index])
    return torch.cat(output).contiguous(), coords.contiguous()
