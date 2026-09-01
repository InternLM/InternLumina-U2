import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from ..renderers import GaussianRenderer
from ..representations import Gaussian
from .random_utils import sphere_hammersley_sequence


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    device = torch.device("cuda")
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov_rad = torch.deg2rad(torch.tensor(float(fov), device=device))
        yaw_t = torch.tensor(float(yaw), device=device)
        pitch_t = torch.tensor(float(pitch), device=device)

        # Camera position on a sphere
        orig = torch.stack([
            torch.sin(yaw_t) * torch.cos(pitch_t),
            torch.cos(yaw_t) * torch.cos(pitch_t),
            torch.sin(pitch_t),
        ], dim=0) * r

        # Look-at view matrix (world -> camera)
        target = torch.zeros(3, device=device)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        forward = (target - orig)
        forward = forward / (forward.norm() + 1e-8)
        right = torch.cross(forward, world_up)
        right = right / (right.norm() + 1e-8)
        up = torch.cross(right, forward)

        extr = torch.eye(4, device=device)
        extr[0, :3] = right
        extr[1, :3] = up
        extr[2, :3] = -forward
        extr[0, 3] = -torch.dot(right, orig)
        extr[1, 3] = -torch.dot(up, orig)
        extr[2, 3] = torch.dot(forward, orig)

        # Simple pinhole intrinsics from symmetric FOV, normalized coords
        intr = torch.eye(3, device=device)
        fx = 0.5 / torch.tan(fov_rad / 2)
        fy = fx
        intr[0, 0] = fx
        intr[1, 1] = fy
        intr[0, 2] = 0.5
        intr[1, 2] = 0.5
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    if isinstance(sample, Gaussian) or "Gaussian" in str(type(sample)):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = options.get("resolution", 512)
        renderer.rendering_options.near = options.get("near", 0.8)
        renderer.rendering_options.far = options.get("far", 1.6)
        renderer.rendering_options.bg_color = options.get("bg_color", (0, 0, 0))
        renderer.rendering_options.ssaa = options.get("ssaa", 1)
        renderer.pipe.kernel_size = kwargs.get("kernel_size", 0.1)
        renderer.pipe.use_mip_gaussian = True
        device = sample.device
    else:
        raise ValueError(f"Unsupported sample type: {type(sample)}")
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
        if 'color' not in rets: rets['color'] = []
        if 'depth' not in rets: rets['depth'] = []
        rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        if 'percent_depth' in res:
            rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
        elif 'depth' in res:
            rets['depth'].append(res['depth'].detach().cpu().numpy())
        else:
            rets['depth'].append(None)
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)
