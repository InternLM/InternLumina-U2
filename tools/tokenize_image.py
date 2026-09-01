# -*- coding: utf-8 -*-
"""Encode images into AToken token files (and optionally verify by reconstruction).

Produces the precomputed-token format consumed by the inference entrypoints
(``--input_token_paths`` for editing, ``--token_path`` for understanding): a
``torch.save`` dict with ``indices`` ``[N, 8]`` and ``coords`` ``[N, 5]``.

Examples:
    # Encode one image (writes <output_dir>/<stem>.pkl)
    python tools/tokenize_image.py --image imgs/edit_source.png --output_dir tokens/

    # Encode a folder of images at 512 long-edge and save reconstruction
    # previews (original | decoded side by side) for visual verification
    python tools/tokenize_image.py --image path/to/folder --max_edge 512 \
        --reconstruct --output_dir tokens/

Requires a GPU and the AToken weights (``ATOKEN_MODEL_PATH`` or
``atoken_inference/checkpoints/atoken-sod.pt``).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PATCH = 16  # AToken spatial patch size; encoded grids are (H/16) x (W/16).


def prepare_image(path: Path, max_edge: int) -> torch.Tensor:
    """Load an RGB image, resize its long edge, and center-crop to a 16-multiple."""
    img = Image.open(path).convert("RGB")
    if max_edge > 0:
        scale = max_edge / max(img.size)
        new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
        if new_size != img.size:
            img = img.resize(new_size, Image.LANCZOS)
    width, height = img.size
    crop_w, crop_h = (width // PATCH) * PATCH, (height // PATCH) * PATCH
    if crop_w < PATCH or crop_h < PATCH:
        raise ValueError(f"Image too small for encode after resize: {img.size} ({path})")
    if (crop_w, crop_h) != (width, height):
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h))
    tensor = torch.from_numpy(np.array(img)).cuda()
    return (tensor.float() / 255.0) * 2 - 1  # [H, W, C] in [-1, 1]


def unnormalize(img: torch.Tensor) -> np.ndarray:
    img = img.permute(0, 2, 3, 1)
    img = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    return img.cpu().numpy()


def save_reconstruction(original: torch.Tensor, wrapper, indices, coords, out_path: Path) -> None:
    """Decode the saved tokens and write an original|reconstruction preview."""
    from atoken_inference.model.utils import sparse_to_img_list

    decoded = wrapper.extract_decode(indices.cuda(), coords.cuda().to(torch.int32))
    rec = sparse_to_img_list(decoded.cpu(), [4, PATCH, PATCH], task_types=["image"])[0]
    rec_pil = Image.fromarray(unnormalize(rec)[0])

    orig = ((original + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    orig_pil = Image.fromarray(orig)
    if rec_pil.size != orig_pil.size:
        rec_pil = rec_pil.resize(orig_pil.size, Image.LANCZOS)

    combined = Image.new("RGB", (orig_pil.width * 2, orig_pil.height))
    combined.paste(orig_pil, (0, 0))
    combined.paste(rec_pil, (orig_pil.width, 0))
    combined.save(out_path)


def collect_images(inputs) -> list:
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(item)
    if not paths:
        raise ValueError("No images found in the given --image inputs")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", nargs="+", required=True, help="Image file(s) or folder(s)")
    parser.add_argument("--output_dir", type=str, default="tokens", help="Where to write <stem>.pkl")
    parser.add_argument("--max_edge", type=int, default=1024, help="Resize long edge before encode (0 = keep size)")
    parser.add_argument(
        "--reconstruct", action="store_true",
        help="Also decode each token file and save an original|reconstruction preview PNG",
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--atoken_config", type=str,
        default=os.getenv("ATOKEN_CONFIG_PATH", os.path.join(repo_root, "atoken_inference/configs/atoken-sod.yaml")),
    )
    parser.add_argument(
        "--atoken_model", type=str,
        default=os.getenv("ATOKEN_MODEL_PATH", os.path.join(repo_root, "atoken_inference/checkpoints/atoken-sod.pt")),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AToken encoding")
    if not os.path.isfile(args.atoken_model):
        raise FileNotFoundError(
            f"AToken checkpoint not found: {args.atoken_model}. Set ATOKEN_MODEL_PATH (see README)."
        )

    from atoken_inference.atoken_wrapper import ATokenWrapper

    wrapper = ATokenWrapper(args.atoken_config, args.atoken_model).cuda().to(torch.bfloat16).eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in collect_images(args.image):
        tensor = prepare_image(path, args.max_edge)
        height, width = int(tensor.shape[0]), int(tensor.shape[1])

        with torch.inference_mode():
            sparse = wrapper.image_video_to_sparse_tensor([tensor])
            indices, coords = wrapper.extract_encode(sparse)
        indices = indices.to(torch.int32).cpu()
        coords = coords.to(torch.int32).cpu()

        out_path = out_dir / f"{path.stem}.pkl"
        torch.save(
            {
                "indices": indices,
                "coords": coords,
                "image_width": width,
                "image_height": height,
            },
            out_path,
        )
        print(f"saved={out_path} grid={height // PATCH}x{width // PATCH} token_length={len(indices)}")

        if args.reconstruct:
            preview = out_dir / f"{path.stem}_recon.png"
            with torch.inference_mode():
                save_reconstruction(tensor, wrapper, indices, coords, preview)
            print(f"preview={preview} (left: input, right: decoded from tokens)")


if __name__ == "__main__":
    main()
