#!/usr/bin/env python3
"""Generate a tiny, fully local SFT dataset so the training launcher runs as-is.

``data/examples/example_sft.jsonl`` documents the record *schema*, but its
``token_path`` fields are placeholders -- following the README with it verbatim
dies in the dataloader on a missing file. This writes a dataset whose token
files actually exist, so

    python tools/make_example_data.py --output_dir data/example_run
    SFT_TRAIN_PATH=data/example_run/train.yaml MODEL_INIT_PATH=... bash train_sft.sh

reaches the training loop without any object storage or real images.

The visual tokens are random codes: the point is to exercise the pipeline
(packing, masking, loss weighting, checkpointing), not to learn anything. For
real training, encode real images with ``tools/tokenize_image.py``.
"""
import argparse
import json
import os

import torch

# Matches the AToken contract that utils/media_token_io.py validates.
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 4096
VAE_SCALE = 16

TEXT_ANSWERS = [
    "The idea reduces to keeping the interface narrow and the state explicit.",
    "It works by splitting the problem into parts that can be checked separately.",
    "The short version: measure first, then change one thing at a time.",
    "Treat it as a budget, and spend it where the error is largest.",
    "The ordering matters more than the individual steps do.",
    "It trades a little memory for a much shorter critical path.",
    "Most of the cost sits in one place, so start there.",
    "The invariant is simple: nothing observes a value before it is written.",
]

IMAGE_CAPTIONS = [
    "A dense grid of randomly coloured cells.",
    "A synthetic test pattern with no recognisable subject.",
    "Uniform noise arranged on a square lattice.",
    "A checkerboard of unrelated colour patches.",
]


def write_image_tokens(path, px, generator):
    """Write one image-token file: {indices [N,8], coords [N,5]}."""
    grid = px // VAE_SCALE
    n = grid * grid
    coords = torch.zeros(n, 5, dtype=torch.long)
    k = 0
    for y in range(grid):
        for x in range(grid):
            coords[k, 2] = y   # col 2 = x-dim index in the AToken layout
            coords[k, 3] = x   # col 3 = y-dim index
            k += 1
    indices = torch.randint(0, CODEBOOK_SIZE, (n, NUM_CODEBOOKS),
                            dtype=torch.long, generator=generator)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The loader uses torch.load, not pickle.load -- a pickle-written file fails
    # with "Invalid magic number".
    torch.save({"indices": indices, "coords": coords}, path)
    return {"token_path": os.path.abspath(path), "token_length": n,
            "image_width": px, "image_height": px}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output_dir", default="data/example_run")
    ap.add_argument("--image_size", type=int, default=256,
                    help="square output size in pixels; must be a multiple of 16")
    ap.add_argument("--samples_per_task", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.image_size % VAE_SCALE:
        raise SystemExit(f"--image_size must be a multiple of {VAE_SCALE}")

    gen = torch.Generator().manual_seed(a.seed)
    out = os.path.abspath(a.output_dir)
    tok_dir = os.path.join(out, "tokens")
    os.makedirs(out, exist_ok=True)

    records = []
    for i in range(a.samples_per_task):
        # Text answers must differ across records. With one answer string
        # repeated, the model fits it within a step or two and the reported
        # `loss_text` rounds to 0.00, which reads as a broken text branch
        # rather than as a 16-sample dataset being memorized. Measured on real
        # data the same code path reports loss_text ~1.1.
        answer = TEXT_ANSWERS[i % len(TEXT_ANSWERS)]
        caption = IMAGE_CAPTIONS[i % len(IMAGE_CAPTIONS)]
        img_a = write_image_tokens(os.path.join(tok_dir, f"a{i}.pt"), a.image_size, gen)
        img_b = write_image_tokens(os.path.join(tok_dir, f"b{i}.pt"), a.image_size, gen)
        part_a = {"type": "image_url", "token": img_a}
        part_b = {"type": "image_url", "token": img_b}

        records += [
            {"id": f"text-{i}", "task_type": "text",
             "system_prompt": "You are a helpful assistant.",
             "messages": [
                 {"role": "user", "content": [
                     {"type": "text", "text": f"Explain idea number {i} in one sentence."}]},
                 {"role": "assistant", "content": answer}]},
            {"id": f"t2i-{i}", "task_type": "t2i",
             "system_prompt": "Generate an image according to the text prompt.",
             "messages": [
                 {"role": "user", "content": [
                     {"type": "text", "text": f"A synthetic test pattern, variant {i}."}]},
                 {"role": "assistant", "content": [dict(part_a)]}]},
            {"id": f"under-{i}", "task_type": "under",
             "system_prompt": "Answer using the provided image.",
             "messages": [
                 {"role": "user", "content": [
                     dict(part_a),
                     {"type": "text", "text": "<IMG_CONTEXT>\nWhat is in this image?"}]},
                 {"role": "assistant", "content": caption}]},
            {"id": f"edit-{i}", "task_type": "single_edit",
             "system_prompt": ("Generate an image applying the following editing "
                               "instruction based on the original image."),
             "messages": [
                 {"role": "user", "content": [
                     dict(part_a),
                     {"type": "text", "text": "<IMG_CONTEXT>\nShift the palette."}]},
                 {"role": "assistant", "content": [dict(part_b)]}]},
        ]

    jsonl = os.path.join(out, "example_sft.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    yaml_path = os.path.join(out, "train.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Generated by tools/make_example_data.py -- random tokens, for\n"
                "# pipeline smoke-testing only.\nMETA:\n"
                f"  - path: '{jsonl}'\n")

    n_tok = len(records) // 4 * 2
    print(f"wrote {len(records)} records across 4 task types")
    print(f"  tokens : {tok_dir} ({n_tok} files, {a.image_size}x{a.image_size})")
    print(f"  jsonl  : {jsonl}")
    print(f"  meta   : {yaml_path}")
    print(f"\nSFT_TRAIN_PATH={yaml_path} MODEL_INIT_PATH=<hf_ckpt> bash train_sft.sh")


if __name__ == "__main__":
    main()
