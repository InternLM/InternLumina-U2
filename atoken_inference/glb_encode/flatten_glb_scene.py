#!/usr/bin/env python3
"""Flatten a GLB scene graph while preserving per-mesh materials."""

import argparse
from pathlib import Path

import trimesh


def flatten_glb(input_path: Path, output_path: Path) -> None:
    source = trimesh.load(input_path, force="scene", process=False)
    flattened = trimesh.Scene()
    for index, node_name in enumerate(source.graph.nodes_geometry):
        transform, geometry_name = source.graph[node_name]
        mesh = source.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        if len(mesh.faces) and mesh.area > 0:
            flattened.add_geometry(mesh, node_name=f"mesh_{index:04d}")
    if not flattened.geometry:
        raise ValueError(f"No triangular geometry in {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flattened.export(output_path, file_type="glb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    flatten_glb(args.input, args.output)
    print(f"flattened={args.input} output={args.output}")


if __name__ == "__main__":
    main()
