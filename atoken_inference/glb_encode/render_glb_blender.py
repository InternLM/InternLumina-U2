#!/usr/bin/env python3
"""Render deterministic RGB/depth views for v48 3D AToken encoding.

Run this file with Blender, not CPython. The input GLB must already have its
scene graph flattened by flatten_glb_scene.py.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-views", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--fov-deg", type=float, default=40.0)
    parser.add_argument("--camera-radius", type=float, default=2.0)
    parser.add_argument("--render-samples", type=int, default=8)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def object_bounds(objects):
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type in {"MESH", "CURVE"}
        for corner in obj.bound_box
    ]
    if not points:
        raise ValueError("Imported asset contains no mesh or curve bounds")
    minimum = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maximum = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return minimum, maximum


def normalize_asset(objects):
    minimum, maximum = object_bounds(objects)
    center = (minimum + maximum) * 0.5
    extent = max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z)
    root = bpy.data.objects.new("AssetRoot", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in objects:
        if obj.parent not in objects:
            matrix = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = matrix
    scale = 1.0 / max(extent, 1e-8)
    root.scale = (scale, scale, scale)
    root.location = -center * scale
    bpy.context.view_layer.update()
    return object_bounds(objects)


def look_at(obj, target=(0.0, 0.0, 0.0)):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def camera_angles(num_views):
    side = int(round(math.sqrt(num_views)))
    if side * side != num_views:
        raise ValueError("num_views must be a square number, for example 16 or 64")
    return [
        (yaw, pitch)
        for pitch in np.linspace(-60.0, 60.0, side)
        for yaw in np.linspace(0.0, 360.0, side, endpoint=False)
    ]


def setup_scene(args):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = args.render_samples
    scene.render.resolution_x = args.image_size
    scene.render.resolution_y = args.image_size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_layers[0].use_pass_z = True

    world = bpy.data.worlds.new("ATokenWorld") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 0.3

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.angle = math.radians(args.fov_deg)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new("CameraFill", type="AREA")
    light_data.energy = 50.0
    light_data.shape = "DISK"
    light_data.size = 5.0
    light = bpy.data.objects.new("CameraFill", light_data)
    scene.collection.objects.link(light)

    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    depth_output = nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = str(args.output_dir / "depth")
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    scene.node_tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    color_output = nodes.new("CompositorNodeOutputFile")
    color_output.base_path = str(args.output_dir / "color")
    color_output.format.file_format = "PNG"
    color_output.format.color_mode = "RGBA"
    scene.node_tree.links.new(render_layers.outputs["Image"], color_output.inputs[0])
    return scene, camera, light, depth_output, color_output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "depth").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "color").mkdir(parents=True, exist_ok=True)
    clear_scene()
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(args.asset), merge_vertices=False)
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    normalized_min, normalized_max = normalize_asset(imported)
    scene, camera, light, depth_output, color_output = setup_scene(args)

    all_depth, all_w2c, views = [], [], []
    blender_to_source = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    for index, (yaw_deg, pitch_deg) in enumerate(camera_angles(args.num_views)):
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        camera.location = args.camera_radius * Vector((
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ))
        look_at(camera)
        light.location = camera.location
        look_at(light)
        bpy.context.view_layer.update()
        depth_output.file_slots[0].path = f"depth_{index:03d}_"
        color_output.file_slots[0].path = f"color_{index:03d}_"
        scene.frame_set(1)
        bpy.ops.render.render()

        depth_path = sorted((args.output_dir / "depth").glob(f"depth_{index:03d}_*.exr"))[-1]
        color_path = sorted((args.output_dir / "color").glob(f"color_{index:03d}_*.png"))[-1]
        depth_image = bpy.data.images.load(str(depth_path), check_existing=False)
        depth = np.asarray(depth_image.pixels[:], dtype=np.float32).reshape(
            args.image_size, args.image_size, 4
        )[:, :, 0]
        bpy.data.images.remove(depth_image)
        color_image = bpy.data.images.load(str(color_path), check_existing=False)
        alpha = np.asarray(color_image.pixels[:], dtype=np.float32).reshape(
            args.image_size, args.image_size, 4
        )[:, :, 3]
        bpy.data.images.remove(color_image)

        depth = np.flipud(depth).copy()
        alpha = np.flipud(alpha)
        invalid = (~np.isfinite(depth)) | (depth <= 0) | (depth >= camera.data.clip_end) | (alpha <= 0.01)
        depth[invalid] = -1
        all_depth.append(depth)

        pose = np.asarray(camera.matrix_world, dtype=np.float64)
        c2w_source_cv = blender_to_source @ pose
        c2w_source_cv[:3, 1:3] *= -1
        all_w2c.append(np.linalg.inv(c2w_source_cv))
        views.append({"index": index, "yaw_deg": float(yaw_deg), "pitch_deg": float(pitch_deg)})
        print(f"view {index + 1}/{args.num_views}: {(~invalid).sum()} pixels", flush=True)

    focal = 0.5 / math.tan(math.radians(args.fov_deg) / 2.0)
    intrinsics = np.array([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]], dtype=np.float32)
    np.save(args.output_dir / "all_depth.npy", np.stack(all_depth).astype(np.float32))
    np.save(args.output_dir / "all_w2c.npy", np.stack(all_w2c).astype(np.float32))
    np.save(args.output_dir / "all_intrinsics.npy", np.repeat(intrinsics[None], args.num_views, axis=0))
    metadata = {
        "asset": str(args.asset), "num_views": args.num_views, "image_size": args.image_size,
        "fov_deg": args.fov_deg, "camera_radius": args.camera_radius,
        "normalized_bounds_blender": [list(normalized_min), list(normalized_max)], "views": views,
    }
    (args.output_dir / "render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
