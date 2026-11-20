"""1194MのCAD素材から原CG画像と正解内周座標をBlenderで生成する。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg.blender_common import (
    configure_render_scene,
    configure_sun_spherical,
    look_at,
    project_target_points,
    spherical_direction,
)


def _expand_cameras(grid: dict[str, Any]) -> list[dict[str, Any]]:
    settings = []
    for tilt_deg in grid["tilt_deg"]:
        azimuths = grid["azimuth_deg"][:1] if (float(tilt_deg) == 0.0 and grid.get("collapse_axis_azimuth", True)) else grid["azimuth_deg"]
        for azimuth_deg, distance, target_idx in itertools.product(
            azimuths, grid["distances"], range(len(grid["targets"]))
        ):
            location = spherical_direction(float(tilt_deg), float(azimuth_deg)) * float(distance)
            settings.append(
                {
                    "id": f"camera_t{int(tilt_deg):03d}_a{int(azimuth_deg):03d}_d{float(distance):05.1f}_o{target_idx:02d}",
                    "tilt_deg": float(tilt_deg),
                    "azimuth_deg": float(azimuth_deg),
                    "location": list(location),
                    "target": grid["targets"][target_idx],
                    "lens_mm": float(grid["lens_mm"]),
                }
            )
    return settings


def _expand_lights(grid: dict[str, Any]) -> list[dict[str, Any]]:
    settings = []
    energies = grid["energies"] if "energies" in grid else [grid["energy"]]
    for tilt_deg in grid["tilt_deg"]:
        azimuths = grid["azimuth_deg"][:1] if (float(tilt_deg) == 0.0 and grid.get("collapse_axis_azimuth", True)) else grid["azimuth_deg"]
        for azimuth_deg, energy in itertools.product(azimuths, energies):
            settings.append(
                {
                    "id": f"light_t{int(tilt_deg):03d}_a{int(azimuth_deg):03d}_e{float(energy):04.1f}",
                    "tilt_deg": float(tilt_deg),
                    "azimuth_deg": float(azimuth_deg),
                    "angle_deg": float(grid.get("angle_deg", 0.0)),
                    "energy": float(energy),
                }
            )
    return settings


def _import_and_center(model_path: Path, normalize_axes: bool) -> list[bpy.types.Object]:
    bpy.ops.wm.obj_import(filepath=str(model_path), forward_axis="NEGATIVE_Z", up_axis="Y")
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"メッシュを読み込めませんでした: {model_path}")
    if normalize_axes:
        for obj in objects:
            obj.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector((min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners)))
    maximum = Vector((max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners)))
    center = (minimum + maximum) / 2.0
    for obj in objects:
        obj.location -= center
    return objects


def _assign_target_ring(objects: list[bpy.types.Object], settings: dict[str, Any]):
    target_object = None
    target_indices: list[int] = []
    target_z = float(settings["z"])
    target_radius = float(settings["radius"])
    tolerance = float(settings["tolerance"])

    for obj in objects:
        indices = [
            v.index
            for v in obj.data.vertices
            if abs(v.co.z - target_z) <= tolerance
            and abs(math.hypot(v.co.x, v.co.y) - target_radius) <= tolerance
        ]
        if not indices:
            continue
        if target_object is not None:
            raise RuntimeError("正解内周エッジが複数オブジェクトに存在します。")
        target_object = obj
        target_indices = indices

    expected = int(settings["expected_vertex_count"])
    if target_object is None or len(target_indices) != expected:
        raise RuntimeError(f"正解内周エッジを一意に特定できません: {len(target_indices)} vertices")

    old_group = target_object.vertex_groups.get("GT_INNER_RING")
    if old_group is not None:
        target_object.vertex_groups.remove(old_group)
    group = target_object.vertex_groups.new(name="GT_INNER_RING")
    group.add(target_indices, 1.0, "REPLACE")
    return target_object, target_indices


def _apply_materials(settings: dict[str, Any]) -> None:
    for material in bpy.data.materials:
        values = settings.get(material.name)
        if values is None:
            continue
        material.use_nodes = True
        principled = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue
        principled.inputs["Base Color"].default_value = (*values["base_color"], 1.0)
        principled.inputs["Metallic"].default_value = float(values["metallic"])
        principled.inputs["Roughness"].default_value = float(values["roughness"])


def render_1194m(
    config_path: Path,
    base_dir: Path,
    *,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_dir = (
        (base_dir / config["output_dir"]).resolve()
        if output_dir is None
        else output_dir.resolve()
    )
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    model_path = (base_dir / config["model_path"]).resolve()
    objects = _import_and_center(model_path, bool(config.get("normalize_model_axes", True)))
    target_object, target_indices = _assign_target_ring(objects, config["target_ring"])
    _apply_materials(config["materials"])

    scene = bpy.context.scene
    render_cfg = config["render"]
    configure_render_scene(
        scene,
        engine=render_cfg["engine"],
        width=int(render_cfg["width"]),
        height=int(render_cfg["height"]),
        world_color=tuple(config["world"]["color"]),
        world_strength=float(config["world"]["strength"]),
    )

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    bpy.ops.object.light_add(type="SUN")
    sun = bpy.context.object

    cameras = _expand_cameras(config["camera_grid"])
    lights = _expand_lights(config["lighting_grid"])
    samples = []

    for cam_cfg, light_cfg in itertools.product(cameras, lights):
        if limit is not None and len(samples) >= limit:
            break
        sample_id = f"{cam_cfg['id']}__{light_cfg['id']}"
        camera.location = Vector(cam_cfg["location"])
        camera.data.lens = float(cam_cfg["lens_mm"])
        look_at(camera, Vector(cam_cfg["target"]))
        bpy.context.view_layer.update()
        ray_direction = configure_sun_spherical(
            sun,
            tilt_deg=light_cfg["tilt_deg"],
            azimuth_deg=light_cfg["azimuth_deg"],
            energy=light_cfg["energy"],
            angle_deg=light_cfg["angle_deg"],
        )

        image_path = images_dir / f"{sample_id}.png"
        label_path = labels_dir / f"{sample_id}.json"
        scene.render.filepath = str(image_path)
        image_points = project_target_points(scene, camera, target_object, target_indices)
        label = {
            "sample_id": sample_id,
            "vertex_group": "GT_INNER_RING",
            "object": target_object.name,
            "vertex_indices": target_indices,
            "image_width": scene.render.resolution_x,
            "image_height": scene.render.resolution_y,
            "image_points": image_points,
            "conditions": {
                "camera_id": cam_cfg["id"],
                "lighting_id": light_cfg["id"],
                "camera": cam_cfg,
                "lighting": {**light_cfg, "direction_mode": "object_spherical", "ray_direction": ray_direction},
            },
        }
        label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        bpy.ops.render.render(write_still=True)

        samples.append(
            {
                "sample_id": sample_id,
                "image": image_path.relative_to(dataset_dir).as_posix(),
                "label": label_path.relative_to(dataset_dir).as_posix(),
                "conditions": label["conditions"],
            }
        )
        print(f"Generated: {sample_id}", flush=True)

    manifest = {
        "dataset_id": config["dataset_id"],
        "target_ring": config["target_ring"],
        "camera_count": len(cameras),
        "lighting_count": len(lights),
        "sample_count": len(samples),
        "samples": samples,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if config.get("save_preview_blend", True):
        bpy.ops.wm.save_as_mainfile(filepath=str(dataset_dir / "dataset_preview.blend"))
    print(f"Dataset saved: {dataset_dir}", flush=True)
    return dataset_dir


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="1194M原CGデータセットをBlenderで生成する。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(arguments)
    render_1194m(args.config, args.base_dir, output_dir=args.output, limit=args.limit)


if __name__ == "__main__":
    main()
