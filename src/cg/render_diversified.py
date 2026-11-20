"""多様形状PAFの原CG画像と正解内周座標をBlenderで生成する。"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg.blender_common import (
    configure_render_scene,
    configure_sun_spherical,
    create_closed_mesh,
    look_at,
    project_target_points,
    spherical_direction,
)
from cg.geometry_model import (
    build_dimensions,
    generate_diversified_plan,
    make_geometry,
    read_json,
    sha256_file,
    write_json,
)


def _build_paf_object(parameters: dict[str, float], segments: int):
    vertices, faces, rim_indices, profile = make_geometry(parameters, segments)
    material = bpy.data.materials.new("PAF_material")
    material.use_nodes = True
    obj = create_closed_mesh("PAF", vertices, faces, material)
    obj.vertex_groups.new(name="target_inner_rim").add(rim_indices, 1.0, "REPLACE")
    return obj, rim_indices, profile


def render_diversified(
    config_path: Path,
    base_dir: Path,
    *,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> Path:
    config = read_json(config_path)
    out = (
        (base_dir / config["output_dir"]).resolve()
        if output_dir is None
        else output_dir.resolve()
    )
    planned = generate_diversified_plan(config_path, out)
    ratio_keys = ("bottom_outer_diameter_ratio", "height_ratio", "hole_diameter_ratio")

    # 寸法範囲の端8点すべてで閉じた正体積メッシュになることを描画前に確認する。
    for ratios_tuple in itertools.product(*(config[k] for k in ratio_keys)):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        _build_paf_object(build_dimensions(config, dict(zip(ratio_keys, ratios_tuple))), int(config["segments"]))

    samples_to_render = planned["samples"][:limit] if limit is not None else planned["samples"]
    records = []
    for index, sample in enumerate(samples_to_render):
        sid = sample["sample_id"]
        image_path = out / f"images/{sid}.png"
        label_path = out / f"labels/{sid}.json"

        if image_path.exists() and label_path.exists():
            label = read_json(label_path)
            if label["parameters"] != sample["parameters"] or label["conditions"] != sample["conditions"]:
                raise ValueError(f"保存済み画像の寸法・撮影条件が計画と一致しません: {sid}")
        else:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene = bpy.context.scene
            obj, rim_indices, profile = _build_paf_object(sample["parameters"], int(config["segments"]))
            configure_render_scene(
                scene,
                engine="BLENDER_EEVEE",
                width=int(config["resolution"]),
                height=int(config["resolution"]),
            )
            conditions = sample["conditions"]
            bsdf = obj.data.materials[0].node_tree.nodes["Principled BSDF"]
            brightness = float(conditions["brightness"])
            bsdf.inputs["Base Color"].default_value = (brightness, brightness, brightness, 1.0)
            bsdf.inputs["Metallic"].default_value = float(conditions["metallic"])
            bsdf.inputs["Roughness"].default_value = float(conditions["roughness"])

            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.data.lens = float(config["camera"]["lens_mm"])
            scene.camera = camera

            height = float(sample["parameters"]["height"])
            center = Vector((0.0, 0.0, -height / 2.0))
            direction = spherical_direction(conditions["camera_tilt"], conditions["camera_azimuth"])
            extent = math.hypot(float(sample["parameters"]["bottom_outer_diameter"]) / 2.0, height / 2.0)
            distance = 2.0 * extent / math.tan(camera.data.angle / 2.0)
            margin = float(config["camera"]["margin"])

            for _ in range(24):
                camera.location = direction * distance
                look_at(camera, center)
                bpy.context.view_layer.update()
                projected = [world_to_camera_view(scene, camera, obj.matrix_world @ v.co) for v in obj.data.vertices]
                if all(p.z > 0.0 and margin <= p.x <= 1.0 - margin and margin <= p.y <= 1.0 - margin for p in projected):
                    break
                distance *= 1.12
            else:
                raise ValueError(f"モデル全体を画像内へ収められません: {sid}")

            bpy.ops.object.light_add(type="SUN")
            configure_sun_spherical(
                bpy.context.object,
                tilt_deg=conditions["light_tilt"],
                azimuth_deg=conditions["light_azimuth"],
                energy=conditions["energy"],
                angle_deg=0.53,
            )
            scene.view_settings.exposure = float(conditions["exposure"])
            points = project_target_points(scene, camera, obj, rim_indices)

            label = {
                "sample_id": sid,
                "image_width": int(config["resolution"]),
                "image_height": int(config["resolution"]),
                "image_points": points,
                "parameters": sample["parameters"],
                "ratios": sample["ratios"],
                "conditions": conditions,
                "camera_distance": distance,
                "camera_location": list(camera.location),
                "camera_target": list(center),
                "meridian_profile": profile,
                "target_present": True,
            }
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(image_path)
            bpy.ops.render.render(write_still=True)
            write_json(label_path, label)
            if index == 0:
                bpy.ops.wm.save_as_mainfile(filepath=str(out / "sample_model.blend"))

        records.append(
            {
                **sample,
                "split": "train",
                "shape_id": sid,
                "image": image_path.relative_to(out).as_posix(),
                "label": label_path.relative_to(out).as_posix(),
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
            }
        )
        if (index + 1) % 32 == 0:
            print(f"Dataset {index + 1}/{len(samples_to_render)}", flush=True)

    write_json(out / "manifest.json", {"dataset_id": config["dataset_id"], "samples": records})
    write_json(
        out / "provenance.json",
        {
            "config_sha256": sha256_file(config_path),
            "plan_sha256": sha256_file(out / "plan.json"),
            "blender": bpy.app.version_string,
            "boundary_mesh_checks": 8,
            "sample_count": len(records),
        },
    )
    print(f"Dataset saved: {out}", flush=True)
    return out


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="多様形状PAF原CGデータセットをBlenderで生成する。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(arguments)
    render_diversified(args.config, args.base_dir, output_dir=args.output, limit=args.limit)


if __name__ == "__main__":
    main()
