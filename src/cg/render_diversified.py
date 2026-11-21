"""多様形状PAFの原CG画像（元構図・投影拡張・外観分解各ビュー）と正解内周座標をBlenderで生成する。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

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
    RATIO_KEYS,
    build_dimensions,
    generate_diversified_plan,
    make_geometry,
    read_json,
    sha256_file,
    write_json,
)

SURFACE_MODES = ("plain", "albedo", "stripe", "noise", "checker")


def _parse_view_spec(view_name: str) -> tuple[str, str]:
    """ビュー名を (構図モード: base|projection, 外観モード) へ分解する。"""
    if view_name == "base":
        return "base", "plain"
    if view_name.startswith("base_"):
        mode = view_name[len("base_") :]
        if mode in SURFACE_MODES:
            return "base", mode
    if view_name.startswith("projection_"):
        mode = view_name[len("projection_") :]
        if mode in SURFACE_MODES:
            return "projection", mode
    raise ValueError(f"未対応のビュー名です: {view_name}")


def _build_paf_object(parameters: dict[str, float], segments: int):
    vertices, faces, rim_indices, profile = make_geometry(parameters, segments)
    material = bpy.data.materials.new("PAF_material")
    if material.node_tree is None:
        material.use_nodes = True
    obj = create_closed_mesh("PAF", vertices, faces, material)
    obj.vertex_groups.new(name="target_inner_rim").add(rim_indices, 1.0, "REPLACE")
    return obj, rim_indices, profile


def _apply_surface_material(
    material: bpy.types.Material,
    mode: str,
    conditions: dict[str, Any],
    surface: dict[str, Any],
) -> None:
    """指定した外観モードに応じてPrincipled BSDFノードを再構成する。"""
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    if mode == "plain":
        v = float(conditions["brightness"])
        bsdf.inputs["Base Color"].default_value = (v, v, v, 1.0)
        bsdf.inputs["Metallic"].default_value = float(conditions["metallic"])
        bsdf.inputs["Roughness"].default_value = float(conditions["roughness"])
        return

    bsdf.inputs["Metallic"].default_value = float(surface["metallic"])
    bsdf.inputs["Roughness"].default_value = float(surface["roughness"])

    if mode == "albedo":
        v = float(surface["albedo_mean"])
        bsdf.inputs["Base Color"].default_value = (v, v, v, 1.0)
        return

    lo = float(surface["albedo_low"])
    hi = float(surface["albedo_high"])
    tex_coord = tree.nodes.new("ShaderNodeTexCoord")

    if mode == "stripe":
        # 方位角方向の周期濃淡により、実機PAFの放射状リブ・パネル継ぎ目の明暗を模擬する。
        sep = tree.nodes.new("ShaderNodeSeparateXYZ")
        atan2 = tree.nodes.new("ShaderNodeMath")
        atan2.operation = "ARCTAN2"
        mul = tree.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = float(surface["stripe_cycles"])
        add = tree.nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        add.inputs[1].default_value = float(surface["stripe_phase"])
        cos_node = tree.nodes.new("ShaderNodeMath")
        cos_node.operation = "COSINE"
        remap = tree.nodes.new("ShaderNodeMapRange")
        remap.inputs["From Min"].default_value = -1.0
        remap.inputs["From Max"].default_value = 1.0
        remap.inputs["To Min"].default_value = lo
        remap.inputs["To Max"].default_value = hi
        combine = tree.nodes.new("ShaderNodeCombineColor")

        tree.links.new(tex_coord.outputs["Object"], sep.inputs["Vector"])
        tree.links.new(sep.outputs["Y"], atan2.inputs[0])
        tree.links.new(sep.outputs["X"], atan2.inputs[1])
        tree.links.new(atan2.outputs["Value"], mul.inputs[0])
        tree.links.new(mul.outputs["Value"], add.inputs[0])
        tree.links.new(add.outputs["Value"], cos_node.inputs[0])
        tree.links.new(cos_node.outputs["Value"], remap.inputs["Value"])
        for ch in ("Red", "Green", "Blue"):
            tree.links.new(remap.outputs["Result"], combine.inputs[ch])
        tree.links.new(combine.outputs["Color"], bsdf.inputs["Base Color"])
        return

    if mode == "noise":
        noise = tree.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = float(surface["noise_scale"])
        noise.inputs["Detail"].default_value = 4.0
        noise.inputs["Roughness"].default_value = 0.55
        remap = tree.nodes.new("ShaderNodeMapRange")
        remap.clamp = True
        remap.inputs["From Min"].default_value = 0.25
        remap.inputs["From Max"].default_value = 0.75
        remap.inputs["To Min"].default_value = lo
        remap.inputs["To Max"].default_value = hi
        combine = tree.nodes.new("ShaderNodeCombineColor")

        tree.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
        tree.links.new(noise.outputs["Fac"], remap.inputs["Value"])
        for ch in ("Red", "Green", "Blue"):
            tree.links.new(remap.outputs["Result"], combine.inputs[ch])
        tree.links.new(combine.outputs["Color"], bsdf.inputs["Base Color"])
        return

    if mode == "checker":
        checker = tree.nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = float(surface["checker_scale"])
        checker.inputs["Color1"].default_value = (lo, lo, lo, 1.0)
        checker.inputs["Color2"].default_value = (hi, hi, hi, 1.0)
        tree.links.new(tex_coord.outputs["Object"], checker.inputs["Vector"])
        tree.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
        return

    raise ValueError(f"未対応の外観モードです: {mode}")


def _place_base_camera(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    obj: bpy.types.Object,
    sample: dict[str, Any],
    margin: float,
) -> tuple[float, Vector]:
    """モデル全体が余白margin内に収まるカメラ距離・注視点を配置する。"""
    conditions = sample["conditions"]
    height = float(sample["parameters"]["height"])
    center = Vector((0.0, 0.0, -height / 2.0))
    direction = spherical_direction(conditions["camera_tilt"], conditions["camera_azimuth"])
    extent = math.hypot(float(sample["parameters"]["bottom_outer_diameter"]) / 2.0, height / 2.0)
    distance = 2.0 * extent / math.tan(camera.data.angle / 2.0)

    for _ in range(24):
        camera.location = direction * distance
        look_at(camera, center)
        bpy.context.view_layer.update()
        projected = [world_to_camera_view(scene, camera, obj.matrix_world @ v.co) for v in obj.data.vertices]
        if all(p.z > 0.0 and margin <= p.x <= 1.0 - margin and margin <= p.y <= 1.0 - margin for p in projected):
            return distance, center
        distance *= 1.12
    raise ValueError(f"モデル全体を画像内へ収められません: {sample['sample_id']}")


def _rim_span_px(points: list[list[float]]) -> tuple[float, float, float]:
    """投影頂点群の重心(cx, cy)と最大直径[px]を返す。"""
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    half = n // 2
    diam = max(math.hypot(points[i][0] - points[i + half][0], points[i][1] - points[i + half][1]) for i in range(half))
    return cx, cy, diam


def _place_projection_camera(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    obj: bpy.types.Object,
    rim_indices: list[int],
    sample: dict[str, Any],
    resolution: int,
) -> tuple[float, Vector, list[list[float]]]:
    """内周楕円が目標長径・扁平率・中心位置へ投影されるようカメラを配置する。"""
    proj = sample["projection"]
    hole = float(sample["parameters"]["hole_diameter"])
    target_major = float(proj["target_major_px_256"]) * (resolution / 256.0)
    aspect = max(0.1, min(0.996, float(proj["target_aspect_ratio"])))
    tilt_deg = max(5.0, min(84.26, math.degrees(math.acos(aspect))))
    direction = spherical_direction(tilt_deg, float(proj["camera_azimuth"]))

    focal_px = (camera.data.lens / camera.data.sensor_width) * resolution
    distance = focal_px * hole / target_major
    target_x = float(proj["target_cx"]) * resolution
    target_y = float(proj["target_cy"]) * resolution
    target = Vector((0.0, 0.0, 0.0))

    for _ in range(6):
        camera.location = target + direction * distance
        look_at(camera, target)
        points = project_target_points(scene, camera, obj, rim_indices)
        cx, cy, diam = _rim_span_px(points)
        if diam > 1e-6:
            distance *= diam / target_major
        # 画像面での重心誤差をカメラ基底へ逆投影して注視点を微調整する。
        dx_px = cx - target_x
        dy_px = cy - target_y
        rot = camera.matrix_world.to_3x3()
        right = rot @ Vector((1.0, 0.0, 0.0))
        up = rot @ Vector((0.0, 1.0, 0.0))
        world_per_px = distance / focal_px
        target = target + right * (dx_px * world_per_px) - up * (dy_px * world_per_px)

    # すべての内周頂点が画像内に収まるよう必要時のみ距離を微調整する。
    for _ in range(16):
        camera.location = target + direction * distance
        look_at(camera, target)
        points = project_target_points(scene, camera, obj, rim_indices)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if min(xs) >= 2.0 and max(xs) <= resolution - 2.0 and min(ys) >= 2.0 and max(ys) <= resolution - 2.0:
            return distance, target, points
        distance *= 1.06
    return distance, target, points


def render_diversified(
    config_path: Path,
    base_dir: Path,
    *,
    output_dir: Path | None = None,
    limit: int | None = None,
    views: list[str] | None = None,
    fixed_ratios: dict[str, float] | None = None,
    projected_splits: tuple[str, ...] = ("train",),
) -> Path:
    config = read_json(config_path)
    out = (base_dir / config["output_dir"]).resolve() if output_dir is None else output_dir.resolve()
    planned = generate_diversified_plan(config_path, out, fixed_ratios=fixed_ratios)

    # 寸法範囲の端8点すべてで閉じた正体積メッシュになることを描画前に確認する。
    for ratios_tuple in itertools.product(*(config[k] for k in RATIO_KEYS)):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        _build_paf_object(build_dimensions(config, dict(zip(RATIO_KEYS, ratios_tuple))), int(config["segments"]))

    multi_view = views is not None
    active_views = ["base"] if not multi_view else list(views)
    parsed_views = [(name, *_parse_view_spec(name)) for name in active_views]
    resolution = int(config["resolution"])
    margin = float(config["camera"]["margin"])

    samples_to_render = planned["samples"][:limit] if limit is not None else planned["samples"]
    records = []

    for index, sample in enumerate(samples_to_render):
        sid = sample["sample_id"]
        split = sample.get("split", "train")

        # 既存ファイルの再利用判定
        needed_views = []
        for view_name, framing, mode in parsed_views:
            if (framing == "projection" or mode != "plain") and split not in projected_splits:
                continue
            if fixed_ratios is not None and split == "test":
                continue
            view_root = (out / view_name) if multi_view else out
            img_path = view_root / f"images/{sid}.png"
            needs_label = (framing == "base" and view_name == "base") or (
                framing == "projection" and view_name == "projection_plain"
            )
            lbl_path = view_root / f"labels/{sid}.json"
            if not img_path.exists() or (needs_label and not lbl_path.exists()):
                needed_views.append((view_name, framing, mode, view_root, img_path, lbl_path if needs_label else None))

        if needed_views:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene = bpy.context.scene
            obj, rim_indices, profile = _build_paf_object(sample["parameters"], int(config["segments"]))
            configure_render_scene(scene, engine="BLENDER_EEVEE", width=resolution, height=resolution)
            conditions = sample["conditions"]
            surface = sample["surface"]

            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.data.lens = float(config["camera"]["lens_mm"])
            scene.camera = camera

            bpy.ops.object.light_add(type="SUN")
            configure_sun_spherical(
                bpy.context.object,
                tilt_deg=conditions["light_tilt"],
                azimuth_deg=conditions["light_azimuth"],
                energy=conditions["energy"],
                angle_deg=0.53,
            )
            scene.view_settings.exposure = float(conditions["exposure"])

            # 構図ごとにカメラ配置を1回だけ計算し、複数外観モードを高速描画する。
            by_framing: dict[str, list[Any]] = {"base": [], "projection": []}
            for item in needed_views:
                by_framing[item[1]].append(item)

            for framing in ("base", "projection"):
                group = by_framing[framing]
                if not group:
                    continue
                if framing == "base":
                    distance, target = _place_base_camera(scene, camera, obj, sample, margin)
                    points = project_target_points(scene, camera, obj, rim_indices)
                else:
                    distance, target, points = _place_projection_camera(
                        scene, camera, obj, rim_indices, sample, resolution
                    )

                for view_name, _, mode, view_root, img_path, lbl_path in group:
                    _apply_surface_material(obj.data.materials[0], mode, conditions, surface)
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    scene.render.filepath = str(img_path)
                    bpy.ops.render.render(write_still=True)
                    if lbl_path is not None:
                        write_json(
                            lbl_path,
                            {
                                "sample_id": sid,
                                "split": split,
                                "framing": framing,
                                "image_width": resolution,
                                "image_height": resolution,
                                "image_points": points,
                                "parameters": sample["parameters"],
                                "ratios": sample["ratios"],
                                "conditions": conditions,
                                "projection": sample["projection"] if framing == "projection" else None,
                                "camera_distance": distance,
                                "camera_location": list(camera.location),
                                "camera_target": list(target),
                                "meridian_profile": profile,
                                "target_present": True,
                            },
                        )

            if not multi_view and index == 0:
                bpy.ops.wm.save_as_mainfile(filepath=str(out / "sample_model.blend"))

        if not multi_view:
            image_path = out / f"images/{sid}.png"
            label_path = out / f"labels/{sid}.json"
            records.append(
                {
                    **sample,
                    "shape_id": sid,
                    "image": image_path.relative_to(out).as_posix(),
                    "label": label_path.relative_to(out).as_posix(),
                    "image_sha256": sha256_file(image_path),
                    "label_sha256": sha256_file(label_path),
                }
            )
        if (index + 1) % 50 == 0 or (index + 1) == len(samples_to_render):
            print(f"Rendered {index + 1}/{len(samples_to_render)} shapes ({','.join(active_views)})", flush=True)

    if not multi_view:
        write_json(out / "manifest.json", {"dataset_id": config["dataset_id"], "samples": records})
    write_json(
        out / "provenance.json",
        {
            "config_sha256": sha256_file(config_path),
            "plan_sha256": sha256_file(out / "plan.json"),
            "blender": bpy.app.version_string,
            "views": active_views,
            "fixed_ratios": fixed_ratios,
            "sample_count": len(samples_to_render),
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
    parser.add_argument("--views", nargs="+", default=None, help="複数ビューを一括描画する際のビュー名一覧")
    parser.add_argument("--fixed-ratios", type=str, default=None, help="固定寸法比JSON（単一代表円錐対照用）")
    args = parser.parse_args(arguments)
    fixed = json.loads(args.fixed_ratios) if args.fixed_ratios else None
    render_diversified(
        args.config,
        args.base_dir,
        output_dir=args.output,
        limit=args.limit,
        views=args.views,
        fixed_ratios=fixed,
    )


if __name__ == "__main__":
    main()
