"""Blender内で実行する簡易PAFの独立したモデル生成・プレビュー入口。"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

import bpy
import bmesh
from mathutils import Vector

ROOT = Path(__file__).resolve().parent
SEGMENTS = 192


def mesh_object(name, vertices, faces, material):
    """明示した頂点と面からメッシュを作る。"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    if mesh.validate():
        raise ValueError(f"不正なメッシュ: {name}")
    # 閉じた各部品の法線を外向きに統一し、穴・退化面を検出する。
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    if not all(edge.is_manifold for edge in bm.edges) or bm.calc_volume(signed=True) <= 0:
        bm.free()
        raise ValueError(f"閉じた立体になっていません: {name}")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def validate(p):
    """空洞が残る寸法と、設定値の型・範囲を確認する。"""
    for key in ("top_outer_diameter", "bottom_outer_diameter", "taper_height", "wall_thickness"):
        if not isinstance(p[key], (int, float)) or not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key}には有限の正数を指定してください")
    if p["bottom_outer_diameter"] < p["top_outer_diameter"]:
        raise ValueError("下端外径は上端外径以上にしてください")
    if 2 * p["wall_thickness"] >= p["top_outer_diameter"]:
        raise ValueError("肉厚が大きすぎて開口部がなくなります")
    if type(p["rib_count"]) is not int or not 0 <= p["rib_count"] <= 48:
        raise ValueError("rib_countは0〜48の整数にしてください")
    if len(p["color"]) != 3 or not all(math.isfinite(c) and 0 <= c <= 1 for c in p["color"]):
        raise ValueError("colorは0〜1のRGB値を3つ指定してください")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", p["name"]):
        raise ValueError("モデル名は英数字・ハイフン・アンダースコアにしてください")


def build_model(p):
    """断面を回転して中空本体を作る。上端内周中心を原点に固定する。"""
    top = p["top_outer_diameter"] / 2
    bottom = p["bottom_outer_diameter"] / 2
    height = p["taper_height"]
    wall = p["wall_thickness"]
    base_height = 0.10 * p["top_outer_diameter"]
    material = bpy.data.materials.new("PAF_material")
    material.diffuse_color = (*p["color"], 1)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*p["color"], 1)
    bsdf.inputs["Metallic"].default_value = 0.25
    bsdf.inputs["Roughness"].default_value = 0.40

    # 半径・高さの断面。内外壁の半径差を一定にし、上下の開口に蓋はしない。
    profile = [(top, 0), (bottom, -height), (bottom, -height-base_height),
               (bottom-wall, -height-base_height), (bottom-wall, -height), (top-wall, 0)]
    vertices = [(r*math.cos(2*math.pi*i/SEGMENTS), r*math.sin(2*math.pi*i/SEGMENTS), z)
                for r, z in profile for i in range(SEGMENTS)]
    faces = []
    for j in range(len(profile)):
        for i in range(SEGMENTS):
            ni, nj = (i+1) % SEGMENTS, (j+1) % len(profile)
            faces.append((j*SEGMENTS+i, nj*SEGMENTS+i, nj*SEGMENTS+ni, j*SEGMENTS+ni))
    body = mesh_object("PAF_body", vertices, faces, material)
    for poly in body.data.polygons:
        poly.use_smooth = poly.index // SEGMENTS not in (2, 5)
    group = body.vertex_groups.new(name="target_inner_rim")
    group.add(list(range(5*SEGMENTS, 6*SEGMENTS)), 1.0, "REPLACE")
    objects = [body]

    # リブは外壁に沿う補強板。幅・突出量は上端外径に対する固定比率。
    width = 0.025 * p["top_outer_diameter"]
    depth = 0.035 * p["top_outer_diameter"]
    inset = 0.002 * p["top_outer_diameter"]
    for i in range(p["rib_count"]):
        angle = 2*math.pi*i/p["rib_count"]
        # 接線方向に幅を持つ板が外壁から浮かないよう、根元を少し埋め込む。
        section = [(top-inset, 0), (bottom-inset, -height), (bottom+depth, -height), (top+depth, 0)]
        rib_vertices = [(r*math.cos(angle)-y*math.sin(angle),
                         r*math.sin(angle)+y*math.cos(angle), z)
                        for y in (-width/2, width/2) for r, z in section]
        rib_faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                     (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        objects.append(mesh_object(f"rib_{i:02d}", rib_vertices, rib_faces, material))
    return objects, vertices[5*SEGMENTS:6*SEGMENTS], base_height


def setup_preview():
    """全モデル共通のカメラ・照明による形状確認用の描画設定。"""
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.world.color = (0.18, 0.18, 0.18)
    bpy.ops.object.camera_add(location=(2.5, -3.5, 2.6))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, -0.2))-camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.5
    scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(0, -3, 4))
    light = bpy.context.object
    light.data.energy = 500
    light.data.shape = "DISK"
    light.data.size = 3
    light.rotation_euler = (-light.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = [config["defaults"] | override for override in config["models"]]
    for p in models:
        validate(p)
    if len({p["name"] for p in models}) != len(models):
        raise ValueError("モデル名が重複しています")
    for p in models:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.scene.world = bpy.data.worlds.new("World")
        objects, ring, base_height = build_model(p)
        out = args.output.resolve() / p["name"]
        out.mkdir(parents=True, exist_ok=True)
        # OBJにはモデルだけを出力し、カメラ等はblendに残す。
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.wm.obj_export(filepath=str(out / "model.obj"), export_selected_objects=True,
                              forward_axis="Y", up_axis="Z")
        metadata = {"parameters": p, "units": "meters", "base_height": base_height,
                    "target_inner_diameter": p["top_outer_diameter"]-2*p["wall_thickness"],
                    "target_ring_xyz": ring, "segments": SEGMENTS,
                    "note": "上端内周中心が原点、軸は+Z。実在PAFの忠実な再現ではない。"}
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        setup_preview()
        bpy.context.scene.render.filepath = str(out / "preview.png")
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "model.blend"))
        bpy.ops.render.render(write_still=True)
        print(f"Generated: {out}", flush=True)


if __name__ == "__main__":
    main()
