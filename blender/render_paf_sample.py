import json
import math
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "3d_models" / "1194M_3" / "1194M_3.obj"
OUTPUT_DIR = PROJECT_ROOT / "output" / "blender"
RENDER_PATH = OUTPUT_DIR / "paf_sample.png"
BLEND_PATH = OUTPUT_DIR / "paf_sample.blend"
GROUND_TRUTH_DIR = PROJECT_ROOT / "output" / "ground_truth"
GROUND_TRUTH_PATH = GROUND_TRUTH_DIR / "paf_sample_inner_ring_points.json"

# 1194Mモデル内で、検出対象とするPAF内周エッジのモデル座標。
# 同じ内周半径にある2本のうち、カメラ側（手前側）のエッジを正解とする。
TARGET_RING_Z = 4.73574
TARGET_RING_RADIUS = 5.41875
TARGET_RING_TOLERANCE = 1e-4


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


# 毎回同じ結果になるよう、起動時のシーンを空にする。
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fusionから出力したPAFモデルを読み込む。
bpy.ops.wm.obj_import(filepath=str(MODEL_PATH), forward_axis="NEGATIVE_Z", up_axis="Y")
model_objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
if not model_objects:
    raise RuntimeError(f"メッシュを読み込めませんでした: {MODEL_PATH}")

# モデル全体の中心を原点に合わせ、カメラ配置を再現可能にする。
corners = [obj.matrix_world @ Vector(corner) for obj in model_objects for corner in obj.bound_box]
model_center = sum(corners, Vector()) / len(corners)
for obj in model_objects:
    obj.location -= model_center

# 内周エッジを明示的な頂点グループとして.blend内に保存する。
target_ring_object = None
target_ring_indices = []
for obj in model_objects:
    indices = [
        vertex.index
        for vertex in obj.data.vertices
        if abs(vertex.co.z - TARGET_RING_Z) <= TARGET_RING_TOLERANCE
        and abs(math.hypot(vertex.co.x, vertex.co.y) - TARGET_RING_RADIUS)
        <= TARGET_RING_TOLERANCE
    ]
    if not indices:
        continue
    if target_ring_object is not None:
        raise RuntimeError("内周エッジの候補が複数オブジェクトに存在します")
    target_ring_object = obj
    target_ring_indices = indices

if target_ring_object is None or len(target_ring_indices) != 88:
    raise RuntimeError(
        f"内周エッジを一意に特定できませんでした: {len(target_ring_indices)} vertices"
    )

old_group = target_ring_object.vertex_groups.get("GT_INNER_RING")
if old_group is not None:
    target_ring_object.vertex_groups.remove(old_group)
target_ring_group = target_ring_object.vertex_groups.new(name="GT_INNER_RING")
target_ring_group.add(target_ring_indices, 1.0, "REPLACE")

# MTLには拡散色しかないため、最低限の材質特性をBlender側で補う。
material_settings = {
    "アルミニウム_-_サテン": {
        "base_color": (0.62, 0.65, 0.70, 1.0),
        "metallic": 0.85,
        "roughness": 0.32,
    },
    "PEEK": {
        "base_color": (0.38, 0.22, 0.09, 1.0),
        "metallic": 0.0,
        "roughness": 0.45,
    },
}
for material in bpy.data.materials:
    settings = material_settings.get(material.name)
    if settings is None:
        continue
    material.use_nodes = True
    principled = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if principled is None:
        continue
    principled.inputs["Base Color"].default_value = settings["base_color"]
    principled.inputs["Metallic"].default_value = settings["metallic"]
    principled.inputs["Roughness"].default_value = settings["roughness"]

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.filepath = str(RENDER_PATH)

# Worldノードが有効な場合にも、背景と環境光が完全な黒になるよう設定する。
scene.world.use_nodes = True
world_background = next(
    node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND"
)
world_background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world_background.inputs["Strength"].default_value = 0.0

# 斜め視点にして、PAF内周が画像上で楕円になる条件を作る。
bpy.ops.object.camera_add(location=(0.0, -34.0, 23.0))
camera = bpy.context.object
camera.data.lens = 55
look_at(camera, Vector((0.0, 0.0, 0.0)))
scene.camera = camera
bpy.context.view_layer.update()

# 指定した3D内周頂点を画像座標へ投影し、検出処理とは独立した正解点を作る。
ground_truth_points = []
for vertex_index in target_ring_indices:
    vertex = target_ring_object.data.vertices[vertex_index]
    world_coordinate = target_ring_object.matrix_world @ vertex.co
    camera_coordinate = world_to_camera_view(scene, camera, world_coordinate)
    ground_truth_points.append(
        [
            camera_coordinate.x * scene.render.resolution_x,
            (1.0 - camera_coordinate.y) * scene.render.resolution_y,
        ]
    )

GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_PATH.write_text(
    json.dumps(
        {
            "vertex_group": "GT_INNER_RING",
            "object": target_ring_object.name,
            "vertex_indices": target_ring_indices,
            "image_width": scene.render.resolution_x,
            "image_height": scene.render.resolution_y,
            "image_points": ground_truth_points,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# 太陽に相当する平行光だけを使い、影側へ補助光を入れない。
bpy.ops.object.light_add(type="SUN", location=(8.0, -10.0, 18.0))
sun = bpy.context.object
sun.data.energy = 3.0
sun.rotation_euler = (0.35, -0.45, -0.4)

bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
bpy.ops.render.render(write_still=True)

print(f"Saved blend: {BLEND_PATH}")
print(f"Saved render: {RENDER_PATH}")
print(f"Saved ground truth: {GROUND_TRUTH_PATH}")
