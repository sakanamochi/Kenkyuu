"""比較機Cを評価用形式へ書き出し、構造を数値検証する。"""

from pathlib import Path
import json

import bpy
from mathutils import Vector


WORKSPACE = Path(r"C:\Users\siomi\OneDrive\デスクトップ\Kenkyuu")
OUTPUT_DIR = WORKSPACE / "model_generalization_study" / "output" / "mcp_model"
BLEND_PATH = OUTPUT_DIR / "comparison_c_from_1194M_3.blend"
OBJ_PATH = OUTPUT_DIR / "comparison_c_from_1194M_3.obj"
GLB_PATH = OUTPUT_DIR / "comparison_c_from_1194M_3.glb"
MANIFEST_PATH = OUTPUT_DIR / "comparison_c_manifest.json"

MODEL_COLLECTIONS = (
    "COMPARISON_C_WORK",
    "COMPARISON_C_MAJOR",
    "COMPARISON_C_DETAILS",
)


def world_bounds(objects):
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
        if hasattr(obj, "bound_box")
    ]
    return {
        "min": [round(min(point[axis] for point in corners), 5) for axis in range(3)],
        "max": [round(max(point[axis] for point in corners), 5) for axis in range(3)],
    }


source = bpy.data.objects.get("SOURCE_1194M_3_01")
base = bpy.data.objects.get("PAF_BASE_01")
if source is None or base is None:
    raise RuntimeError("元モデルまたは編集用基準モデルが見つかりません")

# 元モデルと基準コピーのメッシュ規模が一致することを検証する。
source_signature = (
    len(source.data.vertices),
    len(source.data.edges),
    len(source.data.polygons),
)
base_signature = (
    len(base.data.vertices),
    len(base.data.edges),
    len(base.data.polygons),
)
if source_signature != base_signature:
    raise RuntimeError(
        f"元モデルのメッシュ構造が変化しています: source={source_signature}, base={base_signature}"
    )

model_objects = []
collection_counts = {}
for collection_name in MODEL_COLLECTIONS:
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        raise RuntimeError(f"モデルコレクションがありません: {collection_name}")
    objects = list(collection.objects)
    collection_counts[collection_name] = len(objects)
    model_objects.extend(objects)

if len(model_objects) < 100:
    raise RuntimeError(f"ディテール数が想定より少なすぎます: {len(model_objects)}")

# 評価用書き出しでは元モデル保存用コレクションと撮影用カメラ・ライトを除外する。
bpy.ops.object.select_all(action="DESELECT")
for obj in model_objects:
    obj.hide_set(False)
    obj.select_set(True)
if model_objects:
    bpy.context.view_layer.objects.active = model_objects[0]

bpy.ops.wm.obj_export(
    filepath=str(OBJ_PATH),
    export_selected_objects=True,
    apply_modifiers=True,
    export_materials=True,
    export_normals=True,
    export_uv=True,
    forward_axis="NEGATIVE_Z",
    up_axis="Y",
)

bpy.ops.export_scene.gltf(
    filepath=str(GLB_PATH),
    export_format="GLB",
    use_selection=True,
    export_apply=True,
)

type_counts = {}
for obj in model_objects:
    type_counts[obj.type] = type_counts.get(obj.type, 0) + 1

manifest = {
    "model_name": "comparison_c_from_1194M_3",
    "purpose": "CNNの機体モデル依存性を調べる比較用PAFモデル",
    "source_model": "assets/1194M_3/1194M_3.obj",
    "source_mesh_signature": {
        "vertices": source_signature[0],
        "edges": source_signature[1],
        "polygons": source_signature[2],
    },
    "source_preserved_in_blend": True,
    "source_collection_hidden": bool(
        bpy.data.collections["SOURCE_1194M_3"].hide_viewport
        and bpy.data.collections["SOURCE_1194M_3"].hide_render
    ),
    "model_object_count": len(model_objects),
    "collection_object_counts": collection_counts,
    "object_type_counts": type_counts,
    "world_bounds": world_bounds(model_objects),
    "artifacts": {
        "blend": str(BLEND_PATH),
        "obj": str(OBJ_PATH),
        "glb": str(GLB_PATH),
        "hero_render": str(OUTPUT_DIR / "comparison_c_hero.png"),
        "front_render": str(OUTPUT_DIR / "comparison_c_front.png"),
        "side_render": str(OUTPUT_DIR / "comparison_c_side.png"),
    },
}

MANIFEST_PATH.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
print(json.dumps(manifest, ensure_ascii=False))
