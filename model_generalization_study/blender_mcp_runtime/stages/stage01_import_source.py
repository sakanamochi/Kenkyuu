"""元の1194M_3 OBJを変更せずBlenderへ読み込む。"""

from pathlib import Path

import bpy


WORKSPACE = Path(r"C:\Users\siomi\OneDrive\デスクトップ\Kenkyuu")
SOURCE_OBJ = WORKSPACE / "assets" / "1194M_3" / "1194M_3.obj"
OUTPUT_DIR = WORKSPACE / "model_generalization_study" / "output" / "mcp_model"
BLEND_PATH = OUTPUT_DIR / "comparison_c_from_1194M_3.blend"


# 初期キューブ・カメラ・ライトを含め、空のシーンから開始する。
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for collection in list(bpy.data.collections):
    if collection.name != "Collection":
        bpy.data.collections.remove(collection)

root_collection = bpy.context.scene.collection
default_collection = bpy.data.collections.get("Collection")
default_collection.name = "SOURCE_1194M_3"

# 元OBJは単位・頂点・マテリアルを維持して読み込む。
bpy.ops.wm.obj_import(filepath=str(SOURCE_OBJ), forward_axis="NEGATIVE_Z", up_axis="Y")
source_objects = list(bpy.context.selected_objects)
if not source_objects:
    raise RuntimeError("1194M_3.objからオブジェクトを読み込めませんでした")

for index, obj in enumerate(source_objects, start=1):
    obj.name = f"SOURCE_1194M_3_{index:02d}"
    obj["source_file"] = str(SOURCE_OBJ)
    obj["source_role"] = "ユーザー作成の元モデル（変更禁止の基準形状）"

# 原形を残したまま、以降の編集対象となるリンク複製を別コレクションへ作る。
work_collection = bpy.data.collections.new("COMPARISON_C_WORK")
root_collection.children.link(work_collection)
for index, source in enumerate(source_objects, start=1):
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.name = f"PAF_BASE_{index:02d}"
    work_collection.objects.link(duplicate)

# 元形状はファイル内に保存するが、制作・レンダー時には非表示にする。
default_collection.hide_viewport = True
default_collection.hide_render = True

# 編集対象を選択し、3Dビューへ全体表示する。
bpy.ops.object.select_all(action="DESELECT")
for obj in work_collection.objects:
    obj.select_set(True)
if work_collection.objects:
    bpy.context.view_layer.objects.active = work_collection.objects[0]

for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type != "VIEW_3D":
            continue
        region = next((item for item in area.regions if item.type == "WINDOW"), None)
        if region is None:
            continue
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.view3d.view_selected(use_all_regions=False)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))

print(
    {
        "source_objects": len(source_objects),
        "work_objects": len(work_collection.objects),
        "blend_path": str(BLEND_PATH),
    }
)
