"""比較機Cのスタジオ照明と3方向プレビューを生成する。"""

from pathlib import Path

import bpy
from mathutils import Vector


WORKSPACE = Path(r"C:\Users\siomi\OneDrive\デスクトップ\Kenkyuu")
OUTPUT_DIR = WORKSPACE / "model_generalization_study" / "output" / "mcp_model"
BLEND_PATH = OUTPUT_DIR / "comparison_c_from_1194M_3.blend"


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, energy, size, color, collection):
    data = bpy.data.lights.new(name=f"{name}_データ", type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    look_at(obj, (0, -0.4, 0))
    collection.objects.link(obj)
    return obj


# 再実行時はレンダー設定だけを作り直す。
old_collection = bpy.data.collections.get("COMPARISON_C_RENDER")
if old_collection:
    for obj in list(old_collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(old_collection)

render_collection = bpy.data.collections.new("COMPARISON_C_RENDER")
bpy.context.scene.collection.children.link(render_collection)

camera_data = bpy.data.cameras.new("C3_カメラ_データ")
camera_data.lens = 58
camera_data.sensor_width = 36
camera = bpy.data.objects.new("C3_カメラ", camera_data)
render_collection.objects.link(camera)
bpy.context.scene.camera = camera

add_area_light(
    "C3_キーライト",
    (10.0, -15.0, 19.0),
    1700,
    9.0,
    (0.78, 0.88, 1.0),
    render_collection,
)
add_area_light(
    "C3_フィルライト",
    (-17.0, -10.0, 5.0),
    1650,
    8.0,
    (1.0, 0.66, 0.38),
    render_collection,
)
add_area_light(
    "C3_リムライト",
    (6.0, 12.0, 17.0),
    2100,
    7.0,
    (0.42, 0.62, 1.0),
    render_collection,
)
add_area_light(
    "C3_前面補助光",
    (-3.0, -18.0, -8.0),
    1200,
    6.0,
    (0.82, 0.88, 1.0),
    render_collection,
)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 1100
scene.render.resolution_y = 850
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = False
scene.render.image_settings.color_depth = "8"

scene.world.color = (0.004, 0.007, 0.014)
scene.world.use_nodes = True
background = scene.world.node_tree.nodes.get("Background")
background.inputs["Color"].default_value = (0.003, 0.007, 0.018, 1.0)
background.inputs["Strength"].default_value = 0.28

try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    pass

views = {
    "comparison_c_hero.png": ((28.0, -40.0, 22.0), (0.0, -0.55, 0.0), 58),
    "comparison_c_front.png": ((0.0, -48.0, 0.0), (0.0, -0.65, 0.0), 58),
    "comparison_c_side.png": ((46.0, -7.0, 3.0), (0.0, -0.1, 0.0), 58),
}

for filename, (location, target, lens) in views.items():
    camera.location = location
    camera.data.lens = lens
    look_at(camera, target)
    scene.render.filepath = str(OUTPUT_DIR / filename)
    bpy.ops.render.render(write_still=True)

bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
print(
    {
        "renders": [str(OUTPUT_DIR / filename) for filename in views],
        "blend_path": str(BLEND_PATH),
    }
)
