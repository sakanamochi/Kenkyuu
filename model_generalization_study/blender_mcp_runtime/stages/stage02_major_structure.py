"""比較機Cの外周骨格と機器ポッドを追加する。"""

from pathlib import Path
import math

import bpy
from mathutils import Matrix, Vector


WORKSPACE = Path(r"C:\Users\siomi\OneDrive\デスクトップ\Kenkyuu")
BLEND_PATH = (
    WORKSPACE
    / "model_generalization_study"
    / "output"
    / "mcp_model"
    / "comparison_c_from_1194M_3.blend"
)


def get_material(name, color, metallic=0.0, roughness=0.4):
    """再実行しても同じマテリアルを再利用する。"""
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    return material


def move_to_collection(obj, collection):
    """生成オブジェクトを指定コレクションだけへ所属させる。"""
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def apply_scale(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)


def add_beveled_box(name, center, dimensions, orientation, material, collection, bevel=0.12):
    bpy.ops.mesh.primitive_cube_add(location=center)
    obj = bpy.context.object
    obj.name = name
    obj.matrix_world = Matrix.Translation(Vector(center)) @ orientation
    obj.dimensions = dimensions
    apply_scale(obj)
    modifier = obj.modifiers.new("角部R", "BEVEL")
    modifier.width = bevel
    modifier.segments = 4
    modifier.limit_method = "ANGLE"
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.materials.append(material)
    move_to_collection(obj, collection)
    return obj


def add_cylinder_between(name, start, end, radius, material, collection, vertices=32):
    start = Vector(start)
    end = Vector(end)
    direction = end - start
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=direction.length,
        location=(start + end) / 2,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(direction.normalized())
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    move_to_collection(obj, collection)
    return obj


def add_torus(name, center, major_radius, minor_radius, material, collection):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=128,
        minor_segments=16,
        location=center,
        rotation=(math.pi / 2, 0, 0),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    move_to_collection(obj, collection)
    return obj


def local_frame(angle):
    """ローカルX=接線、Y=機体軸、Z=外向き半径の座標系を返す。"""
    radial = Vector((math.cos(angle), 0.0, math.sin(angle)))
    tangent = Vector((-math.sin(angle), 0.0, math.cos(angle)))
    axial = Vector((0.0, 1.0, 0.0))
    orientation = Matrix((tangent, axial, radial)).transposed().to_4x4()
    return tangent, axial, radial, orientation


# 再実行時はこの段階の成果物だけ作り直す。
old_collection = bpy.data.collections.get("COMPARISON_C_MAJOR")
if old_collection:
    for obj in list(old_collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(old_collection)

major_collection = bpy.data.collections.new("COMPARISON_C_MAJOR")
bpy.context.scene.collection.children.link(major_collection)

titanium = get_material("C3_チタン", (0.32, 0.37, 0.42), metallic=0.9, roughness=0.24)
graphite = get_material("C3_グラファイト", (0.025, 0.035, 0.045), metallic=0.25, roughness=0.3)
gold = get_material("C3_金色断熱材", (0.78, 0.38, 0.045), metallic=0.72, roughness=0.27)
white = get_material("C3_白色セラミック", (0.72, 0.76, 0.79), metallic=0.08, roughness=0.32)
red = get_material("C3_識別赤", (0.55, 0.018, 0.012), metallic=0.2, roughness=0.25)

# 基準機の円錐面に食い込む二重サービスレール。
add_torus("C3_前側サービスレール", (0, -0.85, 0), 10.47, 0.20, titanium, major_collection)
add_torus("C3_後側サービスレール", (0, 1.65, 0), 11.18, 0.20, titanium, major_collection)

# PAF開口の輪郭は維持しつつ、比較機固有の前縁保護リングを追加する。
add_torus("C3_前縁保護リング", (0, -4.76, 0), 6.18, 0.19, graphite, major_collection)
add_torus("C3_前縁識別バンド", (0, -4.82, 0), 6.48, 0.075, gold, major_collection)

# 12本の長手ストリンガーを円錐面へ沿わせ、滑らかな基準機と異なる分割外観を作る。
for index in range(12):
    angle = math.radians(index * 30 + 15)
    radial = Vector((math.cos(angle), 0.0, math.sin(angle)))
    front = radial * 9.85 + Vector((0, -0.75, 0))
    rear = radial * 11.12 + Vector((0, 1.75, 0))
    add_cylinder_between(
        f"C3_ストリンガー_{index + 1:02d}",
        front,
        rear,
        0.13,
        graphite if index % 2 else titanium,
        major_collection,
        vertices=24,
    )

# 四方向へ一体化した機器ポッドを配置し、外周シルエットを明確に変える。
for pod_index, angle_deg in enumerate((0, 90, 180, 270), start=1):
    angle = math.radians(angle_deg)
    tangent, axial, radial, orientation = local_frame(angle)
    center = radial * 11.12 + axial * 0.50

    pod = add_beveled_box(
        f"C3_機器ポッド_{pod_index:02d}",
        center,
        (3.15, 2.65, 1.15),
        orientation,
        graphite,
        major_collection,
        bevel=0.24,
    )
    pod["role"] = "比較機固有の外周機器ポッド"

    face_center = center + radial * 0.62
    add_beveled_box(
        f"C3_ポッド外板_{pod_index:02d}",
        face_center,
        (2.62, 2.05, 0.14),
        orientation,
        white if pod_index % 2 else titanium,
        major_collection,
        bevel=0.09,
    )

    # 外板の6本ボルトは局所座標で配置し、必ずポッド表面へ密着させる。
    for bolt_index, (tx, ay) in enumerate(
        ((-1.05, -0.75), (0.0, -0.75), (1.05, -0.75), (-1.05, 0.75), (0.0, 0.75), (1.05, 0.75)),
        start=1,
    ):
        bolt_center = face_center + tangent * tx + axial * ay + radial * 0.11
        add_cylinder_between(
            f"C3_ポッド{pod_index:02d}_ボルト_{bolt_index:02d}",
            bolt_center - radial * 0.07,
            bolt_center + radial * 0.07,
            0.095,
            gold,
            major_collection,
            vertices=24,
        )

    # ポッド中央の識別キャップ。
    cap_center = face_center + radial * 0.18
    add_cylinder_between(
        f"C3_識別キャップ_{pod_index:02d}",
        cap_center - radial * 0.09,
        cap_center + radial * 0.09,
        0.34,
        red if pod_index == 1 else gold,
        major_collection,
        vertices=48,
    )

    # ポッド両端の円錐面側ブラケット。
    for side in (-1, 1):
        bracket_center = center + tangent * side * 1.36 - radial * 0.45
        add_beveled_box(
            f"C3_ポッド{pod_index:02d}_取付座_{'L' if side < 0 else 'R'}",
            bracket_center,
            (0.34, 2.9, 0.52),
            orientation,
            titanium,
            major_collection,
            bevel=0.10,
        )

bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
print({"major_objects": len(major_collection.objects), "blend_path": str(BLEND_PATH)})
