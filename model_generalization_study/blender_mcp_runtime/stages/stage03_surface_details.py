"""比較機Cへ表面機器、配線、内周ラッチを追加する。"""

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


def material(name):
    result = bpy.data.materials.get(name)
    if result is None:
        raise RuntimeError(f"必要なマテリアルがありません: {name}")
    return result


def move_to_collection(obj, collection):
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def local_frame(angle):
    radial = Vector((math.cos(angle), 0.0, math.sin(angle)))
    tangent = Vector((-math.sin(angle), 0.0, math.cos(angle)))
    axial = Vector((0.0, 1.0, 0.0))
    orientation = Matrix((tangent, axial, radial)).transposed().to_4x4()
    return tangent, axial, radial, orientation


def add_box(name, center, dimensions, orientation, mat, collection, bevel=0.08):
    bpy.ops.mesh.primitive_cube_add(location=center)
    obj = bpy.context.object
    obj.name = name
    obj.matrix_world = Matrix.Translation(Vector(center)) @ orientation
    obj.dimensions = dimensions
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    modifier = obj.modifiers.new("角部R", "BEVEL")
    modifier.width = bevel
    modifier.segments = 3
    modifier.limit_method = "ANGLE"
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.materials.append(mat)
    move_to_collection(obj, collection)
    return obj


def add_cylinder(name, center, axis, radius, depth, mat, collection, vertices=32):
    axis = Vector(axis).normalized()
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=center,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(axis)
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    move_to_collection(obj, collection)
    return obj


def add_cable(name, points, mat, collection, radius=0.055):
    curve_data = bpy.data.curves.new(name=f"{name}_形状", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 16
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4

    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, coordinate in zip(spline.bezier_points, points):
        point.co = coordinate
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"

    obj = bpy.data.objects.new(name, curve_data)
    collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def add_text(name, body, center, orientation, mat, collection, size=0.38):
    curve = bpy.data.curves.new(f"{name}_形状", "FONT")
    curve.body = body
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = size
    curve.extrude = 0.018
    curve.bevel_depth = 0.006
    obj = bpy.data.objects.new(name, curve)
    obj.matrix_world = Matrix.Translation(Vector(center)) @ orientation
    obj.data.materials.append(mat)
    collection.objects.link(obj)
    return obj


# 再実行時は表面ディテールだけを作り直す。
old_collection = bpy.data.collections.get("COMPARISON_C_DETAILS")
if old_collection:
    for obj in list(old_collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(old_collection)

details = bpy.data.collections.new("COMPARISON_C_DETAILS")
bpy.context.scene.collection.children.link(details)

titanium = material("C3_チタン")
graphite = material("C3_グラファイト")
gold = material("C3_金色断熱材")
white = material("C3_白色セラミック")
red = material("C3_識別赤")

glass = bpy.data.materials.get("C3_センサーガラス") or bpy.data.materials.new("C3_センサーガラス")
glass.diffuse_color = (0.015, 0.16, 0.22, 1.0)
glass.use_nodes = True
glass_bsdf = glass.node_tree.nodes.get("Principled BSDF")
glass_bsdf.inputs["Base Color"].default_value = (0.005, 0.08, 0.12, 1.0)
glass_bsdf.inputs["Metallic"].default_value = 0.55
glass_bsdf.inputs["Roughness"].default_value = 0.12

# 前方円錐面の8台センサーユニット。基台はシェルへ0.25埋め込む。
for sensor_index in range(8):
    angle = math.radians(sensor_index * 45 + 22.5)
    tangent, axial, radial, orientation = local_frame(angle)
    center = radial * 8.72 + axial * -2.05

    add_box(
        f"C3_センサー基台_{sensor_index + 1:02d}",
        center,
        (1.30, 1.25, 0.52),
        orientation,
        graphite,
        details,
        bevel=0.13,
    )
    face_center = center + radial * 0.32
    add_box(
        f"C3_センサー外板_{sensor_index + 1:02d}",
        face_center,
        (1.02, 0.92, 0.12),
        orientation,
        titanium,
        details,
        bevel=0.055,
    )
    lens_center = face_center + radial * 0.14
    add_cylinder(
        f"C3_センサーレンズ_{sensor_index + 1:02d}",
        lens_center,
        radial,
        0.25,
        0.18,
        glass,
        details,
        vertices=48,
    )
    for side in (-1, 1):
        screw_center = face_center + tangent * side * 0.38 + radial * 0.10
        add_cylinder(
            f"C3_センサー{sensor_index + 1:02d}_ねじ_{'L' if side < 0 else 'R'}",
            screw_center,
            radial,
            0.055,
            0.10,
            gold,
            details,
            vertices=20,
        )

# 各センサー群から対応する機器ポッドまで、円錐面へ沿う二重配線を引く。
for cable_index, (start_deg, end_deg) in enumerate(
    ((22.5, 0), (67.5, 90), (157.5, 180), (247.5, 270)),
    start=1,
):
    points = []
    for step in range(7):
        ratio = step / 6
        angle = math.radians(start_deg + (end_deg - start_deg) * ratio)
        y = -1.85 + 2.25 * ratio
        surface_radius = 8.9 + 2.20 * ratio
        points.append(
            Vector(
                (
                    math.cos(angle) * surface_radius,
                    y,
                    math.sin(angle) * surface_radius,
                )
            )
        )
    add_cable(
        f"C3_主配線_{cable_index:02d}_A",
        points,
        gold,
        details,
        radius=0.062,
    )
    offset_points = [point + Vector((0, 0.10, 0)) for point in points]
    add_cable(
        f"C3_主配線_{cable_index:02d}_B",
        offset_points,
        graphite,
        details,
        radius=0.052,
    )

# 後部円筒面は黒い熱制御パネルで8分割する。
for panel_index in range(8):
    angle = math.radians(panel_index * 45)
    tangent, axial, radial, orientation = local_frame(angle)
    center = radial * 11.03 + axial * 2.75
    add_box(
        f"C3_後部熱制御パネル_{panel_index + 1:02d}",
        center,
        (2.65, 1.48, 0.13),
        orientation,
        graphite,
        details,
        bevel=0.07,
    )
    for side in (-1, 1):
        fastener_center = center + tangent * side * 1.07 + radial * 0.10
        add_cylinder(
            f"C3_熱制御パネル{panel_index + 1:02d}_留め具_{'L' if side < 0 else 'R'}",
            fastener_center,
            radial,
            0.07,
            0.10,
            titanium,
            details,
            vertices=20,
        )

# PAF内周へ6台の小型ラッチを追加するが、中心開口は塞がない。
for latch_index in range(6):
    angle = math.radians(latch_index * 60)
    tangent, axial, radial, orientation = local_frame(angle)
    base_center = radial * 5.72 + axial * -4.61
    add_box(
        f"C3_内周ラッチ基台_{latch_index + 1:02d}",
        base_center,
        (0.78, 0.44, 0.40),
        orientation,
        titanium,
        details,
        bevel=0.08,
    )
    roller_center = radial * 5.38 + axial * -4.83
    add_cylinder(
        f"C3_内周ラッチローラー_{latch_index + 1:02d}",
        roller_center,
        tangent,
        0.15,
        0.50,
        gold,
        details,
        vertices=32,
    )

# 可視性の高い二つのポッドへ機体識別を刻む。
for label_index, angle_deg in enumerate((0, 90), start=1):
    angle = math.radians(angle_deg)
    _, axial, radial, orientation = local_frame(angle)
    center = radial * 11.94 + axial * 1.23
    add_text(
        f"C3_識別文字_{label_index:02d}",
        "PAF-C3",
        center,
        orientation,
        red if label_index == 1 else gold,
        details,
        size=0.30,
    )

bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
print({"detail_objects": len(details.objects), "blend_path": str(BLEND_PATH)})
