"""Blender描画で1194M・多様形状が共有するカメラ・照明・投影・閉メッシュ検証処理。"""

from __future__ import annotations

import math
from typing import Any

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    """オブジェクトの-Z軸をtarget方向へ向ける。"""
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def spherical_direction(tilt_deg: float, azimuth_deg: float) -> Vector:
    """PAF軸+Zからの傾きと、-Yを0度とする方位角から単位方向ベクトルを作る。"""
    tilt = math.radians(tilt_deg)
    azimuth = math.radians(azimuth_deg)
    return Vector(
        (
            math.sin(azimuth) * math.sin(tilt),
            -math.cos(azimuth) * math.sin(tilt),
            math.cos(tilt),
        )
    )


def configure_sun_spherical(
    sun: bpy.types.Object,
    *,
    tilt_deg: float,
    azimuth_deg: float,
    energy: float,
    angle_deg: float = 0.0,
) -> list[float]:
    """オブジェクト球面座標で太陽光を設定し、光線方向ベクトルを返す。"""
    sun.data.energy = float(energy)
    sun.data.angle = math.radians(float(angle_deg))
    ray_direction = -spherical_direction(float(tilt_deg), float(azimuth_deg))
    sun.rotation_euler = ray_direction.to_track_quat("-Z", "Y").to_euler()
    return list(ray_direction)


def configure_render_scene(
    scene: bpy.types.Scene,
    *,
    engine: str,
    width: int,
    height: int,
    world_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    world_strength: float = 0.0,
) -> None:
    """黒背景・PNG RGB出力のBlenderシーンを設定する。"""
    scene.render.engine = engine
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    if scene.world is None:
        scene.world = bpy.data.worlds.new("Black_background")
    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (*world_color, 1.0)
    background.inputs["Strength"].default_value = float(world_strength)


def project_target_points(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    target_object: bpy.types.Object,
    target_indices: list[int],
) -> list[list[float]]:
    """正解内周エッジの頂点群を画像座標(x, y)[px]へ投影する。"""
    bpy.context.view_layer.update()
    points: list[list[float]] = []
    for index in target_indices:
        world_coord = target_object.matrix_world @ target_object.data.vertices[index].co
        projected = world_to_camera_view(scene, camera, world_coord)
        points.append(
            [
                projected.x * scene.render.resolution_x,
                (1.0 - projected.y) * scene.render.resolution_y,
            ]
        )
    return points


def create_closed_mesh(
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    material: bpy.types.Material,
) -> bpy.types.Object:
    """頂点・面から閉じた正体積メッシュを作成し、法線を外向きに統一する。"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    if mesh.validate():
        raise ValueError(f"不正なメッシュ: {name}")

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    if not all(edge.is_manifold for edge in bm.edges) or bm.calc_volume(signed=True) <= 0.0:
        bm.free()
        raise ValueError(f"閉じた正体積メッシュになっていません: {name}")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj
