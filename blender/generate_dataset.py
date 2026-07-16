import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="PAFの条件付きCGデータセットを生成する")
    parser.add_argument("--config", required=True, help="実験設定JSON")
    parser.add_argument("--limit-samples", type=int, default=None, help="描画確認用の上限")
    return parser.parse_args(arguments)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def spherical_direction(tilt_deg: float, azimuth_deg: float) -> Vector:
    """PAF軸+Zからの傾きと、-Yを0°とする方位から単位方向を作る。"""
    tilt = math.radians(tilt_deg)
    azimuth = math.radians(azimuth_deg)
    return Vector(
        (
            math.sin(azimuth) * math.sin(tilt),
            -math.cos(azimuth) * math.sin(tilt),
            math.cos(tilt),
        )
    )


def expand_camera_settings(config: dict) -> list[dict]:
    if "camera_grid" not in config:
        return config["cameras"]
    grid = config["camera_grid"]
    settings = []
    distances = grid["distances"] if "distances" in grid else [grid["distance"]]
    targets = grid.get("targets", [grid.get("target", [0.0, 0.0, 0.0])])
    for tilt_deg in grid["tilt_deg"]:
        azimuths = grid["azimuth_deg"]
        if float(tilt_deg) == 0.0 and grid.get("collapse_axis_azimuth", True):
            azimuths = azimuths[:1]
        for azimuth_deg, distance, target_index in itertools.product(
            azimuths, distances, range(len(targets))
        ):
            direction = spherical_direction(float(tilt_deg), float(azimuth_deg))
            location = direction * float(distance)
            target = targets[target_index]
            settings.append(
                {
                    "id": (
                        f"camera_t{int(tilt_deg):03d}_a{int(azimuth_deg):03d}"
                        f"_d{float(distance):05.1f}_o{target_index:02d}"
                    ),
                    "tilt_deg": float(tilt_deg),
                    "azimuth_deg": float(azimuth_deg),
                    "location": list(location),
                    "target": target,
                    "lens_mm": float(grid["lens_mm"]),
                }
            )
    return settings


def expand_lighting_settings(config: dict) -> list[dict]:
    if "lighting_grid" not in config:
        return config["lighting"]
    grid = config["lighting_grid"]
    settings = []
    energies = grid["energies"] if "energies" in grid else [grid["energy"]]
    for tilt_deg in grid["tilt_deg"]:
        azimuths = grid["azimuth_deg"]
        if float(tilt_deg) == 0.0 and grid.get("collapse_axis_azimuth", True):
            azimuths = azimuths[:1]
        for azimuth_deg, energy in itertools.product(azimuths, energies):
            settings.append(
                {
                    "id": (
                        f"light_t{int(tilt_deg):03d}_a{int(azimuth_deg):03d}"
                        f"_e{float(energy):04.1f}"
                    ),
                    "tilt_deg": float(tilt_deg),
                    "azimuth_deg": float(azimuth_deg),
                    "angle_deg": float(grid.get("angle_deg", 0.0)),
                    "energy": float(energy),
                }
            )
    return settings


def configure_sun(
    sun: bpy.types.Object,
    settings: dict,
    camera_location: Vector,
    target: Vector,
    direction_mode: str,
) -> list[float]:
    """0°をカメラ光軸、180°をその正反対として太陽光を設定する。"""
    sun.data.energy = float(settings["energy"])
    sun.data.angle = math.radians(float(settings.get("angle_deg", 0.0)))
    if direction_mode == "camera_great_circle":
        front_source_direction = (camera_location - target).normalized()
        side_direction = Vector((1.0, 0.0, 0.0))
        side_direction -= front_source_direction * side_direction.dot(
            front_source_direction
        )
        side_direction.normalize()
        sweep_angle = math.radians(float(settings["azimuth_deg"]))
        source_direction = (
            math.cos(sweep_angle) * front_source_direction
            + math.sin(sweep_angle) * side_direction
        )
        ray_direction = -source_direction
        sun.rotation_euler = ray_direction.to_track_quat("-Z", "Y").to_euler()
        return list(ray_direction)

    if direction_mode == "object_spherical":
        source_direction = spherical_direction(
            float(settings["tilt_deg"]), float(settings["azimuth_deg"])
        )
        ray_direction = -source_direction
        sun.rotation_euler = ray_direction.to_track_quat("-Z", "Y").to_euler()
        return list(ray_direction)

    if "azimuth_deg" not in settings:
        sun.rotation_euler = tuple(
            math.radians(value) for value in settings["rotation_euler_deg"]
        )
        ray_direction = sun.rotation_euler.to_matrix() @ Vector((0.0, 0.0, -1.0))
        return list(ray_direction)

    azimuth = math.radians(float(settings["azimuth_deg"]))
    elevation = math.radians(float(settings["elevation_deg"]))
    source_direction = Vector(
        (
            math.sin(azimuth) * math.cos(elevation),
            -math.cos(azimuth) * math.cos(elevation),
            math.sin(elevation),
        )
    )
    ray_direction = -source_direction
    sun.rotation_euler = ray_direction.to_track_quat("-Z", "Y").to_euler()
    return list(ray_direction)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_and_center_model(
    model_path: Path, *, normalize_model_axes: bool
) -> list[bpy.types.Object]:
    bpy.ops.wm.obj_import(
        filepath=str(model_path),
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
    )
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"メッシュを読み込めませんでした: {model_path}")

    if normalize_model_axes:
        # OBJローカルのPAF軸+ZをBlenderワールド+Zへ一致させる。
        for obj in objects:
            obj.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners)))
    maximum = Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners)))
    center = (minimum + maximum) / 2
    for obj in objects:
        obj.location -= center
    return objects


def assign_target_ring(objects: list[bpy.types.Object], settings: dict):
    target_object = None
    target_indices = []
    target_z = float(settings["z"])
    target_radius = float(settings["radius"])
    tolerance = float(settings["tolerance"])

    for obj in objects:
        indices = [
            vertex.index
            for vertex in obj.data.vertices
            if abs(vertex.co.z - target_z) <= tolerance
            and abs(math.hypot(vertex.co.x, vertex.co.y) - target_radius) <= tolerance
        ]
        if not indices:
            continue
        if target_object is not None:
            raise RuntimeError("正解内周エッジが複数オブジェクトに存在します")
        target_object = obj
        target_indices = indices

    expected_count = int(settings["expected_vertex_count"])
    if target_object is None or len(target_indices) != expected_count:
        raise RuntimeError(
            f"正解内周エッジを一意に特定できませんでした: {len(target_indices)} vertices"
        )

    old_group = target_object.vertex_groups.get("GT_INNER_RING")
    if old_group is not None:
        target_object.vertex_groups.remove(old_group)
    group = target_object.vertex_groups.new(name="GT_INNER_RING")
    group.add(target_indices, 1.0, "REPLACE")
    return target_object, target_indices


def apply_materials(settings: dict) -> None:
    for material in bpy.data.materials:
        values = settings.get(material.name)
        if values is None:
            continue
        material.use_nodes = True
        principled = next(
            (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        if principled is None:
            continue
        principled.inputs["Base Color"].default_value = (*values["base_color"], 1.0)
        principled.inputs["Metallic"].default_value = float(values["metallic"])
        principled.inputs["Roughness"].default_value = float(values["roughness"])


def _planet_material(name: str, colors: list[list[float]], scale: float):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    noise.inputs["Scale"].default_value = float(scale)
    noise.inputs["Detail"].default_value = 7.0
    noise.inputs["Roughness"].default_value = 0.72
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[1])
    positions = [0.0, 0.38, 0.58, 0.78]
    for index, (position, color) in enumerate(zip(positions, colors)):
        element = ramp.color_ramp.elements[0] if index == 0 else ramp.color_ramp.elements.new(position)
        element.position = position
        element.color = (*color, 1.0)
    principled.inputs["Roughness"].default_value = 0.9
    if "Emission Color" in principled.inputs:
        links.new(ramp.outputs["Color"], principled.inputs["Emission Color"])
        principled.inputs["Emission Strength"].default_value = 0.35
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_background_object():
    """実写素材に依存しないEarth/Moon風の背景天体を作る。"""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    background = bpy.context.object
    background.name = "PROCEDURAL_BACKGROUND_BODY"
    background.hide_render = True
    try:
        background.visible_shadow = False
    except AttributeError:
        pass
    earth = _planet_material(
        "BACKGROUND_EARTH",
        [[0.01, 0.03, 0.12], [0.02, 0.18, 0.55], [0.05, 0.42, 0.12], [0.82, 0.86, 0.78]],
        3.2,
    )
    moon = _planet_material(
        "BACKGROUND_MOON",
        [[0.025, 0.025, 0.03], [0.12, 0.12, 0.13], [0.35, 0.34, 0.33], [0.68, 0.66, 0.62]],
        5.5,
    )
    background.data.materials.append(earth)
    return background, {"earth": earth, "moon": moon}


def configure_background(
    background,
    materials: dict,
    setting: dict,
    camera,
    target: Vector,
) -> None:
    background_type = setting.get("type", "space")
    background.hide_render = background_type == "space"
    if background.hide_render:
        return
    if background_type not in materials:
        raise ValueError(f"未対応の背景です: {background_type}")
    view_direction = (target - camera.location).normalized()
    camera_rotation = camera.matrix_world.to_quaternion()
    right = camera_rotation @ Vector((1.0, 0.0, 0.0))
    up = camera_rotation @ Vector((0.0, 1.0, 0.0))
    offset = setting.get("offset", [0.0, 0.0])
    background.location = (
        target
        + view_direction * float(setting.get("depth", 45.0))
        + right * float(offset[0])
        + up * float(offset[1])
    )
    radius = float(setting.get("radius", 20.0))
    background.scale = (radius, radius, radius)
    background.data.materials[0] = materials[background_type]


def project_target_points(scene, camera, target_object, target_indices) -> list[list[float]]:
    bpy.context.view_layer.update()
    points = []
    for index in target_indices:
        world_coordinate = target_object.matrix_world @ target_object.data.vertices[index].co
        projected = world_to_camera_view(scene, camera, world_coordinate)
        points.append(
            [
                projected.x * scene.render.resolution_x,
                (1.0 - projected.y) * scene.render.resolution_y,
            ]
        )
    return points


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(config["output_dir"])
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()
    model_objects = import_and_center_model(
        resolve_project_path(config["model_path"]),
        normalize_model_axes=bool(config.get("normalize_model_axes", False)),
    )
    target_object, target_indices = assign_target_ring(
        model_objects, config["target_ring"]
    )
    apply_materials(config["materials"])

    scene = bpy.context.scene
    render = config["render"]
    scene.render.engine = render["engine"]
    scene.render.resolution_x = int(render["width"])
    scene.render.resolution_y = int(render["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    scene.world.use_nodes = True
    world_background = next(
        node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND"
    )
    world_background.inputs["Color"].default_value = (*config["world"]["color"], 1.0)
    world_background.inputs["Strength"].default_value = float(
        config["world"]["strength"]
    )

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    bpy.ops.object.light_add(type="SUN")
    sun = bpy.context.object
    background_object, background_materials = create_background_object()

    camera_settings = expand_camera_settings(config)
    lighting_settings = expand_lighting_settings(config)
    background_settings = config.get("backgrounds", [{"id": "space", "type": "space"}])
    include_background_id = "backgrounds" in config
    samples = []
    for camera_setting, lighting_setting, background_setting in itertools.product(
        camera_settings, lighting_settings, background_settings
    ):
        if args.limit_samples is not None and len(samples) >= args.limit_samples:
            break
        sample_id = f"{camera_setting['id']}__{lighting_setting['id']}"
        if include_background_id:
            sample_id += f"__bg_{background_setting['id']}"
        camera.location = Vector(camera_setting["location"])
        camera.data.lens = float(camera_setting["lens_mm"])
        target = Vector(camera_setting.get("target", [0.0, 0.0, 0.0]))
        look_at(camera, target)
        bpy.context.view_layer.update()
        configure_background(
            background_object,
            background_materials,
            background_setting,
            camera,
            target,
        )

        ray_direction = configure_sun(
            sun,
            lighting_setting,
            camera.location,
            target,
            config.get("lighting_direction_mode", "world_azimuth"),
        )

        image_path = images_dir / f"{sample_id}.png"
        label_path = labels_dir / f"{sample_id}.json"
        scene.render.filepath = str(image_path)
        image_points = project_target_points(
            scene, camera, target_object, target_indices
        )
        label = {
            "sample_id": sample_id,
            "vertex_group": "GT_INNER_RING",
            "object": target_object.name,
            "vertex_indices": target_indices,
            "image_width": scene.render.resolution_x,
            "image_height": scene.render.resolution_y,
            "image_points": image_points,
            "conditions": {
                "camera_id": camera_setting["id"],
                "lighting_id": lighting_setting["id"],
                "camera": camera_setting,
                "lighting": {
                    **lighting_setting,
                    "direction_mode": config.get(
                        "lighting_direction_mode", "world_azimuth"
                    ),
                    "ray_direction": ray_direction,
                },
                **({"background": background_setting} if include_background_id else {}),
                **config.get("sample_conditions", {}),
            },
        }
        label_path.write_text(
            json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        bpy.ops.render.render(write_still=True)

        samples.append(
            {
                "sample_id": sample_id,
                "image": image_path.relative_to(dataset_dir).as_posix(),
                "label": label_path.relative_to(dataset_dir).as_posix(),
                "conditions": label["conditions"],
                **({"split": config["default_split"]} if "default_split" in config else {}),
            }
        )
        print(f"Generated: {sample_id}")

    manifest = {
        "experiment_id": config["experiment_id"],
        "config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "target_ring": config["target_ring"],
        "camera_count": len(camera_settings),
        "lighting_count": len(lighting_settings),
        "background_count": len(background_settings),
        "sample_count": len(samples),
        "samples": samples,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if config.get("save_preview_blend", True):
        bpy.ops.wm.save_as_mainfile(filepath=str(dataset_dir / "dataset_preview.blend"))
    print(f"Dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
