"""既存1194Mと簡易PAFを共通の黒背景・視点・照明で描画する。"""

import itertools
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from generate import build_model, validate
from blender.generate_dataset import (
    import_and_center_model, assign_target_ring, apply_materials,
    spherical_direction, look_at, configure_sun, project_target_points,
)


def main():
    settings = json.loads((HERE / "comparison.json").read_text(encoding="utf-8"))
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / "config/render_base.json").read_text(encoding="utf-8"))
    models = [config["defaults"] | p for p in config["models"]]
    # 上端外径1.4 mのモデルを形状比較に含める。
    models.append(config["defaults"] | {"name": "wide_top", "top_outer_diameter": 1.4})
    samples = []
    output = HERE / "output/comparison"
    for p in [{"name": "reference_1194M"}] + models:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        if p["name"] == "reference_1194M":
            objects = import_and_center_model(ROOT / original["model_path"], normalize_model_axes=True)
            body, indices = assign_target_ring(objects, original["target_ring"])
            apply_materials(original["materials"])
            bpy.context.view_layer.update()
            center = sum((body.matrix_world @ body.data.vertices[i].co for i in indices), Vector()) / len(indices)
            # 座標系を上端内周中心にそろえる。形状比は変更しない。
            radius = float(original["target_ring"]["radius"])
            scale = (config["defaults"]["top_outer_diameter"]-2*config["defaults"]["wall_thickness"])/(2*radius)
            for obj in objects:
                obj.location = (obj.location-center)*scale
                obj.scale *= scale
            inner_diameter = 2*radius*scale
        else:
            validate(p)
            objects, ring, _ = build_model(p)
            body = objects[0]
            group = body.vertex_groups["target_inner_rim"].index
            indices = [v.index for v in body.data.vertices if any(g.group == group for g in v.groups)]
            inner_diameter = p["top_outer_diameter"]-2*p["wall_thickness"]
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.resolution_x = scene.render.resolution_y = settings["resolution"]
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.film_transparent = False
        scene.world = bpy.data.worlds.new("Black_world")
        scene.world.use_nodes = True
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.data.lens = settings["lens_mm"]
        scene.camera = camera
        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.object
        folder = output / p["name"]
        folder.mkdir(parents=True, exist_ok=True)
        conditions = itertools.product(settings["camera_tilts"], settings["camera_azimuths"],
                                       settings["light_tilts"], settings["light_azimuths"])
        for tilt, azimuth, light_tilt, light_azimuth in conditions:
            condition_id = f"c{tilt:02d}_{azimuth:03d}_l{light_tilt:02d}_{light_azimuth:03d}"
            camera.location = spherical_direction(tilt, azimuth)*inner_diameter*settings["distance_per_inner_diameter"]
            look_at(camera, Vector())
            lighting = {"tilt_deg": light_tilt, "azimuth_deg": light_azimuth,
                        "energy": settings["sun_energy"], "angle_deg": settings["sun_angle_deg"]}
            configure_sun(sun, lighting, camera.location, Vector(), "object_spherical")
            points = project_target_points(scene, camera, body, indices)
            if not all(0 <= x < settings["resolution"] and 0 <= y < settings["resolution"] for x, y in points):
                raise ValueError("正解内周が画面外です")
            image_path = folder / f"{condition_id}.png"
            scene.render.filepath = str(image_path)
            bpy.ops.render.render(write_still=True)
            samples.append({"model": p["name"], "condition_id": condition_id,
                            "image": image_path.relative_to(output).as_posix(),
                            "image_points": points, "camera_tilt": tilt,
                            "camera_azimuth": azimuth, "light_tilt": light_tilt,
                            "light_azimuth": light_azimuth})
        print(f"Finished {p['name']}: 48 conditions", flush=True)
    manifest = {"settings": settings, "models": models, "reference_config": original,
                "samples": samples}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
