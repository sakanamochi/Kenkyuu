"""既存比較の正解・幾何学的可視性を検査し、柔らかい照明の補足画像を描画する。"""

import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from parametric_paf.generate import build_model, validate
from blender.generate_dataset import spherical_direction, look_at, project_target_points


def project_circle(scene, camera, radius, z):
    """正解以外の同心境界も、生成形状から独立に投影する。"""
    values = []
    for i in range(192):
        a = 2*math.pi*i/192
        p = world_to_camera_view(scene, camera, Vector((radius*math.cos(a), radius*math.sin(a), z)))
        values.append([p.x*scene.render.resolution_x, (1-p.y)*scene.render.resolution_y])
    return values


def main():
    source = HERE/'output/comparison'
    output = HERE/'output/diagnostic_geometry_v1'
    if (output/'manifest.json').exists():
        raise ValueError('既存の補足実験があります。上書きせず保存先を変更してください')
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    settings = manifest['settings']
    geometries, samples = [], []
    for p in manifest['models']:
        if p['name'] not in ('baseline', 'low_taper', 'thick_wall'):
            continue
        validate(p)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        objects, ring, _ = build_model(p)
        body = objects[0]
        group = body.vertex_groups['target_inner_rim'].index
        indices = [v.index for v in body.data.vertices if any(g.group==group for g in v.groups)]
        inner = p['top_outer_diameter']-2*p['wall_thickness']
        scene.render.engine = 'BLENDER_EEVEE'
        scene.render.resolution_x = scene.render.resolution_y = settings['resolution']
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.film_transparent = False
        scene.world = bpy.data.worlds.new('Black_world')
        scene.world.use_nodes = True
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.data.lens = settings['lens_mm']
        scene.camera = camera
        # 大きい面光源は影を緩和するための補足条件。宇宙照明の再現とはしない。
        bpy.ops.object.light_add(type='AREA')
        light = bpy.context.object
        light.data.shape = 'DISK'
        light.data.size = 4*inner
        light.data.energy = 800*inner**2
        light.location = spherical_direction(60, 0)*4*inner
        look_at(light, Vector())
        for tilt in settings['camera_tilts']:
            for azimuth in settings['camera_azimuths']:
                camera.location = spherical_direction(tilt, azimuth)*inner*settings['distance_per_inner_diameter']
                look_at(camera, Vector())
                points = project_target_points(scene, camera, body, indices)
                cid = f'c{tilt:02d}_{azimuth:03d}_l60_000'
                previous = next(s for s in manifest['samples'] if s['model']==p['name'] and s['condition_id']==cid)
                residual = max(math.dist(a, b) for a, b in zip(points, previous['image_points']))
                if residual > 1e-3:
                    raise ValueError(f'既存投影と不一致: {p["name"]} {cid} {residual}')
                depsgraph = bpy.context.evaluated_depsgraph_get()
                visibility = []
                # 頂点より十分手前までに遮蔽物があるかを判定し、自己交差を避ける。
                for xyz in ring:
                    delta = Vector(xyz)-camera.location
                    hit = scene.ray_cast(depsgraph, camera.location, delta.normalized(), distance=delta.length-1e-4*inner)[0]
                    visibility.append(not hit)
                boundaries = {
                    'top_outer': project_circle(scene, camera, p['top_outer_diameter']/2, 0),
                    'bottom_inner': project_circle(scene, camera, p['bottom_outer_diameter']/2-p['wall_thickness'], -p['taper_height']),
                    'bottom_outer': project_circle(scene, camera, p['bottom_outer_diameter']/2, -p['taper_height']),
                }
                geometries.append({'model':p['name'], 'camera_tilt':tilt, 'camera_azimuth':azimuth,
                                   'image_points':points, 'visible':visibility, 'boundaries':boundaries,
                                   'projection_max_residual_px':residual, 'parameters':p})
                path = output/p['name']/f'{cid}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
                samples.append({'model':p['name'], 'condition_id':cid, 'image':path.relative_to(output).as_posix(),
                                'image_points':points, 'camera_tilt':tilt, 'camera_azimuth':azimuth,
                                'light_tilt':60, 'light_azimuth':0, 'lighting':'area_soft'})
                print(f'Finished {p["name"]} {cid} visible={sum(visibility)}/{len(visibility)}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output/'geometry.json').write_text(json.dumps(geometries, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps({'settings':settings, 'samples':samples,
        'lighting':{'type':'AREA', 'size_per_inner':4, 'distance_per_inner':4,
                    'energy_per_inner_squared':800, 'tilt_deg':60, 'azimuth_deg':0},
        'purpose':'陰影を緩和した診断用。軌道照明の再現ではない。',
        'blender_version':bpy.app.version_string}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
