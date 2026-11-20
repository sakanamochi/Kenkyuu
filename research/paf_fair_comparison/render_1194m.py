"""1194Mのtrainだけに、採用多様形状モデルと同じ投影・外観の4ビューを用意する。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_generalization.render_materials import read, write, digest
from research.paf_generalization.render_framing import center_axes
from research.paf_generalization.render_surface import texture_material

OUT = ROOT/'output/paf_fair_comparison_v1/augmentation_1194m'
BASE = ROOT/'output/hole_region_comparison_v1/datasets/1194m'


def render(limit=None):
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import (import_and_center_model, assign_target_ring,
        apply_materials, look_at, configure_sun, project_target_points, spherical_direction)

    config = read(ROOT/'config/render_base.json')
    manifest = read(BASE/'manifest.json')
    originals = [s for s in manifest['samples'] if s['split'] == 'train'
                 and s['conditions']['degradation'] == 'clean']
    # 同じ順番の学習例へ同じ拡張パラメータを対応させ、分布の差を追加しない。
    source_plan = [p for p in read(ROOT/'output/paf_generalization_v2/projection_coverage/plan.json')['plan']
                   if p['split'] == 'train']
    textures = [p['texture'] for p in read(ROOT/'output/paf_generalization_v1/surface_counterfactual/plan.json')['plan']
                if p['split'] == 'train']
    assert len(originals) == len(source_plan) == len(textures) == 497
    plan = [{'base_sample_id': s['conditions']['base_sample_id'], 'split': 'train',
             'center_480': p['center_480'], 'major_axis_256': p['major_axis_256'],
             'target_axis_ratio': p['target_axis_ratio'], 'texture': t}
            for s, p, t in zip(originals, source_plan, textures)]
    dependencies = [Path(__file__), ROOT/'blender/generate_dataset.py', ROOT/'config/render_base.json',
                    ROOT/config['model_path'], BASE/'manifest.json',
                    ROOT/'research/paf_generalization/render_surface.py',
                    ROOT/'research/paf_generalization/render_framing.py',
                    ROOT/'output/paf_generalization_v2/projection_coverage/plan.json',
                    ROOT/'output/paf_generalization_v1/surface_counterfactual/plan.json']
    specification = {'files': {str(p.relative_to(ROOT)): digest(p) for p in dependencies},
        'training_only': True, 'texture_coordinates': 'top-plane origin, upper outer diameter = 1', 'plan': plan}
    if (OUT/'plan.json').exists():
        assert read(OUT/'plan.json') == specification
    else:
        write(OUT/'plan.json', specification)
    # CADの読込は一度だけ。形状・元材質・照明は各例の原データを再現する。
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objects = import_and_center_model(ROOT/config['model_path'], normalize_model_axes=True)
    obj, rim = assign_target_ring(objects, config['target_ring'])
    apply_materials(config['materials'])
    bpy.context.view_layer.update()
    rim_center = sum((obj.matrix_world @ obj.data.vertices[i].co for i in rim), Vector())/len(rim)
    radius = config['target_ring']['radius']
    z = config['target_ring']['z']
    upper_diameter = 2*max(np.hypot(v.co.x, v.co.y) for o in objects for v in o.data.vertices
                          if abs(v.co.z-z) < .001)
    materials = list({slot.material.name: slot.material for o in objects for slot in o.material_slots
                      if slot.material and slot.material.use_nodes}.values())
    originals_materials = {m.name: m for m in materials}
    surface_materials = {}
    for material in materials:
        surface_materials[material.name] = material.copy()
    scene = bpy.context.scene
    scene.render.engine = config['render']['engine']
    scene.render.resolution_x = scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.world = bpy.data.worlds.new('Black_background')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.object

    def set_surface(texture):
        # 材質スロットごとに同じ模様を適用し、元の金属度・粗さは維持する。
        for name, material in surface_materials.items():
            nodes = material.node_tree.nodes
            for node in list(nodes):
                if node.type not in ('OUTPUT_MATERIAL', 'BSDF_PRINCIPLED'):
                    nodes.remove(node)
            texture_material(material, texture)
            coordinates = next(n for n in nodes if n.type == 'TEX_COORD')
            mapping = next(n for n in nodes if n.type == 'VECT_MATH')
            subtract = nodes.new('ShaderNodeVectorMath')
            subtract.operation = 'SUBTRACT'
            subtract.inputs[1].default_value = (0, 0, z)
            scale = nodes.new('ShaderNodeVectorMath')
            scale.operation = 'SCALE'
            scale.inputs[3].default_value = 1/upper_diameter
            material.node_tree.links.new(coordinates.outputs['Object'], subtract.inputs[0])
            material.node_tree.links.new(subtract.outputs['Vector'], scale.inputs[0])
            material.node_tree.links.new(scale.outputs['Vector'], mapping.inputs[0])

    def style(surface):
        for o in objects:
            for slot in o.material_slots:
                name = slot.material.name
                original_name = next((k for k, v in surface_materials.items() if v.name == name), name)
                slot.material = surface_materials[original_name] if surface else originals_materials[original_name]

    for index, (sample, item) in enumerate(zip(originals, plan)):
        if limit is not None and index >= limit:
            break
        sid = item['base_sample_id']
        record = OUT/f'labels/{sid}.json'
        paths = {view: OUT/f'{view}/images/{sid}.png'
                 for view in ('original_surface', 'projection_plain', 'projection_surface')}
        if record.exists() and all(p.exists() for p in paths.values()):
            assert all(read(record)['image_hashes'][v] == digest(p) for v, p in paths.items())
            continue
        label = read(BASE/sample['label'])
        original_camera = label['conditions']['camera']
        camera.location = Vector(original_camera['location'])
        camera.data.lens = original_camera['lens_mm']
        camera.data.shift_x = camera.data.shift_y = 0
        look_at(camera, Vector(original_camera['target']))
        configure_sun(sun, label['conditions']['lighting'], camera.location,
                      Vector(original_camera['target']), 'object_spherical')
        points = np.asarray(project_target_points(scene, camera, obj, rim))
        original_error = float(np.max(np.abs(points-np.asarray(label['image_points']))))
        assert original_error < .01, original_error
        set_surface(item['texture'])
        style(True)
        paths['original_surface'].parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(paths['original_surface'])
        bpy.ops.render.render(write_still=True)
        # 円の中心を注視した透視投影で、指定された軸比を満たす視点を求める。
        distance = float((camera.location-rim_center).length)
        q, k = item['target_axis_ratio'], radius/distance
        tilt = float(np.rad2deg(np.arccos(q*np.sqrt((1-k*k)/(1-q*q*k*k)))))
        azimuth = float(original_camera.get('azimuth_deg', 0))
        camera.location = rim_center+spherical_direction(tilt, azimuth)*distance
        look_at(camera, rim_center)
        points = np.asarray(project_target_points(scene, camera, obj, rim))
        center, axes = center_axes(points)
        assert abs(min(axes)/max(axes)-q) < 1e-4
        target = np.asarray(item['center_480'])
        extent = np.max(np.abs(points-center), axis=0)
        factor = min(item['major_axis_256']*480/256/max(axes),
                     float(np.min(np.minimum(target-12, 468-target)/extent)))
        expected = (points-center)*factor+target
        camera.data.lens *= factor
        points = np.asarray(project_target_points(scene, camera, obj, rim))
        center, _ = center_axes(points)
        jacobian = []
        for attribute in ('shift_x', 'shift_y'):
            setattr(camera.data, attribute, .01)
            shifted = np.asarray(project_target_points(scene, camera, obj, rim))
            jacobian.append((shifted-points).mean(axis=0)/.01)
            setattr(camera.data, attribute, 0)
        shift = np.linalg.solve(np.stack(jacobian, axis=1), target-center)
        camera.data.shift_x, camera.data.shift_y = shift
        points = np.asarray(project_target_points(scene, camera, obj, rim))
        error = float(np.max(np.abs(points-expected)))
        assert error < .01 and float(points.min()) >= 11.99 and float(points.max()) <= 468.01
        for view in ('projection_plain', 'projection_surface'):
            style(view.endswith('surface'))
            paths[view].parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(paths[view])
            bpy.ops.render.render(write_still=True)
        center, axes = center_axes(points)
        write(record, {**label, 'image_points': points.tolist(), 'projection_plan': item,
            'camera_location': list(camera.location), 'camera_target': list(rim_center),
            'camera_lens_mm': camera.data.lens, 'camera_shift': shift.tolist(),
            'original_projection_error_px': original_error, 'projection_transform_max_error_px': error,
            'actual_major_axis_256': float(max(axes)*256/480), 'actual_axis_ratio': float(min(axes)/max(axes)),
            'texture_upper_diameter': float(upper_diameter),
            'image_hashes': {v: digest(p) for v, p in paths.items()}})
        print(f'FAIR_RENDER {index+1}/{len(plan)}', flush=True)
    if limit is not None:
        return
    for view in ('original_surface', 'projection_plain', 'projection_surface'):
        rows = []
        for sample in manifest['samples']:
            if sample['split'] != 'train':
                continue
            sid = sample['conditions']['base_sample_id']
            image = OUT/f'{view}/images/{sid}.png'
            label = (BASE/sample['label']).resolve() if view == 'original_surface' else OUT/f'labels/{sid}.json'
            rows.append({**sample, 'image': str(image), 'label': str(label),
                         'image_sha256': digest(image), 'label_sha256': digest(label)})
        write(OUT/f'{view}/manifest.json', {'samples': rows})
    write(OUT/'verification.json', {'n_originals': len(plan), 'n_images': 3*len(plan),
        'blender_version': bpy.app.version_string, 'plan_sha256': digest(OUT/'plan.json'),
        'manifest_hashes': {v: digest(OUT/f'{v}/manifest.json')
                           for v in ('original_surface', 'projection_plain', 'projection_surface')}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    render(args.limit)
