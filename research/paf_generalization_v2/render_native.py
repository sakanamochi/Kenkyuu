"""内周の投影サイズと位置を本体寸法から独立に抽選し、対応するCGを描画する。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_generalization.render_materials import read, write, digest
from research.paf_generalization.render_framing import center_axes
from research.paf_generalization.render_surface import texture_material

OUTPUT = ROOT/'output/paf_generalization_v2/native_framing'
BASE = ROOT/'output/hole_region_comparison_v1/datasets/diversified'
SURFACE = ROOT/'output/paf_generalization_v1/surface_counterfactual'


def render():
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import look_at, configure_sun, project_target_points
    from parametric_paf.random_dataset import build
    config = read(ROOT/'parametric_paf/random_paf/config.json')
    manifest = read(BASE/'manifest.json')
    originals = [s for s in manifest['samples'] if s['split'] in ('train', 'validation')
                 and s['conditions']['degradation'] == 'clean']
    textures = {r['base_sample_id']: r['texture'] for r in read(SURFACE/'plan.json')['plan']}
    rng = np.random.default_rng(2026102511)
    plan = []
    for sample in originals:
        label = read(BASE/sample['label'])
        points = np.asarray(label['image_points'])
        center, axes = center_axes(points)
        target = rng.uniform(.3, .7, 2)*480
        major = float(np.exp(rng.uniform(np.log(24), np.log(180))))*480/256
        # 内周全体を12pxの余白内に保つ。機体全体の切り出しは許容する。
        extent = np.max(np.abs(points-center), axis=0)
        factor = min(major/max(axes), float(np.min(np.minimum(target-12, 468-target)/extent)))
        plan.append({'base_sample_id': sample['conditions']['base_sample_id'],
                     'split': sample['split'], 'scale': factor, 'center_480': target.tolist(),
                     'source_center_480': center.tolist(), 'major_axis_256': float(max(axes)*factor*256/480)})
    dependencies = [Path(__file__), ROOT/'research/paf_generalization/render_framing.py',
                    ROOT/'research/paf_generalization/render_surface.py',
                    ROOT/'parametric_paf/random_dataset.py', ROOT/'blender/generate_dataset.py',
                    ROOT/'parametric_paf/random_paf/config.json', BASE/'manifest.json', SURFACE/'plan.json']
    specification = {'seed': 2026102511, 'size_range_256': [24, 180], 'center_range': [.3, .7],
                     'source_only': True, 'pose_light_material_unchanged': True,
                     'files': {str(p.relative_to(ROOT)): digest(p) for p in dependencies}, 'plan': plan}
    if (OUTPUT/'plan.json').exists():
        assert read(OUTPUT/'plan.json') == specification
    else:
        write(OUTPUT/'plan.json', specification)
    for index, (sample, item) in enumerate(zip(originals, plan)):
        sid = item['base_sample_id']
        paths = {style: OUTPUT/f'native_{style}/images/{sid}.png' for style in ('plain', 'surface')}
        record = OUTPUT/f'labels/{sid}.json'
        if record.exists() and all(p.exists() for p in paths.values()):
            saved = read(record)
            assert all(saved['image_hashes'][style] == digest(p) for style, p in paths.items())
            continue
        original = read(BASE/sample['label'])
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        obj, rim, _ = build(sample['parameters'], config)
        conditions = original['conditions']
        material = obj.data.materials[0]
        bsdf = material.node_tree.nodes['Principled BSDF']
        value = conditions['brightness']
        bsdf.inputs['Base Color'].default_value = (value, value, value, 1)
        bsdf.inputs['Metallic'].default_value = conditions['metallic']
        bsdf.inputs['Roughness'].default_value = conditions['roughness']
        scene.render.engine = 'BLENDER_EEVEE'
        scene.render.resolution_x = scene.render.resolution_y = 480
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.world = bpy.data.worlds.new('Black_background')
        scene.world.use_nodes = True
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.data.lens = config['camera']['lens_mm']*item['scale']
        camera.location = Vector(original['camera_location'])
        look_at(camera, Vector(original['camera_target']))
        scene.camera = camera
        bpy.context.view_layer.update()
        points = np.asarray(project_target_points(scene, camera, obj, rim))
        projected_center, _ = center_axes(points)
        jacobian = []
        for attribute in ('shift_x', 'shift_y'):
            setattr(camera.data, attribute, .01)
            shifted = np.asarray(project_target_points(scene, camera, obj, rim))
            jacobian.append((shifted-points).mean(axis=0)/.01)
            setattr(camera.data, attribute, 0)
        shift = np.linalg.solve(np.stack(jacobian, axis=1), np.asarray(item['center_480'])-projected_center)
        camera.data.shift_x, camera.data.shift_y = shift
        points = np.asarray(project_target_points(scene, camera, obj, rim))
        expected = ((np.asarray(original['image_points'])-item['source_center_480'])*item['scale']
                    + item['center_480'])
        error = float(np.max(np.abs(points-expected)))
        assert error < .01, (sid, error)
        assert float(points.min()) >= 11.99 and float(points.max()) <= 468.01
        bpy.ops.object.light_add(type='SUN')
        configure_sun(bpy.context.object, {'tilt_deg': conditions['light_tilt'],
                      'azimuth_deg': conditions['light_azimuth'], 'energy': conditions['energy'], 'angle_deg': .53},
                      camera.location, Vector(), 'object_spherical')
        scene.view_settings.exposure = conditions['exposure']
        for style in ('plain', 'surface'):
            if style == 'surface':
                texture_material(material, textures[sid])
            paths[style].parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(paths[style])
            bpy.ops.render.render(write_still=True)
        write(record, {**original, 'image_points': points.tolist(), 'framing': item,
                       'camera_lens_mm': camera.data.lens, 'camera_shift': shift.tolist(),
                       'projection_transform_max_error_px': error,
                       'image_hashes': {style: digest(p) for style, p in paths.items()}})
        print(f'NATIVE {index+1}/{len(plan)}', flush=True)
    for style in ('plain', 'surface'):
        rows = []
        for sample in manifest['samples']:
            if sample['split'] not in ('train', 'validation'):
                continue
            sid = sample['conditions']['base_sample_id']
            image, label = paths[style].parent/f'{sid}.png', OUTPUT/f'labels/{sid}.json'
            rows.append({**sample, 'image': str(image), 'label': str(label),
                         'image_sha256': digest(image), 'label_sha256': digest(label)})
        write(OUTPUT/f'native_{style}/manifest.json', {'samples': rows})
    write(OUTPUT/'verification.json', {'n_shapes': len(plan), 'n_images': 2*len(plan),
          'projection_tolerance_px': .01, 'plan_sha256': digest(OUTPUT/'plan.json'),
          'manifest_hashes': {style: digest(OUTPUT/f'native_{style}/manifest.json') for style in ('plain', 'surface')},
          'blender_version': bpy.app.version_string})


if __name__ == '__main__':
    render()
