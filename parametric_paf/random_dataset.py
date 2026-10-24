"""上端外径D基準のランダムPAF画像・正解・寸法記録を生成する。"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parametric_paf.paf_model import make_geometry

CONFIG = ROOT / 'parametric_paf/random_paf/config.json'
OUTPUT = ROOT / 'parametric_paf/output/paf_random_outer_d_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(config, ratios):
    """上端外径Dから実寸法を復元し、穴径と肉厚を独立に指定する。"""
    diameter = config['top_outer_diameter']
    return {'top_outer_diameter': diameter,
            'bottom_outer_diameter': diameter * ratios['bottom_outer_diameter_ratio'],
            'height': diameter * ratios['height_ratio'],
            'hole_diameter': diameter * ratios['hole_diameter_ratio'],
            'wall_thickness': diameter * config['wall_thickness_ratio']}


def training_framing(samples, config, experiment):
    """寸法分割のtestを固定し、独立乱数でtrain/validationの穴サイズだけを指定する。"""
    bounds = config['camera'].get('training_hole_long_axis_fraction')
    if bounds is None:
        return
    if not 0 < bounds[0] <= bounds[1] < 1:
        raise ValueError('穴の画面内長径比は0より大きく1未満で指定してください')
    groups = sorted(s['sample_id'] for s in samples)
    random.Random(experiment['sources']['diversified']['split_seed']).shuffle(groups)
    end = int(len(groups) * experiment['split']['train_fraction'])
    end += round(len(groups) * experiment['split']['validation_fraction'])
    eligible = set(groups[:end])
    for sample in samples:
        if sample['sample_id'] in eligible:
            rng = random.Random(f"{config['camera']['framing_seed']}:{sample['sample_id']}")
            sample['conditions']['hole_long_axis_fraction'] = rng.uniform(*bounds)


def plan():
    """画像ごとに寸法を独立一様抽選し、重複しない固定データを作る。"""
    config = read(CONFIG)
    rng = random.Random(config['seed'])
    samples, identities = [], set()
    while len(samples) < config['image_count']:
        ratios = {key: round(rng.uniform(*config[key]), 6) for key in
                  ('bottom_outer_diameter_ratio', 'height_ratio', 'hole_diameter_ratio')}
        identity = tuple(ratios.values())
        if identity in identities:
            continue
        identities.add(identity)
        appearance = config['appearance']
        conditions = {'camera_tilt': rng.uniform(*config['camera']['tilt_deg']),
                      'camera_azimuth': rng.uniform(*config['camera']['azimuth_deg']),
                      'brightness': rng.uniform(*appearance['brightness']),
                      'metallic': rng.uniform(*appearance['metallic']),
                      'roughness': rng.uniform(*appearance['roughness']),
                      'light_tilt': rng.uniform(*appearance['light_tilt_deg']),
                      'light_azimuth': rng.uniform(0, 360), 'energy': rng.uniform(*appearance['energy']),
                      'exposure': rng.uniform(*appearance['exposure'])}
        samples.append({'sample_id': f'paf_{len(samples):04d}', 'ratios': ratios,
                        'parameters': parameters(config, ratios), 'conditions': conditions})
    training_framing(samples, config, read(ROOT / 'config/experiment.json'))
    value = {'config_sha256': digest(CONFIG), 'samples': samples}
    destination = OUTPUT / 'plan.json'
    if destination.exists():
        previous = read(destination)['samples']
        # 撮影サイズ以外の寸法・外観・視点は、保存済みの計画と同一に保つ。
        def identity(sample):
            return {**sample, 'conditions': {k: v for k, v in sample['conditions'].items()
                                            if k != 'hole_long_axis_fraction'}}
        if [identity(s) for s in previous] != [identity(s) for s in samples]:
            raise ValueError('既存の寸法・撮影計画と異なります')
    write(destination, value)
    return value


def build(parameters_value, config):
    """同じ回転断面からメッシュを作り、上端の穴の縁に頂点グループを付ける。"""
    import bpy
    from parametric_paf.generate import mesh_object
    vertices, faces, rim_indices, profile = make_geometry(parameters_value, config['segments'])
    material = bpy.data.materials.new('PAF_material')
    material.use_nodes = True
    obj = mesh_object('PAF', vertices, faces, material)
    obj.vertex_groups.new(name='target_inner_rim').add(rim_indices, 1.0, 'REPLACE')
    return obj, rim_indices, profile


def render():
    """testの全体構図を保ち、学習側は穴サイズを指定してPNGと正解を描画する。"""
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    from blender.generate_dataset import look_at, spherical_direction, configure_sun, project_target_points
    config = read(CONFIG)
    planned = plan()
    # 寸法範囲の端でも閉じた正体積メッシュになることを描画前に確認する。
    for ratios_tuple in itertools.product(*[config[key] for key in
            ('bottom_outer_diameter_ratio', 'height_ratio', 'hole_diameter_ratio')]):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        values = dict(zip(('bottom_outer_diameter_ratio', 'height_ratio', 'hole_diameter_ratio'), ratios_tuple))
        build(parameters(config, values), config)
    records = []
    for index, sample in enumerate(planned['samples']):
        sid = sample['sample_id']
        image = OUTPUT / f'images/{sid}.png'
        label_path = OUTPUT / f'labels/{sid}.json'
        cached = image.exists() and label_path.exists()
        if cached:
            label = read(label_path)
            cached = label['parameters'] == sample['parameters'] and label['conditions'] == sample['conditions']
        if not cached:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene = bpy.context.scene
            obj, rim_indices, profile = build(sample['parameters'], config)
            scene.render.engine = 'BLENDER_EEVEE'
            scene.render.resolution_x = scene.render.resolution_y = config['resolution']
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = 'PNG'
            scene.render.image_settings.color_mode = 'RGB'
            scene.render.film_transparent = False
            scene.world = bpy.data.worlds.new('Black_background')
            scene.world.use_nodes = True
            scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
            conditions = sample['conditions']
            bsdf = obj.data.materials[0].node_tree.nodes['Principled BSDF']
            value = conditions['brightness']
            bsdf.inputs['Base Color'].default_value = (value, value, value, 1)
            bsdf.inputs['Metallic'].default_value = conditions['metallic']
            bsdf.inputs['Roughness'].default_value = conditions['roughness']
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.data.lens = config['camera']['lens_mm']
            scene.camera = camera
            height = sample['parameters']['height']
            center = Vector((0, 0, -height / 2))
            direction = spherical_direction(conditions['camera_tilt'], conditions['camera_azimuth'])
            extent = math.hypot(sample['parameters']['bottom_outer_diameter'] / 2, height / 2)
            distance = 2 * extent / math.tan(camera.data.angle / 2)
            margin = config['camera']['margin']
            for _ in range(24):
                camera.location = direction * distance
                look_at(camera, center)
                bpy.context.view_layer.update()
                projected = [world_to_camera_view(scene, camera, obj.matrix_world @ v.co) for v in obj.data.vertices]
                if all(p.z > 0 and margin <= p.x <= 1-margin and margin <= p.y <= 1-margin for p in projected):
                    break
                distance *= 1.12
            else:
                raise ValueError('モデル全体を画像内に配置できません')
            requested = conditions.get('hole_long_axis_fraction')
            if requested is not None:
                base_distance = distance
                rotation = camera.rotation_euler.copy()
                # 開口への方向とカメラ姿勢を固定し、距離だけで投影サイズを調節する。
                for _ in range(16):
                    points = project_target_points(scene, camera, obj, rim_indices)
                    span = max(max(p[axis] for p in points) - min(p[axis] for p in points)
                               for axis in (0, 1)) / config['resolution']
                    if abs(span / requested - 1) < 1e-5:
                        break
                    distance *= span / requested
                    camera.location = direction * distance
                    camera.rotation_euler = rotation
                # 穴の全周は画面内へ残し、外側の本体だけは画面外を許容する。
                for _ in range(40):
                    points = project_target_points(scene, camera, obj, rim_indices)
                    margin_px = margin * config['resolution']
                    if all(margin_px <= coordinate <= config['resolution'] - margin_px
                           for point in points for coordinate in point):
                        break
                    distance *= 1.02
                    camera.location = direction * distance
                else:
                    raise ValueError('穴の全周を画像内に配置できません')
                center *= distance / base_distance
            bpy.ops.object.light_add(type='SUN')
            configure_sun(bpy.context.object, {'tilt_deg': conditions['light_tilt'],
                          'azimuth_deg': conditions['light_azimuth'], 'energy': conditions['energy'],
                          'angle_deg': 0.53}, camera.location, Vector(), 'object_spherical')
            scene.view_settings.exposure = conditions['exposure']
            points = project_target_points(scene, camera, obj, rim_indices)
            label = {'sample_id': sid, 'image_width': config['resolution'], 'image_height': config['resolution'],
                     'image_points': points, 'parameters': sample['parameters'], 'ratios': sample['ratios'],
                     'conditions': conditions, 'camera_distance': distance, 'camera_location': list(camera.location),
                     'camera_target': list(center), 'meridian_profile': profile, 'target_present': True}
            image.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(image)
            bpy.ops.render.render(write_still=True)
            write(label_path, label)
            if index == 0:
                bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / 'sample_model.blend'))
        records.append({**sample, 'split': 'train', 'shape_id': sid,
                        'image': image.relative_to(OUTPUT).as_posix(), 'label': label_path.relative_to(OUTPUT).as_posix(),
                        'image_sha256': digest(image), 'label_sha256': digest(label_path)})
        if (index + 1) % 32 == 0:
            print(f'Dataset {index + 1}/{len(planned["samples"])}', flush=True)
    write(OUTPUT / 'manifest.json', {'dataset_id': config['dataset_id'], 'samples': records})
    write(OUTPUT / 'provenance.json', {'config_sha256': digest(CONFIG), 'plan_sha256': digest(OUTPUT / 'plan.json'),
          'experiment_sha256': digest(ROOT / 'config/experiment.json'),
          'generator_sha256': digest(Path(__file__)), 'model_sha256': digest(ROOT / 'parametric_paf/paf_model.py'),
          'blender': bpy.app.version_string, 'boundary_mesh_checks': 8, 'sample_count': len(records)})


def verify():
    """保存した全画像・寸法・正解・hashを検査し、実データから確認用一覧を作る。"""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    config = read(CONFIG)
    planned = read(OUTPUT / 'plan.json')
    manifest = read(OUTPUT / 'manifest.json')
    if len(manifest['samples']) != config['image_count']:
        raise ValueError('生成枚数が不一致')
    identities, sizes = set(), {'train_validation': [], 'test': []}
    for sample, expected in zip(manifest['samples'], planned['samples']):
        if (sample['sample_id'] != expected['sample_id'] or sample['parameters'] != expected['parameters']
                or sample['conditions'] != expected['conditions']):
            raise ValueError('寸法計画と生成データが不一致')
        ratios = sample['ratios']
        identity = tuple(ratios[key] for key in ('bottom_outer_diameter_ratio', 'height_ratio', 'hole_diameter_ratio'))
        identities.add(identity)
        for key in ('bottom_outer_diameter_ratio', 'height_ratio', 'hole_diameter_ratio'):
            if not config[key][0] <= ratios[key] <= config[key][1]:
                raise ValueError('寸法比が範囲外')
        for key in ('image', 'label'):
            if digest(OUTPUT / sample[key]) != sample[key + '_sha256']:
                raise ValueError('画像または正解のhash不一致')
        label = read(OUTPUT / sample['label'])
        if label['conditions'] != expected['conditions']:
            raise ValueError('ラベルの撮影条件が計画と不一致')
        points = np.asarray(label['image_points'])
        if not np.isfinite(points).all() or points.min() < 0 or points.max() >= config['resolution']:
            raise ValueError('正解輪郭が画面外')
        image = cv2.imdecode(np.fromfile(OUTPUT / sample['image'], np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (config['resolution'], config['resolution'], 3):
            raise ValueError('画像の読込または寸法が不正')
        values = parameters(config, ratios)
        if values != label['parameters'] or values != sample['parameters']:
            raise ValueError('ラベル寸法が不一致')
        make_geometry(values, config['segments'])
        # 実際の投影正解から、学習側と固定testのサイズ分布を分けて記録する。
        group = 'train_validation' if 'hole_long_axis_fraction' in sample['conditions'] else 'test'
        ellipse = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
        sizes[group].append(max(ellipse[1]) * 256 / config['resolution'])
    if len(identities) != config['image_count']:
        raise ValueError('寸法の組合せが重複')
    # 保存済みデータの先頭16枚を成否や明るさで選ばず、そのまま提示する。
    canvas = Image.new('RGB', (4*256, 4*294), 'white')
    for index, sample in enumerate(manifest['samples'][:16]):
        image = Image.open(OUTPUT / sample['image']).convert('RGB').resize((256, 256))
        x, y = index % 4 * 256, index // 4 * 294
        canvas.paste(image, (x, y))
        ratios = sample['ratios']
        draw = ImageDraw.Draw(canvas)
        draw.text((x+3, y+257), sample['sample_id'] + '  D=1', fill='black')
        draw.text((x+3, y+275), 'B={:.2f} H={:.2f} hole={:.2f}'.format(*[ratios[k] for k in
                  ('bottom_outer_diameter_ratio', 'height_ratio', 'hole_diameter_ratio')]), fill='black')
    canvas.save(OUTPUT / 'dataset_preview.png')
    write(OUTPUT / 'verification.json', {'status': 'passed', 'sample_count': len(manifest['samples']),
          'unique_dimensions': len(identities), 'manifest_sha256': digest(OUTPUT / 'manifest.json'),
          'projected_long_axis_px_at_256': {key: {'count': len(values),
              'min_median_max': np.quantile(values, [0, 0.5, 1]).tolist()}
              for key, values in sizes.items() if values},
          'checks': ['all_image_decodes', 'all_image_label_hashes', 'all_parameter_ranges',
                     'independent_hole_diameter', 'fixed_wall_thickness', 'all_labels_in_frame',
                     'all_planned_camera_conditions']})
    print(json.dumps(read(OUTPUT / 'verification.json'), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'render', 'verify'])
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    {'plan': plan, 'render': render, 'verify': verify}[args.stage]()


if __name__ == '__main__':
    main()
