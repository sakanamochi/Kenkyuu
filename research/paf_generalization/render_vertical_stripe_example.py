"""既存形状の縦縞を、元の撮影条件または向きを変更した条件で1枚描画する。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_generalization.render_materials import read, write, digest


def main(changed_view=False, thin_stripes=False, wide_light=False):
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import look_at, configure_sun, project_target_points, spherical_direction
    from parametric_paf.random_dataset import build

    sid = 'paf_0268'
    base = ROOT / 'output/hole_region_comparison_v1/datasets/diversified'
    out = ROOT / 'output/paf_vertical_stripe_example'
    if changed_view:
        out = out / 'changed_view'
    if thin_stripes:
        out = out / 'thin_stripes'
    if wide_light:
        out = out / 'wide_light'
    # cosの半角式で明色帯と濃淡遷移の角度幅を半分にする。
    ramp_positions = [((1 + p) / 2) ** .5 for p in (.70, .94)] if thin_stripes else [.70, .94]
    # 明色を基調とする指定では、細い帯だけを暗色にする。
    ramp_values = (.70, .055) if wide_light else (.055, .70)
    config = read(ROOT / 'parametric_paf/random_paf/config.json')
    sample = next(s for s in read(base / 'manifest.json')['samples']
                  if s['conditions']['base_sample_id'] == sid and s['conditions']['degradation'] == 'clean')
    label = read(base / sample['label'])
    conditions = dict(label['conditions'])
    if changed_view:
        # 距離・注視点・光量を保ち、カメラと光源の方向だけを変更する。
        conditions['camera_tilt'] += 12
        conditions['camera_azimuth'] = (conditions['camera_azimuth'] + 25) % 360
        conditions['light_tilt'] -= 20
        conditions['light_azimuth'] = (conditions['light_azimuth'] + 40) % 360
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    obj, rim, _ = build(sample['parameters'], config)
    plain = obj.data.materials[0]
    bsdf = plain.node_tree.nodes['Principled BSDF']
    value = conditions['brightness']
    bsdf.inputs['Base Color'].default_value = (value, value, value, 1)
    bsdf.inputs['Metallic'].default_value = conditions['metallic']
    bsdf.inputs['Roughness'].default_value = conditions['roughness']
    striped = plain.copy()
    striped.name = 'Vertical_stripes'
    obj.data.materials.append(striped)
    # 外側の側壁だけに模様を付け、上端環・内壁・底面には元材質を使う。
    for polygon in obj.data.polygons:
        polygon.material_index = 1 if polygon.index // config['segments'] == 1 else 0
    nodes, links = striped.node_tree.nodes, striped.node_tree.links
    coordinates = nodes.new('ShaderNodeTexCoord')
    xyz = nodes.new('ShaderNodeSeparateXYZ')
    links.new(coordinates.outputs['Object'], xyz.inputs[0])
    # 方位角にのみ依存させることで、円錐側面でも高さ方向に連続する縞になる。
    angle = nodes.new('ShaderNodeMath')
    angle.operation = 'ARCTAN2'
    links.new(xyz.outputs['Y'], angle.inputs[0])
    links.new(xyz.outputs['X'], angle.inputs[1])
    repeat = nodes.new('ShaderNodeMath')
    repeat.operation = 'MULTIPLY'
    repeat.inputs[1].default_value = 32
    links.new(angle.outputs[0], repeat.inputs[0])
    wave = nodes.new('ShaderNodeMath')
    wave.operation = 'COSINE'
    links.new(repeat.outputs[0], wave.inputs[0])
    ramp = nodes.new('ShaderNodeValToRGB')
    for element, position, value in zip(ramp.color_ramp.elements, ramp_positions, ramp_values):
        element.position = position
        element.color = (value, value, value, 1)
    links.new(wave.outputs[0], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], nodes['Principled BSDF'].inputs['Base Color'])

    # 指定方向のカメラ・光源を設定し、既存例の解像度と露光を使う。
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = scene.render.resolution_y = config['resolution']
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.world = bpy.data.worlds.new('Black_background')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = config['camera']['lens_mm']
    camera.location = Vector(label['camera_location'])
    if changed_view:
        camera.location = spherical_direction(conditions['camera_tilt'], conditions['camera_azimuth']) * label['camera_distance']
    look_at(camera, Vector(label['camera_target']))
    scene.camera = camera
    bpy.context.view_layer.update()
    bpy.ops.object.light_add(type='SUN')
    configure_sun(bpy.context.object, {'tilt_deg': conditions['light_tilt'],
        'azimuth_deg': conditions['light_azimuth'], 'energy': conditions['energy'], 'angle_deg': .53},
        camera.location, Vector(), 'object_spherical')
    scene.view_settings.exposure = conditions['exposure']
    points = project_target_points(scene, camera, obj, rim)
    error = float(np.max(np.abs(np.asarray(points)-np.asarray(label['image_points']))))
    if not changed_view:
        assert error < 1e-3
    assert np.isfinite(points).all()
    out.mkdir(parents=True, exist_ok=True)
    image = out / 'vertical_stripes.png'
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)
    write(out / 'example.json', {
        'base_sample_id': sid, 'split': sample['split'], 'original_label': str(base/sample['label']),
        'texture': {'kind': 'angular_vertical_stripes', 'count': 32, 'dark': .055, 'light': .70,
                    'cosine_ramp_positions': ramp_positions, 'ramp_values': ramp_values,
                    'surface': 'outer_wall', 'geometry_changed': False},
        'same_shape_camera_light': not changed_view,
        'conditions': conditions, 'camera_location': list(camera.location),
        'camera_target': label['camera_target'], 'image_points': points,
        'projection_max_error_px': error if not changed_view else None,
        'image_sha256': digest(image), 'renderer_sha256': digest(Path(__file__)),
        'blender_version': bpy.app.version_string,
        'scope': '表面模様の参考画像1枚。学習データや学習済みモデルには未追加。',
    })


if __name__ == '__main__':
    main(changed_view='--changed-view' in sys.argv, thin_stripes='--thin-stripes' in sys.argv,
         wide_light='--wide-light' in sys.argv)
