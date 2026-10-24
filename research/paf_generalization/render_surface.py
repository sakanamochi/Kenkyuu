"""形状と撮影を固定し、表面の模様だけを独立に変える対応付きCG。"""

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.paf_generalization.render_materials import read,write,digest


def texture_material(material,settings):
    """物体座標の模様を反射色へ接続し、陰影と投影形状を維持する。"""
    nodes=material.node_tree.nodes
    links=material.node_tree.links
    coordinates=nodes.new('ShaderNodeTexCoord')
    mapping=nodes.new('ShaderNodeVectorMath')
    mapping.operation='ADD'
    mapping.inputs[1].default_value=settings['offset']
    links.new(coordinates.outputs['Object'],mapping.inputs[0])
    if settings['kind']=='noise':
        texture=nodes.new('ShaderNodeTexNoise')
        texture.inputs['Detail'].default_value=2
    elif settings['kind']=='bands':
        texture=nodes.new('ShaderNodeTexWave')
        texture.wave_type='BANDS'
        texture.bands_direction=settings['axis']
        texture.inputs['Distortion'].default_value=settings['distortion']
    else:
        texture=nodes.new('ShaderNodeTexChecker')
    texture.inputs['Scale'].default_value=settings['scale']
    links.new(mapping.outputs['Vector'],texture.inputs['Vector'])
    ramp=nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position=.25
    ramp.color_ramp.elements[1].position=.75
    for element,value in zip(ramp.color_ramp.elements,(settings['dark'],settings['light'])):
        element.color=(value,value,value,1)
    links.new(texture.outputs['Fac'],ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'],nodes['Principled BSDF'].inputs['Base Color'])


def render(split):
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import look_at,configure_sun,project_target_points
    from parametric_paf.random_dataset import build
    baseline=ROOT/'output/hole_region_comparison_v1/datasets/diversified'
    output=ROOT/'output/paf_generalization_v1/surface_counterfactual'
    config=read(ROOT/'parametric_paf/random_paf/config.json')
    manifest=read(baseline/'manifest.json')
    originals=[s for s in manifest['samples'] if s['split'] in ('train','validation') and s['conditions']['degradation']=='clean']
    plan=[]
    for index,sample in enumerate(originals):
        rng=np.random.default_rng(2026102505+index)
        plan.append({'base_sample_id':sample['conditions']['base_sample_id'],'split':sample['split'],
            'texture':{'kind':str(rng.choice(['noise','bands','checker'])),
                       'axis':str(rng.choice(['X','Y','Z'])),'scale':float(rng.uniform(2,12)),
                       'distortion':float(rng.uniform(0,3)),'offset':rng.uniform(-5,5,3).tolist(),
                       'dark':float(rng.uniform(.01,.15)),'light':float(rng.uniform(.4,.95))}})
    specification={'seed':2026102505,'renderer_sha256':digest(Path(__file__)),
        'baseline_manifest_sha256':digest(baseline/'manifest.json'),'plan':plan}
    if (output/'plan.json').exists():
        assert read(output/'plan.json')==specification
    else:
        write(output/'plan.json',specification)
    selected=[(s,p) for s,p in zip(originals,plan) if split=='all' or s['split']==split]
    for index,(sample,item) in enumerate(selected):
        sid=item['base_sample_id']
        image=output/f'images/{sid}.png'
        record=output/f'labels/{sid}.json'
        label=read((baseline/sample['label']).resolve())
        if image.exists() and record.exists():
            assert read(record)['image_sha256']==digest(image)
            continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene=bpy.context.scene
        obj,rim,_=build(sample['parameters'],config)
        conditions=label['conditions']
        material=obj.data.materials[0]
        bsdf=material.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Metallic'].default_value=conditions['metallic']
        bsdf.inputs['Roughness'].default_value=conditions['roughness']
        texture_material(material,item['texture'])
        scene.render.engine='BLENDER_EEVEE'
        scene.render.resolution_x=scene.render.resolution_y=config['resolution']
        scene.render.resolution_percentage=100
        scene.render.image_settings.file_format='PNG'
        scene.render.image_settings.color_mode='RGB'
        scene.world=bpy.data.worlds.new('Black_background')
        scene.world.use_nodes=True
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
        bpy.ops.object.camera_add()
        camera=bpy.context.object
        camera.data.lens=config['camera']['lens_mm']
        camera.location=Vector(label['camera_location'])
        look_at(camera,Vector(label['camera_target']))
        scene.camera=camera
        bpy.context.view_layer.update()
        bpy.ops.object.light_add(type='SUN')
        configure_sun(bpy.context.object,{'tilt_deg':conditions['light_tilt'],
            'azimuth_deg':conditions['light_azimuth'],'energy':conditions['energy'],'angle_deg':.53},
            camera.location,Vector(),'object_spherical')
        scene.view_settings.exposure=conditions['exposure']
        points=project_target_points(scene,camera,obj,rim)
        error=float(np.max(np.abs(np.asarray(points)-np.asarray(label['image_points']))))
        assert error<1e-3,(sid,error)
        image.parent.mkdir(parents=True,exist_ok=True)
        scene.render.filepath=str(image)
        bpy.ops.render.render(write_still=True)
        write(record,{**label,'texture':item['texture'],'projection_max_error_px':error,'image_sha256':digest(image)})
        print(f'SURFACE {split} {index+1}/{len(selected)}',flush=True)
    available={p['split'] for p in plan if all((output/f"images/{s['base_sample_id']}.png").exists()
               for s in plan if s['split']==p['split'])}
    rows=[]
    for sample in manifest['samples']:
        if sample['split'] not in available:
            continue
        sid=sample['conditions']['base_sample_id']
        image=output/f'images/{sid}.png'
        label=(baseline/sample['label']).resolve()
        rows.append({**sample,'image':str(image),'label':str(label),
                     'image_sha256':digest(image),'label_sha256':digest(label)})
    write(output/'manifest.json',{'samples':rows})
    write(output/'verification.json',{'available_splits':sorted(available),'projection_tolerance_px':1e-3,
        'n_images':sum(1 for p in plan if p['split'] in available),'blender_version':bpy.app.version_string,
        'plan_sha256':digest(output/'plan.json'),'manifest_sha256':digest(output/'manifest.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['train','validation','all'],default='all')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    render(args.split)
