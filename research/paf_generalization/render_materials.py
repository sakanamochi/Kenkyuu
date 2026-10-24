"""形状・投影・照明を固定し、部位別材質だけを変える反実仮想CGを描画する。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(limit=None):
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import look_at, configure_sun, project_target_points
    from parametric_paf.random_dataset import build
    baseline=ROOT/'output/hole_region_comparison_v1/datasets/diversified'
    output=ROOT/'output/paf_generalization_v1/material_counterfactual'
    config=read(ROOT/'parametric_paf/random_paf/config.json')
    manifest=read(baseline/'manifest.json')
    # train/validationの同じ原画像だけを使う。testの形状・材質は参照しない。
    samples=[s for s in manifest['samples'] if s['split'] in ('train','validation') and s['conditions']['degradation']=='clean']
    plan=[]
    for index,sample in enumerate(samples):
        rng=np.random.default_rng(2026102503+index)
        materials=[]
        for name in ('top','outer_wall','outer_bottom','inner_bottom','inner_wall','underside','hole_wall'):
            materials.append({'name':name,'brightness':float(rng.uniform(.02,.95)),
                              'metallic':float(rng.uniform(0,.85)),'roughness':float(rng.uniform(.2,.65))})
        plan.append({'base_sample_id':sample['conditions']['base_sample_id'],'split':sample['split'],'materials':materials})
    specification={'seed':2026102503,'parts':7,'n_shapes':len(samples),
                   'baseline_manifest_sha256':digest(baseline/'manifest.json'),
                   'renderer_sha256':digest(Path(__file__)), 'plan':plan}
    if (output/'plan.json').exists():
        assert read(output/'plan.json')==specification
    else:
        write(output/'plan.json',specification)
    for index,(sample,item) in enumerate(zip(samples,plan)):
        if limit is not None and index>=limit:
            break
        sid=item['base_sample_id']
        image=output/f'images/{sid}.png'
        record=output/f'labels/{sid}.json'
        if image.exists() and record.exists():
            assert read(record)['materials']==item['materials']
            continue
        label=read((baseline/sample['label']).resolve())
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene=bpy.context.scene
        obj,rim_indices,_=build(sample['parameters'],config)
        obj.data.materials.clear()
        for settings in item['materials']:
            material=bpy.data.materials.new(settings['name'])
            material.use_nodes=True
            bsdf=material.node_tree.nodes['Principled BSDF']
            value=settings['brightness']
            bsdf.inputs['Base Color'].default_value=(value,value,value,1)
            bsdf.inputs['Metallic'].default_value=settings['metallic']
            bsdf.inputs['Roughness'].default_value=settings['roughness']
            obj.data.materials.append(material)
        assert len(obj.data.polygons)==7*config['segments']
        for polygon in obj.data.polygons:
            polygon.material_index=polygon.index//config['segments']
        scene.render.engine='BLENDER_EEVEE'
        scene.render.resolution_x=scene.render.resolution_y=config['resolution']
        scene.render.resolution_percentage=100
        scene.render.image_settings.file_format='PNG'
        scene.render.image_settings.color_mode='RGB'
        scene.render.film_transparent=False
        scene.world=bpy.data.worlds.new('Black_background')
        scene.world.use_nodes=True
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
        bpy.ops.object.camera_add()
        camera=bpy.context.object
        camera.data.lens=config['camera']['lens_mm']
        scene.camera=camera
        camera.location=Vector(label['camera_location'])
        look_at(camera,Vector(label['camera_target']))
        bpy.context.view_layer.update()
        conditions=label['conditions']
        bpy.ops.object.light_add(type='SUN')
        configure_sun(bpy.context.object,{'tilt_deg':conditions['light_tilt'],
            'azimuth_deg':conditions['light_azimuth'],'energy':conditions['energy'],'angle_deg':.53},
            camera.location,Vector(),'object_spherical')
        scene.view_settings.exposure=conditions['exposure']
        points=project_target_points(scene,camera,obj,rim_indices)
        maximum_error=float(np.max(np.abs(np.asarray(points)-np.asarray(label['image_points']))))
        assert maximum_error<1e-3, (sid,maximum_error)
        image.parent.mkdir(parents=True,exist_ok=True)
        scene.render.filepath=str(image)
        bpy.ops.render.render(write_still=True)
        write(record,{**label,'materials':item['materials'],'projection_max_error_px':maximum_error,
                      'image_sha256':digest(image),'original_image':str((baseline/sample['image']).resolve())})
        print(f'MATERIAL {index+1}/{len(samples)} {sid}',flush=True)
    if limit is not None:
        return
    rows=[]
    for sample in manifest['samples']:
        if sample['split'] not in ('train','validation'):
            continue
        sid=sample['conditions']['base_sample_id']
        image=output/f'images/{sid}.png'
        label=(baseline/sample['label']).resolve()
        rows.append({**sample,'image':str(image),'label':str(label),
                     'image_sha256':digest(image),'label_sha256':digest(label)})
    write(output/'manifest.json',{'source':'diversified_material_counterfactual',
        'source_manifest_sha256':digest(baseline/'manifest.json'),'samples':rows})
    write(output/'verification.json',{'n_shapes':len(plan),'n_train':sum(s['split']=='train' for s in rows),
        'n_validation':sum(s['split']=='validation' for s in rows),'projection_tolerance_px':1e-3,
        'manifest_sha256':digest(output/'manifest.json'),'plan_sha256':digest(output/'plan.json'),
        'blender_version':bpy.app.version_string})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    render(args.limit)
