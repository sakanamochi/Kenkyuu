"""上端環を固定し、内部の棒・小環を独立に乱数化した合成画像を描画する。"""

import argparse
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.paf_generalization.render_materials import read,write,digest
from research.paf_generalization.render_structure import profile_geometry


def interior_plan(parameters,rng):
    """特定機体の部品配置を使わず、開口の下の紛らわしい輪郭を作る。"""
    p=parameters
    diameter=p['top_outer_diameter']
    height=p['height']
    slope=(p['bottom_outer_diameter']-diameter)/(2*height)
    offset=p['wall_thickness']*math.sqrt(1+slope*slope)
    components=[]
    for i in range(int(rng.integers(3,9))):
        depth=float(rng.uniform(.06*diameter,min(.8*height,1.2*diameter)))
        radius=diameter/2+slope*depth-offset
        components.append({'kind':'bar','depth':depth,'cavity_radius':radius,
            'angle':float(rng.uniform(0,2*math.pi)),'offset':float(rng.uniform(-.7,.7)*radius),
            'thickness':float(rng.uniform(.015,.07)*diameter),
            'brightness':float(rng.uniform(.05,.95)),'metallic':float(rng.uniform(0,.8)),
            'roughness':float(rng.uniform(.2,.7))})
    for i in range(int(rng.integers(1,4))):
        depth=float(rng.uniform(.08*diameter,min(.8*height,1.2*diameter)))
        radius=diameter/2+slope*depth-offset
        components.append({'kind':'ring','depth':depth,'outer_radius':float(rng.uniform(.25,.95)*radius),
            'hole_ratio':float(rng.uniform(.55,.9)),'thickness':.015*diameter,
            'brightness':float(rng.uniform(.05,.95)),'metallic':float(rng.uniform(0,.8)),
            'roughness':float(rng.uniform(.2,.7))})
    return components


def build_interior(components,segments):
    import bpy
    from mathutils import Vector
    from parametric_paf.generate import mesh_object
    for index,c in enumerate(components):
        material=bpy.data.materials.new(f'Interior_material_{index}')
        material.use_nodes=True
        bsdf=material.node_tree.nodes['Principled BSDF']
        value=c['brightness']
        bsdf.inputs['Base Color'].default_value=(value,value,value,1)
        bsdf.inputs['Metallic'].default_value=c['metallic']
        bsdf.inputs['Roughness'].default_value=c['roughness']
        if c['kind']=='bar':
            a=c['angle']
            direction=Vector((math.cos(a),math.sin(a),0))
            center=Vector((-math.sin(a)*c['offset'],math.cos(a)*c['offset'],-c['depth']))
            length=2*math.sqrt(c['cavity_radius']**2-c['offset']**2)
            bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=c['thickness']/2,depth=length,location=center)
            obj=bpy.context.object
            obj.rotation_euler=direction.to_track_quat('Z','Y').to_euler()
            obj.data.materials.append(material)
        else:
            p={'top_outer_diameter':2*c['outer_radius'],'bottom_outer_diameter':2*c['outer_radius'],
               'hole_diameter':2*c['outer_radius']*c['hole_ratio'],'height':c['thickness'],
               'wall_thickness':c['thickness']}
            vertices,faces,_,_=profile_geometry(p,segments,'ring_only')
            obj=mesh_object(f'Interior_ring_{index}',vertices,faces,material)
            obj.location.z=-c['depth']


def render(split):
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import look_at,configure_sun,project_target_points
    from parametric_paf.random_dataset import build
    baseline=ROOT/'output/hole_region_comparison_v1/datasets/diversified'
    output=ROOT/'output/paf_generalization_v1/interior_counterfactual'
    config=read(ROOT/'parametric_paf/random_paf/config.json')
    manifest=read(baseline/'manifest.json')
    originals=[s for s in manifest['samples'] if s['split'] in ('train','validation') and s['conditions']['degradation']=='clean']
    plan=[{'base_sample_id':s['conditions']['base_sample_id'],'split':s['split'],
           'components':interior_plan(s['parameters'],np.random.default_rng(2026102504+i))}
          for i,s in enumerate(originals)]
    specification={'seed':2026102504,'renderer_sha256':digest(Path(__file__)),
        'geometry_helper_sha256':digest(Path(__file__).with_name('render_structure.py')),
        'baseline_manifest_sha256':digest(baseline/'manifest.json'),'plan':plan}
    if (output/'plan.json').exists():
        assert read(output/'plan.json')==specification
    else:
        write(output/'plan.json',specification)
    selected=[(s,item) for s,item in zip(originals,plan) if split=='all' or s['split']==split]
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
        bsdf=obj.data.materials[0].node_tree.nodes['Principled BSDF']
        value=conditions['brightness']
        bsdf.inputs['Base Color'].default_value=(value,value,value,1)
        bsdf.inputs['Metallic'].default_value=conditions['metallic']
        bsdf.inputs['Roughness'].default_value=conditions['roughness']
        build_interior(item['components'],config['segments'])
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
        write(record,{**label,'components':item['components'],'projection_max_error_px':error,
                      'image_sha256':digest(image)})
        print(f'INTERIOR {split} {index+1}/{len(selected)}',flush=True)
    # 存在する分割だけを記録し、未描画の画像を学習器へ渡さない。
    available={item['split'] for item in plan if all((output/f"images/{s['base_sample_id']}.png").exists()
               for s in plan if s['split']==item['split'])}
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
        'n_images':sum(1 for item in plan if item['split'] in available),'blender_version':bpy.app.version_string,
        'plan_sha256':digest(output/'plan.json'),'manifest_sha256':digest(output/'manifest.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['train','validation','all'],default='all')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    render(args.split)
