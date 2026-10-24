"""未学習の周辺構造と元の回転断面形状を、同じ上端環・撮影条件で描画する。"""

import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.paf_generalization.render_materials import read,write,digest
from research.paf_generalization.render_structure import profile_geometry
from research.paf_generalization.render_framing import center_axes

OUTPUT=ROOT/'output/paf_generalization_v1/family_holdout_v1'


def make_plan():
    """新しい寸法・撮影乱数と評価対象の実装を、推定を見る前に固定する。"""
    import numpy as np
    config=read(Path(__file__).with_name('holdout.json'))
    source=read(ROOT/'parametric_paf/random_paf/config.json')
    rng=np.random.default_rng(config['seed'])
    plans=[]
    for family in config['families']:
        for index in range(config['shapes_per_family']):
            params={'top_outer_diameter':1.,'bottom_outer_diameter':float(rng.uniform(1,3)),
                    'height':float(rng.uniform(.5,3)),'hole_diameter':float(rng.uniform(.4,.95)),'wall_thickness':.02}
            appearance={k:float(rng.uniform(*source['appearance'][k])) for k in
                        ('brightness','metallic','roughness','energy','exposure')}
            appearance.update(light_tilt=float(rng.uniform(*source['appearance']['light_tilt_deg'])),
                              light_azimuth=float(rng.uniform(0,360)))
            plans.append({'group_id':f'{family}_{index:03d}','family':family,'parameters':params,
                'appearance':appearance,'camera_tilt':float(rng.uniform(*config['camera_tilt_degrees'])),
                'camera_azimuth':float(rng.uniform(0,360)),'major_axis_256px':float(rng.uniform(*config['major_axis_256px'])),
                'center_fraction':rng.uniform(*config['center_fraction'],2).tolist(),
                'support_count':int(rng.integers(5,15)),'support_phase':float(rng.uniform(0,2*math.pi)),
                'effect_seed':int(rng.integers(0,2**31))})
    frozen=[Path(__file__),Path(__file__).with_name('holdout.json'),
            Path(__file__).with_name('surface.py'),Path(__file__).with_name('surface.json'),
            Path(__file__).with_name('boundary.py'),Path(__file__).with_name('boundary.json'),
            Path(__file__).with_name('experiment.py'),Path(__file__).with_name('experiment.json'),
            Path(__file__).with_name('context.py'),Path(__file__).with_name('detector.py'),
            Path(__file__).with_name('render_structure.py'),Path(__file__).with_name('render_framing.py'),
            ROOT/'config/experiment.json',ROOT/'parametric_paf/paf_model.py',ROOT/'parametric_paf/random_dataset.py']
    specification={'config':config,'source_render_config':source,'samples':plans,
                   'frozen_files':{str(p.relative_to(ROOT)):digest(p) for p in frozen}}
    if (OUTPUT/'plan.json').exists():
        assert read(OUTPUT/'plan.json')==specification
    else:
        write(OUTPUT/'plan.json',specification)
    return specification


def add_rod(name,start,end,radius,material):
    """同じ材質の円柱を二点間へ配置する。"""
    import bpy
    from mathutils import Vector
    a,b=Vector(start),Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=16,radius=radius,depth=(b-a).length,location=(a+b)/2)
    obj=bpy.context.object
    obj.name=name
    obj.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler()
    obj.data.materials.append(material)


def build_shape(item,control,source):
    """上端環を共有する、閉じた角形胴・開放支持・リブ付き胴の三系列。"""
    import bpy
    from parametric_paf.random_dataset import build
    from parametric_paf.paf_model import make_geometry
    from parametric_paf.generate import mesh_object
    p=item['parameters']; segments=source['segments']
    if control or item['family']=='ribbed_shell':
        obj,rim,_=build(p,source)
        material=obj.data.materials[0]
    else:
        material=bpy.data.materials.new('PAF_material'); material.use_nodes=True
        if item['family']=='square_shell':
            vertices,faces,rim,_=make_geometry(p,segments)
            transformed=[]
            for x,y,z in vertices:
                radius=math.hypot(x,y)
                square_factor=(math.sqrt(.5)*radius/max(abs(x),abs(y))) if radius else 1
                factor=1+(-z/p['height'])*(square_factor-1)
                transformed.append((x*factor,y*factor,z))
            obj=mesh_object('Square_shell',transformed,faces,material)
        else:
            vertices,faces,rim,_=profile_geometry(p,segments,'ring_only')
            obj=mesh_object('Top_ring',vertices,faces,material)
            bottom={**p,'top_outer_diameter':p['bottom_outer_diameter'],
                    'hole_diameter':p['bottom_outer_diameter']-.12}
            bv,bf,_,_=profile_geometry(bottom,segments,'ring_only')
            lower=mesh_object('Lower_support_ring',bv,bf,material)
            lower.location.z=-p['height']
    if not control and item['family'] in ('open_support','ribbed_shell'):
        for index in range(item['support_count']):
            theta=item['support_phase']+2*math.pi*index/item['support_count']
            upper=.49 if item['family']=='open_support' else .515
            lower=p['bottom_outer_diameter']/2-.03 if item['family']=='open_support' else p['bottom_outer_diameter']/2+.015
            add_rod(f'Support_{index}',(upper*math.cos(theta),upper*math.sin(theta),-.02),
                    (lower*math.cos(theta),lower*math.sin(theta),-p['height']),.025,material)
    return obj,rim,material


def render():
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import look_at,configure_sun,project_target_points,spherical_direction
    plan=make_plan(); config=plan['config']; source=plan['source_render_config']
    rows=[]
    for index,item in enumerate(plan['samples']):
        paired_points=None
        for view in ('control','new_family'):
            sid=item['group_id']+'_'+view
            image=OUTPUT/f'images/{sid}.png'; label_path=OUTPUT/f'labels/{sid}.json'
            if image.exists() and label_path.exists():
                label=read(label_path)
                assert label['image_sha256']==digest(image)
            else:
                bpy.ops.wm.read_factory_settings(use_empty=True)
                scene=bpy.context.scene
                obj,rim,material=build_shape(item,view=='control',source)
                a=item['appearance']; bsdf=material.node_tree.nodes['Principled BSDF']
                value=a['brightness']; bsdf.inputs['Base Color'].default_value=(value,value,value,1)
                bsdf.inputs['Metallic'].default_value=a['metallic']; bsdf.inputs['Roughness'].default_value=a['roughness']
                scene.render.engine='BLENDER_EEVEE'
                scene.render.resolution_x=scene.render.resolution_y=480; scene.render.resolution_percentage=100
                scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGB'
                scene.world=bpy.data.worlds.new('Black_background'); scene.world.use_nodes=True
                scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
                bpy.ops.object.camera_add(); camera=bpy.context.object; scene.camera=camera
                camera.data.lens=55
                distance=6*max(item['parameters']['height'],item['parameters']['bottom_outer_diameter'])
                camera.location=spherical_direction(item['camera_tilt'],item['camera_azimuth'])*distance
                look_at(camera,Vector()); bpy.context.view_layer.update()
                points=np.asarray(project_target_points(scene,camera,obj,rim)); _,axes=center_axes(points)
                camera.data.lens*=item['major_axis_256px']*480/256/max(axes)
                points=np.asarray(project_target_points(scene,camera,obj,rim)); center,_=center_axes(points)
                jacobian=[]
                for attribute in ('shift_x','shift_y'):
                    setattr(camera.data,attribute,.01)
                    shifted=np.asarray(project_target_points(scene,camera,obj,rim))
                    jacobian.append((shifted-points).mean(axis=0)/.01)
                    setattr(camera.data,attribute,0)
                shift=np.linalg.solve(np.stack(jacobian,axis=1),np.asarray(item['center_fraction'])*480-center)
                camera.data.shift_x,camera.data.shift_y=shift
                points=np.asarray(project_target_points(scene,camera,obj,rim))
                assert np.all(points>0) and np.all(points<480)
                bpy.ops.object.light_add(type='SUN')
                configure_sun(bpy.context.object,{'tilt_deg':a['light_tilt'],'azimuth_deg':a['light_azimuth'],
                    'energy':a['energy'],'angle_deg':.53},camera.location,Vector(),'object_spherical')
                scene.view_settings.exposure=a['exposure']
                image.parent.mkdir(parents=True,exist_ok=True); scene.render.filepath=str(image)
                bpy.ops.render.render(write_still=True)
                label={'sample_id':sid,'image_width':480,'image_height':480,'image_points':points.tolist(),
                    'parameters':item['parameters'],'family':item['family'],'view':view,'target_present':True,
                    'camera_location':list(camera.location),'camera_lens_mm':camera.data.lens,'camera_shift':shift.tolist(),
                    'image_sha256':digest(image),'group_id':item['group_id']}
                write(label_path,label)
            points=np.asarray(label['image_points'])
            if paired_points is not None:
                assert np.max(np.abs(points-paired_points))<.001
            paired_points=points
            for effect,severity in config['effects'].items():
                rows.append({'sample_id':sid+'__'+effect,'group_id':item['group_id'],'split':'test',
                    'image':str(image),'label':str(label_path),'image_sha256':digest(image),'label_sha256':digest(label_path),
                    'parameters':item['parameters'],'family':item['family'],'view':view,
                    'conditions':{'base_sample_id':sid,'degradation':effect,'severity':severity},
                    'augmentation':None if effect=='clean' else {'name':effect,'severity':severity,'seed':item['effect_seed']}})
        print(f'HOLDOUT {index+1}/{len(plan["samples"])}',flush=True)
    write(OUTPUT/'manifest.json',{'samples':rows,'plan_sha256':digest(OUTPUT/'plan.json')})
    write(OUTPUT/'verification.json',{'n_pairs':len(plan['samples']),'n_images':2*len(plan['samples']),
        'n_evaluations_per_model':len(rows),'paired_projection_tolerance_px':.001,
        'plan_sha256':digest(OUTPUT/'plan.json'),'manifest_sha256':digest(OUTPUT/'manifest.json'),
        'blender_version':bpy.app.version_string})


if __name__=='__main__':
    render()
