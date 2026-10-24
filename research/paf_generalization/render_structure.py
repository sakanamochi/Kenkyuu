"""内周・投影・材質を固定し、開口の下にある底と胴体への依存を調べるCG。"""

import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.paf_generalization.render_materials import read,write,digest


def profile_geometry(parameters,segments,variant):
    """上面の環を共有し、底なし胴体または独立した環の閉じた立体を作る。"""
    p=parameters
    inner,outer,bottom=p['hole_diameter']/2,p['top_outer_diameter']/2,p['bottom_outer_diameter']/2
    height,thickness=p['height'],p['wall_thickness']
    if variant=='open_body':
        slope=(bottom-outer)/height
        offset=thickness*math.sqrt(1+slope*slope)
        profile=[(inner,0),(outer,0),(bottom,-height),(bottom-offset,-height),
                 (outer+slope*thickness-offset,-thickness),(inner,-thickness)]
    elif variant=='ring_only':
        profile=[(inner,0),(outer,0),(outer,-thickness),(inner,-thickness)]
    else:
        raise ValueError(variant)
    vertices=[(radius*math.cos(2*math.pi*i/segments),radius*math.sin(2*math.pi*i/segments),z)
              for radius,z in profile for i in range(segments)]
    faces=[]
    for k in range(len(profile)):
        following=(k+1)%len(profile)
        for i in range(segments):
            j=(i+1)%segments
            faces.append((k*segments+i,following*segments+i,following*segments+j,k*segments+j))
    return vertices,faces,list(range(segments)),profile


def render():
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import look_at,configure_sun,project_target_points
    from parametric_paf.generate import mesh_object
    baseline=ROOT/'output/hole_region_comparison_v1/datasets/diversified'
    output=ROOT/'output/paf_generalization_v1/structure_counterfactual'
    config=read(ROOT/'parametric_paf/random_paf/config.json')
    samples=[s for s in read(baseline/'manifest.json')['samples']
             if s['split']=='validation' and s['conditions']['degradation']=='clean']
    specification={'variants':['open_body','ring_only'],'n_shapes':len(samples),
                   'renderer_sha256':digest(Path(__file__)),
                   'baseline_manifest_sha256':digest(baseline/'manifest.json'),
                   'purpose':'同じ上面内外周・投影・材質・照明を共有する構造介入。学習には使用しない。'}
    if (output/'plan.json').exists():
        assert read(output/'plan.json')==specification
    else:
        write(output/'plan.json',specification)
    for variant in specification['variants']:
        rows=[]
        for index,sample in enumerate(samples):
            sid=sample['conditions']['base_sample_id']
            image=output/variant/f'images/{sid}.png'
            record=output/variant/f'labels/{sid}.json'
            label_path=(baseline/sample['label']).resolve()
            label=read(label_path)
            if not (image.exists() and record.exists()):
                bpy.ops.wm.read_factory_settings(use_empty=True)
                scene=bpy.context.scene
                material=bpy.data.materials.new('Shared_material')
                material.use_nodes=True
                vertices,faces,rim,profile=profile_geometry(sample['parameters'],config['segments'],variant)
                obj=mesh_object('PAF',vertices,faces,material)
                conditions=label['conditions']
                bsdf=material.node_tree.nodes['Principled BSDF']
                value=conditions['brightness']
                bsdf.inputs['Base Color'].default_value=(value,value,value,1)
                bsdf.inputs['Metallic'].default_value=conditions['metallic']
                bsdf.inputs['Roughness'].default_value=conditions['roughness']
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
                write(record,{**label,'structure_variant':variant,'meridian_profile':profile,
                              'projection_max_error_px':error,'image_sha256':digest(image)})
            assert read(record)['image_sha256']==digest(image)
            rows.append({**sample,'image':str(image),'label':str(label_path),
                         'image_sha256':digest(image),'label_sha256':digest(label_path)})
            print(f'STRUCTURE {variant} {index+1}/{len(samples)}',flush=True)
        write(output/variant/'manifest.json',{'samples':rows})
    write(output/'verification.json',{'n_shapes':len(samples),'n_images':2*len(samples),
        'projection_tolerance_px':1e-3,'closed_mesh_verified':True,'blender_version':bpy.app.version_string,
        'plan_sha256':digest(output/'plan.json'),
        'manifests':{v:digest(output/v/'manifest.json') for v in specification['variants']}})


if __name__=='__main__':
    render()
