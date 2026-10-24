"""元の像の拡大と同じ投影を、高解像度の輪郭を保ったカメラで描画する。"""

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.paf_generalization.render_materials import read,write,digest


def center_axes(points):
    """正規化した二次曲線から楕円中心と全軸長を復元する。"""
    import numpy as np
    xy=np.asarray(points)/480
    x,y=xy.T
    design=np.stack([x*x,x*y,y*y,x,y,np.ones_like(x)],axis=1)
    _,_,vectors=np.linalg.svd(design,full_matrices=False)
    a,b,c,d,e,f=vectors[-1]
    q=np.array([[a,b/2],[b/2,c]])
    linear=np.array([d,e])
    center=-.5*np.linalg.solve(q,linear)
    scale=-(center@q@center+linear@center+f)
    values=np.linalg.eigvalsh(q/scale)
    assert min(values)>0
    return center*480,2/np.sqrt(values)*480


def render():
    import bpy
    import numpy as np
    from mathutils import Vector
    from blender.generate_dataset import look_at,configure_sun,project_target_points
    from parametric_paf.random_dataset import build
    baseline=ROOT/'output/hole_region_comparison_v1/datasets/diversified'
    output=ROOT/'output/paf_generalization_v1/framing_counterfactual'
    config=read(ROOT/'parametric_paf/random_paf/config.json')
    samples=[s for s in read(baseline/'manifest.json')['samples']
             if s['split']=='validation' and s['conditions']['degradation']=='clean']
    specification={'renderer_sha256':digest(Path(__file__)),'n_shapes':len(samples),
        'scale':3.,'maximum_major_axis_256px':224,'baseline_manifest_sha256':digest(baseline/'manifest.json')}
    if (output/'plan.json').exists():
        assert read(output/'plan.json')==specification
    else:
        write(output/'plan.json',specification)
    rows=[]
    for index,sample in enumerate(samples):
        sid=sample['conditions']['base_sample_id']
        image=output/f'images/{sid}.png'
        record=output/f'labels/{sid}.json'
        original=read((baseline/sample['label']).resolve())
        center,axes=center_axes(original['image_points'])
        factor=min(3.,420/max(axes))
        if not (image.exists() and record.exists()):
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene=bpy.context.scene
            obj,rim,_=build(sample['parameters'],config)
            conditions=original['conditions']
            bsdf=obj.data.materials[0].node_tree.nodes['Principled BSDF']
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
            camera.data.lens=config['camera']['lens_mm']*factor
            camera.location=Vector(original['camera_location'])
            look_at(camera,Vector(original['camera_target']))
            scene.camera=camera
            bpy.context.view_layer.update()
            points=np.asarray(project_target_points(scene,camera,obj,rim))
            projected_center,_=center_axes(points)
            # レンズシフトの実際の投影変位を測り、拡大診断と同じ中心へ移す。
            jacobian=[]
            for attribute in ('shift_x','shift_y'):
                setattr(camera.data,attribute,.01)
                shifted=np.asarray(project_target_points(scene,camera,obj,rim))
                jacobian.append((shifted-points).mean(axis=0)/.01)
                setattr(camera.data,attribute,0)
            shift=np.linalg.solve(np.stack(jacobian,axis=1),np.array([240.,240.])-projected_center)
            camera.data.shift_x,camera.data.shift_y=shift
            points=np.asarray(project_target_points(scene,camera,obj,rim))
            expected=(np.asarray(original['image_points'])-center)*factor+240
            error=float(np.max(np.abs(points-expected)))
            assert error<.01,(sid,error)
            bpy.ops.object.light_add(type='SUN')
            configure_sun(bpy.context.object,{'tilt_deg':conditions['light_tilt'],
                'azimuth_deg':conditions['light_azimuth'],'energy':conditions['energy'],'angle_deg':.53},
                camera.location,Vector(),'object_spherical')
            scene.view_settings.exposure=conditions['exposure']
            image.parent.mkdir(parents=True,exist_ok=True)
            scene.render.filepath=str(image)
            bpy.ops.render.render(write_still=True)
            write(record,{**original,'image_points':points.tolist(),'framing_scale':factor,
                'camera_lens_mm':camera.data.lens,'camera_shift':shift.tolist(),
                'projection_transform_max_error_px':error,'image_sha256':digest(image)})
        assert read(record)['image_sha256']==digest(image)
        rows.append({**sample,'image':str(image),'label':str(record),
            'image_sha256':digest(image),'label_sha256':digest(record)})
        print(f'FRAMING {index+1}/{len(samples)}',flush=True)
    write(output/'manifest.json',{'samples':rows})
    write(output/'verification.json',{'n_images':len(rows),'projection_tolerance_px':.01,
        'plan_sha256':digest(output/'plan.json'),'manifest_sha256':digest(output/'manifest.json'),
        'blender_version':bpy.app.version_string})


if __name__=='__main__':
    render()
