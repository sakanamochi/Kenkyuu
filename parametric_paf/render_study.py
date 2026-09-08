"""固定済み2×2計画のCG・正解を生成する。CNN学習やテスト推論は行わない。"""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parametric_paf.generate import build_model, mesh_object, SEGMENTS
from parametric_paf.study import PLAN, OUTPUT, read, write, sha, shape_parameters
from blender.generate_dataset import spherical_direction, look_at, configure_sun, project_target_points


def build_shape(shape):
    """テーパを学習族とし、円筒・段付き断面は構造変化の補足族に限定する。"""
    p=shape['parameters']
    objects,ring,_=build_model(p)
    if shape['family']=='tapered': return objects,ring
    body=objects[0]; material=body.data.materials[0]
    bpy.data.objects.remove(body,do_unlink=True)
    top=p['top_outer_diameter']/2; bottom=p['bottom_outer_diameter']/2
    h=p['taper_height']; w=p['wall_thickness']
    if shape['family']=='cylindrical_flange':
        profile=[(top,0),(top,-h),(top-w,-h),(top-w,0)]
    elif shape['family']=='stepped':
        profile=[(top,0),(top,-h*.35),(bottom,-h*.35),(bottom,-h),
                 (bottom-w,-h),(bottom-w,-h*.35),(top-w,-h*.35),(top-w,0)]
    else: raise ValueError('未知の断面族')
    vertices=[(r*math.cos(2*math.pi*i/SEGMENTS),r*math.sin(2*math.pi*i/SEGMENTS),z)
              for r,z in profile for i in range(SEGMENTS)]
    faces=[(j*SEGMENTS+i,((j+1)%len(profile))*SEGMENTS+i,
            ((j+1)%len(profile))*SEGMENTS+(i+1)%SEGMENTS,j*SEGMENTS+(i+1)%SEGMENTS)
           for j in range(len(profile)) for i in range(SEGMENTS)]
    body=mesh_object('PAF_body',vertices,faces,material)
    group=body.vertex_groups.new(name='target_inner_rim')
    group.add(list(range((len(profile)-1)*SEGMENTS,len(profile)*SEGMENTS)),1,'REPLACE')
    return [body]+objects[1:],vertices[-SEGMENTS:]


def setup(shape,protocol):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    objects,ring=build_shape(shape)
    scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=scene.render.resolution_y=protocol['camera']['resolution']
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False
    scene.world=bpy.data.worlds.new('Black_world'); scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
    bpy.ops.object.camera_add(); camera=bpy.context.object
    camera.data.lens=protocol['camera']['lens_mm']; scene.camera=camera
    bpy.ops.object.light_add(type='SUN'); sun=bpy.context.object
    body=objects[0]; group=body.vertex_groups['target_inner_rim'].index
    indices=[v.index for v in body.data.vertices if any(g.group==group for g in v.groups)]
    return scene,camera,sun,objects,indices


def render_group(folder,shape,conditions,appearance,split,protocol):
    scene,camera,sun,objects,indices=setup(shape,protocol)
    results=[]
    for c in conditions:
        a=c['appearance'][appearance]
        # 内径1mでカメラ距離と下部寸法の比を明示する。
        camera.location=spherical_direction(c['camera_tilt'],c['camera_azimuth'])*c['distance']
        look_at(camera,Vector())
        camera.data.shift_x=c.get('image_shift_fraction',[0,0])[0]
        camera.data.shift_y=c.get('image_shift_fraction',[0,0])[1]
        configure_sun(sun,{'tilt_deg':a['light_tilt'],'azimuth_deg':a['light_azimuth'],
                          'energy':a['energy'],'angle_deg':a['angle']},camera.location,Vector(),'object_spherical')
        for obj in objects:
            bsdf=obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
            bsdf.inputs['Base Color'].default_value=(*a['color'],1)
            bsdf.inputs['Metallic'].default_value=a['metallic']; bsdf.inputs['Roughness'].default_value=a['roughness']
        scene.view_settings.exposure=a['exposure']
        points=project_target_points(scene,camera,objects[0],indices)
        if not all(0<=x<480 and 0<=y<480 for x,y in points): raise ValueError('内周が画面外')
        sid=shape['shape_id']+'__'+c['condition_id']
        image=folder/'images'/f'{sid}.png'; label=folder/'labels'/f'{sid}.json'
        condition={k:c[k] for k in ['condition_id','camera_tilt','camera_azimuth','distance']}
        condition['appearance']=a; condition['image_shift_fraction']=c.get('image_shift_fraction',[0,0])
        metadata={'image_width':480,'image_height':480,'image_points':points,'shape_id':shape['shape_id'],
                  'family':shape['family'],'parameters':shape['parameters'],'conditions':condition}
        if image.exists() and label.exists():
            if read(label)!=metadata: raise ValueError('既存画像の条件が異なります')
        else:
            image.parent.mkdir(parents=True,exist_ok=True)
            scene.render.filepath=str(image)
            bpy.ops.render.render(write_still=True)
            write(label,metadata)
        results.append({'sample_id':sid,'shape_id':shape['shape_id'],'split':split,
                        'image':image.relative_to(folder).as_posix(),'label':label.relative_to(folder).as_posix(),
                        'image_sha256':sha(image),'label_sha256':sha(label),'condition':condition})
    print(f'Completed {folder.name}/{shape["shape_id"]}: {len(results)}',flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['data','qa','test'],default='data')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    protocol=read(PLAN/'protocol.json'); shapes=read(PLAN/'shapes.json'); conditions=read(PLAN/'conditions.json')
    provenance={'plan_hashes':{p.name:sha(p) for p in [PLAN/'protocol.json',PLAN/'shapes.json',PLAN/'conditions.json']},
                'code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'parametric_paf/generate.py',ROOT/'blender/generate_dataset.py']},
                'blender':bpy.app.version_string,'stage':args.stage}
    provpath=OUTPUT/f'render_{args.stage}_provenance.json'
    if provpath.exists() and read(provpath)!=provenance: raise ValueError('描画条件が固定記録と不一致')
    write(provpath,provenance)
    if args.stage=='qa':
        # 最終テスト用IDや寸法を見ず、別の開発用形状でメッシュ・描画だけを確認する。
        rows=[]
        for family in protocol['geometry']['test_structure_families']:
            ratios={'bottom_diameter_ratio':1.6,'height_ratio':0.3,'wall_ratio':0.08}
            shape={'shape_id':'qa_'+family,'family':family,'parameters':shape_parameters(ratios,'qa_'+family)}
            rows.extend(render_group(OUTPUT/'qa',shape,conditions['validation'][:4],'basic','development',protocol))
        write(OUTPUT/'qa/manifest.json',{'samples':rows})
        return
    if args.stage=='test':
        rows=[]
        for s in shapes:
            if s['split'].startswith('test'):
                rows.extend(render_group(OUTPUT/'test',s,conditions['test'],'expanded',s['split'],protocol))
                rows.extend(render_group(OUTPUT/'test',s,conditions['supplements'],'expanded',s['split']+'_capture',protocol))
            elif s['split']=='train':
                rows.extend(render_group(OUTPUT/'test',s,conditions['test'],'expanded','known_shape_monitor',protocol))
        write(OUTPUT/'test/manifest.json',{'samples':rows,'purpose':'凍結済み独立評価。学習・調整に使用しない。'})
        return
    validation=[]
    for s in shapes:
        if s['split']=='validation':
            validation.extend(render_group(OUTPUT/'validation',s,conditions['validation'],'expanded','validation',protocol))
    write(OUTPUT/'validation/manifest.json',{'samples':validation})
    train_shapes=[s for s in shapes if s['split']=='train']
    for arm,settings in protocol['arms'].items():
        rows=[]
        for i,s in enumerate(train_shapes):
            selected=conditions['train'] if settings['shape']=='single' and i==0 else (
                [c for c in conditions['train'] if c['shape_slot']==i] if settings['shape']=='diverse' else [])
            if selected: rows.extend(render_group(OUTPUT/arm,s,selected,settings['appearance'],'train',protocol))
        rows.sort(key=lambda r:r['condition']['condition_id'])
        rows.extend([{**r,'image':'../validation/'+r['image'],'label':'../validation/'+r['label']} for r in validation])
        write(OUTPUT/arm/'manifest.json',{'experiment_id':protocol['experiment_id'],'arm':arm,'samples':rows})


if __name__=='__main__': main()
