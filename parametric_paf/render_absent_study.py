"""穴のない円盤63条件と黒画像1条件を、対象なしの補足テストとして描画する。"""

import sys
import argparse
from pathlib import Path

import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parametric_paf.study import PLAN,OUTPUT,read,write,sha
from blender.generate_dataset import spherical_direction,look_at,configure_sun


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa',action='store_true',help='開発用条件で黒画像・円盤を1枚ずつ確認する')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    protocol=read(PLAN/'protocol.json'); conditions=read(PLAN/'conditions.json')
    folder=OUTPUT/('development_absent' if args.qa else 'absent_test')
    if (folder/'manifest.json').exists(): raise ValueError('対象なしテストは生成済みです')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.engine='BLENDER_EEVEE'; scene.render.resolution_x=scene.render.resolution_y=480
    scene.render.resolution_percentage=100; scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'; scene.render.film_transparent=False
    scene.world=bpy.data.worlds.new('Black_world'); scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
    bpy.ops.object.camera_add(); camera=bpy.context.object; camera.data.lens=55; scene.camera=camera
    bpy.ops.object.light_add(type='SUN'); sun=bpy.context.object
    bpy.ops.mesh.primitive_cylinder_add(vertices=192,radius=0.65,depth=0.25,location=(0,0,-0.125))
    disk=bpy.context.object
    material=bpy.data.materials.new('disk_material'); material.use_nodes=True; disk.data.materials.append(material)
    bsdf=material.node_tree.nodes.get('Principled BSDF')
    rows=[]
    # テスト用乱数から固定済みの条件を使う。黒画像を重複させて分母を増やさない。
    selected=conditions['validation'][:1] if args.qa else (conditions['test']+conditions['supplements'])[:63]
    for i,c in enumerate([selected[0]]+selected):
        a=c['appearance']['expanded']; disk.hide_render=(i==0)
        camera.location=spherical_direction(c['camera_tilt'],c['camera_azimuth'])*c['distance'];look_at(camera,Vector())
        bsdf.inputs['Base Color'].default_value=(*a['color'],1);bsdf.inputs['Metallic'].default_value=a['metallic']
        bsdf.inputs['Roughness'].default_value=a['roughness'];scene.view_settings.exposure=a['exposure']
        configure_sun(sun,{'tilt_deg':a['light_tilt'],'azimuth_deg':a['light_azimuth'],'energy':a['energy'],'angle_deg':a['angle']},camera.location,Vector(),'object_spherical')
        sid=f'absent_{i:03d}';image=folder/'images'/f'{sid}.png';label=folder/'labels'/f'{sid}.json'
        image.parent.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
        condition={k:c[k] for k in ['condition_id','camera_tilt','camera_azimuth','distance']};condition['appearance']=a
        write(label,{'target_present':False,'image_width':480,'image_height':480,'image_points':[],
                     'kind':'black' if i==0 else 'solid_disk','conditions':condition})
        rows.append({'sample_id':sid,'shape_id':'black' if i==0 else 'solid_disk',
                     'split':'development_absent' if args.qa else 'absent_test',
                     'image':image.relative_to(folder).as_posix(),'label':label.relative_to(folder).as_posix(),
                     'image_sha256':sha(image),'label_sha256':sha(label),'condition':condition})
    write(folder/'manifest.json',{'samples':rows,'purpose':'対象なしの補足。形状数ではなく黒背景1種・円盤1形状。',
                                 'development_only':args.qa})
    write(folder/'provenance.json',{'code_sha256':sha(Path(__file__)),'conditions_sha256':sha(PLAN/'conditions.json'),
                                  'protocol_sha256':sha(PLAN/'protocol.json'),'blender':bpy.app.version_string,
                                  'development_only':args.qa})


if __name__=='__main__':main()
