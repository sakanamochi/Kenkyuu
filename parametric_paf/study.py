"""形状・外観2×2実験の計画固定、学習前検査、明示指定時の学習入口。"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT/'parametric_paf/study'
OUTPUT = ROOT/'parametric_paf/output/paf_shape_appearance_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape_parameters(ratios, name):
    """内径を1mにそろえ、全寸法を内径比から復元する。"""
    return {'name':name, 'top_outer_diameter':1+2*ratios['wall_ratio'],
            'bottom_outer_diameter':ratios['bottom_diameter_ratio'],
            'taper_height':ratios['height_ratio'], 'wall_thickness':ratios['wall_ratio'],
            'rib_count':0, 'color':[0.55,0.58,0.62]}


def draw_shape(rng, protocol):
    return {k:round(rng.uniform(*v),6) for k,v in protocol['shape_ranges'].items()}


def draw_condition(rng, protocol):
    """同じ画像番号で全armのカメラと外観乱数を対応させる。"""
    cam=protocol['camera']; basic=protocol['appearance_basic']; exp=protocol['appearance_expanded']
    u=[rng.random() for _ in range(16)]
    def mix(bounds,index): return bounds[0]+(bounds[1]-bounds[0])*u[index]
    color=[min(1,mix(exp['color_brightness'],6)*mix(exp['color_channel_multiplier'],7+i)) for i in range(3)]
    return {'camera_tilt':mix(cam['tilt_deg'],0), 'camera_azimuth':mix(cam['azimuth_deg'],1),
            'distance':mix(cam['distance_per_inner_diameter'],2),
            'appearance':{name:{'color':basic['color'] if name=='basic' else color,
              'metallic':basic['metallic'] if name=='basic' else mix(exp['metallic'],10),
              'roughness':basic['roughness'] if name=='basic' else mix(exp['roughness'],11),
              'light_tilt':mix(a['light_tilt_deg'],3), 'light_azimuth':mix(a['light_azimuth_deg'],4),
              'energy':mix(a['sun_energy'],5), 'angle':mix(a['sun_angle_deg'],12),
              'exposure':mix(a['exposure'],13)} for name,a in [('basic',basic),('expanded',exp)]}}


def plan():
    if (PLAN/'freeze.json').exists():
        raise ValueError('計画は凍結済みです。変更には新しい実験IDが必要です')
    protocol=read(PLAN/'protocol.json')
    rng=random.Random(protocol['shape_seed'])
    shapes=[]
    for split,count in [('train',protocol['train_shape_count']),('validation',protocol['validation_shape_count']),
                        ('test_interpolation',protocol['test_interpolation_shape_count']),
                        ('test_extrapolation',protocol['test_extrapolation_shape_count'])]:
        for i in range(count):
            ratios=protocol['single_shape_ratios'].copy() if split=='train' and i==0 else draw_shape(rng,protocol)
            if split=='test_extrapolation':
                axis=list(protocol['shape_ranges'])[i//4]
                intervals={'bottom_diameter_ratio':[1.28,2.05],'height_ratio':[0.10,0.52],'wall_ratio':[0.025,0.15]}
                ratios[axis]=intervals[axis][(i//2)%2]
                if axis=='bottom_diameter_ratio' and ratios[axis]==1.28:
                    ratios['wall_ratio']=min(ratios['wall_ratio'],0.1)
            sid=f'{split}_{i:02d}'
            shapes.append({'shape_id':sid,'split':split,'family':'tapered','ratios':ratios,
                           'parameters':shape_parameters(ratios,sid)})
    for family in protocol['geometry']['test_structure_families']:
        for i in range(protocol['test_structure_shapes_per_family']):
            ratios=draw_shape(rng,protocol); sid=f'test_structure_{family}_{i:02d}'
            shapes.append({'shape_id':sid,'split':'test_structure','family':family,'ratios':ratios,
                           'parameters':shape_parameters(ratios,sid)})
    render=random.Random(protocol['render_seed'])
    slots=[i%16 for i in range(protocol['train_images_per_arm'])]; render.shuffle(slots)
    training=[{'condition_id':f'train_{i:04d}','shape_slot':slot,**draw_condition(render,protocol)} for i,slot in enumerate(slots)]
    validation=[{'condition_id':f'validation_{i:03d}',**draw_condition(render,protocol)}
                for i in range(protocol['validation_views_per_shape'])]
    testing=random.Random(protocol['test_seed'])
    test=[{'condition_id':f'test_{i:03d}',**draw_condition(testing,protocol)} for i in range(protocol['test_views_per_shape'])]
    # 撮像条件外の補足も学習前に数値を固定し、後から都合のよい条件を選ばない。
    supplements=[]
    for i in range(16):
        c=draw_condition(testing,protocol)
        c['condition_id']=f'supplement_{i:03d}'
        c['camera_tilt']=protocol['test_supplements']['camera_tilt_deg'][i%2]
        c['appearance']['expanded']['exposure']=protocol['test_supplements']['exposure'][(i//2)%2]
        c['image_shift_fraction']=[protocol['test_supplements']['image_shift_fraction'][(i//4)%2],0]
        supplements.append(c)
    values={'shapes.json':shapes,'conditions.json':{'train':training,'validation':validation,'test':test,'supplements':supplements}}
    for name,value in values.items():
        path=PLAN/name
        if path.exists() and read(path)!=value:
            raise ValueError('固定済み計画が異なります。新しい実験IDが必要です')
        write(path,value)
    config=read(ROOT/'config/experiment.json')
    for arm in protocol['arms']:
        for seed in protocol['training_seeds']:
            c=json.loads(json.dumps(config)); c['experiment_id']=protocol['experiment_id']+'_'+arm
            c['seed']=seed; c['cnn'].update({k:v for k,v in protocol['training'].items() if k in c['cnn']})
            c['paths']['training_dataset']=(OUTPUT/arm).relative_to(ROOT).as_posix()
            c['paths']['checkpoint']=(OUTPUT/'models'/arm/str(seed)/'cnn_best.pt').relative_to(ROOT).as_posix()
            c['paths']['results']=(OUTPUT/'evaluation'/arm/str(seed)).relative_to(ROOT).as_posix()
            write(PLAN/'configs'/f'{arm}_{seed}.json',c)
    print('計画と12学習設定を作成。学習は開始していません。')


def check():
    """全ファイルと形状分離を確認し、学習には進まない。"""
    import cv2
    import numpy as np
    import torch
    from paf_ring_detection.data import RingDataset, label_ellipse
    from paf_ring_detection.methods.cnn import TinyUNet
    verify_freeze(required=False)
    protocol=read(PLAN/'protocol.json'); shapes=read(PLAN/'shapes.json')
    ids=[s['shape_id'] for s in shapes]
    if len(ids)!=len(set(ids)): raise ValueError('形状ID重複')
    fingerprints=[(s['family'],tuple(sorted(s['ratios'].items()))) for s in shapes]
    if len(fingerprints)!=len(set(fingerprints)): raise ValueError('同一形状が複数splitに存在します')
    shape_by_id={s['shape_id']:s for s in shapes}
    manifests={}
    for arm in protocol['arms']:
        folder=OUTPUT/arm; manifest=read(folder/'manifest.json'); manifests[arm]=manifest
        validation_count=protocol['validation_shape_count']*protocol['validation_views_per_shape']
        for split,expected in [('train',protocol['train_images_per_arm']),('validation',validation_count)]:
            samples=[s for s in manifest['samples'] if s['split']==split]
            if len(samples)!=expected: raise ValueError(f'{arm}/{split} 件数不一致')
            if len({s['sample_id'] for s in samples})!=expected: raise ValueError('サンプルID重複')
            for sample in samples:
                if shape_by_id[sample['shape_id']]['split']!=split: raise ValueError('形状split漏洩')
                image=cv2.imdecode(np.fromfile(folder/sample['image'],np.uint8),cv2.IMREAD_COLOR)
                if image is None or image.shape!=(480,480,3): raise ValueError('画像形式不正')
                if sha(folder/sample['image'])!=sample['image_sha256']: raise ValueError('画像hash不一致')
                if sha(folder/sample['label'])!=sample['label_sha256']: raise ValueError('ラベルhash不一致')
                label=read(folder/sample['label']); points=np.asarray(label['image_points'])
                if label['shape_id']!=sample['shape_id'] or label['conditions']!=sample['condition']:
                    raise ValueError('画像manifestとラベルの対応不正')
                if not np.isfinite(points).all() or points.min()<0 or points.max()>=480: raise ValueError('正解が画面外')
                ellipse=label_ellipse(label)
                if min(ellipse[1])<=0: raise ValueError('正解楕円不正')
            dataset=RingDataset(folder,split,256)
            for i in [0,len(dataset)//2,len(dataset)-1]:
                image,mask,_=dataset[i]
                if image.shape!=(3,256,256) or mask.shape!=(1,256,256): raise ValueError('学習入力形状不正')
                if not (0<=float(mask.min())<=float(mask.max())<=1): raise ValueError('教師範囲不正')
    # 同じ外観のarm間はカメラ・材質・照明を厳密に一致させる。
    for a,b in [('A','C'),('B','D')]:
        for x,y in zip(manifests[a]['samples'],manifests[b]['samples']):
            if x['condition']!=y['condition']: raise ValueError('対応条件不一致')
    for arm in 'BCD':
        for x,y in zip(manifests['A']['samples'],manifests[arm]['samples']):
            for key in ['condition_id','camera_tilt','camera_azimuth','distance']:
                if x['condition'][key]!=y['condition'][key]: raise ValueError('arm間のカメラ条件不一致')
            px=read(OUTPUT/'A'/x['label'])['image_points'];py=read(OUTPUT/arm/y['label'])['image_points']
            if not np.allclose(px,py,atol=1e-3,rtol=0): raise ValueError('arm間の投影内周寸法不一致')
    for arm in protocol['arms']:
        counts={}
        for s in manifests[arm]['samples']:
            if s['split']=='train': counts[s['shape_id']]=counts.get(s['shape_id'],0)+1
        if arm in 'CD' and (len(counts)!=16 or set(counts.values())!={64}): raise ValueError('多様化形状予算不均等')
        if arm in 'AB' and len(counts)!=1: raise ValueError('単一形状条件不正')
    settings=protocol['training']
    updates=(protocol['train_images_per_arm']//settings['batch_size'])*settings['epochs']
    if protocol['train_images_per_arm']%settings['batch_size'] or updates!=settings['updates_per_arm_seed']:
        raise ValueError('更新回数の比較予算不一致')
    for arm in protocol['arms']:
        for seed in protocol['training_seeds']:
            c=read(PLAN/'configs'/f'{arm}_{seed}.json')
            if any(c['cnn'][k]!=settings[k] for k in c['cnn']): raise ValueError('学習設定の不一致')
            if c['seed']!=seed or c['paths']['training_dataset']!=(OUTPUT/arm).relative_to(ROOT).as_posix():
                raise ValueError('学習設定のデータ対応不一致')
    selection=read(OUTPUT/'geometry_validation/selection.json')
    if selection['test_used'] or selection['manifest_sha256']!=sha(OUTPUT/'validation/manifest.json'):
        raise ValueError('幾何手法のvalidation根拠が不一致')
    # 勾配もoptimizerも作らず、本番batch形状で前向き計算だけを確認する。
    torch.set_num_threads(4);torch.manual_seed(protocol['training_seeds'][0])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TinyUNet(settings['base_channels']).to(device).eval()
    dataset=RingDataset(OUTPUT/'D','train',256)
    batch=torch.stack([dataset[i][0] for i in range(settings['batch_size'])]).to(device)
    with torch.inference_mode(): predicted=model(batch)
    if predicted.shape!=(settings['batch_size'],1,256,256) or not torch.isfinite(predicted).all():
        raise ValueError('CNN前向き計算不正')
    result={'status':'passed','training_started':any((OUTPUT/'models').rglob('*.pt')) if (OUTPUT/'models').exists() else False,
            'shape_count':len(shapes),'train_images_per_arm':1024,'validation_images_per_arm':192,
            'checks':['shape_identity_disjoint','image_and_label_hashes','all_image_decodes','all_labels_in_frame',
                      'paired_nuisance_conditions','balanced_shape_budget','RingDataset_input_and_mask',
                      'identical_projected_inner_rim','unique_sample_ids','equal_update_budget',
                      'all_training_configs','geometry_validation_only','forward_batch_without_training'],
            'plan_hashes':{p.name:sha(p) for p in [PLAN/'protocol.json',PLAN/'shapes.json',PLAN/'conditions.json']},
            'manifest_hashes':{a:sha(OUTPUT/a/'manifest.json') for a in protocol['arms']}}
    write(OUTPUT/'readiness.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def verify_freeze(required=True):
    """凍結時の計画・設定・実装・manifestからの変更を検出する。"""
    path=PLAN/'freeze.json'
    if not path.exists():
        if required: raise ValueError('実験をfreezeしてから学習してください')
        return
    frozen=read(path)
    for relative,digest in frozen['sha256'].items():
        if not (ROOT/relative).exists() or sha(ROOT/relative)!=digest:
            raise ValueError(f'凍結後の変更を検出: {relative}')
    if runtime_versions()!=frozen['runtime_versions']:
        raise ValueError('凍結時と主要ライブラリ版が異なります')


def runtime_versions():
    """数値再現性に関わるPython・推論ライブラリの版を記録する。"""
    import cv2
    import numpy as np
    import torch
    return {'python':platform.python_version(),'opencv':cv2.__version__,
            'numpy':np.__version__,'torch':torch.__version__}


def freeze():
    if (PLAN/'freeze.json').exists():
        verify_freeze(); print('凍結内容は一致しています'); return
    result=check()
    if result['training_started']: raise ValueError('学習開始後の事前凍結はできません')
    files=list((ROOT/'paf_ring_detection').rglob('*.py'))+list((ROOT/'parametric_paf').glob('*.py'))
    files += [PLAN/'protocol.json',PLAN/'shapes.json',PLAN/'conditions.json']+list((PLAN/'configs').glob('*.json'))
    files += [OUTPUT/a/'manifest.json' for a in 'ABCD']+[OUTPUT/'validation/manifest.json',
              OUTPUT/'geometry_validation/selection.json',OUTPUT/'geometry_validation/selected_config.json']
    packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,encoding='utf-8')
    (OUTPUT/'environment.txt').write_text(packages,encoding='utf-8')
    write(PLAN/'freeze.json',{'experiment_id':read(PLAN/'protocol.json')['experiment_id'],
          'date':'2026-09-07','training_started':False,'final_test_evaluated':False,
          'runtime_versions':runtime_versions(),
          'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
          'environment_sha256':sha(OUTPUT/'environment.txt')})
    print('学習前の計画・設定・コード・入力を凍結しました。')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','check','freeze','train'])
    parser.add_argument('--arm',choices=list('ABCD')); parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.stage=='plan': plan()
    elif args.stage=='check': check()
    elif args.stage=='freeze': freeze()
    else:
        if args.arm is None or args.seed not in read(PLAN/'protocol.json')['training_seeds']:
            parser.error('trainには--armと計画済み--seedが必要です')
        verify_freeze()
        check()
        config=read(PLAN/'configs'/f'{args.arm}_{args.seed}.json')
        if (ROOT/config['paths']['checkpoint']).exists(): raise ValueError('既存重みは上書きしません')
        from paf_ring_detection.train import train
        train(config)


if __name__=='__main__': main()
