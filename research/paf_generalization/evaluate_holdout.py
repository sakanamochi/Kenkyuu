"""事前固定した未学習形状の対を、既存方式と候補方式で同じ条件のまま評価する。"""

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,load_samples
from paf_ring_detection.evaluate import _geometry
from paf_ring_detection.methods.cnn import load_model,TinyUNet
from paf_ring_detection.methods.cnn_region import region_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.context import ContextUNet
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_generalization.detector import PAFDetector
from research.paf_generalization.diagnose import BASE,OUT,sha,image_and_truth

DATA=OUT/'family_holdout_v1'


def verify():
    """全画像・教師・事前固定した候補実装を照合する。"""
    plan=read_json(DATA/'plan.json')
    check=read_json(DATA/'verification.json')
    assert check['plan_sha256']==sha(DATA/'plan.json')
    assert check['manifest_sha256']==sha(DATA/'manifest.json')
    for path,digest in plan['frozen_files'].items():
        assert sha(ROOT/path)==digest,path
    checked=set()
    for sample in read_json(DATA/'manifest.json')['samples']:
        assert sample['split']=='test'
        for field in ('image','label'):
            path=Path(sample[field])
            if path not in checked:
                assert sha(path)==sample[field+'_sha256']
                checked.add(path)
    protocol={'evaluator_sha256':sha(Path(__file__)),'manifest_sha256':sha(DATA/'manifest.json'),
        'success_iou':.8,'uniform_input_policy':'一様画像は全方式で情報不足として未検出にする',
        'model_selection':'元の多様形状validation損失最小。追加形状の結果では調整しない。',
        'geometry_config_sha256':sha(ROOT/'config/experiment.json')}
    if (DATA/'evaluation_protocol.json').exists():
        assert read_json(DATA/'evaluation_protocol.json')==protocol
    else:
        write_json(DATA/'evaluation_protocol.json',protocol)
    return protocol


def evaluate(methods):
    torch.set_num_threads(2)
    protocol=verify()
    config=read_json(ROOT/'config/experiment.json')
    cases=[(s,*image_and_truth(DATA,s)) for s in load_samples(DATA,'test')]
    jobs=[]
    if 'geometry' in methods:
        jobs.extend((name,None) for name in ('canny','zhang2019'))
    if 'baseline' in methods:
        jobs.extend(('cnn_diversified',s) for s in (20260907,20260908,20260909))
    if 'surface' in methods:
        jobs.extend(('surface_context_spatial',s) for s in (20260907,20260908,20260909))
        jobs.append(('surface_ensemble',None))
    if 'boundary' in methods:
        jobs.append(('boundary_spatial',20260907))
    for name,seed in jobs:
        model=None; ensemble=None; hashes=[]
        if name=='cnn_diversified':
            path=BASE/f'models/diversified/{seed}/best.pt'
            model=load_model(path,torch.device('cuda')); hashes=[sha(path)]
        elif name in ('surface_context_spatial','boundary_spatial'):
            folder=OUT/'runs'/name/str(seed); path=folder/'best.pt'
            assert read_json(folder/'complete.json')['checkpoint_sha256']==sha(path)
            cls=ContextUNet if name=='surface_context_spatial' else TinyUNet
            model=cls(16).cuda().eval()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state'])
            hashes=[sha(path)]
        elif name=='surface_ensemble':
            ensemble=PAFDetector()
            hashes=[c['sha256'] for c in ensemble.provenance['checkpoints']]
        metadata={**protocol,'method':name,'seed':seed,'checkpoint_sha256':hashes}
        destination=DATA/'results'/f'{name}_{seed if seed is not None else "fixed"}.json'
        if destination.exists():
            assert read_json(destination)['metadata']==metadata
            continue
        rows=[]
        for index,(sample,image,truth) in enumerate(cases):
            status='evaluated'
            if np.ptp(image)==0:
                ellipse=None; status='insufficient_image_evidence'
            elif ensemble is not None:
                prediction=ensemble.predict(image); ellipse=prediction['ellipse']; status=prediction['status']
            elif model is not None:
                x=torch.from_numpy(image.transpose(2,0,1).copy()).float().cuda()[None]/127.5-1
                with torch.inference_mode():
                    probability=torch.sigmoid(model(x))[0,0].cpu().numpy()
                ellipse=boundary_ellipse(probability) if name=='boundary_spatial' else region_ellipse(probability)
            else:
                # 対になった二形状は同じRANSAC乱数seedで比較する。
                identity=sample['group_id']+sample['conditions']['degradation']
                random_seed=(config['evaluation_seed']+zlib.crc32(identity.encode('utf-8')))%(2**32)
                ellipse=_geometry(image,name,config,random_seed)
            iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id':sample['sample_id'],'group_id':sample['group_id'],'family':sample['family'],
                'view':sample['view'],'degradation':sample['conditions']['degradation'],
                'severity':sample['conditions']['severity'],'ellipse':ellipse,'truth_ellipse':truth,
                'ellipse_iou':iou,'success':iou>=.8,'status':status})
            if (index+1)%96==0:
                print('HOLDOUT',name,seed,index+1,len(cases),flush=True)
        write_json(destination,{'metadata':metadata,'rows':rows})
        print('DONE',name,seed,sum(r['success'] for r in rows),len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=['geometry','baseline','surface','boundary'],
                        default=['geometry','baseline','surface','boundary'])
    evaluate(parser.parse_args().methods)
