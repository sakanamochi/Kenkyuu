"""事前固定した境界・表面模様の組合せ候補を、追加形状集合へ一回適用する。"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,load_samples
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_generalization.evaluate_holdout import DATA,verify
from research.paf_generalization.diagnose import OUT,sha,image_and_truth


def evaluate():
    torch.set_num_threads(2)
    protocol=verify()
    additional=read_json(DATA/'additional_candidate_plan.json')
    assert additional['holdout_plan_sha256']==sha(DATA/'plan.json')
    for path,digest in additional['frozen_files'].items():
        assert sha(ROOT/path)==digest
    folder=OUT/'runs/boundary_surface/20260907'; path=folder/'best.pt'
    assert read_json(folder/'complete.json')['checkpoint_sha256']==sha(path)
    model=TinyUNet(16).cuda().eval()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state'])
    settings=read_json(Path(__file__).with_name('boundary_surface.json'))['ransac']
    metadata={**protocol,'method':'boundary_surface','seed':20260907,'checkpoint_sha256':[sha(path)],
        'additional_candidate_plan_sha256':sha(DATA/'additional_candidate_plan.json'),
        'additional_evaluator_sha256':sha(Path(__file__))}
    destination=DATA/'results/boundary_surface_20260907.json'
    if destination.exists():
        assert read_json(destination)['metadata']==metadata
        return
    rows=[]
    samples=load_samples(DATA,'test')
    for index,sample in enumerate(samples):
        image,truth=image_and_truth(DATA,sample)
        if np.ptp(image)==0:
            ellipse=None; status='insufficient_image_evidence'
        else:
            x=torch.from_numpy(image.transpose(2,0,1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                probability=torch.sigmoid(model(x))[0,0].cpu().numpy()
            ellipse=boundary_ellipse(probability,settings); status='evaluated'
        iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
        rows.append({'sample_id':sample['sample_id'],'group_id':sample['group_id'],'family':sample['family'],
            'view':sample['view'],'degradation':sample['conditions']['degradation'],
            'severity':sample['conditions']['severity'],'ellipse':ellipse,'truth_ellipse':truth,
            'ellipse_iou':iou,'success':iou>=.8,'status':status})
        if (index+1)%96==0:
            print('HOLDOUT boundary_surface',index+1,len(samples),flush=True)
    write_json(destination,{'metadata':metadata,'rows':rows})


if __name__=='__main__':
    evaluate()
