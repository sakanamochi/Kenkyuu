"""境界CNNの証拠を固定し、単純な楕円当てはめと既存RANSACを比較する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import load_samples,read_json,write_json
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.methods.ransac import _is_valid_ellipse,_point_distances
from research.paf_generalization.boundary import skeleton_points,boundary_ellipse
from research.paf_generalization.diagnose import BASE,OUT as OLD,sha,image_and_truth

OUT=ROOT/'output/paf_generalization_v2/fitting'
METHODS=('raw_direct','component_direct','skeleton_direct')


def direct_fit(points,shape,settings):
    """既存と同じ最低点数・楕円の有効範囲で、一度だけ最小二乗を解く。"""
    if len(points)<settings['min_inliers']:
        return None
    try:
        ellipse=cv2.fitEllipseDirect(np.asarray(points,np.float32).reshape(-1,1,2))
    except cv2.error:
        return None
    return ellipse if _is_valid_ellipse(ellipse,shape,settings) else None


def estimate(probability,method,settings):
    """最大連結成分は画素面積のみで選び、正解や対象サイズを参照しない。"""
    binary=(probability>=.5).astype(np.uint8)
    if method=='skeleton_direct':
        points,_=skeleton_points(probability)
    else:
        if method=='component_direct':
            count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
            if count<=1:
                return None
            binary=(labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))).astype(np.uint8)
        ys,xs=np.where(binary)
        points=np.stack([xs,ys],axis=1)
    return direct_fit(points,probability.shape,settings)


def load():
    folder=OLD/'runs/boundary_surface/20260907'
    checkpoint=folder/'best.pt'
    config=read_json(folder/'provenance.json')['config']
    assert read_json(folder/'complete.json')['checkpoint_sha256']==sha(checkpoint)
    model=TinyUNet(16).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'])
    protocol={'checkpoint_sha256':sha(checkpoint),'implementation_sha256':sha(Path(__file__)),
              'boundary_sha256':sha(ROOT/'research/paf_generalization/boundary.py'),
              'ransac_sha256':sha(ROOT/'paf_ring_detection/methods/ransac.py'),
              'methods':list(METHODS),'threshold':.5,'ransac':config['ransac'],
              'selection':'元および表面模様のsource validation平均成功率。testでは選び直さない。',
              'oracle':'正解境界から2px以内のCNN点だけの当てはめは診断専用で、採用候補に含めない。'}
    path=OUT/'protocol.json'
    if path.exists():
        assert read_json(path)==protocol
    else:
        write_json(path,protocol)
    return model,protocol


def evaluate(model,protocol,folder,split,name,methods,reference=None):
    samples=load_samples(folder,split); rows=[]; settings=protocol['ransac']
    for start in range(0,len(samples),8):
        batch=samples[start:start+8]
        cases=[image_and_truth(folder,s) for s in batch]
        x=torch.from_numpy(np.stack([im.transpose(2,0,1) for im,_ in cases]).copy()).float().cuda()/127.5-1
        with torch.inference_mode():
            probabilities=torch.sigmoid(model(x))[:,0].cpu().numpy()
        for sample,(image,truth),probability in zip(batch,cases,probabilities):
            points,weights=skeleton_points(probability)
            distances=_point_distances(points,truth) if len(points) else np.array([])
            near=distances<=2.
            ellipses={m:estimate(probability,m,settings) for m in methods}
            if reference is None:
                ellipses['ransac']=boundary_ellipse(probability,settings)
            else:
                old=reference[sample['sample_id']]['ellipse']
                ellipses['ransac']=(tuple(old[0]),tuple(old[1]),old[2]) if old else None
            ellipses['oracle_near_truth']=direct_fit(points[near],probability.shape,settings)
            for method,ellipse in ellipses.items():
                iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
                rows.append({'sample_id':sample['sample_id'],'group_id':sample['group_id'],
                    'degradation':sample['conditions']['degradation'],'severity':sample['conditions']['severity'],
                    'method':method,'ellipse':ellipse,'ellipse_iou':iou,'success':iou>=.8,
                    'cnn_boundary_points':len(points),'points_near_truth':int(near.sum()),
                    'boundary_precision':float(np.mean(near)) if len(near) else None})
        if (start+len(batch))%160==0:
            print(name,start+len(batch),len(samples),flush=True)
    summary=[]
    for method in list(methods)+['ransac','oracle_near_truth']:
        for effect in ('all','clean','black_rectangle','sensor_whiteout','sensor_black_crush'):
            selected=[r for r in rows if r['method']==method and (effect=='all' or r['degradation']==effect)]
            summary.append({'method':method,'degradation':effect,'success_rate':float(np.mean([r['success'] for r in selected])),
                            'n_images':len(selected)})
    write_json(OUT/f'{name}.json',{'protocol':protocol,'manifest_sha256':sha(folder/'manifest.json'),
                                  'summary':summary,'rows':rows})
    print(name,[(r['method'],round(100*r['success_rate'],2)) for r in summary if r['degradation']=='all'],flush=True)
    return summary


def main(stage):
    model,protocol=load()
    if stage=='validation':
        summaries=[]
        for name,folder in [('source_validation',BASE/'datasets/diversified'),
                            ('surface_validation',OLD/'surface_counterfactual')]:
            summaries.append(evaluate(model,protocol,folder,'validation',name,METHODS))
        scores={m:float(np.mean([next(r['success_rate'] for r in s if r['method']==m and r['degradation']=='all')
                                for s in summaries])) for m in METHODS+('ransac',)}
        selected=max(METHODS,key=lambda m:scores[m])
        write_json(OUT/'selection.json',{'scores':scores,'selected_simple_method':selected,
            'selection_inputs':[sha(OUT/f'{n}.json') for n in ('source_validation','surface_validation')],
            'interpretation':'単純方式の最良をsource validationで固定し、RANSACとの差をtestで確認する。'})
    else:
        selected=read_json(OUT/'selection.json')['selected_simple_method']
        for source in ('1194m','diversified'):
            saved=read_json(OLD/f'runs/boundary_surface/20260907/test_{source}.json')
            assert saved['checkpoint_sha256']==protocol['checkpoint_sha256']
            reference={r['sample_id']:r for r in saved['rows']}
            evaluate(model,protocol,BASE/'datasets'/source,'test',f'test_{source}',[selected],reference)


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['validation','test'])
    main(parser.parse_args().stage)
