"""固定した指標でCNNと幾何手法を評価する。学習や閾値調整は行わない。"""

import argparse
import csv
import json
from pathlib import Path
import sys
import zlib

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parametric_paf.study import PLAN,OUTPUT,read,write,sha
from parametric_paf.diagnosis_metrics import contour_metrics,probability_support
from parametric_paf.tune_geometry import run_geometry
from paf_ring_detection.data import read_image,label_ellipse,scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import load_model
from paf_ring_detection.methods.cnn_ransac import detect_from_probability


def evaluate_prediction(ellipse,truth,image_shape,settings):
    """対象ありの誤楕円と、対象なしの誤検出を別々に定義する。"""
    if truth is None:
        return {'target_present':False,'detected':ellipse is not None,'strict_success':None,
                'iou_success':None,'missed':None,'wrong_output':None,
                'absent_false_positive':ellipse is not None,'ellipse_iou':None,
                'contour_mean_px':None,'contour_p95_px':None,'contour_p95_ratio':None}
    metrics=evaluate_ellipses(ellipse,truth,image_shape) if ellipse else {'ellipse_iou':0.0}
    distances=contour_metrics(ellipse,truth)
    iou_success=bool(ellipse and metrics['ellipse_iou']>=settings['success_iou'])
    strict=bool(iou_success and distances['contour_p95_px']<=settings['contour_p95_px'])
    return {'target_present':True,'detected':ellipse is not None,'strict_success':strict,
            'iou_success':iou_success,'missed':ellipse is None,'wrong_output':bool(ellipse and not strict),
            'absent_false_positive':None,'ellipse_iou':metrics['ellipse_iou'],**distances}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=OUTPUT/'test')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if (args.output/'results.csv').exists(): raise ValueError('既存評価は上書きしません')
    protocol=read(PLAN/'protocol.json')
    config=read(OUTPUT/'geometry_validation/selected_config.json')
    manifest=read(args.dataset/'manifest.json')
    samples=manifest['samples'][:args.limit] if args.limit else manifest['samples']
    torch.set_num_threads(4); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_model(args.checkpoint,device); rows=[]
    for i,s in enumerate(samples):
        image=cv2.resize(read_image(args.dataset/s['image']),(256,256),interpolation=cv2.INTER_AREA)
        label=read(args.dataset/s['label'])
        truth=scaled_ellipse(label_ellipse(label),256/label['image_width']) if label.get('target_present',True) else None
        seed=(protocol['test_seed']+zlib.crc32(s['condition']['condition_id'].encode()))%2**32
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()
        with torch.inference_mode():
            probability=torch.sigmoid(model(torch.from_numpy(rgb).float().unsqueeze(0).to(device)/127.5-1))[0,0].cpu().numpy()
        support=probability_support(probability,truth)[2] if truth else {}
        for method in ['cnn','canny','zhang']:
            result=detect_from_probability(probability,config['ransac'],seed) if method=='cnn' else run_geometry(image,method,config,seed)
            ellipse=result['ellipse'] if result else None
            metrics=evaluate_prediction(ellipse,truth,image.shape,protocol['evaluation'])
            rows.append({'sample_id':s['sample_id'],'shape_id':s['shape_id'],'split':s['split'],
                'condition_id':s['condition']['condition_id'],'camera_tilt':s['condition']['camera_tilt'],
                'method':method,'ransac_seed':seed,**metrics,
                'estimated_ellipse':json.dumps(ellipse) if ellipse is not None else '',
                'probability_full_contour_coverage':support.get('full_contour_coverage') if method=='cnn' else None})
        if (i+1)%12==0: print(f'Evaluation {i+1}/{len(samples)}',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for split,shape,method in sorted({(r['split'],r['shape_id'],r['method']) for r in rows}):
        a=[r for r in rows if (r['split'],r['shape_id'],r['method'])==(split,shape,method)]
        present=[r for r in a if r['target_present']]; absent=[r for r in a if not r['target_present']]
        summaries.append({'split':split,'shape_id':shape,'method':method,'n_present':len(present),'n_absent':len(absent),
            'strict_successes':sum(r['strict_success'] for r in present),
            'iou_successes':sum(r['iou_success'] for r in present),
            'missed':sum(r['missed'] for r in present),'wrong_output':sum(r['wrong_output'] for r in present),
            'absent_false_positive':sum(r['absent_false_positive'] for r in absent)})
    write(args.output/'summary.json',summaries)
    write(args.output/'provenance.json',{'checkpoint_sha256':sha(args.checkpoint),
        'manifest_sha256':sha(args.dataset/'manifest.json'),'protocol_sha256':sha(PLAN/'protocol.json'),
        'selected_config_sha256':sha(OUTPUT/'geometry_validation/selected_config.json'),
        'code_sha256':sha(Path(__file__)),'sample_count':len(samples),'limited_preflight':args.limit is not None})


if __name__=='__main__':main()
