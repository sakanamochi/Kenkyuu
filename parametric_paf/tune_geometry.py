"""共通validationだけで幾何手法のCanny閾値を選び、その設定を固定する。"""

import copy
import csv
from pathlib import Path
import sys
import zlib

import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parametric_paf.study import PLAN,OUTPUT,read,write,sha
from parametric_paf.diagnosis_metrics import contour_metrics
from paf_ring_detection.data import read_image,label_ellipse,scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.canny_contour_ransac import detect_canny_contour_ransac
from paf_ring_detection.methods.zhang2019 import detect_zhang_arc_candidates
from paf_ring_detection.methods.zhang2019_paf import score_zhang_candidates_for_paf,select_zhang_inner_boundary


def run_geometry(image,method,config,seed):
    """候補生成と最終内周選択は標準の実装を共通利用する。"""
    if method=='canny':
        return detect_canny_contour_ransac(image,config['canny_contour_ransac'],config['ransac'],seed)[0]
    z=config['zhang2019']
    candidates,stages=detect_zhang_arc_candidates(image,z['preprocess'],z['detector'])
    candidates=score_zhang_candidates_for_paf(candidates,stages,z['paf_postprocess']['candidate_validation'])
    return select_zhang_inner_boundary(candidates,z['paf_postprocess']['selector'])


def main():
    protocol=read(PLAN/'protocol.json'); config=read(ROOT/'config/experiment.json')
    folder=OUTPUT/'validation'; manifest=read(folder/'manifest.json')
    if any(s['split']!='validation' for s in manifest['samples']): raise ValueError('validation以外の混入')
    out=OUTPUT/'geometry_validation'
    if (out/'selection.json').exists(): raise ValueError('幾何設定は選択済みです')
    rows=[]
    for index,s in enumerate(manifest['samples']):
        image=cv2.resize(read_image(folder/s['image']),(256,256),interpolation=cv2.INTER_AREA)
        truth=scaled_ellipse(label_ellipse(read(folder/s['label'])),256/480)
        seed=(protocol['render_seed']+zlib.crc32(s['condition']['condition_id'].encode()))%2**32
        for method in ['canny','zhang']:
            for candidate,thresholds in enumerate(protocol['geometry_validation_candidates']):
                c=copy.deepcopy(config)
                c['canny_contour_ransac' if method=='canny' else 'zhang2019']['preprocess'].update(thresholds)
                result=run_geometry(image,method,c,seed)
                ellipse=result['ellipse'] if result else None
                metrics=evaluate_ellipses(ellipse,truth,image.shape) if ellipse else {'ellipse_iou':0.0}
                distances=contour_metrics(ellipse,truth)
                success=bool(ellipse and metrics['ellipse_iou']>=protocol['evaluation']['success_iou'] and
                             distances['contour_p95_px']<=protocol['evaluation']['contour_p95_px'])
                rows.append({'sample_id':s['sample_id'],'shape_id':s['shape_id'],'method':method,
                             'candidate':candidate,'detected':ellipse is not None,'strict_success':success,
                             'iou_success':metrics['ellipse_iou']>=0.8,'iou':metrics['ellipse_iou'],**distances})
        if (index+1)%12==0: print(f'Geometry validation {index+1}/{len(manifest["samples"])}',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    selection={};selected=copy.deepcopy(config)
    for method in ['canny','zhang']:
        scores=[]
        for candidate in range(len(protocol['geometry_validation_candidates'])):
            a=[r for r in rows if r['method']==method and r['candidate']==candidate]
            scores.append({'candidate':candidate,'strict_successes':sum(r['strict_success'] for r in a),
                           'iou_successes':sum(r['iou_success'] for r in a),'n':len(a)})
        # 同数なら変更量の小さい標準設定(index=1)、それ以外は小さいindexを優先する。
        best=max(scores,key=lambda s:(s['strict_successes'],s['iou_successes'],s['candidate']==1,-s['candidate']))
        selection[method]={'candidates':scores,'selected_index':best['candidate'],
                           'thresholds':protocol['geometry_validation_candidates'][best['candidate']]}
        selected['canny_contour_ransac' if method=='canny' else 'zhang2019']['preprocess'].update(selection[method]['thresholds'])
    write(out/'selected_config.json',selected)
    write(out/'selection.json',{'selection':selection,'manifest_sha256':sha(folder/'manifest.json'),
        'protocol_sha256':sha(PLAN/'protocol.json'),'code_sha256':sha(Path(__file__)),
        'selection_rule':'strict成功数、IoU成功数、標準設定、小さいindexの順。形状ごと24枚で均等。',
        'test_used':False})
    print(selection,flush=True)


if __name__=='__main__':main()
