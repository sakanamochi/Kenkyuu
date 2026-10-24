"""幾何候補とCNN候補の相補性を、正解を使う診断上限として測る。"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json
from research.paf_generalization.diagnose import BASE,OUT


def main(variant,seed):
    rows=[]
    for source in ('1194m','diversified'):
        methods={}
        for method in ('canny','zhang2019'):
            methods[method]={r['sample_id']:r for r in read_json(BASE/f'results/{source}/{method}_fixed.json')['rows']}
        cnn=read_json(OUT/f'runs/{variant}/{seed}/test_{source}.json')['rows']
        for row in cnn:
            sid=row['sample_id']
            geometry=any(methods[m][sid]['success'] for m in methods)
            rows.append({'dataset':source,'sample_id':sid,'degradation':row['degradation'],'severity':row['severity'],
                'geometry_union':geometry,'cnn_success':row['success'],'oracle_union':geometry or row['success'],
                'cnn_only_success':row['success'] and not geometry,'geometry_only_success':geometry and not row['success']})
    summary=[]
    for source in ('1194m','diversified'):
        for effect in ('all','clean','black_rectangle','sensor_whiteout','sensor_black_crush'):
            selected=[r for r in rows if r['dataset']==source and (effect=='all' or r['degradation']==effect)]
            summary.append({'dataset':source,'degradation':effect,'n':len(selected),
                **{key:float(np.mean([r[key] for r in selected])) for key in
                   ('geometry_union','cnn_success','oracle_union','cnn_only_success','geometry_only_success')}})
    write_json(OUT/f'coverage_{variant}_{seed}.json',{'variant':variant,'seed':seed,'rows':rows,'summary':summary,
        'interpretation':'正解により成功候補を選ぶ診断上限。実行可能な検出器の性能ではない。幾何は各方式が選択した最終楕円のみを含む。'})
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',default='spatial')
    parser.add_argument('--seed',type=int,default=20260907)
    args=parser.parse_args()
    main(args.variant,args.seed)
