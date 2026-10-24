"""表面模様＋境界学習の固定構成を追加seedで反復し、初回固定評価と区別する。"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.boundary_detector import BoundaryPAFDetector
from research.paf_generalization.diagnose import BASE,OUT,sha,image_and_truth
from research.paf_generalization.evaluate_holdout import DATA,verify
from research.paf_generalization.report import paired_interval
from research.paf_generalization.report_holdout import paired_shape_difference

DEST=DATA/'replications'
SEEDS=(20260907,20260908,20260909)


def plan():
    """変更しない実装・seed・評価規則と、この反復の解釈を固定する。"""
    path=DEST/'plan.json'
    if path.exists():
        return read_json(path)
    files=['boundary.py','boundary.json','boundary_surface.py','boundary_surface.json',
           'boundary_detector.py','experiment.py','context.py','replication.py']
    frozen={str((Path(__file__).parent/f).relative_to(ROOT)):sha(Path(__file__).parent/f) for f in files}
    data={'created_utc':datetime.now(timezone.utc).isoformat(),'variant':'boundary_surface','seeds':list(SEEDS),
          'selection':'各seedで元の多様形状validation損失最小。最良のtest seedを選ばず3seed全てを報告。',
          'scope':'初回の固定評価で選んだ構成の反復確認。初回評価11設定とは別に保存し、新たな独立形状集合とは呼ばない。',
          'frozen_files':frozen,'original_plan_sha256':sha(DATA/'plan.json'),
          'initial_result_sha256':sha(DATA/'results/boundary_surface_20260907.json'),
          'uniform_input_policy':'一様画像は未検出。全画像を分母に維持。'}
    write_json(path,data)
    return data


def evaluate():
    """元の固定形状に対して追加seedだけを推論し、初回の出力は保持する。"""
    protocol=verify(); specification=read_json(DEST/'plan.json')
    for name,digest in specification['frozen_files'].items():
        assert sha(ROOT/name)==digest
    cases=[(sample,*image_and_truth(DATA,sample)) for sample in load_samples(DATA,'test')]
    for seed in SEEDS[1:]:
        detector=BoundaryPAFDetector(seed=seed)
        metadata={**protocol,'seed':seed,'method':'boundary_surface',
                  'replication_plan_sha256':sha(DEST/'plan.json'),'detector':detector.provenance}
        destination=DEST/'results'/f'boundary_surface_{seed}.json'
        if destination.exists():
            assert read_json(destination)['metadata']==metadata
            continue
        rows=[]
        for sample,image,truth in cases:
            result=detector.predict(image); ellipse=result['ellipse']
            iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id':sample['sample_id'],'group_id':sample['group_id'],
                'family':sample['family'],'view':sample['view'],
                'degradation':sample['conditions']['degradation'],'severity':sample['conditions']['severity'],
                'ellipse':ellipse,'truth_ellipse':truth,'ellipse_iou':iou,'success':iou>=.8,'status':result['status']})
        write_json(destination,{'metadata':metadata,'rows':rows})
        print('replication evaluated',seed,len(rows),flush=True)


def report():
    """全3seedを元画像・正解に照合し、対応する既存CNNとの差も算出する。"""
    specification=read_json(DEST/'plan.json')
    for name,digest in specification['frozen_files'].items():
        assert sha(ROOT/name)==digest
    assert specification['initial_result_sha256']==sha(DATA/'results/boundary_surface_20260907.json')
    samples=load_samples(DATA,'test'); records=[]; references=[]; checks=0
    for seed in SEEDS:
        path=(DATA/'results' if seed==SEEDS[0] else DEST/'results')/f'boundary_surface_{seed}.json'
        saved=read_json(path)
        assert [r['sample_id'] for r in saved['rows']]==[s['sample_id'] for s in samples]
        if seed!=SEEDS[0]:
            assert saved['metadata']['replication_plan_sha256']==sha(DEST/'plan.json')
            assert saved['metadata']['detector']['checkpoint_sha256']==sha(OUT/f'runs/boundary_surface/{seed}/best.pt')
        for sample,row in zip(samples,saved['rows']):
            _,truth=image_and_truth(DATA,sample)
            e=row['ellipse']; ellipse=(tuple(e[0]),tuple(e[1]),e[2]) if e else None
            iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
            assert iou==row['ellipse_iou'] and row['success']==(iou>=.8)
            for key in ('group_id','family','view'):
                assert row[key]==sample[key]
            records.append({**row,'seed':seed}); checks+=1
        references.extend({**r,'seed':seed} for r in read_json(DATA/f'results/cnn_diversified_{seed}.json')['rows'])
    summaries=[]
    for family in ['all']+sorted({r['family'] for r in records}):
        for view in ('control','new_family'):
            for effect in ('all','clean','black_rectangle','sensor_whiteout','sensor_black_crush'):
                rows=[r for r in records if r['view']==view and (family=='all' or r['family']==family)
                      and (effect=='all' or r['degradation']==effect)]
                summaries.append({'family':family,'view':view,'degradation':effect,
                    'success_rate':float(np.mean([r['success'] for r in rows])),
                    'per_seed':{str(s):float(np.mean([r['success'] for r in rows if r['seed']==s])) for s in SEEDS},
                    'n_images':len({r['sample_id'] for r in rows}),'n_predictions':len(rows)})
    new=[r for r in records if r['view']=='new_family']
    comparison=paired_interval(new,[r for r in references if r['view']=='new_family'])
    shape_difference=paired_shape_difference(records)
    write_json(DEST/'summary.json',{'rows':summaries,'paired_baseline_difference':comparison,
        'paired_shape_difference':shape_difference,'prediction_checks':checks,'plan':specification})
    lines=['# 固定構成の3seed反復確認','',specification['scope'],'',
           '閾値・構成・学習回数は固定し、各seedの元validation損失最小を使用する。3モデルの成功率の平均であり、確率平均の統合推論ではない。','',
           '|対象|条件|3seed平均|seed07|seed08|seed09|','|---|---|---:|---:|---:|---:|']
    for row in summaries:
        if row['family']=='all':
            values=[row['success_rate']]+list(row['per_seed'].values())
            lines.append('|'+row['view']+'|'+row['degradation']+'|'+'|'.join(f'{100*v:.2f}%' for v in values)+'|')
    lines+=['',f"全{checks:,}推定のIoUを照合。新構造での既存CNNとの差：{100*comparison['delta']:+.2f}pp、95%区間 [{100*comparison['ci95'][0]:.2f}, {100*comparison['ci95'][1]:.2f}]pp。",
            '区間は同じ形状群で対応付けた再標本化。学習seed分布全体、未知の形状族全体、モデル選択の多重性を含まない。']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('replication checked',checks,'predictions')


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','evaluate','report'])
    args=parser.parse_args()
    {'plan':plan,'evaluate':evaluate,'report':report}[args.stage]()
