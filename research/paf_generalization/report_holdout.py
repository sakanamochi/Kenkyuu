"""追加形状の全推定を照合し、対応する元形状との差を群単位で集計する。"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,load_samples,label_ellipse,scaled_ellipse,read_image
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.evaluate_holdout import DATA,verify
from research.paf_generalization.diagnose import BASE,OUT,sha

NAMES={'cnn_diversified':'既存CNN（多様形状学習）','surface_context_spatial':'表面模様学習CNN：seed別成績の平均',
       'surface_ensemble':'表面模様学習CNN：3モデルの確率平均','boundary_spatial':'CNN境界確率＋RANSAC',
       'boundary_surface':'表面模様学習＋CNN境界確率＋RANSAC',
       'canny':'Canny＋輪郭別RANSAC','zhang2019':'Zhang 2019型（再現実装）'}


def paired_shape_difference(rows):
    """同じ形状・劣化・学習seedの新構造と元構造の差を対応付ける。"""
    controls={(r['group_id'],r['degradation'],r['seed']):r for r in rows if r['view']=='control'}
    groups=defaultdict(list)
    families={}
    for row in rows:
        if row['view']!='new_family':
            continue
        key=(row['group_id'],row['degradation'],row['seed'])
        groups[row['group_id']].append(float(row['success'])-float(controls[key]['success']))
        families[row['group_id']]=row['family']
    rng=np.random.default_rng(2026102602)
    totals=np.zeros(2000); denominator=0
    for family in sorted(set(families.values())):
        keys=sorted(k for k in groups if families[k]==family)
        values=np.array([np.mean(groups[k]) for k in keys])
        selected=rng.integers(0,len(values),(2000,len(values)))
        totals+=values[selected].sum(1); denominator+=len(values)
    return {'delta':float(np.mean([np.mean(v) for v in groups.values()])),
            'ci95':np.quantile(totals/denominator,[.025,.975]).tolist(),'n_shape_pairs':len(groups),
            'scope':'固定した形状族内の形状対を再標本化。新しい形状族全体や学習seed分布への区間ではない。'}


def main():
    protocol=verify()
    inventory=read_json(OUT/'backup/baseline_20261025/inventory.json')
    for item in inventory['files']:
        if not item['path'].startswith('.codex/'):
            assert sha(ROOT/item['path'])==item['sha256']
    samples=load_samples(DATA,'test')
    lookup={s['sample_id']:s for s in samples}
    truths={sid:scaled_ellipse(label_ellipse(read_json(Path(s['label']))),256/480) for sid,s in lookup.items()}
    records=[]; checks=0; model_files=[]
    for path in sorted((DATA/'results').glob('*.json')):
        saved=read_json(path); metadata=saved['metadata']
        assert all(metadata[k]==v for k,v in protocol.items())
        name,seed=metadata['method'],metadata['seed']
        if name=='cnn_diversified':
            expected_hashes=[sha(BASE/f'models/diversified/{seed}/best.pt')]
        elif name=='surface_ensemble':
            expected_hashes=[sha(OUT/f'runs/surface_context_spatial/{s}/best.pt') for s in (20260907,20260908,20260909)]
        elif name in ('canny','zhang2019'):
            expected_hashes=[]
        else:
            expected_hashes=[sha(OUT/f'runs/{name}/{seed}/best.pt')]
        assert metadata['checkpoint_sha256']==expected_hashes
        if name=='boundary_surface':
            additional=read_json(DATA/'additional_candidate_plan.json')
            assert metadata['additional_candidate_plan_sha256']==sha(DATA/'additional_candidate_plan.json')
            assert metadata['additional_evaluator_sha256']==sha(Path(__file__).with_name('evaluate_boundary_surface.py'))
            for frozen,digest in additional['frozen_files'].items():
                assert sha(ROOT/frozen)==digest
        assert [r['sample_id'] for r in saved['rows']]==list(lookup)
        for row in saved['rows']:
            sample=lookup[row['sample_id']]
            for field in ('family','view','group_id'):
                assert row[field]==sample[field]
            for field in ('degradation','severity'):
                assert row[field]==sample['conditions'][field]
            e=row['ellipse']; ellipse=(tuple(e[0]),tuple(e[1]),e[2]) if e else None
            truth=truths[row['sample_id']]
            iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
            assert iou==row['ellipse_iou'] and row['success']==(iou>=.8)
            assert np.allclose(np.array([*truth[0],*truth[1],truth[2]]),
                               np.array([*row['truth_ellipse'][0],*row['truth_ellipse'][1],row['truth_ellipse'][2]]))
            records.append({**row,'method':metadata['method'],'seed':metadata['seed']}); checks+=1
        model_files.append({'path':str(path),'sha256':sha(path),'method':metadata['method'],'seed':metadata['seed']})
    summary=[]; pairs=[]
    for method in NAMES:
        rows=[r for r in records if r['method']==method]
        if not rows:
            continue
        pairs.append({'method':method,**paired_shape_difference(rows)})
        for family in ['all']+sorted({r['family'] for r in rows}):
            for view in ('control','new_family'):
                for effect in ('all','clean','black_rectangle','sensor_whiteout','sensor_black_crush'):
                    subset=[r for r in rows if r['view']==view and (family=='all' or r['family']==family)
                            and (effect=='all' or r['degradation']==effect)]
                    summary.append({'method':method,'family':family,'view':view,'degradation':effect,
                        'success_rate':float(np.mean([r['success'] for r in subset])),
                        'mean_iou':float(np.mean([r['ellipse_iou'] for r in subset])),
                        'n_predictions':len(subset),'n_images':len({r['sample_id'] for r in subset}),
                        'n_shape_pairs':len({r['group_id'] for r in subset}),
                        'seeds':sorted({r['seed'] for r in subset},key=str)})
    # 画面外へ切れた部分もあるため、構造介入が画像に及ぼした差を併記する。
    appearance=[]
    groups={s['group_id']:s['family'] for s in samples}
    for group,family in groups.items():
        control=read_image(DATA/f'images/{group}_control.png').astype(float)
        changed=read_image(DATA/f'images/{group}_new_family.png').astype(float)
        appearance.append({'group_id':group,'family':family,'mean_absolute_pixel_difference':float(np.abs(control-changed).mean())})
    write_json(DATA/'summary.json',{'rows':summary,'paired_shape_differences':pairs,'appearance':appearance})
    expected={('canny',None),('zhang2019',None),('surface_ensemble',None),('boundary_spatial',20260907),('boundary_surface',20260907)}
    expected.update((m,s) for m in ('cnn_diversified','surface_context_spatial') for s in (20260907,20260908,20260909))
    actual={(r['method'],r['seed']) for r in model_files}
    assert len(actual)==len(model_files)
    write_json(DATA/'audit.json',{'prediction_checks':checks,'result_files':model_files,'expected_result_files':11,
        'all_planned_results_present':actual==expected,'baseline_unchanged':True,'input_manifest_sha256':sha(DATA/'manifest.json')})
    lines=['# 未学習の周辺構造に対する固定評価','',
        '新規96形状と対応する元形状の計192画像を、通常・半画面遮蔽・白飛び0.5・黒つぶれ0.5で評価する。各方式768推定、元/新構造は各384画像。既存比較の13条件平均とは直接比較しない。',
        '', '形状族は角形胴・開放支持・リブ付き胴の3種類。上端内周・撮影・材質・照明・劣化乱数を対で共有し、未知の周辺形状への感度を測る。黒背景・単一上端開口の合成画像であり、実画像・対象なし・任意rollは未検証。',
        '',f'照合済み結果ファイル：{len(model_files)}/11。全モデルの追加形状に対する閾値・重みの調整は行わない。',
        '', '|方式|元構造：4条件|新構造：4条件|新構造：通常|新−元（pp）|95%区間（pp）|',
        '|---|---:|---:|---:|---:|---|']
    for pair in pairs:
        method=pair['method']
        def score(view,effect):
            return next(r['success_rate'] for r in summary if r['method']==method and r['family']=='all' and r['view']==view and r['degradation']==effect)
        ci=pair['ci95']
        lines.append(f"|{NAMES[method]}|{100*score('control','all'):.2f}%|{100*score('new_family','all'):.2f}%|{100*score('new_family','clean'):.2f}%|{100*pair['delta']:+.2f}|[{100*ci[0]:.2f}, {100*ci[1]:.2f}]|")
    lines.extend(['','区間は各形状族内で対応する形状対を再標本化したもの。未知の全形状族や学習seed全体への保証ではない。seed別成績の平均と、確率を平均して一つの推定を出す方式を区別する。'])
    (DATA/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('checked',checks,'predictions;',len(model_files),'/11 result files')


if __name__=='__main__':
    main()
