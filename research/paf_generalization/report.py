"""探索実験の全推定を照合し、対応付き集計と研究用比較画像を保存する。"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE, OUT, sha

LABELS = {'cnn_1194m':'穴領域CNN（1194M学習）', 'cnn_diversified':'穴領域CNN（多様形状学習）',
          'canny':'Canny＋輪郭別RANSAC', 'zhang2019':'Zhang 2019型（再現実装）',
          'spatial':'位置・サイズ変換', 'context':'参照範囲拡大', 'context_spatial':'参照範囲拡大＋位置・サイズ変換',
          'geometry':'楕円制約', 'geometry_projection':'楕円制約（モーメント出力）',
          'spatial_geometry':'位置・サイズ変換＋楕円制約',
          'structure_context_spatial':'境界の強さ・向き＋参照範囲拡大＋位置・サイズ変換',
          'annular_context_spatial':'内外周の包含学習＋参照範囲拡大＋位置・サイズ変換',
          'interior_context_spatial':'内部構造の乱数化＋参照範囲拡大＋位置・サイズ変換',
          'surface_context_spatial':'表面模様の乱数化＋参照範囲拡大＋位置・サイズ変換',
          'affine_context':'扁平率・向き・位置・サイズ変換＋参照範囲拡大',
          'boundary_spatial':'CNN境界確率＋重み付きRANSAC＋位置・サイズ変換',
          'boundary_surface':'表面模様学習＋CNN境界確率＋重み付きRANSAC',
          'surface_ensemble':'表面模様学習CNN：3モデル確率平均＋一様画像保留',
          'spatial_geometry_projection':'位置・サイズ変換＋楕円制約（モーメント出力）'}


def paired_interval(rows, baseline_rows):
    """同一group・同一seedの差についてcluster bootstrapの95%区間を計算する。"""
    baseline = {(r['seed'], r['sample_id']): r for r in baseline_rows}
    differences = defaultdict(list)
    for row in rows:
        ref = baseline[(row['seed'], row['sample_id'])]
        differences[row['group_id']].append(float(row['success'])-float(ref['success']))
    # 群サイズを保持し、画像平均に対応した差を再標本化する。
    keys = sorted(differences)
    sums = np.array([sum(differences[k]) for k in keys])
    counts = np.array([len(differences[k]) for k in keys])
    rng = np.random.default_rng(20261025)
    selected = rng.integers(0, len(keys), (2000, len(keys)))
    delta = sums[selected].sum(1)/counts[selected].sum(1)
    return {'delta': float(sums.sum()/counts.sum()), 'ci95': np.quantile(delta, [.025, .975]).tolist(),
            'n_clusters': len(keys), 'bootstrap_replicates': 2000,
            'uncertainty_scope': '画像群のばらつき。seed数が少ない場合の学習分布全体の不確実性は含まない。'}


def collect():
    records, audit = [], {'prediction_checks': 0, 'checkpoints': [], 'baseline_unchanged': False}
    checked_inputs = set()
    for source in ('1194m', 'diversified'):
        folder = BASE/'datasets'/source
        for sample in read_json(folder/'manifest.json')['samples']:
            for key in ('image', 'label'):
                path = (folder/sample[key]).resolve()
                if path not in checked_inputs:
                    assert sha(path) == sample[key+'_sha256'], str(path)
                    checked_inputs.add(path)
    audit['input_hash_checks'] = len(checked_inputs)
    lookups = {source: {s['sample_id']: s for s in load_samples(BASE/'datasets'/source, 'test')}
               for source in ('1194m', 'diversified')}
    inventory = read_json(OUT/'backup/baseline_20261025/inventory.json')
    for item in inventory['files']:
        # 引継ぎ文書は研究中に更新するため、固定比較の成果物と実装を照合する。
        if item['path'].startswith('.codex/'):
            continue
        assert sha(ROOT/item['path']) == item['sha256'], item['path']
    audit['baseline_unchanged'] = True
    for source in ('1194m', 'diversified'):
        for path in sorted((BASE/'results'/source).glob('*.json')):
            saved = read_json(path)
            if 'rows' not in saved:
                continue
            for row in saved['rows']:
                records.append({**row, 'dataset': source, 'variant': saved['method'], 'seed': saved['seed'],
                                'group_id': lookups[source][row['sample_id']]['group_id']})
    for complete in sorted((OUT/'runs').glob('*/*/complete.json')):
        folder = complete.parent
        metadata = read_json(folder/'provenance.json')
        history = read_json(folder/'training_history.json')
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        assert len(history) == metadata['config']['epochs']
        assert checkpoint['validation_loss'] == min(row['validation_loss'] for row in history)
        assert history[checkpoint['epoch']-1]['validation_loss'] == checkpoint['validation_loss']
        assert all(torch.isfinite(t).all() for t in checkpoint['model_state'].values())
        assert read_json(complete)['checkpoint_sha256'] == sha(folder/'best.pt')
        for path, digest in metadata['files'].items():
            assert sha(ROOT/path) == digest, path
        if 'context_implementation_sha256' in metadata['config']:
            assert sha(Path(__file__).with_name('context.py')) == metadata['config']['context_implementation_sha256']
        if 'structure_implementation_sha256' in metadata['config']:
            assert sha(Path(__file__).with_name('structure.py')) == metadata['config']['structure_implementation_sha256']
        if 'annular_implementation_sha256' in metadata['config']:
            assert sha(Path(__file__).with_name('annular.py')) == metadata['config']['annular_implementation_sha256']
        if 'interior_implementation_sha256' in metadata['config']:
            assert sha(Path(__file__).with_name('interior.py')) == metadata['config']['interior_implementation_sha256']
            for path, digest in metadata['config']['interior_data_hashes'].items():
                assert sha(ROOT/path) == digest
        if 'surface_implementation_sha256' in metadata['config']:
            assert sha(Path(__file__).with_name('surface.py')) == metadata['config']['surface_implementation_sha256']
            for path, digest in metadata['config']['surface_data_hashes'].items():
                assert sha(ROOT/path) == digest
        if 'affine_implementation_sha256' in metadata['config']:
            assert sha(Path(__file__).with_name('affine.py')) == metadata['config']['affine_implementation_sha256']
        if 'boundary_implementation_sha256' in metadata['config']:
            assert sha(Path(__file__).with_name('boundary.py')) == metadata['config']['boundary_implementation_sha256']
            assert sha(ROOT/'paf_ring_detection/methods/ransac.py') == metadata['config']['ransac_implementation_sha256']
        if 'boundary_surface_implementation_sha256' in metadata['config']:
            assert sha(Path(__file__).with_name('boundary_surface.py')) == metadata['config']['boundary_surface_implementation_sha256']
            for path,digest in metadata['config']['surface_data_hashes'].items():
                assert sha(ROOT/path)==digest
        audit['checkpoints'].append({'variant': metadata['variant'], 'seed': metadata['seed'],
            'epochs': len(history), 'selected_epoch': checkpoint['epoch'], 'validation_loss': checkpoint['validation_loss']})
        for source in ('1194m', 'diversified'):
            path = folder/f'test_{source}.json'
            if not path.exists():
                continue
            saved = read_json(path)
            assert saved['checkpoint_sha256'] == sha(folder/'best.pt')
            assert [r['sample_id'] for r in saved['rows']] == list(lookups[source])
            for row in saved['rows']:
                # 元ラベルから毎回正解を復元し、保存済み正解への自己照合を避ける。
                sample = lookups[source][row['sample_id']]
                from paf_ring_detection.data import label_ellipse, scaled_ellipse
                label = read_json(BASE/'datasets'/source/sample['label'])
                truth = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
                assert np.allclose(np.array([*truth[0], *truth[1], truth[2]]),
                                   np.array([*row['truth_ellipse'][0], *row['truth_ellipse'][1], row['truth_ellipse'][2]]))
                ellipse = row['ellipse']
                ellipse = (tuple(ellipse[0]), tuple(ellipse[1]), ellipse[2]) if ellipse else None
                iou = evaluate_ellipses(ellipse, truth, (256,256))['ellipse_iou'] if ellipse else 0.
                assert iou == row['ellipse_iou'] and row['success'] == (iou >= .8)
                records.append({**row, 'dataset': source, 'variant': metadata['variant'], 'seed': metadata['seed']})
                if 'projected_ellipse' in row:
                    ell = row['projected_ellipse']
                    ell = (tuple(ell[0]), tuple(ell[1]), ell[2]) if ell else None
                    iou = evaluate_ellipses(ell, truth, (256,256))['ellipse_iou'] if ell else 0.
                    assert iou == row['projected_iou'] and row['projected_success'] == (iou >= .8)
                    records.append({**row, 'dataset': source, 'variant': metadata['variant']+'_projection',
                        'seed': metadata['seed'], 'ellipse_iou': iou, 'success': iou >= .8,
                        'ellipse': row['projected_ellipse']})
                audit['prediction_checks'] += 1
    detector_folder=OUT/'detectors/surface_context_spatial_20260907_20260908_20260909'
    for source in ('1194m','diversified'):
        path=detector_folder/f'test_{source}.json'
        if not path.exists():
            continue
        saved=read_json(path); provenance=saved['provenance']
        assert provenance==read_json(detector_folder/'provenance.json')
        assert provenance['implementation_sha256']==sha(Path(__file__).with_name('detector.py'))
        for relative,digest in provenance['implementation_files'].items():
            assert sha(ROOT/relative)==digest
        for model in provenance['checkpoints']:
            assert sha(Path(model['path']))==model['sha256']
        assert [r['sample_id'] for r in saved['rows']]==list(lookups[source])
        for row in saved['rows']:
            sample=lookups[source][row['sample_id']]
            label=read_json(BASE/'datasets'/source/sample['label'])
            truth=scaled_ellipse(label_ellipse(label),256/label['image_width'])
            e=row['ellipse']; ellipse=(tuple(e[0]),tuple(e[1]),e[2]) if e else None
            iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
            assert iou==row['ellipse_iou'] and row['success']==(iou>=.8)
            assert row['group_id']==sample['group_id']
            assert row['degradation']==sample['conditions']['degradation']
            assert row['severity']==sample['conditions']['severity']
            records.append({**row,'dataset':source,'variant':'surface_ensemble','seed':'fixed3'})
            audit['prediction_checks']+=1
    return records, audit


def report():
    records, audit = collect()
    summary, comparisons = [], []
    for source in ('1194m', 'diversified'):
        variants = sorted({r['variant'] for r in records if r['dataset'] == source})
        for variant in variants:
            rows = [r for r in records if r['dataset'] == source and r['variant'] == variant]
            seeds = sorted({r['seed'] for r in rows}, key=str)
            subsets = [('all', None, rows)]
            for effect in ('clean', 'black_rectangle', 'sensor_whiteout', 'sensor_black_crush'):
                selected = [r for r in rows if r['degradation'] == effect]
                subsets.append((effect, None, selected))
                if effect != 'clean':
                    for severity in (.25, .5, .75, 1.):
                        subsets.append((effect, severity, [r for r in selected if r['severity'] == severity]))
            baseline = [r for r in records if r['dataset'] == source and r['variant'] == 'cnn_diversified' and r['seed'] in seeds]
            if variant=='surface_ensemble':
                # 一つの統合出力と、同じ画像の基準3seedの平均成功を比較する。
                original=[r for r in records if r['dataset']==source and r['variant']=='cnn_diversified']
                grouped=defaultdict(list)
                for row in original:
                    grouped[row['sample_id']].append(row)
                baseline=[{**group[0],'seed':'fixed3','success':float(np.mean([r['success'] for r in group]))}
                          for group in grouped.values()]
            for effect, severity, selected in subsets:
                per_seed = {str(s): float(np.mean([r['success'] for r in selected if r['seed'] == s])) for s in seeds}
                item = {'dataset': source, 'variant': variant, 'degradation': effect, 'severity': severity,
                    'n_seeds': len(seeds), 'n_predictions': len(selected),
                    'n_images': len({r['sample_id'] for r in selected}),
                    'success_rate': float(np.mean([r['success'] for r in selected])),
                    'mean_iou': float(np.mean([r['ellipse_iou'] for r in selected])), 'per_seed': per_seed}
                item['mean_mask_iou'] = (float(np.mean([r['mask_iou'] for r in selected]))
                                         if all(r.get('mask_iou') is not None for r in selected) else None)
                summary.append(item)
                if variant not in ('cnn_diversified', 'cnn_1194m', 'canny', 'zhang2019') and effect in ('all', 'clean'):
                    comparisons.append({**item, **paired_interval(selected, baseline)})
    # 視点は投影径・内部の見え方・照明と交絡するため、原因ではなく層別成績として扱う。
    cameras={s['sample_id']:s['conditions']['camera']['tilt_deg']
             for s in load_samples(BASE/'datasets/1194m','test')}
    viewpoint=[]
    for variant in sorted({r['variant'] for r in records if r['dataset']=='1194m'}):
        for angle in sorted(set(cameras.values())):
            selected=[r for r in records if r['dataset']=='1194m' and r['variant']==variant
                      and r['degradation']=='clean' and cameras[r['sample_id']]==angle]
            seeds=sorted({r['seed'] for r in selected},key=str)
            viewpoint.append({'variant':variant,'tilt_deg':angle,'n_images':len({r['sample_id'] for r in selected}),
                'success_rate':float(np.mean([r['success'] for r in selected])),
                'per_seed':{str(s):float(np.mean([r['success'] for r in selected if r['seed']==s])) for s in seeds}})
    visibility={ (r['dataset'],r['sample_id']):r['remaining_projected_rim_fraction']
                 for r in read_json(OUT/'observability.json')['rows'] if r['split']=='test'}
    coverage=[]
    for source in ('1194m','diversified'):
        for variant in sorted({r['variant'] for r in records if r['dataset']==source}):
            for name,low,high in [('0',-1.,0.),('(0,0.25]',0.,.25),('(0.25,0.5]',.25,.5),
                                  ('(0.5,0.75]',.5,.75),('(0.75,1]',.75,1.000001)]:
                selected=[r for r in records if r['dataset']==source and r['variant']==variant
                          and r['degradation']=='black_rectangle'
                          and low<visibility[source,r['sample_id']]<=high]
                if selected:
                    coverage.append({'dataset':source,'variant':variant,'remaining_rim_fraction':name,
                        'n_images':len({r['sample_id'] for r in selected}),'n_predictions':len(selected),
                        'success_rate':float(np.mean([r['success'] for r in selected]))})
    write_json(OUT/'summary.json', {'rows': summary, 'paired_comparisons': comparisons,'viewpoint_diagnosis':viewpoint,
        'projected_rim_coverage_diagnosis':coverage,
        'coverage_interpretation':'矩形の外に残る正解投影内周の割合。光学的可視性や他の本体情報の有無とは異なる。',
        'observability_sha256':sha(OUT/'observability.json')})
    write_json(OUT/'audit.json', audit)
    fields = ['dataset','variant','degradation','severity','n_seeds','n_images','n_predictions','success_rate','mean_iou','mean_mask_iou']
    with (OUT/'scores.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    lines = ['# PAF汎化探索：確認済みの比較結果', '', '更新日：2026-10-25（JST）。成功条件は楕円IoU≥0.8。既存testを設計に参照した探索的評価。', '',
        '各方式のseed数を併記する。1seedの探索結果と3seed平均を同じ確度で扱わない。幾何は学習seedなしの一回評価。', '',
        '|方式|seed数|1194M 全条件|1194M 通常|多様形状 全条件|多様形状 通常|', '|---|---:|---:|---:|---:|---:|']
    lookup = {(r['dataset'],r['variant'],r['degradation']): r for r in summary if r['severity'] is None}
    for variant in sorted({r['variant'] for r in summary}):
        values = [lookup.get((s,variant,e)) for s in ('1194m','diversified') for e in ('all','clean')]
        if not all(values):
            continue
        seed_label = '—（1回）' if variant in ('canny','zhang2019') else str(values[0]['n_seeds'])
        if variant=='surface_ensemble':
            seed_label='3モデル統合'
        lines.append('|'+LABELS.get(variant,variant)+'|'+seed_label+'|'+'|'.join(f"{100*r['success_rate']:.2f}%" for r in values)+'|')
    lines += ['', '## 多様形状学習CNNとの差', '',
        '対応する同一seedの基準との差。95%区間は1194Mのカメラ群、多様形状の形状群を単位に2,000回再標本化する。正の値が改善。モデル選択の多重性を補正した確証的検定ではない。', '',
        '|対象|方式|条件|差（pp）|95%区間（pp）|群数|', '|---|---|---|---:|---|---:|']
    for r in comparisons:
        lo,hi=r['ci95']
        lines.append(f"|{r['dataset']}|{LABELS.get(r['variant'],r['variant'])}|{r['degradation']}|{100*r['delta']:.2f}|[{100*lo:.2f}, {100*hi:.2f}]|{r['n_clusters']}|")
    lines += ['', '## 検証', '', f"固定比較のhash照合：{audit['baseline_unchanged']}。新規推定の元ラベル照合：{audit['prediction_checks']:,}件。",
              '全完了モデルのepoch数・最小validation損失の重み・有限値を照合。条件・強度・seed別成績はscores.csvとsummary.json、全推定はruns内のtest JSON。', '',
              '## 制約', '', '未学習の周辺構造3形状族の固定評価はfamily_holdout_v1/REPORT.mdを参照。実画像、対象なし判定は未実施。全黒など情報が消失した条件も主指標の分母に含む。',
              '確率平均モデルは3モデルから一つの出力を得る。対応差は既存3seedの画像ごとの平均成功との差であり、統合モデル自体の学習反復の区間ではない。',
              'spatialは位置と大きさを同時に変える。geometryは学習損失とcheckpoint選択の総損失が変わる。contextはパラメータ数が同じだが参照構造が変わる。',
              '方式の詳細、出典、未検証の研究仮説はresearch/paf_generalization/README.mdを参照する。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    report()
