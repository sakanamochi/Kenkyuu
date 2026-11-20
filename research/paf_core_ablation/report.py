"""完了した対照の全推定を原ラベルから照合し、同条件の要因差を集計する。"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import sha
from research.paf_generalization.report import paired_interval
from research.paf_core_ablation.experiment import OUT, BASE, CONFIG

LABELS = {'main_1194m': 'main構成：1194M学習', 'main_diversified': 'main構成：多様形状のみ',
          'main_surface': '＋表面模様', 'main_projection': '＋投影',
          'main_projection_surface': '＋投影＋表面模様',
          'canny': 'Canny＋輪郭別RANSAC', 'zhang2019': 'Zhang 2019型（再現実装）'}


def factorial_interactions(records, conditions):
    """4セル共通の画像・seedで差の差を測り、順序依存の寄与説明を避ける。"""
    results = []
    variants = ('main_diversified', 'main_projection', 'main_surface', 'main_projection_surface')
    for source in ('1194m', 'diversified'):
        cells = {variant: {(r['seed'], r['sample_id']): r for r in records
                           if r['dataset'] == source and r['variant'] == variant} for variant in variants}
        common = sorted(set.intersection(*(set(cell) for cell in cells.values())))
        if not common:
            continue
        for effect, severity in conditions:
            keys = [key for key in common if effect == 'all' or
                    (cells[variants[0]][key]['degradation'] == effect and cells[variants[0]][key]['severity'] == severity)]
            if not keys:
                continue
            with_projection, without_projection = [], []
            for key in keys:
                original, projected, surface, both = [cells[v][key] for v in variants]
                assert len({r['group_id'] for r in (original, projected, surface, both)}) == 1
                # 模様の効果が投影の有無でどれだけ変わるか。正なら相乗的な差。
                with_projection.append({**original, 'success': float(both['success'])-float(projected['success'])})
                without_projection.append({**original, 'success': float(surface['success'])-float(original['success'])})
            seeds = sorted({key[0] for key in keys})
            results.append({'dataset': source, 'degradation': effect, 'severity': severity, 'seeds': seeds,
                            'n_images': len({key[1] for key in keys}),
                            'formula': '(projection_surface - projection) - (surface - original)',
                            'scope': '全3seedの対応比較' if len(seeds) == 3 else '探索：反復未完了',
                            **paired_interval(with_projection, without_projection)})
    return results


def main():
    config, plan = read_json(CONFIG), read_json(OUT/'plan.json')
    for path, digest in plan['files'].items():
        assert sha(ROOT/path) == digest, path
    for path, digest in plan['input_files'].items():
        assert sha(Path(path)) == digest, path
    truth, metadata = {}, {}
    for source in ('1194m', 'diversified'):
        samples = load_samples(BASE/source, 'test')
        metadata[source] = {s['sample_id']: s for s in samples}
        truth[source] = {}
        for sample in samples:
            label = read_json(BASE/source/sample['label'])
            truth[source][sample['sample_id']] = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    records, files, incomplete = [], {}, []
    checks = 0
    for variant in config['variants']:
        for seed in config['seeds']:
            folder = OUT/'runs'/variant/str(seed)
            if not all((folder/name).exists() for name in ('complete.json', 'test_1194m.json', 'test_diversified.json')):
                incomplete.append({'variant': variant, 'seed': seed})
                continue
            completed = read_json(folder/'complete.json')
            assert completed['epochs'] == config['epochs']
            assert completed['checkpoint_sha256'] == sha(folder/'best.pt')
            history = read_json(folder/'training_history.json')
            assert len(history) == config['epochs']
            assert min(r['validation_loss'] for r in history) == completed['best_validation_loss']
            checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
            selected = min(history, key=lambda r: r['validation_loss'])
            assert checkpoint['epoch'] == selected['epoch']
            assert checkpoint['validation_loss'] == selected['validation_loss']
            assert checkpoint['provenance'] == read_json(folder/'provenance.json')
            assert all(torch.isfinite(value).all() for value in checkpoint['model_state'].values())
            for name in ('best.pt', 'complete.json', 'training_history.json', 'provenance.json'):
                files[str((folder/name).relative_to(ROOT))] = sha(folder/name)
            for source in ('1194m', 'diversified'):
                path = folder/f'test_{source}.json'
                result = read_json(path)
                assert result['checkpoint_sha256'] == completed['checkpoint_sha256']
                assert result['provenance']['plan_sha256'] == sha(OUT/'plan.json')
                assert result['provenance']['training_source'] == config['variants'][variant]['source']
                assert result['provenance']['target_data_used_for_training'] == (variant == 'main_1194m')
                rows = result['rows']
                assert len(rows) == len(truth[source])
                assert {r['sample_id'] for r in rows} == set(truth[source])
                for row in rows:
                    key = row['sample_id']
                    sample = metadata[source][key]
                    assert row['group_id'] == sample['group_id']
                    assert row['degradation'] == sample['conditions']['degradation']
                    assert row['severity'] == sample['conditions']['severity']
                    assert np.allclose(np.array(row['truth_ellipse'][0]+row['truth_ellipse'][1]+[row['truth_ellipse'][2]]),
                                       np.array(list(truth[source][key][0])+list(truth[source][key][1])+[truth[source][key][2]]))
                    iou = evaluate_ellipses(row['ellipse'], truth[source][key], (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
                    assert abs(iou-row['ellipse_iou']) < 1e-10
                    assert row['success'] == (iou >= .8)
                    records.append({**row, 'variant': variant, 'seed': seed, 'dataset': source})
                    checks += 1
                files[str(path.relative_to(ROOT))] = sha(path)
    # 保存済み幾何比較も同じ入力manifest・全GTで照合し、CNNと同じ条件表へ載せる。
    geometry_checks = 0
    for source in ('1194m', 'diversified'):
        for method in ('canny', 'zhang2019'):
            path = BASE.parent/f'results/{source}/{method}_fixed.json'
            result = read_json(path)
            assert result['manifest_sha256'] == sha(BASE/source/'manifest.json')
            assert result['input_size'] == 256 and result['limited'] is False
            assert len(result['rows']) == len(truth[source])
            assert {r['sample_id'] for r in result['rows']} == set(truth[source])
            for row in result['rows']:
                sample = metadata[source][row['sample_id']]
                assert row['degradation'] == sample['conditions']['degradation']
                assert row['severity'] == sample['conditions']['severity']
                iou = evaluate_ellipses(row['ellipse'], truth[source][row['sample_id']], (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
                assert abs(iou-row['ellipse_iou']) < 1e-10 and row['success'] == (iou >= .8)
                records.append({**row, 'variant': method, 'seed': None, 'dataset': source, 'group_id': sample['group_id']})
                geometry_checks += 1
            files[str(path.relative_to(ROOT))] = sha(path)
    conditions = [('all', None), ('clean', 0.)]+[(effect, strength) for effect in
        ('black_rectangle', 'sensor_whiteout', 'sensor_black_crush') for strength in (.25, .5, .75, 1.)]
    summaries, paired = [], []
    for source in ('1194m', 'diversified'):
        for variant in LABELS:
            selected = [r for r in records if r['dataset'] == source and r['variant'] == variant]
            if not selected:
                continue
            seeds = sorted({r['seed'] for r in selected})
            for effect, strength in conditions:
                rows = [r for r in selected if effect == 'all' or (r['degradation'] == effect and r['severity'] == strength)]
                summaries.append({'dataset': source, 'variant': variant, 'degradation': effect, 'severity': strength,
                                  'seeds': seeds, 'n_images': len({r['sample_id'] for r in rows}),
                                  'success_rate': float(np.mean([r['success'] for r in rows])),
                                  'per_seed': {str(s): float(np.mean([r['success'] for r in rows if r['seed'] == s])) for s in seeds}})
        # 同じ画像・seedが揃った対照だけを比べ、1seedは探索と明示する。
        for before, after, factor in [('main_diversified', 'main_projection', '投影：模様なし'),
                                      ('main_surface', 'main_projection_surface', '投影：模様あり'),
                                      ('main_diversified', 'main_surface', '表面模様：投影なし'),
                                      ('main_projection', 'main_projection_surface', '表面模様：投影あり')]:
            a = {(r['seed'], r['sample_id']): r for r in records if r['dataset'] == source and r['variant'] == before}
            b = {(r['seed'], r['sample_id']): r for r in records if r['dataset'] == source and r['variant'] == after}
            common = sorted(set(a)&set(b))
            if not common:
                continue
            for effect, strength in conditions:
                keys = [k for k in common if effect == 'all' or (a[k]['degradation'] == effect and a[k]['severity'] == strength)]
                seeds = sorted({k[0] for k in keys})
                paired.append({'dataset': source, 'factor': factor, 'before': before, 'after': after,
                               'degradation': effect, 'severity': strength, 'seeds': seeds,
                               'delta_pp': float(np.mean([int(b[k]['success'])-int(a[k]['success']) for k in keys])*100),
                               'paired_cluster_interval': paired_interval([b[k] for k in keys], [a[k] for k in keys]),
                               'scope': '全3seedの対応比較' if len(seeds) == 3 else '探索：反復未完了'})
    # 幾何は1回の推定を同じCNN seedへ対応付け、独立観測が増えたとは解釈しない。
    comparisons = []
    for source in ('1194m', 'diversified'):
        for variant in config['variants']:
            candidate = [r for r in records if r['dataset'] == source and r['variant'] == variant]
            if not candidate:
                continue
            for reference in ('canny', 'zhang2019', 'main_1194m'):
                if variant == reference:
                    continue
                baseline = [r for r in records if r['dataset'] == source and r['variant'] == reference]
                if not baseline:
                    continue
                if reference in ('canny', 'zhang2019'):
                    baseline = [{**r, 'seed': seed} for seed in sorted({r['seed'] for r in candidate}) for r in baseline]
                lookup = {(r['seed'], r['sample_id']): r for r in baseline}
                for effect, severity in conditions:
                    rows = [r for r in candidate if (r['seed'], r['sample_id']) in lookup and
                            (effect == 'all' or (r['degradation'] == effect and r['severity'] == severity))]
                    if not rows:
                        continue
                    ref = [lookup[r['seed'], r['sample_id']] for r in rows]
                    comparisons.append({'dataset': source, 'variant': variant, 'reference': reference,
                                        'degradation': effect, 'severity': severity,
                                        'seeds': sorted({r['seed'] for r in rows}), **paired_interval(rows, ref)})
    write_json(OUT/'summary.json', {'rows': summaries, 'paired_effects': paired,
                                   'factorial_interactions': factorial_interactions(records, conditions),
                                   'reference_comparisons': comparisons, 'prediction_checks': checks,
                                   'geometry_prediction_checks': geometry_checks,
                                   'incomplete_or_unstarted': incomplete, 'files': files,
                                   'interval_implementation_sha256': sha(ROOT/'research/paf_generalization/report.py'),
                                   'plan_sha256': sha(OUT/'plan.json'), 'report_implementation_sha256': sha(Path(__file__))})
    lines = ['# mainを起点とする投影・外観の寄与', '',
             'mainの単一CNN・完全リング教師・閾値点群＋RANSACを保持した2要因比較。学習回数・画像・分割は現行共通条件へ統一する。', '',
             '未完了のモデルは平均に含めない。seed数を明示し、1seedを最終結論に使わない。', '',
             '|評価対象|構成|seed数|全条件|通常|半画面遮蔽|白飛び0.5|黒つぶれ0.5|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for source in ('1194m', 'diversified'):
        for variant in LABELS:
            selected = [r for r in summaries if r['dataset'] == source and r['variant'] == variant]
            if not selected:
                continue
            values = []
            for effect, severity in [('all', None), ('clean', 0.), ('black_rectangle', .5), ('sensor_whiteout', .5), ('sensor_black_crush', .5)]:
                row = next(r for r in selected if r['degradation'] == effect and r['severity'] == severity)
                values.append(f"{row['success_rate']*100:.2f}%")
            seed_count = '—（1回）' if selected[0]['seeds'] == [None] else str(len(selected[0]['seeds']))
            lines.append('|'+ '|'.join([source, LABELS[variant], seed_count]+values)+'|')
    lines += ['', f'元ラベル照合：CNN {checks:,}推定、固定幾何 {geometry_checks:,}推定。全条件別の値と同じseedでの要因差はsummary.json。', '',
              '対応区間はカメラ/形状群を再標本化した95%区間。学習seed母集団や探索の多重性を含まない。幾何の推定をseed数倍の独立画像とは数えない。', '',
              '既存testは参照済みで盲検ではない。1194M学習の参照以外は1194Mをtrain/validation/初期化に使わない。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print({'prediction_checks': checks, 'completed_models': checks//3458, 'incomplete_or_unstarted': len(incomplete)}, flush=True)


if __name__ == '__main__':
    main()
