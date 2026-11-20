"""可視境界CNNを共通にした要因対照を、原ラベル照合後に対応集計する。"""

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
from research.paf_core_ablation.boundary_core import OUT, CONFIG, verify_references

BASE = ROOT/'output/hole_region_comparison_v1/datasets'
LABELS = {'core_base': '可視境界CNN：元の学習分布', 'core_surface': '＋外観',
          'core_projection': '＋投影', 'boundary_projection_mix': '＋投影＋外観',
          'boundary_projection_relative': '＋投影＋外観＋開口幅遮蔽',
          'main_1194m': '参考：mainリング構成・1194M学習（60epoch）',
          'canny': 'Canny＋輪郭別RANSAC', 'zhang2019': 'Zhang 2019型（再現実装）'}
CONDITIONS = [('all', None), ('clean', 0.)]+[(effect, severity) for effect in
    ('black_rectangle', 'sensor_whiteout', 'sensor_black_crush') for severity in (.25, .5, .75, 1.)]
DISPLAY = [('all', None), ('clean', 0.), ('black_rectangle', .5),
           ('sensor_whiteout', .5), ('sensor_black_crush', .5)]
FACTORS = [('core_base', 'boundary_projection_mix', '投影＋外観の同時変更'),
           ('core_base', 'core_projection', '投影：外観なし'),
           ('core_surface', 'boundary_projection_mix', '投影：外観あり'),
           ('core_base', 'core_surface', '外観：投影なし'),
           ('core_projection', 'boundary_projection_mix', '外観：投影あり'),
           ('boundary_projection_mix', 'boundary_projection_relative', '開口幅遮蔽：投影・外観あり')]


def in_condition(row, effect, severity):
    return effect == 'all' or (row['degradation'] == effect and row['severity'] == severity)


def collect():
    """完了した学習と両testのみ取り込み、未完了の重みは性能に含めない。"""
    verify_references()
    config = read_json(CONFIG)
    metadata, truth, files, checked = {}, {}, {}, set()

    def check_file(path, digest=None):
        path = Path(path).resolve()
        if path not in checked:
            files[str(path.relative_to(ROOT))] = sha(path)
            checked.add(path)
        if digest is not None:
            assert files[str(path.relative_to(ROOT))] == digest, str(path)

    for source in ('1194m', 'diversified'):
        folder = BASE/source
        check_file(folder/'manifest.json')
        # 元の入力画像・ラベルを保存時hashと照合する。
        for sample in read_json(folder/'manifest.json')['samples']:
            for key in ('image', 'label'):
                check_file(folder/sample[key], sample[key+'_sha256'])
        metadata[source] = {s['sample_id']: s for s in load_samples(folder, 'test')}
        truth[source] = {}
        for key, sample in metadata[source].items():
            label = read_json(folder/sample['label'])
            truth[source][key] = scaled_ellipse(label_ellipse(label), 256/label['image_width'])

    def checked_rows(result, source, variant, seed):
        rows = result['rows']
        assert len(rows) == len(truth[source])
        assert {r['sample_id'] for r in rows} == set(truth[source])
        output = []
        for row in rows:
            key = row['sample_id']
            sample, target = metadata[source][key], truth[source][key]
            assert row.get('group_id', sample['group_id']) == sample['group_id']
            assert row['degradation'] == sample['conditions']['degradation']
            assert row['severity'] == sample['conditions']['severity']
            saved = row['truth_ellipse']
            assert np.allclose([*saved[0], *saved[1], saved[2]], [*target[0], *target[1], target[2]])
            iou = evaluate_ellipses(row['ellipse'], target, (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
            assert abs(iou-row['ellipse_iou']) < 1e-10 and row['success'] == (iou >= .8)
            output.append({**row, 'variant': variant, 'seed': seed, 'dataset': source,
                           'group_id': sample['group_id']})
        return output

    records, incomplete, models = [], [], []
    for variant in LABELS:
        if variant in ('canny', 'zhang2019'):
            continue
        if variant.startswith('core_'):
            root, seeds = OUT, config['seeds']
        elif variant == 'main_1194m':
            root, seeds = ROOT/'output/paf_core_ablation_v1', [20260907]
        else:
            root, seeds = ROOT/'output/paf_generalization_v2', config['seeds']
        for seed in seeds:
            folder = root/'runs'/variant/str(seed)
            names = ('best.pt', 'complete.json', 'training_history.json', 'provenance.json',
                     'test_1194m.json', 'test_diversified.json')
            if not all((folder/name).exists() for name in names):
                incomplete.append({'variant': variant, 'seed': seed})
                continue
            completed, history, provenance = (read_json(folder/name) for name in names[1:4])
            assert completed['epochs'] == config['epochs'] == len(history)
            assert [r['epoch'] for r in history] == list(range(1, config['epochs']+1))
            checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
            best = min(history, key=lambda r: r['validation_loss'])
            assert checkpoint['epoch'] == best['epoch']
            assert checkpoint['validation_loss'] == completed['best_validation_loss'] == best['validation_loss']
            assert checkpoint['provenance'] == provenance
            assert all(torch.isfinite(value).all() for value in checkpoint['model_state'].values())
            assert provenance['variant'] == variant and provenance['seed'] == seed
            is_target = variant == 'main_1194m'
            assert provenance['training_source'] == ('1194m' if is_target else 'diversified')
            assert provenance['target_data_used_for_training'] == is_target
            if is_target:
                check_file(root/'plan.json', provenance['plan_sha256'])
                frozen = read_json(root/'plan.json')
                for path, digest in frozen['input_files'].items():
                    check_file(Path(path), digest)
            else:
                frozen = provenance
            for path, digest in frozen['files'].items():
                check_file(ROOT/path, digest)
            for name in names:
                check_file(folder/name)
            check_file(folder/'best.pt', completed['checkpoint_sha256'])
            if not is_target:
                for key in ('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'positive_weight', 'ransac'):
                    assert provenance['config'][key] == config[key], (variant, key)
                for source in ('1194m', 'diversified'):
                    check_file(BASE/source/'manifest.json', provenance['manifests'][source])
                for key in ('projection_files', 'surface_data_hashes', 'occlusion_sampling_files'):
                    for path, digest in provenance['config'].get(key, {}).items():
                        check_file(ROOT/path, digest)
                for key, path in {
                    'boundary_surface': 'research/paf_generalization/boundary_surface.py',
                    'boundary': 'research/paf_generalization/boundary.py',
                    'ransac': 'paf_ring_detection/methods/ransac.py',
                }.items():
                    check_file(ROOT/path, provenance['config'][key+'_implementation_sha256'])
                if variant.startswith('core_'):
                    assert provenance['config']['factor_mapping_implementation_sha256'] == sha(CONFIG.with_suffix('.py'))
            for source in ('1194m', 'diversified'):
                result = read_json(folder/f'test_{source}.json')
                assert result['checkpoint_sha256'] == completed['checkpoint_sha256']
                assert result['provenance'] == provenance
                records.extend(checked_rows(result, source, variant, seed))
            models.append({'variant': variant, 'seed': seed, 'selected_epoch': best['epoch'],
                           'validation_loss': best['validation_loss']})
    cnn_checks = len(records)
    for source in ('1194m', 'diversified'):
        for variant in ('canny', 'zhang2019'):
            path = BASE.parent/f'results/{source}/{variant}_fixed.json'
            result = read_json(path)
            check_file(path)
            check_file(BASE/source/'manifest.json', result['manifest_sha256'])
            assert result['input_size'] == 256 and result['limited'] is False
            records.extend(checked_rows(result, source, variant, None))
    return records, {'prediction_checks': cnn_checks, 'geometry_prediction_checks': len(records)-cnn_checks,
                     'models': models, 'incomplete_or_unstarted': incomplete, 'files': files,
                     'references_sha256': sha(OUT/'references.json'),
                     'report_implementation_sha256': sha(Path(__file__)),
                     'interval_implementation_sha256': sha(ROOT/'research/paf_generalization/report.py')}


def aggregate(records):
    """同じseed・同じ画像だけで要因差を測り、参考CNNは因果対照と分ける。"""
    summaries, paired, interactions, comparisons = [], [], [], []
    for source in ('1194m', 'diversified'):
        lookup = {v: {(r['seed'], r['sample_id']): r for r in records
                      if r['dataset'] == source and r['variant'] == v} for v in LABELS}
        for variant, table in lookup.items():
            if not table:
                continue
            for effect, severity in CONDITIONS:
                rows = [r for r in table.values() if in_condition(r, effect, severity)]
                seeds = sorted({r['seed'] for r in rows})
                summaries.append({'dataset': source, 'variant': variant, 'degradation': effect, 'severity': severity,
                                  'seeds': seeds, 'n_images': len({r['sample_id'] for r in rows}),
                                  'success_rate': float(np.mean([r['success'] for r in rows])),
                                  'mean_ellipse_iou': float(np.mean([r['ellipse_iou'] for r in rows])),
                                  'per_seed': {str(s): float(np.mean([r['success'] for r in rows if r['seed'] == s])) for s in seeds}})
        for before, after, factor in FACTORS:
            a, b = lookup[before], lookup[after]
            common = sorted(set(a)&set(b))
            if not common:
                continue
            for effect, severity in CONDITIONS:
                keys = [k for k in common if in_condition(a[k], effect, severity)]
                seeds = sorted({k[0] for k in keys})
                paired.append({'dataset': source, 'factor': factor, 'before': before, 'after': after,
                               'degradation': effect, 'severity': severity, 'seeds': seeds,
                               'n_images': len({k[1] for k in keys}),
                               'scope': '3seed対応' if len(seeds) == 3 else '探索・反復未完了',
                               **paired_interval([b[k] for k in keys], [a[k] for k in keys])})
        cells = [lookup[v] for v in ('core_base', 'core_projection', 'core_surface', 'boundary_projection_mix')]
        common = sorted(set.intersection(*(set(cell) for cell in cells)))
        if common:
            for effect, severity in CONDITIONS:
                keys = [k for k in common if in_condition(cells[0][k], effect, severity)]
                with_projection, without_projection = [], []
                iou_with_projection, iou_without_projection = [], []
                for key in keys:
                    original, projected, surface, both = [c[key] for c in cells]
                    with_projection.append({**original, 'success': float(both['success'])-float(projected['success'])})
                    without_projection.append({**original, 'success': float(surface['success'])-float(original['success'])})
                    # 閾値による成功/失敗の分離だけで交互作用が見えていないかを確認する。
                    iou_with_projection.append({**original, 'success': both['ellipse_iou']-projected['ellipse_iou']})
                    iou_without_projection.append({**original, 'success': surface['ellipse_iou']-original['ellipse_iou']})
                interactions.append({'dataset': source, 'degradation': effect, 'severity': severity,
                                     'seeds': sorted({k[0] for k in keys}),
                                     'formula': '(both - projection) - (surface - base)',
                                     'mean_iou_interaction': paired_interval(iou_with_projection, iou_without_projection),
                                     **paired_interval(with_projection, without_projection)})
        for variant in list(LABELS)[:5]:
            for reference in ('canny', 'zhang2019', 'main_1194m'):
                a, b = lookup[variant], lookup[reference]
                if reference != 'main_1194m':
                    # 決定済み幾何の同じ推定を各CNN seedへ対応させる。観測数は増やさない。
                    b = {(seed, key[1]): {**r, 'seed': seed} for seed in {k[0] for k in a} for key, r in b.items()}
                common = sorted(set(a)&set(b))
                if not common:
                    continue
                for effect, severity in CONDITIONS:
                    keys = [k for k in common if in_condition(a[k], effect, severity)]
                    comparisons.append({'dataset': source, 'variant': variant, 'reference': reference,
                                        'degradation': effect, 'severity': severity, 'seeds': sorted({k[0] for k in keys}),
                                        **paired_interval([a[k] for k in keys], [b[k] for k in keys])})
    return {'rows': summaries, 'paired_effects': paired, 'factorial_interactions': interactions,
            'reference_comparisons': comparisons}


def main():
    records, audit = collect()
    summary = aggregate(records)
    write_json(OUT/'summary.json', {**summary, 'audit': audit})
    lines = ['# 可視境界CNNを基礎とする投影・外観の寄与', '',
             '仮説：適切な内周表現を共通にし、投影・外観の学習分布を変えると機体固有の見え方への依存が緩和する。', '',
             'CNN、可視境界教師、中心線RANSAC、学習回数・損失・validation選択を固定した2×2比較。外観は模様と平均反射色を含む。', '',
             '成功率はIoU≥0.8、全画像を分母に含む。3seed平均はモデル統合ではない。未完了モデルは含めず、1seedを最終結論としない。', '',
             '|対象|構成|seed数|全条件|通常|半画面遮蔽|白飛び0.5|黒つぶれ0.5|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for source in ('1194m', 'diversified'):
        for variant in LABELS:
            rows = [r for r in summary['rows'] if r['dataset'] == source and r['variant'] == variant]
            if not rows:
                continue
            values = [next(r for r in rows if r['degradation'] == e and r['severity'] == s) for e, s in DISPLAY]
            n = '—（1回）' if rows[0]['seeds'] == [None] else str(len(rows[0]['seeds']))
            lines.append('|'+ '|'.join([source, LABELS[variant], n]+[f"{r['success_rate']*100:.2f}%" for r in values])+'|')
    lines += ['', '## 同じseedでの要因差', '',
              '|対象|要因|seed数|全条件差pp|通常差pp|半画面遮蔽差pp|白飛び差pp|黒つぶれ差pp|',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for source in ('1194m', 'diversified'):
        for _, _, factor in FACTORS:
            rows = [r for r in summary['paired_effects'] if r['dataset'] == source and r['factor'] == factor]
            if rows:
                values = [next(r for r in rows if r['degradation'] == e and r['severity'] == s) for e, s in DISPLAY]
                lines.append('|'+ '|'.join([source, factor, str(len(rows[0]['seeds']))]+[f"{r['delta']*100:+.2f}" for r in values])+'|')
    if summary['factorial_interactions']:
        lines += ['', '## 成功率上の交互作用', '',
                  '差の差＝（投影ありでの外観の効果）−（投影なしでの外観の効果）。同じ4セル・同じseed・同じ画像で計算する。正ならこの尺度で併用の利点が大きい。', '',
                  '|対象|seed数|全条件pp|通常pp|半画面遮蔽pp|白飛びpp|黒つぶれpp|',
                  '|---|---:|---:|---:|---:|---:|---:|']
        iou_notes = []
        for source in ('1194m', 'diversified'):
            rows = [r for r in summary['factorial_interactions'] if r['dataset'] == source]
            values = [next(r for r in rows if r['degradation'] == e and r['severity'] == s) for e, s in DISPLAY]
            lines.append('|'+ '|'.join([source, str(len(rows[0]['seeds']))]+[f"{r['delta']*100:+.2f}" for r in values])+'|')
            all_row = values[0]
            iou = all_row['mean_iou_interaction']
            iou_notes.extend(['', f"{source}の全条件・平均IoUでの差の差：{iou['delta']:+.4f}（画像群95%区間 {iou['ci95'][0]:+.4f}〜{iou['ci95'][1]:+.4f}）。"])
        lines += iou_notes
        lines += ['', '連続値IoUでの差の差もsummary.jsonに保存する。成功率上の交互作用から、CNN内部の学習機構や一般的なPAF概念の獲得を断定しない。']
    lines += ['', '全条件別の対応差・95%区間・交互作用はsummary.json。区間はカメラ/形状群の再標本化であり、学習seed母集団・探索の多重性を含まない。', '',
              'main参考CNNは完全リング・帯RANSAC・cosine学習率を使う別構成で、60epochの同じ評価集合による参考値。main保存結果そのものや単一要因対照ではない。', '',
              'main参考以外のCNNは1194Mをtrain・validation・初期化に使用しない。既存testは参照済みで盲検ではない。', '',
              f"元GT照合：CNN {audit['prediction_checks']:,}推定、幾何 {audit['geometry_prediction_checks']:,}推定。",
              '実装・入力・重みの照合hashと未完了構成はsummary.jsonのaudit。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print({'models': len(audit['models']), 'prediction_checks': audit['prediction_checks'],
           'incomplete': audit['incomplete_or_unstarted']}, flush=True)


if __name__ == '__main__':
    main()
