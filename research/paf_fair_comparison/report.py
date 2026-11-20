"""公平比較の全推定を元GTから照合し、平均・各seed・全強度の順位を保存する。"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.report import paired_interval
from research.paf_fair_comparison.experiment import OUT, BASE, CONFIG, REFERENCE, sha, model_folder, prepare

METHODS = {'1194m': '①1194M学習CNN', 'diversified': '②多様形状学習CNN',
           'canny': '③Canny＋輪郭別RANSAC', 'zhang2019': '③Zhang 2019型（再現実装）'}
SHORT = {'1194m': '①', 'diversified': '②', 'canny': '③Canny', 'zhang2019': '③Zhang'}
EFFECTS = {'all': '全13条件', 'clean': '通常', 'black_rectangle': '矩形遮蔽',
           'sensor_whiteout': '白飛び', 'sensor_black_crush': '黒つぶれ'}
CONDITIONS = [('all', None), ('clean', 0.)]+[(effect, severity) for effect in
    ('black_rectangle', 'sensor_whiteout', 'sensor_black_crush') for severity in (.25, .5, .75, 1.)]


def select(rows, effect, severity):
    return [r for r in rows if effect == 'all' or (r['degradation'] == effect and r['severity'] == severity)]


def ranking(values):
    """丸める前の成功数に相当する値で同率を判定する。"""
    groups = []
    for key in sorted(values, key=lambda k: -values[k]):
        if groups and abs(values[key]-values[groups[-1][0]]) < 1e-12:
            groups[-1].append(key)
        else:
            groups.append([key])
    return ' > '.join(' = '.join(SHORT[k] for k in group) for group in groups)


def main():
    torch.set_num_threads(4)
    plan = prepare()
    config = read_json(CONFIG)
    hashes, models, records, changes = {}, [], [], []
    metadata, truth = {}, {}
    for dataset in config['sources']:
        metadata[dataset] = {s['sample_id']: s for s in load_samples(BASE/dataset, 'test')}
        truth[dataset] = {}
        for key, sample in metadata[dataset].items():
            label = read_json(BASE/dataset/sample['label'])
            truth[dataset][key] = scaled_ellipse(label_ellipse(label), 256/label['image_width'])

    def checked_rows(result, dataset, method, seed):
        rows = result['rows']
        assert len(rows) == len(truth[dataset])
        assert {r['sample_id'] for r in rows} == set(truth[dataset])
        for row in rows:
            sample = metadata[dataset][row['sample_id']]
            target = truth[dataset][row['sample_id']]
            saved = row['truth_ellipse']
            assert np.allclose([*saved[0], *saved[1], saved[2]], [*target[0], *target[1], target[2]])
            for field in ('degradation', 'severity'):
                assert row[field] == sample['conditions'][field]
            assert row.get('group_id', sample['group_id']) == sample['group_id']
            iou = evaluate_ellipses(row['ellipse'], target, (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
            assert abs(row['ellipse_iou']-iou) < 1e-10 and row['success'] == (iou >= .8)
            records.append({**row, 'method': method, 'seed': seed, 'dataset': dataset, 'group_id': sample['group_id']})

    def check(path):
        hashes[str(path.relative_to(ROOT))] = sha(path)

    checks_path = OUT/'implementation_checks.json'
    implementation = read_json(checks_path)
    assert implementation['same_getitem_function'] and implementation['matched_augmentation_parameters'] == 497
    assert implementation['max_projection_error_px'] < .01
    assert [r['seed'] for r in implementation['training_checks']] == config['seeds']
    assert all(r['same_training_input_and_target'] == 128 and r['view_rng_equal'] and r['mask_rng_equal']
               for r in implementation['training_checks'])
    for relative, digest in implementation['files'].items():
        assert sha(ROOT/relative) == digest
        check(ROOT/relative)
    check(checks_path)
    for source in config['sources']:
        for seed in config['seeds']:
            folder = model_folder(source, seed)
            checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
            history = read_json(folder/'training_history.json')
            completed = read_json(folder/'complete.json')
            provenance = read_json(folder/'provenance.json')
            assert len(history) == completed['epochs'] == config['epochs']
            assert [r['epoch'] for r in history] == list(range(1, config['epochs']+1))
            assert all(np.isfinite(r['train_loss']) and np.isfinite(r['validation_loss']) for r in history)
            best = min(history, key=lambda r: r['validation_loss'])
            assert checkpoint['epoch'] == best['epoch']
            assert checkpoint['validation_loss'] == best['validation_loss'] == completed['best_validation_loss']
            assert checkpoint['provenance'] == provenance and provenance['training_source'] == source
            assert provenance['target_data_used_for_training'] == (source == '1194m')
            assert all(torch.isfinite(value).all() for value in checkpoint['model_state'].values())
            assert sha(folder/'best.pt') == completed['checkpoint_sha256']
            if source == '1194m':
                assert provenance['plan_sha256'] == sha(OUT/'plan.json')
            else:
                assert provenance['torch'] == plan['torch'] and provenance['device'] == plan['device']
            models.append({'training_source': source, 'seed': seed, 'selected_epoch': best['epoch'],
                           'validation_loss': best['validation_loss'], 'checkpoint_sha256': completed['checkpoint_sha256']})
            for name in ('best.pt', 'training_history.json', 'complete.json', 'provenance.json'):
                check(folder/name)
            for dataset in config['sources']:
                path = OUT/f'results/{source}/{seed}/test_{dataset}.json'
                result = read_json(path)
                assert result['checkpoint_sha256'] == completed['checkpoint_sha256']
                assert result['plan_sha256'] == sha(OUT/'plan.json')
                assert result['training_source'] == source and result['seed'] == seed and result['dataset'] == dataset
                check(path)
                checked_rows(result, dataset, source, seed)
                if source == 'diversified':
                    old_path = REFERENCE/str(seed)/f'test_{dataset}.json'
                    old = {r['sample_id']: r for r in read_json(old_path)['rows']}
                    changes.append({'seed': seed, 'dataset': dataset,
                        'success_changed': sum(r['success'] != old[r['sample_id']]['success'] for r in result['rows']),
                        'max_iou_difference': max(abs(r['ellipse_iou']-old[r['sample_id']]['ellipse_iou']) for r in result['rows']),
                        'changed_success_cases': [{'sample_id': r['sample_id'], 'degradation': r['degradation'],
                            'severity': r['severity'], 'previous_iou': old[r['sample_id']]['ellipse_iou'],
                            'common_evaluation_iou': r['ellipse_iou']} for r in result['rows']
                            if r['success'] != old[r['sample_id']]['success']]})
                    check(old_path)
    geometry_provenance_path = BASE.parent/'provenance.json'
    geometry_provenance = read_json(geometry_provenance_path)
    check(geometry_provenance_path)
    for relative, digest in geometry_provenance['sha256'].items():
        path = ROOT/relative
        assert sha(path) == digest, relative
        check(path)
    for dataset in config['sources']:
        for method in ('canny', 'zhang2019'):
            path = BASE.parent/f'results/{dataset}/{method}_fixed.json'
            result = read_json(path)
            assert result['input_size'] == 256 and result['limited'] is False
            assert result['method'] == method and result['split'] == 'test'
            assert result['config'] == read_json(ROOT/'config/experiment.json')
            assert result['evaluator_sha256'] == sha(ROOT/'paf_ring_detection/evaluate.py')
            assert result['manifest_sha256'] == sha(BASE/dataset/'manifest.json')
            check(path)
            checked_rows(result, dataset, method, None)
    assert len(records) == 27664
    summaries, rankings, seed_rankings, differences = [], [], [], []
    for dataset in config['sources']:
        table = {m: [r for r in records if r['dataset'] == dataset and r['method'] == m] for m in METHODS}
        for effect, severity in CONDITIONS:
            values, ious = {}, {}
            for method in METHODS:
                rows = select(table[method], effect, severity)
                seeds = [None] if method in ('canny', 'zhang2019') else config['seeds']
                assert set(r['seed'] for r in rows) == set(seeds)
                counts = {str(seed): sum(r['seed'] == seed for r in rows) for seed in seeds}
                assert len(set(counts.values())) == 1
                per_seed = {str(seed): float(np.mean([r['success'] for r in rows if r['seed'] == seed])) for seed in seeds}
                value, iou = float(np.mean([r['success'] for r in rows])), float(np.mean([r['ellipse_iou'] for r in rows]))
                values[method], ious[method] = value, iou
                summaries.append({'dataset': dataset, 'method': method, 'degradation': effect, 'severity': severity,
                    'n_images': next(iter(counts.values())), 'seeds': seeds, 'success_rate': value,
                    'mean_iou': iou, 'per_seed': per_seed,
                    'per_seed_success_count': {str(seed): sum(r['success'] for r in rows if r['seed'] == seed) for seed in seeds}})
            expected = all(values['1194m'] > values['diversified'] > values[g] for g in ('canny', 'zhang2019')) if dataset == '1194m' else all(values['diversified'] > values[g] > values['1194m'] for g in ('canny', 'zhang2019'))
            rankings.append({'dataset': dataset, 'degradation': effect, 'severity': severity,
                'success_order': ranking(values), 'mean_iou_order': ranking(ious),
                'requested_strict_order_holds': expected})
            for seed in config['seeds']:
                by_seed = {}
                for method in METHODS:
                    rows = [r for r in select(table[method], effect, severity) if r['seed'] in (seed, None)]
                    by_seed[method] = float(np.mean([r['success'] for r in rows]))
                seed_expected = all(by_seed['1194m'] > by_seed['diversified'] > by_seed[g] for g in ('canny', 'zhang2019')) if dataset == '1194m' else all(by_seed['diversified'] > by_seed[g] > by_seed['1194m'] for g in ('canny', 'zhang2019'))
                seed_rankings.append({'dataset': dataset, 'degradation': effect, 'severity': severity,
                    'seed': seed, 'success_order': ranking(by_seed), 'requested_strict_order_holds': seed_expected})
            differences.append({'dataset': dataset, 'degradation': effect, 'severity': severity,
                'comparison': 'diversified_minus_1194m',
                **paired_interval(select(table['diversified'], effect, severity), select(table['1194m'], effect, severity))})
    summary = {'rows': summaries, 'rankings': rankings, 'seed_rankings': seed_rankings, 'paired_differences': differences,
               'metric': 'IoU >= 0.8; all test images; mean of three independent seeds, no ensemble'}
    write_json(OUT/'summary.json', summary)
    write_json(OUT/'audit.json', {'prediction_checks': len(records), 'cnn_predictions': 20748,
        'geometry_predictions': 6916, 'models': models, 'files': hashes, 'plan_sha256': sha(OUT/'plan.json'),
        'report_sha256': sha(Path(__file__)), 'reevaluation_of_diversified': changes,
        'frozen_file_checks': len(plan['files']), 'incomplete_runs': []})
    with (OUT/'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['dataset', 'method', 'degradation', 'severity', 'n_images', 'success_rate', 'mean_iou']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    lines = ['# 学習元だけを変更した共通境界CNNの比較', '',
        '両CNNは可視境界教師、中心線RANSAC、投影・外観4ビューと開口幅遮蔽、学習条件を共通化した。'
        '各3seed平均で比較し、幾何2方式は固定する。成功はIoU≥0.8。', '',
        'validationは各学習元735例の重み選択用。以下は別の未学習test集合で、1194M1,456例、多様形状2,002例。'
        'testは研究中に参照済みで盲検ではない。全13条件は通常1条件＋3劣化×4強度を同数ずつ含む。', '']
    for dataset in config['sources']:
        lines += [f'## {dataset} test', '', '|条件|①1194M学習|②多様形状学習|③Canny|③Zhang|成功率の順位|',
                  '|---|---:|---:|---:|---:|---|']
        for effect, severity in CONDITIONS:
            selected = {r['method']: r for r in summaries if r['dataset'] == dataset and r['degradation'] == effect and r['severity'] == severity}
            label = EFFECTS[effect]+(f' {severity:g}' if severity else '')
            order = next(r['success_order'] for r in rankings if r['dataset'] == dataset and r['degradation'] == effect and r['severity'] == severity)
            lines.append('|'+label+'|'+'|'.join(f"{selected[m]['success_rate']*100:.2f}%" for m in METHODS)+'|'+order+'|')
        lines += ['', 'CNNの各seed（20260907/08/09）の全条件成功率：', '']
        for method in ('1194m', 'diversified'):
            row = next(r for r in summaries if r['dataset'] == dataset and r['method'] == method and r['degradation'] == 'all')
            lines.append(f"- {METHODS[method]}："+' / '.join(f'{v*100:.2f}%' for v in row['per_seed'].values()))
        lines.append('')
    lines += ['## 全条件の平均IoU', '', '|評価集合|①1194M学習|②多様形状学習|③Canny|③Zhang|平均IoUの順位|',
              '|---|---:|---:|---:|---:|---|']
    for dataset in config['sources']:
        selected = {r['method']: r for r in summaries if r['dataset'] == dataset and r['degradation'] == 'all'}
        order = next(r['mean_iou_order'] for r in rankings if r['dataset'] == dataset and r['degradation'] == 'all')
        lines.append('|'+dataset+'|'+'|'.join(f"{selected[m]['mean_iou']:.4f}" for m in METHODS)+'|'+order+'|')
    lines += ['', '主指標はIoU≥0.8の成功率。平均IoUでは順位が異なる場合があるため、指標を明示して解釈する。', '',
        '## 解釈の範囲', '',
        '①はmainのGaussianリング＋帯点群モデルではなく、②と同じ基礎で1194Mから学習したモデル。'
        '②の既存3seed重みは共通条件と入力・教師の一致を確認して参照し、全6モデルを同じ評価関数で採点した。', '',
        '評価は採用済み単体推論のOpenCV乱数固定・一様画像棄却にも揃えた。②の重みは同じでも、'
        '従来の保存集計とは一部の推定が変わる。変更例はaudit.jsonのreevaluation_of_diversifiedに記録する。', '',
        'ここで変えたのは学習データ源一式である。元データの形状、撮影・照明・材質分布の差は残るため、'
        '形状多様性だけの因果効果とはしない。全画像は黒背景・単一開口CGで、任意機体・実画像は未検証。', '',
        '幾何はCanny＋輪郭別RANSACとZhang 2019型（再現実装）。Zhangの正規出典はSensors 2019, '
        'DOI 10.3390/s19235243。著者コードの移植ではない。', '',
        '入力・分割・実装はplan.json、全27,664推定の元GT照合と最小validation checkpoint確認はaudit.json。'
        '各seed、平均IoU、同率を含む順位、CNN間の対応差はsummary.json。差の95%区間は固定3seedでの画像群bootstrapであり、'
        '学習seed母集団の不確実性は含まない。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('AUDITED', len(records), 'predictions;', len(models), 'models', flush=True)
    for row in rankings:
        if row['degradation'] in ('all', 'clean'):
            print(row, flush=True)


if __name__ == '__main__':
    main()
