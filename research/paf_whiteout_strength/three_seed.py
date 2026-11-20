"""固定した白飛び条件を3学習seedで評価し、単一seed結果と別に集計する。"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_whiteout_strength import run as single
from research.paf_whiteout_strength.report import baseline, NAMES, COLORS, MARKERS, DS
from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.boundary import boundary_ellipse

OUT = ROOT/'output/paf_whiteout_strength_3seed_v1'
SEEDS = (20260907, 20260908, 20260909)
SEVERITIES = (.25, .5, .75, 1.)
CNN = ('1194m', 'diversified')


def prepare():
    """強度選択・既存推定・全重みを固定し、1seed結果の保存先を変更しない。"""
    previous = read_json(single.OUT/'plan.json')
    calibration = read_json(single.OUT/'calibration.json')
    assert calibration['plan_sha256'] == single.sha(single.OUT/'plan.json')
    assert calibration['selected_maximum_strength'] == 2.
    for relative, digest in previous['files'].items():
        assert single.sha(ROOT/relative) == digest, relative
    files = [Path(__file__), single.OUT/'plan.json', single.OUT/'calibration.json', single.OUT/'audit.json']
    files += list((single.OUT/'results/test').glob('*.json'))
    for seed in SEEDS:
        files += list(single.model_paths(seed).values())
        for dataset in single.DATASETS:
            for method in CNN:
                files.append(baseline(dataset, method, seed)[1])
    for dataset in single.DATASETS:
        for method in ('canny', 'zhang2019'):
            files.append(baseline(dataset, method, SEEDS[0])[1])
    plan = {'seeds': list(SEEDS), 'maximum_strength': 2., 'test_severities': list(SEVERITIES),
            'calibration_seed': SEEDS[0], 'recalibrated': False, 'retrained': False,
            'geometry_seed_count': 1, 'files': {str(p.relative_to(ROOT)): single.sha(p) for p in files},
            'previous_files': previous['files']}
    path = OUT/'plan.json'
    if path.exists():
        assert read_json(path) == plan
    else:
        write_json(path, plan)
    return plan


def previous_result(dataset, severity):
    return single.OUT/f'results/test/{dataset}_max2_s{severity:g}.json'


def prediction_path(seed, dataset, severity):
    return OUT/f'results/{seed}/{dataset}_s{severity:g}.json'


def evaluate(plan):
    """残り2seedのCNNだけを追加評価し、幾何方式とseed07は保存済み推定を再利用する。"""
    settings = read_json(ROOT/'research/paf_fair_comparison/no_surface.json')
    models = {seed: single.load_models(seed) for seed in SEEDS[1:]}
    for dataset in single.DATASETS:
        for severity in SEVERITIES:
            paths = {seed: prediction_path(seed, dataset, severity) for seed in SEEDS[1:]}
            if all(path.exists() for path in paths.values()):
                for path in paths.values():
                    assert read_json(path)['plan_sha256'] == single.sha(OUT/'plan.json')
                continue
            reference = read_json(previous_result(dataset, severity))
            reference_hashes = {r['sample_id']: r['image_sha256'] for r in reference['rows']}
            rows = {seed: [] for seed in SEEDS[1:]}
            samples = single.samples_for(dataset, 'test', severity)
            for index, sample in enumerate(samples):
                image, truth = single.inputs(dataset, sample, severity, plan['maximum_strength'])
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                assert digest == reference_hashes[sample['sample_id']]
                tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
                for seed in SEEDS[1:]:
                    for method in CNN:
                        ellipse = None
                        if np.ptp(image) != 0:
                            with torch.inference_mode():
                                probability = torch.sigmoid(models[seed][method](tensor))[0, 0].cpu().numpy()
                            cv2.setRNGSeed(settings['opencv_seed_per_image'])
                            ellipse = boundary_ellipse(probability, settings['ransac'])
                        iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                        rows[seed].append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                                           'method': method, 'ellipse': ellipse, 'truth_ellipse': truth,
                                           'ellipse_iou': iou, 'success': iou >= .8, 'image_sha256': digest})
                if (index+1) % 40 == 0:
                    print(f'EVAL {dataset} severity={severity:g} {index+1}/{len(samples)}', flush=True)
            for seed, path in paths.items():
                result = {'seed': seed, 'dataset': dataset, 'severity': severity,
                          'maximum_strength': plan['maximum_strength'], 'plan_sha256': single.sha(OUT/'plan.json'),
                          'checkpoint_sha256': {m: single.sha(p) for m, p in single.model_paths(seed).items()},
                          'rows': rows[seed]}
                write_json(path, result)
                print('DONE', dataset, severity, seed,
                      {m: round(100*np.mean([r['success'] for r in rows[seed] if r['method'] == m]), 2) for m in CNN}, flush=True)


def report(plan):
    """全推定の正解・分母・seed対応を確認して平均と最小最大を保存する。"""
    records, checked, image_checks = [], 0, 0
    for dataset in single.DATASETS:
        for severity in (0.,)+SEVERITIES:
            truths, hashes = {}, {}
            if severity > 0:
                for sample in single.samples_for(dataset, 'test', severity):
                    image, truth = single.inputs(dataset, sample, severity, plan['maximum_strength'])
                    assert single.sha(single.BASE/dataset/sample['image']) == sample['image_sha256']
                    assert single.sha(single.BASE/dataset/sample['label']) == sample['label_sha256']
                    truths[sample['sample_id']] = truth
                    hashes[sample['sample_id']] = hashlib.sha256(image.tobytes()).hexdigest()
                    image_checks += 1
            for method in single.METHODS:
                seeds = SEEDS if method in CNN else (None,)
                for variant in ('original', 'enhanced'):
                    if severity == 0 and variant == 'enhanced':
                        continue
                    for seed in seeds:
                        if variant == 'original':
                            raw = baseline(dataset, method, seed or SEEDS[0])[0]['rows']
                            rows = [r for r in raw if (r['degradation'] == 'clean' if severity == 0 else
                                    r['degradation'] == 'sensor_whiteout' and r['severity'] == severity)]
                        else:
                            path = (previous_result(dataset, severity) if seed in (None, SEEDS[0])
                                    else prediction_path(seed, dataset, severity))
                            result = read_json(path)
                            if seed not in (None, SEEDS[0]):
                                assert result['plan_sha256'] == single.sha(OUT/'plan.json')
                                assert result['seed'] == seed
                            rows = [r for r in result['rows'] if r['method'] == method]
                        assert len(rows) == (112 if dataset == '1194m' else 154)
                        assert len({r['sample_id'] for r in rows}) == len(rows)
                        for row in rows:
                            if severity > 0:
                                truth = truths[row['sample_id']]
                                assert np.allclose([*row['truth_ellipse'][0], *row['truth_ellipse'][1], row['truth_ellipse'][2]],
                                                   [*truth[0], *truth[1], truth[2]])
                                iou = evaluate_ellipses(row['ellipse'], truth, (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
                                assert abs(iou-row['ellipse_iou']) < 1e-10 and row['success'] == (iou >= .8)
                                if variant == 'enhanced':
                                    assert row['image_sha256'] == hashes[row['sample_id']]
                                checked += 1
                        records.append({'dataset': dataset, 'severity': severity, 'method': method, 'variant': variant,
                                        'seed': seed, 'n': len(rows), 'successes': sum(r['success'] for r in rows),
                                        'success_percent': 100*np.mean([r['success'] for r in rows]),
                                        'mean_iou': np.mean([r['ellipse_iou'] for r in rows])})
    grouped = {}
    for row in records:
        key = (row['dataset'], row['severity'], row['method'], row['variant'])
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for (dataset, severity, method, variant), group in grouped.items():
        expected = 3 if method in CNN else 1
        assert len(group) == expected
        rates = [r['success_percent'] for r in group]
        assert len({r['n'] for r in group}) == 1
        mean = float(np.mean(rates))
        assert abs(mean-100*sum(r['successes'] for r in group)/sum(r['n'] for r in group)) < 1e-10
        aggregates.append({'dataset': dataset, 'severity': severity, 'method': method, 'variant': variant,
                           'seed_count': expected, 'n_per_seed': group[0]['n'], 'success_percent': mean,
                           'min_success_percent': min(rates), 'max_success_percent': max(rates),
                           'mean_iou': float(np.mean([r['mean_iou'] for r in group]))})
    write_json(OUT/'summary.json', {'seeds': list(SEEDS), 'maximum_strength': 2.,
                                   'records': records, 'aggregates': aggregates})
    for name, rows in [('scores_by_seed.csv', records), ('scores_mean.csv', aggregates)]:
        with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(OUT/'audit.json', {'plan_sha256': single.sha(OUT/'plan.json'),
                                 'prediction_checks_including_original': checked, 'shared_image_checks': image_checks,
                                 'additional_cnn_predictions': 4256, 'cnn_seed_count': 3, 'geometry_run_count': 1,
                                 'result_hashes': {str(p.relative_to(ROOT)): single.sha(p) for p in (OUT/'results').rglob('*.json')}})
    figures(aggregates)
    write_report(aggregates, records)
    print('AUDIT', checked, image_checks, flush=True)


def item(rows, dataset, method, variant, severity):
    return next(r for r in rows if r['dataset'] == dataset and r['method'] == method
                and r['variant'] == variant and r['severity'] == severity)


def figures(rows):
    """平均値を線で示し、CNNの3seed最小最大は信頼区間と区別して帯で示す。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Meiryo', 'font.size': 13, 'axes.unicode_minus': False})
    folder = OUT/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    xs = [0., .25, .5, .75, 1.]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, dataset in zip(axes, single.DATASETS):
        for method in single.METHODS:
            values = [item(rows, dataset, method, 'original' if s == 0 else 'enhanced', s) for s in xs]
            ax.plot(xs, [r['success_percent'] for r in values], label=NAMES[method], color=COLORS[method],
                    marker=MARKERS[method], linestyle='-' if method in CNN else '--', linewidth=2)
            if method in CNN:
                ax.fill_between(xs, [r['min_success_percent'] for r in values],
                                [r['max_success_percent'] for r in values], color=COLORS[method], alpha=.13)
        ax.set(title=DS[dataset], xlabel='白飛び強度', xticks=xs, ylim=(-2, 103))
        ax.grid(axis='y', color='#DDDDDD')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('検出成功率 [%]')
    fig.suptitle('白飛び強度の追加評価（CNN：3seed平均、再学習なし）', fontsize=17)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5,.04))
    fig.text(.5,.015,'帯：3seedの最小～最大（信頼区間ではない）　幾何方式：固定1回',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.2,1,.95])
    fig.savefig(folder/'whiteout_success_3seed.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for ri, dataset in enumerate(single.DATASETS):
        for ax, method in zip(axes[ri], single.METHODS):
            for variant, style, label in [('original','--','調整前'),('enhanced','-','調整後')]:
                values = [item(rows,dataset,method,'original' if s == 0 else variant,s) for s in xs]
                ax.plot(xs,[r['success_percent'] for r in values],style,color=COLORS[method],marker=MARKERS[method],
                        markerfacecolor='white' if variant == 'original' else COLORS[method],label=label)
            ax.set_title(NAMES[method].replace('（再現実装）','\n（再現実装）'),fontsize=12)
            ax.set(ylim=(-2,103),xticks=[0,.5,1])
            ax.grid(axis='y',color='#DDDDDD')
            ax.spines[['top','right']].set_visible(False)
            if ri == 1:
                ax.set_xlabel('白飛び強度')
        axes[ri,0].set_ylabel(DS[dataset]+'\n検出成功率 [%]')
    fig.suptitle('白飛び条件の調整前後（CNN：3seed平均、再学習なし）',fontsize=17)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.06,1,.95])
    fig.savefig(folder/'whiteout_before_after_3seed.png',dpi=170)
    plt.close(fig)
    write_json(folder/'figure_data.json',{'aggregates':rows,'source':'../summary.json','band':'min/max of 3 seeds'})


def write_report(rows, records):
    lines = ['# 白飛び強度の追加評価：3seed平均', '',
             '学習seedは20260907・20260908・20260909。既存の採用重みを固定し、再学習は行わない。',
             '最大内部強度2.0は1seedの検証集合で決定した値を固定し、再選択していない。',
             '1194M 112原画像・多様形状154原画像に、同じ乱数で白飛びを付与する。全4強度を評価し、成功条件はIoU≥0.8。未検出も失敗に含める。',
             'CNNは各seedの成功率の算術平均。幾何方式は学習seedに依存しない固定結果を1回分だけ示す。モデル統合ではない。', '',
             '## 白飛び強度1.0', '', '|対象|方式|調整前|調整後|調整後のseed最小～最大|', '|---|---|---:|---:|---:|']
    for dataset in single.DATASETS:
        for method in single.METHODS:
            before = item(rows,dataset,method,'original',1.)
            after = item(rows,dataset,method,'enhanced',1.)
            spread = f"{after['min_success_percent']:.2f}～{after['max_success_percent']:.2f}%" if method in CNN else '固定1回'
            lines.append(f"|{DS[dataset]}|{NAMES[method]}|{before['success_percent']:.2f}%|{after['success_percent']:.2f}%|{spread}|")
    lines += ['', '## 各seedの成功率：調整後の強度1.0', '', '|対象|学習元|20260907|20260908|20260909|', '|---|---|---:|---:|---:|']
    for dataset in single.DATASETS:
        for method in CNN:
            values = [next(r['success_percent'] for r in records if r['dataset']==dataset and r['method']==method
                           and r['variant']=='enhanced' and r['severity']==1. and r['seed']==seed) for seed in SEEDS]
            lines.append(f"|{DS[dataset]}|{NAMES[method]}|"+'|'.join(f'{v:.2f}%' for v in values)+'|')
    lines += ['', '## 根拠と範囲', '',
              'seed07と幾何方式は、入力・設定・推定ファイルのhashを照合して1seed実験から再利用する。追加2seedにも同一画像を渡す。',
              '強化した白飛びを学習に含めた場合や実撮像での性能は未検証。参照済みtestを使った追加評価であり、新規盲検ではない。',
              '帯は3seedの最小～最大であり、信頼区間ではない。1seed結果と本結果は別フォルダに保存する。', '',
              '- summary.json / scores_mean.csv：全強度の3seed平均・範囲',
              '- scores_by_seed.csv：各seedの成功数・成功率・平均IoU',
              '- results/ / plan.json / audit.json：追加推定、依存ファイル、元正解との照合',
              '- figures/whiteout_success_3seed.png：調整後の強度別成功率',
              '- figures/whiteout_before_after_3seed.png：調整前後の比較', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['evaluate','report'])
    args = parser.parse_args()
    plan = prepare()
    evaluate(plan) if args.stage == 'evaluate' else report(plan)
