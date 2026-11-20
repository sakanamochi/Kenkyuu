"""白飛び強度比較の数値・元正解・共通入力を照合し、図と報告を保存する。"""

import csv
import hashlib
import sys
import zlib
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_whiteout_strength.run import (BASE, OUT, PRIOR, METHODS, DATASETS,
                                               sha, inputs, samples_for, load_models)
from paf_ring_detection.data import read_json, write_json, read_image, load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_whiteout_strength.effect import apply_whiteout
from paf_ring_detection.evaluate import _geometry
from research.paf_generalization.boundary import boundary_ellipse

NAMES = {'1194m': '1194M学習CNN', 'diversified': '多様形状学習CNN',
         'canny': 'Canny＋輪郭別RANSAC', 'zhang2019': 'Zhang 2019型（再現実装）'}
COLORS = {'1194m': '#355EAD', 'diversified': '#12856D', 'canny': '#D88919', 'zhang2019': '#9562A6'}
MARKERS = {'1194m': 'o', 'diversified': 's', 'canny': '^', 'zhang2019': 'D'}
DS = {'1194m': '1194M', 'diversified': '多様形状'}


def baseline(dataset, method, seed):
    """同じ重みの保存済み比較を参照し、3seed平均とは混在させない。"""
    path = (PRIOR/f'results/{method}/{seed}/test_{dataset}.json' if method in ('1194m', 'diversified')
            else BASE.parent/f'results/{dataset}/{method}_fixed.json')
    return read_json(path), path


def verify_baseline():
    """成否で選ばない16画像・64推定で、保存済み対照と現在の推論の対応を確認する。"""
    path = OUT/'baseline_reproduction.json'
    if path.exists():
        saved = read_json(path)
        assert saved['plan_sha256'] == sha(OUT/'plan.json')
        return saved
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    seed = read_json(OUT/'plan.json')['config']['seed']
    models = load_models(seed)
    settings = read_json(ROOT/'research/paf_fair_comparison/no_surface.json')
    geometric = read_json(ROOT/'config/experiment.json')
    rows = []
    for dataset in DATASETS:
        references = {m: {r['sample_id']: r for r in baseline(dataset, m, seed)[0]['rows']} for m in METHODS}
        for severity in [.25, .5, .75, 1.]:
            samples = sorted(samples_for(dataset, 'test', severity), key=lambda s: s['sample_id'])
            for sample in [samples[0], samples[len(samples)//2]]:
                image, truth = inputs(dataset, sample, severity, 1.)
                tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
                for method in METHODS:
                    if method in models:
                        with torch.inference_mode():
                            probability = torch.sigmoid(models[method](tensor))[0, 0].cpu().numpy()
                        cv2.setRNGSeed(settings['opencv_seed_per_image'])
                        ellipse = boundary_ellipse(probability, settings['ransac'])
                    else:
                        random_seed = (geometric['evaluation_seed']+zlib.crc32(sample['sample_id'].encode('utf-8'))) % 2**32
                        ellipse = _geometry(image, method, geometric, random_seed)
                    iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                    old = references[method][sample['sample_id']]
                    rows.append({'dataset': dataset, 'sample_id': sample['sample_id'], 'method': method,
                                 'saved_iou': old['ellipse_iou'], 'recomputed_iou': iou,
                                 'success_matches': (iou >= .8) == old['success']})
    result = {'plan_sha256': sha(OUT/'plan.json'), 'prediction_count': len(rows),
              'changed_successes': sum(not r['success_matches'] for r in rows),
              'maximum_iou_difference': max(abs(r['saved_iou']-r['recomputed_iou']) for r in rows), 'rows': rows}
    write_json(path, result)
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cv2.setNumThreads(1)
    plan = read_json(OUT/'plan.json')
    reproduction = verify_baseline()
    assert reproduction['changed_successes'] == 0
    for path, digest in plan['files'].items():
        assert sha(ROOT/path) == digest, path
    calibration = read_json(OUT/'calibration.json')
    maximum = calibration['selected_maximum_strength']
    seed = plan['config']['seed']
    severities = plan['config']['test_severities']
    records, prediction_checks, input_checks, source_hashes = [], 0, 0, {}
    validation_checks = 0
    for candidate in calibration['calibration']:
        rates = []
        for dataset in DATASETS:
            strength = candidate['maximum_strength']
            path = OUT/f'results/validation/{dataset}_max{strength:g}_s1.json'
            result = read_json(path)
            assert result['plan_sha256'] == sha(OUT/'plan.json')
            samples = samples_for(dataset, 'validation', 1.)
            assert len(samples) == 105 and len(result['rows']) == 420
            lookup = {(r['sample_id'], r['method']): r for r in result['rows']}
            assert len(lookup) == 420
            for sample in samples:
                image, truth = inputs(dataset, sample, 1., strength)
                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                for method in METHODS:
                    row = lookup[(sample['sample_id'], method)]
                    assert row['image_sha256'] == image_hash
                    iou = evaluate_ellipses(row['ellipse'], truth, (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
                    assert abs(iou-row['ellipse_iou']) < 1e-10 and row['success'] == (iou >= .8)
                    validation_checks += 1
            for method in METHODS:
                rate = 100*np.mean([r['success'] for r in result['rows'] if r['method'] == method])
                assert abs(rate-candidate['by_dataset'][dataset][method]['success_percent']) < 1e-10
                rates.append(rate)
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
        assert abs(np.mean(rates)-candidate['pooled_success_percent']) < 1e-10
    low, high = plan['config']['acceptable_pooled_success_percent']
    eligible = [v['maximum_strength'] for v in calibration['calibration'][1:]
                if low <= v['pooled_success_percent'] <= high and
                calibration['calibration'][0]['pooled_success_percent']-v['pooled_success_percent']
                >= plan['config']['minimum_drop_from_original_pp']]
    assert maximum == min(eligible)
    # 予測を画像IDで対応付け、元の正解から成功判定を再計算する。
    for dataset in DATASETS:
        originals = {m: baseline(dataset, m, seed) for m in METHODS}
        for method, (result, path) in originals.items():
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
            rows = [r for r in result['rows'] if r['degradation'] == 'clean']
            records.append({'dataset': dataset, 'variant': 'original', 'severity': 0., 'method': method,
                            'n': len(rows), 'successes': sum(r['success'] for r in rows),
                            'success_percent': 100*np.mean([r['success'] for r in rows]),
                            'mean_iou': np.mean([r['ellipse_iou'] for r in rows])})
        for severity in severities:
            path = OUT/f'results/test/{dataset}_max{maximum:g}_s{severity:g}.json'
            result = read_json(path)
            assert result['plan_sha256'] == sha(OUT/'plan.json')
            lookup = {(r['sample_id'], r['method']): r for r in result['rows']}
            samples = samples_for(dataset, 'test', severity)
            assert len(lookup) == len(samples)*4
            for sample in samples:
                image, truth = inputs(dataset, sample, severity, maximum)
                assert sha(BASE/dataset/sample['image']) == sample['image_sha256']
                assert sha(BASE/dataset/sample['label']) == sample['label_sha256']
                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                input_checks += 1
                for method in METHODS:
                    row = lookup[(sample['sample_id'], method)]
                    assert row['image_sha256'] == image_hash
                    iou = evaluate_ellipses(row['ellipse'], truth, (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
                    assert abs(iou-row['ellipse_iou']) < 1e-10 and row['success'] == (iou >= .8)
                    old = next(r for r in originals[method][0]['rows'] if r['sample_id'] == sample['sample_id'])
                    old_iou = evaluate_ellipses(old['ellipse'], truth, (256, 256))['ellipse_iou'] if old['ellipse'] else 0.
                    assert abs(old_iou-old['ellipse_iou']) < 1e-10 and old['success'] == (old_iou >= .8)
                    prediction_checks += 2
            for method in METHODS:
                for variant, rows in [('enhanced', [r for r in result['rows'] if r['method'] == method]),
                                      ('original', [r for r in originals[method][0]['rows']
                                                    if r['degradation'] == 'sensor_whiteout' and r['severity'] == severity])]:
                    assert len(rows) == len(samples)
                    records.append({'dataset': dataset, 'variant': variant, 'severity': severity, 'method': method,
                                    'n': len(rows), 'successes': sum(r['success'] for r in rows),
                                    'success_percent': 100*np.mean([r['success'] for r in rows]),
                                    'mean_iou': np.mean([r['ellipse_iou'] for r in rows])})
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
    write_json(OUT/'summary.json', {'seed': seed, 'selected_maximum_strength': maximum, 'records': records,
                                   'sources': source_hashes, 'retrained': False})
    with (OUT/'scores.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(OUT/'audit.json', {'prediction_checks_including_original': prediction_checks,
                                  'validation_prediction_checks': validation_checks,
                                  'baseline_reproduction': reproduction,
                                  'shared_image_checks': input_checks, 'methods_per_image': 4,
                                  'original_formula_pixel_equality_checks': input_checks,
                                  'plan_sha256': sha(OUT/'plan.json'), 'sources': source_hashes})
    figures(records, maximum, seed)
    report(records, maximum, seed, calibration)
    print('AUDIT', prediction_checks, input_checks, flush=True)


def value(records, dataset, method, variant, severity):
    return next(r for r in records if r['dataset'] == dataset and r['method'] == method
                and r['variant'] == variant and r['severity'] == severity)['success_percent']


def figures(records, maximum, seed):
    """軸・色・分母を揃え、白飛び強度と成功率の関係をPNGへ出力する。"""
    folder = OUT/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Meiryo', 'font.size': 13, 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        for method in METHODS:
            values = [value(records, dataset, method, 'original' if s == 0 else 'enhanced', s)
                      for s in [0., .25, .5, .75, 1.]]
            ax.plot([0, .25, .5, .75, 1], values, label=NAMES[method], color=COLORS[method],
                    marker=MARKERS[method], linestyle='--' if method in ('canny', 'zhang2019') else '-', linewidth=2)
        ax.set(title=DS[dataset], xlabel='白飛び強度', ylim=(-2, 103), xticks=[0, .25, .5, .75, 1])
        ax.grid(axis='y', color='#DDDDDD')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('検出成功率 [%]')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.suptitle(f'白飛び強度の追加評価（CNN：seed {seed}、再学習なし）', fontsize=16)
    fig.tight_layout(rect=[0, .16, 1, .95])
    fig.savefig(folder/'whiteout_success.png', dpi=180)
    plt.close(fig)
    # 同一方式の調整前後を分けて表示し、8本の線が一つの軸に重ならないようにする。
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for row, dataset in enumerate(DATASETS):
        for ax, method in zip(axes[row], METHODS):
            for variant, style, label in [('original', '--', '調整前'), ('enhanced', '-', '調整後')]:
                ys = [value(records, dataset, method, 'original' if s == 0 else variant, s)
                      for s in [0., .25, .5, .75, 1.]]
                ax.plot([0, .25, .5, .75, 1], ys, style, color=COLORS[method], marker=MARKERS[method],
                        markerfacecolor='white' if variant == 'original' else COLORS[method], label=label)
            ax.set_title(NAMES[method].replace('（再現実装）', '\n（再現実装）'), fontsize=12)
            ax.set(ylim=(-2, 103), xticks=[0, .5, 1])
            ax.grid(axis='y', color='#DDDDDD')
            ax.spines[['top', 'right']].set_visible(False)
            if row == 1:
                ax.set_xlabel('白飛び強度')
        axes[row, 0].set_ylabel(DS[dataset]+'\n検出成功率 [%]')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.suptitle(f'白飛び条件の調整前後（CNN：seed {seed}、再学習なし）', fontsize=17)
    fig.tight_layout(rect=[0, .06, 1, .95])
    fig.savefig(folder/'whiteout_before_after.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(10, 13))
    example_sources = []
    for di, dataset in enumerate(DATASETS):
        samples = sorted(samples_for(dataset, 'test', 1.), key=lambda x: x['sample_id'])
        # 成否を使わず、画像ID順の先頭と中央を例として固定する。
        for si, sample in enumerate([samples[0], samples[len(samples)//2]]):
            row = di*2+si
            raw = read_image(BASE/dataset/sample['image'])
            for ax, max_strength, label in zip(axes[row], [0, 1, maximum], ['通常', '調整前：強度1.0', '調整後：強度1.0']):
                image = raw if max_strength == 0 else apply_whiteout(raw, 1., np.random.default_rng(sample['augmentation']['seed']), max_strength)
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
                ax.set_title(label if row == 0 else '', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[row, 0].set_ylabel(DS[dataset]+f' 例{si+1}', fontsize=13)
            example_sources.append({'dataset': dataset, 'sample_id': sample['sample_id']})
    fig.tight_layout()
    fig.savefig(folder/'whiteout_examples.png', dpi=160)
    plt.close(fig)
    write_json(folder/'figure_data.json', {'records': records, 'examples': example_sources,
                                           'source': '../summary.json', 'maximum_strength': maximum})
    # 検証用の例も同じ固定規則で選び、強度範囲を比較できる図として保存する。
    validation = sorted(samples_for('1194m', 'validation', 1.), key=lambda x: x['sample_id'])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for row, sample in enumerate([validation[0], validation[len(validation)//2]]):
        raw = read_image(BASE/'1194m'/sample['image'])
        for ax, strength in zip(axes[row], [0, 1, 1.5, 2]):
            image = raw if strength == 0 else apply_whiteout(raw, 1., np.random.default_rng(sample['augmentation']['seed']), strength)
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255)
            ax.set_title('通常' if strength == 0 else f'最大内部強度 {strength:g}', fontsize=13)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(folder/'calibration_examples.png', dpi=160)
    plt.close(fig)


def report(records, maximum, seed, calibration):
    lines = ['# 白飛び強度の追加評価', '',
             f'CNNは既存seed {seed}の重みを固定し、再学習していない。成功条件はIoU≥0.8、未検出を含む全画像を分母とする。', '',
             '## 白飛び条件', '',
             f'表示強度sを内部強度t={maximum:g}×sへ対応させる。露光倍率2^(5t)、にじみのぼかし幅1+18t px、加算係数0.15+0.85t。',
             f'強度1.0の露光倍率は32倍から{2**(5*maximum):.2f}倍、ぼかし幅は19 pxから{1+18*maximum:g} px、加算係数は1から{.15+.85*maximum:g}となる。',
             '原画像480px上で処理してからモノクロ256pxへ縮小する。光源やレンズの物理モデルを追加したものではない。', '',
             '## 強度の選択', '', calibration['selection_rule'], '',
             '検証用原画像は各105枚。両集合の4方式を等重みで平均する。検証集合も研究中に参照済みであり、新規盲検ではない。', '',
             '|最大内部強度|検証用画像の平均成功率|', '|---|---:|']
    lines += [f"|{v['maximum_strength']:g}|{v['pooled_success_percent']:.2f}%|" for v in calibration['calibration']]
    lines += ['', '## テスト：白飛び強度1.0', '',
              '1194Mは112原画像、多様形状は154原画像。調整前は同じCNN重み・幾何方式による保存済み結果。今回のCNN値は1seedで、既存発表資料の3seed平均とは区別する。', '',
              '|評価対象|方式|調整前|調整後|変化（pp）|', '|---|---|---:|---:|---:|']
    for dataset in DATASETS:
        for method in METHODS:
            before = value(records, dataset, method, 'original', 1.)
            after = value(records, dataset, method, 'enhanced', 1.)
            lines.append(f'|{DS[dataset]}|{NAMES[method]}|{before:.2f}%|{after:.2f}%|{after-before:+.2f}|')
    lines += ['', '## 解釈と制約', '',
              '検証集合で劣化の強さを選んだ探索的な耐性比較であり、方式の順位や方式間の差を最大化して選択したものではない。',
              'CNNは従来の白飛び範囲で学習した既存モデル。強めた条件で再学習した場合の性能は未検証。実際のカメラ・フレアへの妥当性も未検証。',
              '既存共有処理・学習重み・3seedの主比較・PPTXは上書きしない。追加条件は本実験のeffect.pyとcalibration.jsonで定義する。', '',
              '## 図と根拠', '',
              '- figures/whiteout_success.png：調整後の強度別成功率',
              '- figures/whiteout_before_after.png：同一方式の調整前後',
              '- figures/whiteout_examples.png：通常・調整前・調整後の同一原画像',
              '- plan.json / calibration.json：依存ファイル・選択規則・検証結果',
              '- results/ / summary.json / scores.csv / audit.json：推定・集計・元正解との照合', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
