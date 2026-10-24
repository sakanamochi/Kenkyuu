"""固定済みランダム寸法実験の全成果物を照合し、判断に必要な集計を保存する。"""

import ast
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parametric_paf.random_study import OUTPUT, PLAN, checkpoint_path, read, sha, verify_freeze, verify_sources, verify_training, write, write_csv


def summarize_group(rows):
    """未検出を含む全画像を分母にし、距離の分母だけは検出画像と明示する。"""
    present = [r for r in rows if r['target_present']]
    absent = [r for r in rows if not r['target_present']]
    distances = [r['contour_p95_px'] for r in present if r['detected']]
    coverage = [r['probability_full_contour_coverage'] for r in present]
    return {'n': len(rows), 'n_present': len(present), 'n_absent': len(absent),
            'strict_successes': sum(r['strict_success'] for r in present),
            'iou_successes': sum(r['iou_success'] for r in present),
            'missed': sum(r['missed'] for r in present),
            'wrong_output': sum(r['wrong_output'] for r in present),
            'absent_false_positives': sum(r['absent_false_positive'] for r in absent),
            'n_detected_for_distance': len(distances),
            'mean_contour_p95_px_detected': float(np.mean(distances)) if distances else None,
            'mean_full_contour_coverage': float(np.mean(coverage)) if coverage else None}


def camera_bin(tilt):
    """通常条件と高傾斜ストレスを同じ群にまとめず、視点別の分母を残す。"""
    if tilt <= 30:
        return '05_to_30deg'
    if tilt <= 50:
        return '30_to_50deg'
    if tilt <= 70:
        return '50_to_70deg'
    return '75_or_80deg'


def main():
    verify_freeze()
    verify_sources()
    protocol = read(PLAN / 'protocol.json')
    summary = read(OUTPUT / 'report/summary.json')
    if summary['freeze_sha256'] != sha(PLAN / 'freeze.json'):
        raise ValueError('集計と凍結記録が不一致')
    frozen_sha = sha(PLAN / 'freeze.json')
    evidence, populations, view_rows = [], [], []
    all_rows = {}
    manifest_counts = {}
    expected_models = {(arm, seed) for arm in protocol['arms'] for seed in protocol['training_seeds']}
    if {(run['arm'], run['seed']) for run in summary['runs']} != expected_models or len(summary['runs']) != 6:
        raise ValueError('全6モデルの学習検証記録がそろっていません')
    for run in summary['runs']:
        if run != verify_training(run['arm'], run['seed']):
            raise ValueError('学習履歴・採用重みと集計が不一致')
    with (OUTPUT / 'report/shape_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        shape_results = list(csv.DictReader(stream))
    shape_lookup = {(r['arm'], int(r['seed']), r['dataset'], r['split'], r['shape_id']): r for r in shape_results}
    if len(shape_lookup) != len(shape_results):
        raise ValueError('形状別集計のキーが重複')
    # 同じ画像への重複したモデル評価とは分けて、データそのものの全件を照合する。
    for dataset, expected in [('validation', 192), ('test', 1584), ('absent', 64)]:
        folder = OUTPUT / dataset
        samples = read(folder / 'manifest.json')['samples']
        if len(samples) != expected or len({s['sample_id'] for s in samples}) != expected:
            raise ValueError('評価データの枚数または一意性が不正')
        manifest_counts[dataset] = dict(Counter(s['split'] for s in samples))
        for sample in samples:
            for field in ('image', 'label'):
                if sha(folder / sample[field]) != sample[field + '_sha256']:
                    raise ValueError('評価データのhash不一致')
            label = read(folder / sample['label'])
            if label['conditions'] != sample['condition']:
                raise ValueError('ラベルと撮像条件の対応不正')
    expected_runs = {(arm, seed, dataset) for arm in protocol['arms'] for seed in protocol['training_seeds']
                     for dataset in ('validation', 'test', 'absent')}
    found_runs = set()
    for item in summary['evidence']:
        key = item['arm'], item['seed'], item['dataset']
        if key in found_runs:
            raise ValueError('集計根拠の重複')
        found_runs.add(key)
        folder = ROOT / item['path']
        completed = read(folder / 'completion.json')
        if completed['freeze_sha256'] != frozen_sha or completed['status'] != 'complete':
            raise ValueError('評価の仕様または完了状態が不正')
        if completed['manifest_sha256'] != sha(OUTPUT / item['dataset'] / 'manifest.json'):
            raise ValueError('評価と入力manifestのhashが不一致')
        if completed['checkpoint_sha256'] != sha(checkpoint_path(item['arm'], item['seed'])):
            raise ValueError('評価と現存する学習済み重みのhashが不一致')
        for field, filename in [('records_sha256', 'records.jsonl'), ('results_sha256', 'results.csv')]:
            if item[field] != sha(folder / filename) or completed[field] != item[field]:
                raise ValueError('集計根拠の評価hash不一致')
        rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
        manifest = read(OUTPUT / item['dataset'] / 'manifest.json')['samples']
        if [r['sample_id'] for r in rows] != [s['sample_id'] for s in manifest]:
            raise ValueError('全画像と評価原記録の対応不正')
        with (folder / 'results.csv').open(encoding='utf-8-sig', newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        if len(csv_rows) != len(rows) or [r['sample_id'] for r in csv_rows] != [r['sample_id'] for r in rows]:
            raise ValueError('CSVと原記録の対応不正')
        for row, sample in zip(rows, manifest):
            if any(row[field] != sample[field] for field in ('sample_id', 'shape_id', 'split')):
                raise ValueError('評価の形状・分割対応不正')
            if row['camera_tilt'] != sample['condition']['camera_tilt']:
                raise ValueError('評価の視点対応不正')
            if row['detected'] != (row['estimated_ellipse'] is not None):
                raise ValueError('推定楕円と検出フラグが不一致')
            if row['target_present']:
                detected = row['detected']
                iou = detected and row['ellipse_iou'] >= protocol['evaluation']['success_iou']
                success = iou and row['contour_p95_px'] <= protocol['evaluation']['contour_p95_px']
                if row['strict_success'] != success or row['iou_success'] != iou:
                    raise ValueError('主成功または互換IoU成功判定が不一致')
                if row['missed'] != (not detected) or row['wrong_output'] != (detected and not success):
                    raise ValueError('未検出・誤楕円の判定が不一致')
                if detected and not all(math.isfinite(row[k]) for k in ('ellipse_iou', 'contour_mean_px', 'contour_p95_px')):
                    raise ValueError('検出距離が非有限')
            elif row['absent_false_positive'] != row['detected']:
                raise ValueError('対象なし誤検出の判定が不一致')
        all_rows[key] = rows
        grouped = defaultdict(list)
        views = defaultdict(list)
        for row in rows:
            grouped[row['split'], row['family']].append(row)
            if row['target_present']:
                views[row['split'], camera_bin(row['camera_tilt'])].append(row)
        arm, seed, dataset = key
        per_shape = defaultdict(list)
        for row in rows:
            per_shape[row['split'], row['shape_id']].append(row)
        for (split, sid), group in per_shape.items():
            stored = shape_lookup[arm, seed, dataset, split, sid]
            if int(stored['n']) != len(group):
                raise ValueError('形状別集計の分母が原記録と不一致')
            if group[0]['target_present']:
                for field, raw in [('successes', 'strict_success'), ('missed', 'missed'), ('wrong_output', 'wrong_output')]:
                    if int(stored[field]) != sum(r[raw] for r in group):
                        raise ValueError('形状別集計の件数が原記録と不一致')
            elif int(stored['false_positives']) != sum(r['absent_false_positive'] for r in group):
                raise ValueError('形状別の対象なし誤検出数が原記録と不一致')
        for (split, family), group in grouped.items():
            populations.append({'arm': arm, 'seed': seed, 'dataset': dataset, 'split': split,
                                'family': family, **summarize_group(group)})
        for (split, view), group in views.items():
            view_rows.append({'arm': arm, 'seed': seed, 'dataset': dataset, 'split': split,
                              'camera_bin': view, **summarize_group(group)})
        evidence.append({'path': item['path'], 'completion_sha256': sha(folder / 'completion.json'),
                         'records_sha256': sha(folder / 'records.jsonl'), 'n': len(rows)})
    if found_runs != expected_runs:
        raise ValueError('全2条件×3seed×3データ区分の評価がそろっていません')
    for seed in protocol['training_seeds']:
        history = read(OUTPUT / f'models/random/{seed}/training_history.json')
        log = (OUTPUT / f'logs/train_random_{seed}.log').read_text(encoding='utf-8')
        logged = [ast.literal_eval(line) for line in log.splitlines() if line.startswith("{'epoch':")]
        if logged != history:
            raise ValueError('学習ログと全12epochの履歴が不一致')
    reference = ROOT / 'output/models/cnn_best.pt'
    if sha(reference) != 'd9238666ab08572aeb26005b5bbc3bc83d87ae574442c8e89fb7eb4b878fc4f6':
        raise ValueError('元1194M重みが変更されています')
    primary = []
    for seed in protocol['training_seeds']:
        scores = {}
        for arm in protocol['arms']:
            group = [r for r in all_rows[arm, seed, 'test'] if r['split'] == protocol['primary_split']]
            if len(group) != 576 or len({r['shape_id'] for r in group}) != 12:
                raise ValueError('主テストの分母が不正')
            scores[arm] = sum(r['strict_success'] for r in group) / 576 * 100
        primary.append(scores['random'] - scores['single'])
    if not np.isclose(np.mean(primary), summary['primary_mean_difference_pp'], atol=1e-10):
        raise ValueError('主テストの平均差と集計が不一致')
    supported = all(x > 0 for x in primary) and np.mean(primary) >= protocol['evaluation']['minimum_effect_percentage_points']
    if supported != summary['primary_hypothesis_supported']:
        raise ValueError('主仮説の判定が事前基準と不一致')
    for arm in protocol['arms']:
        repro = read(OUTPUT / f'reproducibility/{arm}.json')
        if repro['status'] != 'passed' or repro['freeze_sha256'] != frozen_sha:
            raise ValueError('推論再現性の確認結果が不正')
        if read(OUTPUT / f'reproducibility/{arm}_0.json') != read(OUTPUT / f'reproducibility/{arm}_1.json'):
            raise ValueError('別プロセス推論の原記録が不一致')
    report = OUTPUT / 'report'
    write_csv(report / 'population_results.csv', populations)
    write_csv(report / 'view_results.csv', view_rows)
    # 主張と適用範囲を読み取れるよう、原記録から同じ分母の3seed平均を示す。
    labels = [('test_interpolation', '範囲内の未知寸法（主テスト）', None),
              ('test_extrapolation', '学習範囲外の寸法', None),
              ('test_interpolation_capture', '厳しい撮像条件：範囲内寸法', None),
              ('test_extrapolation_capture', '厳しい撮像条件：範囲外寸法', None),
              ('single_shape_monitor', '元の基準寸法', None)]
    lines = ['# ランダム寸法学習の検証結果', '',
             '問い：機体ごとの詳細CADを学習に使わず、合成PAFの寸法を多様化すれば、未知形状の内周を検出しやすくなるか。', '',
             '単一寸法の1,024枚と、画像ごとに寸法を抽選した1,024形状×1枚を比較した。',
             '画像枚数・学習回数・視点・色・照明・CNN・楕円推定処理はそろえた。学習は各条件3回。',
             '学習・モデル選択に使っていない同じテスト画像で比較し、テストを見た再調整は行っていない。', '',
             '成功はIoU≥0.8かつ輪郭距離p95≤2px。以下は全画像を分母にした3学習seedの平均。', '',
             '|評価する条件|各モデルの画像数|単一寸法|ランダム寸法|差（ポイント）|',
             '|---|---:|---:|---:|---:|']
    for split, label, family in labels:
        rates = {}
        counts = []
        for arm in protocol['arms']:
            values = []
            for seed in protocol['training_seeds']:
                group = [r for r in all_rows[arm, seed, 'test'] if r['split'] == split
                         and (family is None or r['family'] == family)]
                counts.append(len(group))
                values.append(sum(r['strict_success'] for r in group) / len(group) * 100)
            rates[arm] = float(np.mean(values))
        if len(set(counts)) != 1:
            raise ValueError('比較群の画像数が不一致')
        lines.append(f'|{label}|{counts[0]}|{rates["single"]:.2f}%|{rates["random"]:.2f}%|'
                     f'{rates["random"] - rates["single"]:+.2f}|')
    lines += ['', f'主テストの改善幅はseedごとに {" / ".join(f"{value:+.2f}" for value in primary)} ポイント。',
              '事前基準（3回すべてで正、平均10ポイント以上）は' + ('満たした。' if supported else '満たさなかった。'),
              'この結果が示すのは、定義した合成形状族・黒背景・画面内に対象がある条件での効果である。', '',
              '元の基準寸法で各seedの低下を5ポイント以内に収める補足基準は' +
              ('満たした。' if summary['single_shape_monitor_within_5pp_each_seed'] else '満たさなかった。'),
              'このmonitorは単一寸法側が学習した基準形状1個の評価であり、randomの全学習形状を調べたものではない。', '',
              '|対象なし画像|各モデルの画像数|単一寸法の誤検出率|ランダム寸法の誤検出率|',
              '|---|---:|---:|---:|']
    for family, label in [('black', '黒画像'), ('solid_disk', '穴なし円盤')]:
        rates = {}
        for arm in protocol['arms']:
            rates[arm] = float(np.mean([sum(r['absent_false_positive'] for r in all_rows[arm, seed, 'absent']
                                            if r['family'] == family) /
                                        sum(r['family'] == family for r in all_rows[arm, seed, 'absent']) * 100
                                        for seed in protocol['training_seeds']]))
        count = sum(r['family'] == family for r in all_rows['single', protocol['training_seeds'][0], 'absent'])
        lines.append(f'|{label}|{count}|{rates["single"]:.2f}%|{rates["random"]:.2f}%|')
    absent_differences = [r['difference_pp'] for r in summary['absent_false_positive_rates_percent']]
    lines += ['', '対象なし誤検出の増加を各seedで5ポイント以内に収める補足基準は' +
              ('満たした。' if all(value <= 5 for value in absent_differences) else '満たさなかった。'),
              '増加が小さくても、誤検出率そのものが高ければ対象の有無を見分ける性能が十分とは言えない。',
              '円盤63画像は同じ1形状の撮像条件違いであり、独立した63形状ではない。', '',
              '実画像・実在する別PAF・任意の背景への一般化、必要な形状数、十分な学習収束は未検証。',
              '3学習seedは初期化と学習順序の反復であり、寸法抽選データ自体の3反復ではない。',
              '詳しいseed別・形状別の結果はREPORT.md、shape_results.csv、summary.json、',
              '未検出・誤楕円・IoU成功・輪郭距離・支持被覆はpopulation_results.csvとview_results.csvを参照する。',
              '全18評価区分・11,040原記録の入力と集計の対応はaudit.jsonに保存する。', '']
    (report / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    write(report / 'audit.json', {'status': 'passed', 'completed_evaluation_runs': len(found_runs),
          'evaluation_records': sum(len(rows) for rows in all_rows.values()), 'manifest_counts': manifest_counts,
          'primary_seed_differences_pp': primary, 'primary_mean_difference_pp': float(np.mean(primary)),
          'audit_code_sha256': sha(Path(__file__)), 'experiment_freeze_sha256': frozen_sha,
          'summary_sha256': sha(report / 'summary.json'), 'evidence': evidence,
          'population_results_sha256': sha(report / 'population_results.csv'),
          'view_results_sha256': sha(report / 'view_results.csv'),
          'findings_sha256': sha(report / 'FINDINGS.md')})
    print(json.dumps({'status': 'passed', 'completed_evaluation_runs': len(found_runs),
                      'evaluation_records': sum(len(rows) for rows in all_rows.values()),
                      'primary_mean_difference_pp': float(np.mean(primary))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
