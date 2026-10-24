"""ランダム寸法実験の同一画像・固定指標で、既存の幾何2手法を追加比較する。"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parametric_paf.random_study import PLAN, OUTPUT as CNN_OUTPUT, SOURCE_OUTPUT, read, sha, verify_freeze, verify_sources, write, write_csv

OUTPUT = ROOT / 'parametric_paf/output/paf_random_geometry_v1'
FREEZE = PLAN / 'geometry_comparison.json'
METHODS = {'canny': 'Canny＋輪郭別RANSAC', 'zhang': 'Zhang 2019型（再現実装）'}
COUNTS = {'validation': 192, 'test': 1584, 'absent': 64}
CONFIG = SOURCE_OUTPUT / 'geometry_validation/selected_config.json'


def freeze():
    """既存validationで選んだ設定と、評価開始前のコード・入力・CNN結果を固定する。"""
    from parametric_paf.study import runtime_versions
    verify_freeze()
    verify_sources()
    selection = read(SOURCE_OUTPUT / 'geometry_validation/selection.json')
    if selection['test_used']:
        raise ValueError('幾何設定の選定にテストが使用されています')
    if selection['manifest_sha256'] != sha(SOURCE_OUTPUT / 'validation/manifest.json'):
        raise ValueError('幾何設定を選んだvalidationが不一致')
    files = [Path(__file__), CONFIG, SOURCE_OUTPUT / 'geometry_validation/selection.json',
             PLAN / 'freeze.json', PLAN / 'protocol.json', CNN_OUTPUT / 'report/summary.json',
             CNN_OUTPUT / 'report/audit.json', ROOT / 'parametric_paf/tune_geometry.py',
             ROOT / 'parametric_paf/evaluate_study.py', ROOT / 'parametric_paf/diagnosis_metrics.py',
             ROOT / 'paf_ring_detection/data.py', ROOT / 'paf_ring_detection/geometry.py']
    files += list((ROOT / 'paf_ring_detection/methods').glob('*.py'))
    files += [CNN_OUTPUT / dataset / 'manifest.json' for dataset in COUNTS]
    value = {'experiment_id': 'paf_random_geometry_v1', 'input_size': 256,
             'methods': METHODS, 'dataset_counts': COUNTS, 'runtime_versions': runtime_versions(),
             'threshold_selection': '既存の共通validationで選択済みの30/90。追加調整なし。',
             'geometry_repetitions': '学習seedなし。画像単位のNumPy・OpenCV乱数を固定した1推定。',
             'sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}}
    if FREEZE.exists() and read(FREEZE) != value:
        raise ValueError('追加比較の固定記録と不一致。既存結果は上書きしません')
    write(FREEZE, value)


def verify():
    """比較中の入力・設定・コード・数値環境の変更を検出する。"""
    from parametric_paf.study import runtime_versions
    frozen = read(FREEZE)
    for relative, digest in frozen['sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'追加比較の固定記録と不一致: {relative}')
    if runtime_versions() != frozen['runtime_versions']:
        raise ValueError('追加比較のライブラリ版が不一致')


def predict(sample, folder, method):
    """CNNと同じリサイズ・正解・乱数・評価を使い、画像だけを幾何検出器に渡す。"""
    import cv2
    from paf_ring_detection.data import read_image, label_ellipse, scaled_ellipse
    from parametric_paf.tune_geometry import run_geometry
    from parametric_paf.evaluate_study import evaluate_prediction
    cv2.setNumThreads(1)
    protocol = read(PLAN / 'protocol.json')
    image = cv2.resize(read_image(folder / sample['image']), (256, 256), interpolation=cv2.INTER_AREA)
    label = read(folder / sample['label'])
    seed = (protocol['test_seed'] + zlib.crc32(sample['condition']['condition_id'].encode())) % 2**32
    cv2.setRNGSeed(seed & 0x7fffffff)
    truth = scaled_ellipse(label_ellipse(label), 256 / label['image_width']) if label.get('target_present', True) else None
    cv2.setRNGSeed(seed & 0x7fffffff)
    result = run_geometry(image, method, read(CONFIG), seed)
    ellipse = result['ellipse'] if result else None
    return {'sample_id': sample['sample_id'], 'shape_id': sample['shape_id'], 'split': sample['split'],
            'family': label.get('family', label.get('kind')), 'condition_id': sample['condition']['condition_id'],
            'camera_tilt': sample['condition']['camera_tilt'], 'method': method, 'ransac_seed': seed,
            'estimated_ellipse': ellipse, **evaluate_prediction(ellipse, truth, image.shape, protocol['evaluation'])}


def repro_worker(method):
    samples = read(CNN_OUTPUT / 'validation/manifest.json')['samples']
    # 既存の実行間差が見つかった画像も含め、CNNと同じ固定17画像を検査する。
    indices = sorted(set(list(range(0, 192, 12)) + [157]))
    print(json.dumps([predict(samples[i], CNN_OUTPUT / 'validation', method) for i in indices], ensure_ascii=False))


def reproduce(method):
    """独立2プロセスで幾何推定と指標の一致を確認する。"""
    verify()
    outputs = []
    for index in range(2):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__)),
                                 'repro-worker', '--method', method], cwd=ROOT, check=True,
                                capture_output=True, text=True, encoding='utf-8')
        rows = json.loads(result.stdout.strip().splitlines()[-1])
        outputs.append(rows)
        write(OUTPUT / f'reproducibility/{method}_{index}.json', rows)
    if outputs[0] != outputs[1]:
        raise ValueError('別プロセス間の幾何推定が不一致')
    write(OUTPUT / f'reproducibility/{method}.json', {'status': 'passed', 'images': 17,
          'independent_processes': 2, 'freeze_sha256': sha(FREEZE)})
    print(f'Reproducibility passed: {method}', flush=True)


def evaluate(method):
    """学習seedごとに同じ幾何結果を反復せず、各画像を1回ずつ評価する。"""
    verify()
    if read(OUTPUT / f'reproducibility/{method}.json')['status'] != 'passed':
        raise ValueError('推論再現性が未確認')
    for dataset, expected in COUNTS.items():
        folder = CNN_OUTPUT / dataset
        samples = read(folder / 'manifest.json')['samples']
        if len(samples) != expected or len({s['sample_id'] for s in samples}) != expected:
            raise ValueError('データセットの分母不一致')
        for sample in samples:
            for field in ('image', 'label'):
                if sha(folder / sample[field]) != sample[field + '_sha256']:
                    raise ValueError('入力画像・ラベルhash不一致')
        destination = OUTPUT / method / dataset
        destination.mkdir(parents=True, exist_ok=True)
        provenance = {'method': method, 'dataset': dataset, 'sample_count': expected,
                      'manifest_sha256': sha(folder / 'manifest.json'), 'freeze_sha256': sha(FREEZE)}
        prov_path = destination / 'provenance.json'
        if prov_path.exists() and read(prov_path) != provenance:
            raise ValueError('保存済み評価と入力が不一致')
        write(prov_path, provenance)
        record_path = destination / 'records.jsonl'
        rows = [json.loads(line) for line in record_path.read_text(encoding='utf-8').splitlines()] if record_path.exists() else []
        if [r['sample_id'] for r in rows] != [s['sample_id'] for s in samples[:len(rows)]]:
            raise ValueError('原記録と入力画像の対応が不一致')
        with record_path.open('a', encoding='utf-8', newline='\n') as stream:
            for sample in samples[len(rows):]:
                row = predict(sample, folder, method)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                rows.append(row)
                if len(rows) % 24 == 0:
                    print(f'{method}/{dataset}: {len(rows)}/{expected}', flush=True)
        write_csv(destination / 'results.csv', rows)
        write(destination / 'completion.json', {**provenance, 'status': 'complete',
              'records_sha256': sha(record_path), 'results_sha256': sha(destination / 'results.csv')})


def report():
    """全件の対応・判定を検証し、CNN3seedと幾何の固定推定を対応させて集計する。"""
    verify()
    all_rows, evidence, shape_rows = {}, [], []
    seeds = read(PLAN / 'protocol.json')['training_seeds']
    for method in METHODS:
        for dataset, count in COUNTS.items():
            folder = OUTPUT / method / dataset
            complete = read(folder / 'completion.json')
            if complete['status'] != 'complete' or complete['sample_count'] != count:
                raise ValueError('幾何評価が未完了')
            for field, filename in [('records_sha256', 'records.jsonl'), ('results_sha256', 'results.csv')]:
                if complete[field] != sha(folder / filename):
                    raise ValueError('幾何評価hash不一致')
            rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
            samples = read(CNN_OUTPUT / dataset / 'manifest.json')['samples']
            if len(rows) != count or [r['sample_id'] for r in rows] != [s['sample_id'] for s in samples]:
                raise ValueError('幾何評価の全画像対応不正')
            groups = defaultdict(list)
            for row, sample in zip(rows, samples):
                if any(row[key] != sample[key] for key in ('sample_id', 'shape_id', 'split')):
                    raise ValueError('幾何評価の形状・分割対応不正')
                if row['target_present']:
                    success = row['detected'] and row['ellipse_iou'] >= 0.8 and row['contour_p95_px'] <= 2
                    if row['strict_success'] != success or row['missed'] != (not row['detected']):
                        raise ValueError('幾何評価の成功・未検出判定不正')
                    if row['wrong_output'] != (row['detected'] and not success):
                        raise ValueError('幾何評価の誤楕円判定不正')
                elif row['absent_false_positive'] != row['detected']:
                    raise ValueError('幾何評価の対象なし判定不正')
                groups[row['split'], row['shape_id']].append(row)
            for (split, sid), group in groups.items():
                present = group[0]['target_present']
                shape_rows.append({'method': method, 'dataset': dataset, 'split': split, 'shape_id': sid,
                                   'n': len(group), 'successes': sum(r['strict_success'] for r in group) if present else None,
                                   'iou_successes': sum(r['iou_success'] for r in group) if present else None,
                                   'missed': sum(r['missed'] for r in group) if present else None,
                                   'wrong_output': sum(r['wrong_output'] for r in group) if present else None,
                                   'false_positives': sum(r['absent_false_positive'] for r in group) if not present else None})
            all_rows[method, dataset] = rows
            evidence.append({'path': folder.relative_to(ROOT).as_posix(), **complete})
    # CNNの原記録を読み直し、追加比較の画像順と一致することを要求する。
    for arm in ('single', 'random'):
        for seed in seeds:
            for dataset in COUNTS:
                folder = CNN_OUTPUT / f'evaluation/{arm}/{seed}/{dataset}'
                completed = read(folder / 'completion.json')
                if completed['records_sha256'] != sha(folder / 'records.jsonl'):
                    raise ValueError('CNN原記録hash不一致')
                rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
                if [r['sample_id'] for r in rows] != [r['sample_id'] for r in all_rows['canny', dataset]]:
                    raise ValueError('CNNと幾何で評価画像が不一致')
                all_rows[arm, seed, dataset] = rows
    populations = [('test_interpolation', '範囲内の未知寸法', None),
                   ('test_extrapolation', '学習範囲外の寸法', None),
                   ('test_interpolation_capture', '厳しい撮像：範囲内寸法', None),
                   ('test_extrapolation_capture', '厳しい撮像：範囲外寸法', None),
                   ('single_shape_monitor', '元の基準寸法', None)]
    table, contrasts = [], []
    lines = ['# CNNと幾何手法の同一画像比較', '',
             '全方式256×256、同じ独立テスト画像と主成功条件（IoU≥0.8かつ輪郭p95≤2px）で比較する。',
             'CNNは学習seed3本の平均。幾何手法は学習seedを持たず、画像単位で乱数を固定した1推定である。',
             '幾何のCanny閾値30/90は共通validationで選定済み。テストでの追加調整は行わない。', '',
             '|評価条件|単一寸法CNN|ランダム寸法CNN|Canny＋輪郭別RANSAC|Zhang 2019型（再現実装）|',
             '|---|---:|---:|---:|---:|']
    for split, label, family in populations:
        scores = {}
        def select(rows):
            return [r for r in rows if r['split'] == split and (family is None or r['family'] == family)]
        for arm in ('single', 'random'):
            values = []
            for seed in seeds:
                group = select(all_rows[arm, seed, 'test'])
                values.append(sum(r['strict_success'] for r in group) / len(group) * 100)
            scores[arm] = sum(values) / len(values)
        for method in METHODS:
            group = select(all_rows[method, 'test'])
            scores[method] = sum(r['strict_success'] for r in group) / len(group) * 100
            for seed in seeds:
                cnn = select(all_rows['random', seed, 'test'])
                geo = {r['sample_id']: r for r in group}
                differences = defaultdict(list)
                for row in cnn:
                    differences[row['shape_id']].append(int(row['strict_success']) - int(geo[row['sample_id']]['strict_success']))
                contrasts.append({'split': split, 'family': family, 'seed': seed, 'geometry_method': method,
                                  'mean_shape_difference_pp': sum(sum(a) / len(a) for a in differences.values()) / len(differences) * 100})
        table.append({'split': split, 'family': family, **scores})
        lines.append(f'|{label}|' + '|'.join(f'{scores[key]:.2f}%' for key in ('single', 'random', 'canny', 'zhang')) + '|')
    lines += ['', '|対象なし|単一寸法CNN|ランダム寸法CNN|Canny＋輪郭別RANSAC|Zhang 2019型（再現実装）|',
              '|---|---:|---:|---:|---:|']
    absent_rows = []
    for family, label in [('black', '黒画像1枚の誤検出率'), ('solid_disk', '穴なし円盤63枚の誤検出率')]:
        rates = {}
        for arm in ('single', 'random'):
            values = []
            for seed in seeds:
                rows = [r for r in all_rows[arm, seed, 'absent'] if r['family'] == family]
                values.append(sum(r['absent_false_positive'] for r in rows) / len(rows) * 100)
            rates[arm] = sum(values) / len(values)
        for method in METHODS:
            rows = [r for r in all_rows[method, 'absent'] if r['family'] == family]
            rates[method] = sum(r['absent_false_positive'] for r in rows) / len(rows) * 100
        absent_rows.append({'family': family, **rates})
        lines.append(f'|{label}|' + '|'.join(f'{rates[key]:.2f}%' for key in ('single', 'random', 'canny', 'zhang')) + '|')
    lines += ['', 'Cannyは同心・相似な内外周候補対の事前情報を使う統制方式。先行研究の再現実装ではない。',
              'Zhang 2019型（再現実装）は論文構成を参考にした実装とPAF固有後処理であり、著者コードの移植・完全再現ではない。',
              '本比較は既存の2幾何実装との比較であり、幾何手法全般への優越を示すものではない。',
              '円盤の条件違いを独立63形状として扱わない。未検出・誤楕円・形状別件数はshape_results.csvに保存する。', '']
    report_dir = OUTPUT / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write_csv(report_dir / 'shape_results.csv', shape_rows)
    write_csv(report_dir / 'cnn_minus_geometry.csv', contrasts)
    write(report_dir / 'summary.json', {'status': 'passed', 'geometry_records': sum(len(all_rows[m, d]) for m in METHODS for d in COUNTS),
          'comparison': table, 'absent': absent_rows, 'evidence': evidence, 'freeze_sha256': sha(FREEZE),
          'report_sha256': sha(report_dir / 'REPORT.md')})
    print('幾何2方式の全3680推定とCNNの同一画像対応を確認し、比較表を保存しました', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'repro', 'repro-worker', 'evaluate', 'report'])
    parser.add_argument('--method', choices=list(METHODS))
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'report':
        report()
    elif args.method is None:
        parser.error('--methodを指定してください')
    elif args.stage == 'repro':
        reproduce(args.method)
    elif args.stage == 'repro-worker':
        repro_worker(args.method)
    else:
        evaluate(args.method)


if __name__ == '__main__':
    main()
