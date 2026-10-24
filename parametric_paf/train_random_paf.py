"""上端外径D基準の保存済みPAF画像を分離し、学習から独立評価まで実行する。"""

import argparse
import ast
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parametric_paf.random_dataset import read, write, digest

SPEC = ROOT / 'parametric_paf/random_paf/training.json'
OUTPUT = ROOT / 'parametric_paf/output/paf_outer_d_training_v2'
FROZEN = OUTPUT / 'freeze.json'


def versions():
    import cv2
    import numpy as np
    import torch
    return {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
            'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}


def checkpoint(seed):
    return OUTPUT / f'models/{seed}/cnn_best.pt'


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    """原画像を複製せず、形状をまたがない固定分割と入力検査結果を保存する。"""
    import cv2
    import numpy as np
    spec = read(SPEC)
    source = ROOT / spec['source_dataset']
    samples = read(source / 'manifest.json')['samples']
    if len(samples) != sum(spec['split_counts'].values()):
        raise ValueError('分割の合計と元データ枚数が不一致')
    if read(source / 'verification.json')['status'] != 'passed':
        raise ValueError('生成データの全件検査が未完了')
    identities = [tuple(s['parameters'][k] for k in
                  ('top_outer_diameter', 'bottom_outer_diameter', 'height', 'hole_diameter', 'wall_thickness')) for s in samples]
    if len(set(identities)) != len(samples) or len({s['shape_id'] for s in samples}) != len(samples):
        raise ValueError('元データに同一寸法または形状IDの重複があります')
    order = list(range(len(samples)))
    random.Random(spec['split_seed']).shuffle(order)
    folder = OUTPUT / 'dataset'
    prepared, offset = [], 0
    for split, count in spec['split_counts'].items():
        for index in order[offset:offset + count]:
            sample = copy.deepcopy(samples[index])
            sample['split'] = split
            p = sample['parameters']
            if p['top_outer_diameter'] != 1 or p['wall_thickness'] != 0.02 or not (
                    1 <= p['bottom_outer_diameter'] <= 3 and 0.5 <= p['height'] <= 3 and 0.4 <= p['hole_diameter'] <= 0.95):
                raise ValueError('指定した寸法範囲または固定肉厚と不一致')
            for field in ('image', 'label'):
                path = source / sample[field]
                if digest(path) != sample[field + '_sha256']:
                    raise ValueError('元画像・ラベルhash不一致')
                sample[field] = Path(os.path.relpath(path, folder)).as_posix()
            image = cv2.imdecode(np.fromfile(folder / sample['image'], np.uint8), cv2.IMREAD_COLOR)
            label = read(folder / sample['label'])
            points = np.asarray(label['image_points'])
            if image is None or image.shape != (480, 480, 3):
                raise ValueError('元画像の読込・サイズ不正')
            if label['sample_id'] != sample['sample_id'] or label['parameters'] != p or label['conditions'] != sample['conditions']:
                raise ValueError('画像・寸法・ラベルの対応不正')
            if not np.isfinite(points).all() or points.min() < 0 or points.max() >= 480:
                raise ValueError('正解投影不正')
            prepared.append(sample)
        offset += count
    manifest = {'source_manifest_sha256': digest(source / 'manifest.json'),
                'split_seed': spec['split_seed'], 'samples': prepared}
    path = folder / 'manifest.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('保存済み分割と異なります')
    write(path, manifest)
    base = read(ROOT / 'config/experiment.json')
    base['canny_contour_ransac']['preprocess'].update(spec['geometry_thresholds'])
    base['zhang2019']['preprocess'].update(spec['geometry_thresholds'])
    for seed in spec['training_seeds']:
        config = {key: copy.deepcopy(base[key]) for key in ('cnn', 'ransac', 'canny_contour_ransac', 'zhang2019')}
        config.update({'experiment_id': spec['experiment_id'], 'seed': seed, 'input_size': spec['input_size'],
                       'paths': {'training_dataset': folder.relative_to(ROOT).as_posix(),
                                 'checkpoint': checkpoint(seed).relative_to(ROOT).as_posix()}})
        config['cnn'].update(epochs=spec['epochs'], batch_size=spec['batch_size'])
        write(OUTPUT / f'configs/{seed}.json', config)
    write(OUTPUT / 'readiness.json', {'status': 'passed', 'counts': spec['split_counts'],
          'source_manifest_sha256': digest(source / 'manifest.json'), 'manifest_sha256': digest(path),
          'unique_shapes': len(identities), 'shape_and_dimension_splits_disjoint': True,
          'checks': ['all_images_decoded', 'all_source_hashes', 'all_parameter_ranges', 'all_labels_in_frame']})
    files = [Path(__file__), SPEC, ROOT / 'config/experiment.json', path, source / 'manifest.json',
             source / 'verification.json', ROOT / 'parametric_paf/random_paf/config.json',
             ROOT / 'parametric_paf/random_dataset.py', ROOT / 'parametric_paf/paf_model.py',
             ROOT / 'parametric_paf/evaluate_study.py', ROOT / 'parametric_paf/tune_geometry.py',
             ROOT / 'parametric_paf/study.py', ROOT / 'parametric_paf/diagnosis_metrics.py']
    files += list((ROOT / 'paf_ring_detection').rglob('*.py')) + list((OUTPUT / 'configs').glob('*.json'))
    frozen = {'experiment_id': spec['experiment_id'], 'versions': versions(), 'counts': spec['split_counts'],
              'sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(files))}}
    if FROZEN.exists() and read(FROZEN) != frozen:
        raise ValueError('学習前の固定記録と不一致')
    write(FROZEN, frozen)
    print(json.dumps(read(OUTPUT / 'readiness.json'), ensure_ascii=False), flush=True)


def verify_freeze():
    """学習・評価で同じ入力・設定・コード・環境を使っていることを確認する。"""
    frozen = read(FROZEN)
    for relative, value in frozen['sha256'].items():
        if digest(ROOT / relative) != value:
            raise ValueError(f'固定した入力・コードと不一致: {relative}')
    if versions() != frozen['versions']:
        raise ValueError('学習前と実行環境が不一致')


def training_evidence(seed):
    """全epochの履歴と、validation loss最小の採用重みを照合する。"""
    import torch
    spec = read(SPEC)
    path = checkpoint(seed)
    history = read(path.parent / 'training_history.json')
    if [r['epoch'] for r in history] != list(range(1, spec['epochs'] + 1)):
        raise ValueError('全epochの学習が未完了')
    if not all(math.isfinite(r[key]) for r in history for key in ('train_loss', 'validation_loss')):
        raise ValueError('学習lossが非有限')
    model = torch.load(path, map_location='cpu', weights_only=True)
    best = min(history, key=lambda row: row['validation_loss'])
    if model['epoch'] != best['epoch'] or model['validation_loss'] != best['validation_loss']:
        raise ValueError('採用重みと最小validation lossが不一致')
    if not all(torch.isfinite(t).all().item() for t in model['model_state'].values()):
        raise ValueError('学習重みが非有限')
    return {'seed': seed, 'best_epoch': best['epoch'], 'best_validation_loss': best['validation_loss'],
            'checkpoint_sha256': digest(path), 'history_sha256': digest(path.parent / 'training_history.json')}


def train(seed):
    import torch
    from paf_ring_detection.train import train as train_model
    verify_freeze()
    if checkpoint(seed).exists():
        training_evidence(seed)
        return
    torch.set_num_threads(4)
    train_model(read(OUTPUT / f'configs/{seed}.json'))
    training_evidence(seed)


def context(method, seed):
    import cv2
    import torch
    from paf_ring_detection.methods.cnn import load_model
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = read(OUTPUT / f'configs/{seed}.json')
    model = load_model(checkpoint(seed), device) if method == 'cnn' else None
    return model, device, config


def predict(sample, method, model, device, config):
    """検出には画像だけを与え、正解は評価にのみ使用する。"""
    import cv2
    import torch
    from paf_ring_detection.data import read_image, label_ellipse, scaled_ellipse
    from paf_ring_detection.methods.cnn_ransac import detect_from_probability
    from parametric_paf.evaluate_study import evaluate_prediction
    from parametric_paf.tune_geometry import run_geometry
    spec = read(SPEC)
    folder = OUTPUT / 'dataset'
    image = cv2.resize(read_image(folder / sample['image']), (256, 256), interpolation=cv2.INTER_AREA)
    label = read(folder / sample['label'])
    rng_seed = (spec['evaluation_seed'] + zlib.crc32(sample['sample_id'].encode())) % 2**32
    cv2.setRNGSeed(rng_seed & 0x7fffffff)
    truth = scaled_ellipse(label_ellipse(label), 256 / label['image_width'])
    probability_hash = None
    if method == 'cnn':
        import hashlib
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        with torch.inference_mode():
            probability = torch.sigmoid(model(torch.from_numpy(rgb).float().unsqueeze(0).to(device) / 127.5 - 1))[0, 0].cpu().numpy()
        probability_hash = hashlib.sha256(probability.tobytes()).hexdigest()
        cv2.setRNGSeed(rng_seed & 0x7fffffff)
        result = detect_from_probability(probability, config['ransac'], rng_seed)
    else:
        cv2.setRNGSeed(rng_seed & 0x7fffffff)
        result = run_geometry(image, method, config, rng_seed)
    ellipse = result['ellipse'] if result else None
    return {'sample_id': sample['sample_id'], 'shape_id': sample['shape_id'], 'split': sample['split'],
            'method': method, 'ransac_seed': rng_seed, 'estimated_ellipse': ellipse,
            'truth_ellipse': truth, 'probability_sha256': probability_hash,
            **evaluate_prediction(ellipse, truth, image.shape, spec)}


def repro_worker(method, seed):
    validation = [s for s in read(OUTPUT / 'dataset/manifest.json')['samples'] if s['split'] == 'validation']
    selected = [validation[i * 9] for i in range(17)]
    model, device, config = context(method, seed)
    print(json.dumps([predict(s, method, model, device, config) for s in selected], ensure_ascii=False), flush=True)


def reproduce(method, seed):
    """最終評価前に、同じvalidation17画像を独立2プロセスで再現する。"""
    verify_freeze()
    results = []
    for repeat in range(2):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__)), 'repro-worker',
                                 '--method', method, '--seed', str(seed)], cwd=ROOT, check=True,
                                capture_output=True, text=True, encoding='utf-8')
        rows = json.loads(result.stdout.strip().splitlines()[-1])
        results.append(rows)
        write(OUTPUT / f'reproducibility/{method}_{seed}_{repeat}.json', rows)
    if results[0] != results[1]:
        raise ValueError('別プロセス間の推論結果が不一致')
    write(OUTPUT / f'reproducibility/{method}_{seed}.json', {'status': 'passed',
          'images': 17, 'independent_processes': 2, 'freeze_sha256': digest(FROZEN)})
    print(f'Reproducibility passed: {method}/{seed}', flush=True)


def evaluate(method, seed):
    """モデル選択に使わないtest154画像を固定した方法で評価する。"""
    verify_freeze()
    if read(OUTPUT / f'reproducibility/{method}_{seed}.json')['status'] != 'passed':
        raise ValueError('推論再現性の確認が未完了')
    samples = [s for s in read(OUTPUT / 'dataset/manifest.json')['samples'] if s['split'] == 'test']
    if len(samples) != read(SPEC)['split_counts']['test']:
        raise ValueError('最終評価の分母不一致')
    folder = OUTPUT / f'evaluation/{method}/{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    provenance = {'method': method, 'seed': seed, 'n': len(samples), 'freeze_sha256': digest(FROZEN),
                  'checkpoint_sha256': digest(checkpoint(seed)) if method == 'cnn' else None}
    path = folder / 'provenance.json'
    if path.exists() and read(path) != provenance:
        raise ValueError('保存済み評価と条件が不一致')
    write(path, provenance)
    record_path = folder / 'records.jsonl'
    rows = [json.loads(line) for line in record_path.read_text(encoding='utf-8').splitlines()] if record_path.exists() else []
    if [r['sample_id'] for r in rows] != [s['sample_id'] for s in samples[:len(rows)]]:
        raise ValueError('保存済み原記録の対応不正')
    model, device, config = context(method, seed)
    with record_path.open('a', encoding='utf-8', newline='\n') as stream:
        for sample in samples[len(rows):]:
            row = predict(sample, method, model, device, config)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            rows.append(row)
            if len(rows) % 22 == 0:
                print(f'Evaluation {method}/{seed}: {len(rows)}/{len(samples)}', flush=True)
    write_csv(folder / 'results.csv', rows)
    write(folder / 'completion.json', {**provenance, 'status': 'complete',
          'records_sha256': digest(record_path), 'csv_sha256': digest(folder / 'results.csv')})


def report():
    """入力・学習ログ・重み・全評価を照合し、結果と再現用根拠を保存する。"""
    verify_freeze()
    spec = read(SPEC)
    runs = [training_evidence(seed) for seed in spec['training_seeds']]
    for run in runs:
        logged = [ast.literal_eval(line) for line in (OUTPUT / f'logs/train_{run["seed"]}.log').read_text(encoding='utf-8').splitlines()
                  if line.startswith("{'epoch':")]
        if logged != read(checkpoint(run['seed']).parent / 'training_history.json'):
            raise ValueError('学習ログと履歴が不一致')
    expected = [s for s in read(OUTPUT / 'dataset/manifest.json')['samples'] if s['split'] == 'test']
    settings = [('cnn', seed) for seed in spec['training_seeds']] + [('canny', spec['training_seeds'][0]), ('zhang', spec['training_seeds'][0])]
    scores, evidence = [], []
    for method, seed in settings:
        folder = OUTPUT / f'evaluation/{method}/{seed}'
        completed = read(folder / 'completion.json')
        if completed['status'] != 'complete' or completed['freeze_sha256'] != digest(FROZEN):
            raise ValueError('評価が未完了または仕様と不一致')
        for field, filename in [('records_sha256', 'records.jsonl'), ('csv_sha256', 'results.csv')]:
            if completed[field] != digest(folder / filename):
                raise ValueError('評価原記録hash不一致')
        rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
        if [r['sample_id'] for r in rows] != [s['sample_id'] for s in expected] or len(rows) != 154:
            raise ValueError('全test画像の対応不正')
        if method == 'cnn' and completed['checkpoint_sha256'] != digest(checkpoint(seed)):
            raise ValueError('評価した重みと採用モデルが不一致')
        for row in rows:
            detected = row['estimated_ellipse'] is not None
            iou = detected and row['ellipse_iou'] >= spec['success_iou']
            success = iou and row['contour_p95_px'] <= spec['contour_p95_px']
            if row['strict_success'] != success or row['iou_success'] != iou or row['missed'] != (not detected):
                raise ValueError('評価指標・未検出判定不一致')
        if read(OUTPUT / f'reproducibility/{method}_{seed}_0.json') != read(OUTPUT / f'reproducibility/{method}_{seed}_1.json'):
            raise ValueError('推論再現性の原記録が不一致')
        scores.append({'method': method, 'seed': seed, 'n': len(rows),
                       'strict_successes': sum(r['strict_success'] for r in rows),
                       'iou_successes': sum(r['iou_success'] for r in rows),
                       'missed': sum(r['missed'] for r in rows), 'wrong_output': sum(r['wrong_output'] for r in rows)})
        evidence.append({'path': folder.relative_to(ROOT).as_posix(), **completed})
    destination = OUTPUT / 'report'
    destination.mkdir(parents=True, exist_ok=True)
    cnn_mean = sum(row['strict_successes'] / row['n'] * 100 for row in scores if row['method'] == 'cnn') / 3
    write(destination / 'summary.json', {'status': 'passed', 'split_counts': spec['split_counts'], 'runs': runs,
          'scores': scores, 'cnn_mean_strict_success_percent': cnn_mean, 'evaluation_records': 770,
          'freeze_sha256': digest(FROZEN), 'evidence': evidence})
    write_csv(destination / 'scores.csv', scores)
    lines = ['# 上端外径D基準PAFの学習・独立評価', '',
             '指定した1,024画像を学習716・モデル選択154・最終評価154に固定分割した。各画像の寸法は一意で、形状・寸法の重複はない。',
             'Dは上端外径。下端外径1D〜3D、全高0.5D〜3D、穴径0.4D〜0.95D、肉厚0.02D。',
             'Tiny U-Net（16channels）、AdamW、12epoch、batch16、学習率0.001、各540更新。3学習seedをランダム初期化から実行。',
             '全モデルはvalidation loss最小epochを採用。最終評価で選び直さない。',
             'CNNと既存幾何2手法を256入力・同じtest154枚で評価。幾何閾値30/90は固定設定で、今回のデータで再調整していない。', '',
             '主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px。全test画像を分母とする。', '',
             '|方式|学習seed|主成功/154|主成功率|IoU成功/154|未検出|誤楕円|', '|---|---:|---:|---:|---:|---:|---:|']
    names = {'cnn': 'CNN＋重み付きRANSAC', 'canny': 'Canny＋輪郭別RANSAC', 'zhang': 'Zhang 2019型（再現実装）'}
    for row in scores:
        seed = row['seed'] if row['method'] == 'cnn' else 'なし'
        lines.append(f'|{names[row["method"]]}|{seed}|{row["strict_successes"]}|{row["strict_successes"]/154*100:.2f}%|'
                     f'{row["iou_successes"]}|{row["missed"]}|{row["wrong_output"]}|')
    lines += ['', f'CNNの3seed平均主成功率：{cnn_mean:.2f}%。', '',
              'この評価は定義した寸法・描画分布内の未知寸法の推定性能である。1形状1画像なので、同一形状の撮影条件を変えた反復評価ではない。',
              '全画像を同じ合成モデルと描画器から生成しており、実在PAF全般や実画像への汎化を保証しない。',
              '単一寸法学習との比較は含まないため、この結果だけで形状多様化の因果効果を結論しない。',
              '12epochで十分収束したことは未検証。対象なし画像の評価は含まない。',
              'Zhang 2019型（再現実装）は著者コードの完全再現ではなく、論文構成を参考にした実装とPAF固有後処理である。',
              '形状の追加・画像の再生成・テストによる閾値調整は行っていない。', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'全770推定の照合・集計完了。CNN平均 {cnn_mean:.2f}%', flush=True)


def run():
    """保存済みの段階は照合して再利用し、固定した学習・評価だけを順に実行する。"""
    logs = OUTPUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    def step(name, arguments):
        print(f'Start: {name}', flush=True)
        with (logs / f'{name}.log').open('a', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-u', '-X', 'utf8', '-B', str(Path(__file__)), *arguments],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'Complete: {name}', flush=True)
    step('prepare', ['prepare'])
    seeds = read(SPEC)['training_seeds']
    for seed in seeds:
        step(f'train_{seed}', ['train', '--seed', str(seed)])
    for method, seed in [('cnn', seed) for seed in seeds] + [('canny', seeds[0]), ('zhang', seeds[0])]:
        step(f'repro_{method}_{seed}', ['repro', '--method', method, '--seed', str(seed)])
        step(f'evaluate_{method}_{seed}', ['evaluate', '--method', method, '--seed', str(seed)])
    step('report', ['report'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'repro', 'repro-worker', 'evaluate', 'report', 'run'])
    parser.add_argument('--method', choices=['cnn', 'canny', 'zhang'], default='cnn')
    parser.add_argument('--seed', type=int, default=20260907)
    args = parser.parse_args()
    if args.seed not in read(SPEC)['training_seeds']:
        parser.error('指定できる学習seedではありません')
    if args.stage in ('prepare', 'report', 'run'):
        {'prepare': prepare, 'report': report, 'run': run}[args.stage]()
    elif args.stage == 'train':
        train(args.seed)
    else:
        {'repro': reproduce, 'repro-worker': repro_worker, 'evaluate': evaluate}[args.stage](args.method, args.seed)


if __name__ == '__main__':
    main()
