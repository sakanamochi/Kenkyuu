"""保存済み研究データの照合関数と、現行PAFデータ生成の入口。"""

import argparse
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
from parametric_paf.study import read, write, sha, shape_parameters, draw_shape, runtime_versions

PLAN = ROOT / 'parametric_paf/random_study'
OUTPUT = ROOT / 'parametric_paf/output/paf_random_dimensions_v1'
SOURCE_PLAN = ROOT / 'parametric_paf/study'
SOURCE_OUTPUT = ROOT / 'parametric_paf/output/paf_shape_appearance_v1'


def identity(shape):
    """寸法が同じ形状を、異なるIDだけで未知形状と扱わない。"""
    return shape['family'], tuple(sorted(shape['ratios'].items()))


def random_shapes(protocol, reserved):
    """各画像に固有の寸法を割り当て、検証・テスト形状との重複を避ける。"""
    rng = random.Random(protocol['shape_seed'])
    occupied = {identity(s) for s in reserved}
    shapes = []
    while len(shapes) < protocol['train_images_per_arm']:
        ratios = draw_shape(rng, protocol)
        sid = f'random_train_{len(shapes):04d}'
        shape = {'shape_id': sid, 'split': 'train', 'family': 'tapered',
                 'ratios': ratios, 'parameters': shape_parameters(ratios, sid)}
        if identity(shape) not in occupied:
            occupied.add(identity(shape))
            shapes.append(shape)
    return shapes


def rebase_samples(folder, samples, destination):
    """画像を複製せず、保存先からの相対パスで既存成果物を参照する。"""
    return [{**s, **{key: Path(os.path.relpath(folder / s[key], destination)).as_posix()
                    for key in ('image', 'label')}} for s in samples]


def plan():
    protocol = read(PLAN / 'protocol.json')
    reserved = read(SOURCE_PLAN / 'shapes.json')
    conditions = read(SOURCE_PLAN / 'conditions.json')
    source_protocol = read(SOURCE_PLAN / 'protocol.json')
    for key in ('shape_ranges', 'training', 'camera', 'appearance_expanded', 'training_seeds'):
        if protocol[key] != source_protocol[key]:
            raise ValueError(f'単一寸法の既存学習条件と不一致: {key}')
    generated = random_shapes(protocol, reserved)
    value = {'random_shapes': generated, 'evaluation_shapes': [s for s in reserved if s['split'].startswith('test')],
             'single_shape': next(s for s in reserved if s['shape_id'] == 'train_00'),
             'validation_shapes': [s for s in reserved if s['split'] == 'validation'],
             'conditions': conditions}
    target = PLAN / 'plan.json'
    if target.exists() and read(target) != value:
        raise ValueError('保存済み計画と異なります。別実験IDが必要です')
    if not target.exists():
        write(target, value)
    baseline = read(SOURCE_OUTPUT / 'B/manifest.json')['samples']
    write(OUTPUT / 'single/manifest.json', {'samples': rebase_samples(SOURCE_OUTPUT / 'B', baseline, OUTPUT / 'single')})
    validation = read(SOURCE_OUTPUT / 'validation/manifest.json')['samples']
    write(OUTPUT / 'validation/manifest.json', {'samples': rebase_samples(SOURCE_OUTPUT / 'validation', validation, OUTPUT / 'validation')})
    for seed in protocol['training_seeds']:
        config = read(SOURCE_PLAN / f'configs/B_{seed}.json')
        config['experiment_id'] = protocol['experiment_id']
        config['paths']['training_dataset'] = (OUTPUT / 'random').relative_to(ROOT).as_posix()
        config['paths']['checkpoint'] = (OUTPUT / f'models/random/{seed}/cnn_best.pt').relative_to(ROOT).as_posix()
        config['paths']['results'] = (OUTPUT / f'evaluation/random/{seed}').relative_to(ROOT).as_posix()
        write(PLAN / f'configs/random_{seed}.json', config)
    print('計画: 単一寸法1024枚対ランダム寸法1024枚、共通validation192枚', flush=True)


def render(stage):
    """Blenderで固定した寸法と撮影条件を描画し、中断後は検証済み画像を再利用する。"""
    from parametric_paf.render_study import render_group
    protocol = read(PLAN / 'protocol.json')
    planned = read(PLAN / 'plan.json')
    folder = OUTPUT / ('random' if stage == 'train' else 'test')
    jobs = []
    if stage == 'train':
        for shape, condition in zip(planned['random_shapes'], planned['conditions']['train']):
            jobs.append((shape, [condition], 'train'))
    else:
        for shape in planned['evaluation_shapes']:
            jobs.append((shape, planned['conditions']['test'], shape['split']))
            jobs.append((shape, planned['conditions']['supplements'], shape['split'] + '_capture'))
        jobs.append((planned['single_shape'], planned['conditions']['test'], 'single_shape_monitor'))
    progress_path = OUTPUT / f'render_progress/{stage}.json'
    saved = read(progress_path) if progress_path.exists() else {}
    expected_keys = []
    for index, (shape, conditions, split) in enumerate(jobs):
        key = shape['shape_id'] + '/' + split
        expected_keys.append(key)
        if key in saved:
            for sample in saved[key]:
                for field in ('image', 'label'):
                    if sha(folder / sample[field]) != sample[field + '_sha256']:
                        raise ValueError('保存済み描画のhash不一致')
        else:
            saved[key] = render_group(folder, shape, conditions, 'expanded', split, protocol)
            write(progress_path, saved)
        if (index + 1) % 16 == 0:
            print(f'Render jobs {index + 1}/{len(jobs)}', flush=True)
    samples = [s for key in expected_keys for s in saved[key]]
    if stage == 'train':
        validation = read(OUTPUT / 'validation/manifest.json')['samples']
        samples += rebase_samples(OUTPUT / 'validation', validation, folder)
    write(folder / 'manifest.json', {'experiment_id': protocol['experiment_id'], 'samples': samples})
    write(folder / 'render_provenance.json', {'plan_sha256': sha(PLAN / 'plan.json'),
          'protocol_sha256': sha(PLAN / 'protocol.json'), 'code_sha256': sha(Path(__file__)),
          'stage': stage, 'sample_count': len(samples)})
    print(f'Render complete: {stage} {len(samples)} samples', flush=True)


def checkpoint_path(arm, seed):
    """単一寸法の検証済み重みを参照し、ランダム寸法の重みとは保存先を分ける。"""
    if arm == 'single':
        return SOURCE_OUTPUT / f'models/B/{seed}/cnn_best.pt'
    return OUTPUT / f'models/random/{seed}/cnn_best.pt'


def verify_sources():
    """再利用する学習データ・設定・重みを、既存の凍結記録と照合する。"""
    from parametric_paf.study import verify_freeze
    verify_freeze()
    summary = read(SOURCE_OUTPUT / 'training_summary/summary.json')
    for item in summary['evidence']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError(f'再利用元の成果物が変更されています: {item["path"]}')
    protocol = read(PLAN / 'protocol.json')
    for seed in protocol['training_seeds']:
        old = read(SOURCE_PLAN / f'configs/B_{seed}.json')
        new = read(PLAN / f'configs/random_{seed}.json')
        if old['cnn'] != new['cnn'] or old['ransac'] != new['ransac'] or old['seed'] != new['seed']:
            raise ValueError('再利用する単一寸法モデルと学習・推論条件が異なります')
    return summary


def check():
    """全学習画像の対応、寸法分離、正解投影、同じ学習予算を確認する。"""
    import cv2
    import numpy as np
    from paf_ring_detection.data import read_image, RingDataset
    verify_sources()
    protocol = read(PLAN / 'protocol.json')
    planned = read(PLAN / 'plan.json')
    shapes = planned['random_shapes']
    reserved = read(SOURCE_PLAN / 'shapes.json')
    if len(shapes) != 1024 or len({identity(s) for s in shapes}) != 1024:
        raise ValueError('ランダム学習形状数が不正')
    if {identity(s) for s in shapes} & {identity(s) for s in reserved}:
        raise ValueError('学習寸法と検証・テスト寸法が重複')
    for shape in shapes:
        if not all(a <= shape['ratios'][key] <= b for key, (a, b) in protocol['shape_ranges'].items()):
            raise ValueError('学習寸法が範囲外')
    manifests = {arm: read(OUTPUT / arm / 'manifest.json')['samples'] for arm in protocol['arms']}
    labels = {}
    for arm, samples in manifests.items():
        labels[arm] = []
        train_samples = [s for s in samples if s['split'] == 'train']
        validation = [s for s in samples if s['split'] == 'validation']
        if len(train_samples) != 1024 or len(validation) != 192 or len({s['sample_id'] for s in samples}) != 1216:
            raise ValueError('学習・検証枚数または画像IDが不正')
        if len({s['shape_id'] for s in train_samples}) != (1 if arm == 'single' else 1024):
            raise ValueError('条件ごとの形状数が不正')
        for sample in samples:
            folder = OUTPUT / arm
            for field in ('image', 'label'):
                if sha(folder / sample[field]) != sample[field + '_sha256']:
                    raise ValueError('画像またはラベルhash不一致')
            image = read_image(folder / sample['image'])
            label = read(folder / sample['label'])
            points = np.asarray(label['image_points'])
            if image.shape != (480, 480, 3) or not np.isfinite(points).all() or points.min() < 0 or points.max() >= 480:
                raise ValueError('画像形式または正解投影不正')
            if sample['condition'] != label['conditions'] or sample['shape_id'] != label['shape_id']:
                raise ValueError('画像と正解の対応不正')
            if sample['split'] == 'train':
                index = len(labels[arm])
                expected_shape = planned['single_shape'] if arm == 'single' else shapes[index]
                if label['parameters'] != expected_shape['parameters']:
                    raise ValueError('計画した寸法とラベルが不一致')
                labels[arm].append(points)
        dataset = RingDataset(OUTPUT / arm, 'train', 256)
        image, mask, _ = dataset[0]
        if image.shape != (3, 256, 256) or mask.shape != (1, 256, 256):
            raise ValueError('CNN入力または教師マスクの形状不正')
    for index in range(1024):
        a, b = manifests['single'][index], manifests['random'][index]
        if a['condition'] != b['condition']:
            raise ValueError('撮像・外観条件が条件間で不一致')
        if not np.allclose(labels['single'][index], labels['random'][index], atol=1e-3, rtol=0):
            raise ValueError('条件間の内周投影寸法が不一致')
    for a, b in zip(manifests['single'][1024:], manifests['random'][1024:]):
        if a['image_sha256'] != b['image_sha256'] or a['label_sha256'] != b['label_sha256']:
            raise ValueError('共通validationが条件間で不一致')
    result = {'status': 'passed', 'train_images_per_arm': 1024, 'validation_images': 192,
              'single_shapes': 1, 'random_shapes': 1024, 'same_capture_and_appearance': True,
              'same_projected_inner_rim': True, 'shape_split_disjoint': True,
              'manifest_sha256': {arm: sha(OUTPUT / arm / 'manifest.json') for arm in protocol['arms']}}
    write(OUTPUT / 'readiness.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def freeze():
    """新しい学習の前に計画・学習器・評価経路・入力manifest・再利用重みを凍結する。"""
    if (PLAN / 'freeze.json').exists():
        verify_freeze()
        return
    check()
    if list((OUTPUT / 'models').rglob('*.pt')):
        raise ValueError('ランダム寸法学習の開始前に凍結してください')
    protocol = read(PLAN / 'protocol.json')
    files = list((ROOT / 'paf_ring_detection').rglob('*.py'))
    files += list((ROOT / 'parametric_paf').glob('*.py'))
    files += [PLAN / 'protocol.json', PLAN / 'plan.json'] + list((PLAN / 'configs').glob('*.json'))
    files += [OUTPUT / f'{arm}/manifest.json' for arm in protocol['arms']]
    files += [OUTPUT / 'validation/manifest.json', SOURCE_PLAN / 'freeze.json']
    for seed in protocol['training_seeds']:
        files += [checkpoint_path('single', seed), checkpoint_path('single', seed).parent / 'training_history.json']
    write(PLAN / 'freeze.json', {'experiment_id': protocol['experiment_id'], 'date': '2026-10-24',
          'new_training_started': False, 'independent_test_evaluated': False,
          'reused_single_models': True, 'runtime_versions': runtime_versions(),
          'sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}})
    print('ランダム寸法学習前の仕様・入力・コードを凍結しました', flush=True)


def verify_freeze():
    frozen = read(PLAN / 'freeze.json')
    if frozen.get('status') == 'reference_only':
        raise ValueError('現行データ生成にはparametric_paf/random_dataset.pyを使用してください')
    for relative, digest in frozen['sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'凍結内容と不一致: {relative}')
    if runtime_versions() != frozen['runtime_versions']:
        raise ValueError('主要ライブラリ版が凍結時と不一致')


def train_one(seed):
    """共通学習器をそのまま使い、完了済みの重みは上書きしない。"""
    import torch
    from paf_ring_detection.train import train
    verify_freeze()
    config = read(PLAN / f'configs/random_{seed}.json')
    path = checkpoint_path('random', seed)
    if path.exists():
        verify_training('random', seed)
        print(f'Training complete, reusing random/{seed}', flush=True)
        return
    torch.set_num_threads(4)
    train(config)
    verify_training('random', seed)


def verify_training(arm, seed):
    """全epochの履歴と、最小validation lossの採用重みを照合する。"""
    import torch
    path = checkpoint_path(arm, seed)
    history = read(path.parent / 'training_history.json')
    if [r['epoch'] for r in history] != list(range(1, 13)):
        raise ValueError('12epochの学習が未完了')
    if not all(math.isfinite(r[key]) for r in history for key in ('train_loss', 'validation_loss')):
        raise ValueError('学習lossが非有限')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    best = min(history, key=lambda row: row['validation_loss'])
    if checkpoint['epoch'] != best['epoch'] or checkpoint['validation_loss'] != best['validation_loss']:
        raise ValueError('最小validation lossと採用重みが不一致')
    if not all(torch.isfinite(t).all().item() for t in checkpoint['model_state'].values()):
        raise ValueError('重みが非有限')
    return {'arm': arm, 'seed': seed, 'best_epoch': best['epoch'],
            'best_validation_loss': best['validation_loss'], 'checkpoint_sha256': sha(path),
            'history_sha256': sha(path.parent / 'training_history.json')}


def inference_context(arm, seed):
    """両条件の推論で同じ設定と数値実行環境を使用する。"""
    import cv2
    import torch
    from paf_ring_detection.methods.cnn import load_model
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path(arm, seed), device)
    config = read(PLAN / f'configs/random_{seed}.json')
    return model, device, config


def predict_sample(sample, folder, model, device, config):
    """OpenCVとNumPy双方の乱数を画像単位で固定し、正解を検出器に渡さず推定する。"""
    import cv2
    import numpy as np
    import torch
    import hashlib
    from paf_ring_detection.data import read_image, label_ellipse, scaled_ellipse
    from paf_ring_detection.methods.cnn_ransac import detect_from_probability
    from parametric_paf.evaluate_study import evaluate_prediction
    from parametric_paf.diagnosis_metrics import probability_support
    protocol = read(PLAN / 'protocol.json')
    image = cv2.resize(read_image(folder / sample['image']), (256, 256), interpolation=cv2.INTER_AREA)
    label = read(folder / sample['label'])
    rng_seed = (protocol['test_seed'] + zlib.crc32(sample['condition']['condition_id'].encode())) % 2**32
    cv2.setRNGSeed(rng_seed & 0x7fffffff)
    truth = scaled_ellipse(label_ellipse(label), 256 / label['image_width']) if label.get('target_present', True) else None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.from_numpy(rgb).float().unsqueeze(0).to(device) / 127.5 - 1))[0, 0].cpu().numpy()
    cv2.setRNGSeed(rng_seed & 0x7fffffff)
    result = detect_from_probability(probability, config['ransac'], rng_seed)
    ellipse = result['ellipse'] if result else None
    metrics = evaluate_prediction(ellipse, truth, image.shape, protocol['evaluation'])
    support = probability_support(probability, truth)[2] if truth else {}
    return {'sample_id': sample['sample_id'], 'shape_id': sample['shape_id'], 'split': sample['split'],
            'family': label.get('family', label.get('kind')),
            'condition_id': sample['condition']['condition_id'], 'camera_tilt': sample['condition']['camera_tilt'],
            'ransac_seed': rng_seed, 'estimated_ellipse': ellipse, **metrics,
            'probability_full_contour_coverage': support.get('full_contour_coverage'),
            'probability_sha256': hashlib.sha256(probability.tobytes()).hexdigest()}


def repro_worker(arm):
    """各seedで固定した検証画像を評価し、別プロセス間の一致を確認するための出力を返す。"""
    protocol = read(PLAN / 'protocol.json')
    samples = read(OUTPUT / 'validation/manifest.json')['samples']
    selected = [samples[i] for i in sorted(set(list(range(0, 192, 12)) + [157]))]
    rows = []
    for seed in protocol['training_seeds']:
        model, device, config = inference_context(arm, seed)
        for sample in selected:
            rows.append({'seed': seed, **predict_sample(sample, OUTPUT / 'validation', model, device, config)})
    print(json.dumps(rows, ensure_ascii=False), flush=True)


def reproducibility(arm):
    """検証用の同じ17画像を独立2プロセスで推論し、確率・楕円・指標の一致を要求する。"""
    verify_freeze()
    outputs = []
    for repeat in range(2):
        command = [sys.executable, '-X', 'utf8', '-B', str(Path(__file__)), 'repro-worker', '--arm', arm]
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, encoding='utf-8')
        rows = json.loads(result.stdout.strip().splitlines()[-1])
        outputs.append(rows)
        write(OUTPUT / f'reproducibility/{arm}_{repeat}.json', rows)
    passed = outputs[0] == outputs[1]
    write(OUTPUT / f'reproducibility/{arm}.json', {'status': 'passed' if passed else 'failed',
          'independent_processes': 2, 'images_per_seed': 17, 'training_seeds': read(PLAN / 'protocol.json')['training_seeds'],
          'freeze_sha256': sha(PLAN / 'freeze.json'), 'exact_probability_ellipse_metrics_match': passed})
    if not passed:
        raise ValueError('推論の別プロセス間再現性が不一致')
    print(f'Reproducibility passed: {arm}, 17 images x 3 seeds x 2 processes', flush=True)


def validate_test():
    """独立評価の分母と固定済み計画の対応を確認し、途中生成データの評価を防ぐ。"""
    import numpy as np
    from collections import Counter
    samples = read(OUTPUT / 'test/manifest.json')['samples']
    expected = {'test_interpolation': 576, 'test_extrapolation': 576,
                'test_interpolation_capture': 192, 'test_extrapolation_capture': 192,
                'single_shape_monitor': 48}
    if dict(Counter(s['split'] for s in samples)) != expected or len({s['sample_id'] for s in samples}) != 1584:
        raise ValueError('独立テストが未完了、または画像数が不一致')
    planned = read(PLAN / 'plan.json')
    shapes = {s['shape_id']: s for s in planned['evaluation_shapes'] + [planned['single_shape']]}
    for sample in samples:
        for field in ('image', 'label'):
            if sha(OUTPUT / 'test' / sample[field]) != sample[field + '_sha256']:
                raise ValueError('独立テストの画像・ラベルhash不一致')
        label = read(OUTPUT / 'test' / sample['label'])
        points = np.asarray(label['image_points'])
        if label['parameters'] != shapes[sample['shape_id']]['parameters'] or label['conditions'] != sample['condition']:
            raise ValueError('独立テストの固定条件とラベルが不一致')
        if not np.isfinite(points).all() or points.min() < 0 or points.max() >= 480:
            raise ValueError('独立テストの正解が画面外')
    write(OUTPUT / 'test_readiness.json', {'status': 'passed', 'counts': expected,
          'manifest_sha256': sha(OUTPUT / 'test/manifest.json'), 'freeze_sha256': sha(PLAN / 'freeze.json')})


def evaluate(arm, seed, dataset):
    """各モデルに同一の未調整テストを与え、原記録を保存しながら評価する。"""
    verify_freeze()
    verify_training(arm, seed)
    if read(OUTPUT / f'reproducibility/{arm}.json')['status'] != 'passed':
        raise ValueError('検証データで推論再現性を確認してください')
    folder = OUTPUT / dataset
    samples = read(folder / 'manifest.json')['samples']
    if dataset == 'test':
        validate_test()
    elif dataset == 'absent':
        if len(samples) != 64 or any(read(folder / s['label']).get('target_present', True) for s in samples):
            raise ValueError('対象なしテストの構成不正')
    destination = OUTPUT / f'evaluation/{arm}/{seed}/{dataset}'
    provenance = {'checkpoint_sha256': sha(checkpoint_path(arm, seed)),
                  'manifest_sha256': sha(folder / 'manifest.json'), 'freeze_sha256': sha(PLAN / 'freeze.json'),
                  'sample_count': len(samples), 'dataset': dataset, 'arm': arm, 'seed': seed}
    destination.mkdir(parents=True, exist_ok=True)
    prov_path = destination / 'provenance.json'
    if prov_path.exists() and read(prov_path) != provenance:
        raise ValueError('既存評価の入力・仕様と不一致')
    write(prov_path, provenance)
    records = destination / 'records.jsonl'
    rows = [json.loads(line) for line in records.read_text(encoding='utf-8').splitlines()] if records.exists() else []
    if [r['sample_id'] for r in rows] != [s['sample_id'] for s in samples[:len(rows)]]:
        raise ValueError('保存済み原記録の画像対応不正')
    model, device, config = inference_context(arm, seed)
    with records.open('a', encoding='utf-8', newline='\n') as stream:
        for sample in samples[len(rows):]:
            row = predict_sample(sample, folder, model, device, config)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            rows.append(row)
            if len(rows) % 24 == 0:
                print(f'Evaluation {arm}/{seed}/{dataset}: {len(rows)}/{len(samples)}', flush=True)
    write_csv(destination / 'results.csv', rows)
    write(destination / 'completion.json', {**provenance, 'status': 'complete',
          'records_sha256': sha(records), 'results_sha256': sha(destination / 'results.csv')})


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_absent():
    """凍結済みの黒画像1枚・穴なし円盤63枚を共通の補足テストとして参照する。"""
    source = SOURCE_OUTPUT / 'absent_test'
    if not (source / 'manifest.json').exists():
        blender = read(PLAN / 'configs/random_20260907.json')['blender']
        subprocess.run([blender, '--background', '--python-exit-code', '1', '--python',
                        str(ROOT / 'parametric_paf/render_absent_study.py')], cwd=ROOT, check=True)
    samples = read(source / 'manifest.json')['samples']
    for sample in samples:
        for field in ('image', 'label'):
            if sha(source / sample[field]) != sample[field + '_sha256']:
                raise ValueError('対象なし画像・ラベルhash不一致')
    write(OUTPUT / 'absent/manifest.json', {'samples': rebase_samples(source, samples, OUTPUT / 'absent'),
          'source_provenance_sha256': sha(source / 'provenance.json')})


def summarize():
    """全6モデルの完了と全画像の評価を照合し、形状単位の対応差を報告する。"""
    from collections import defaultdict
    import numpy as np
    verify_freeze()
    protocol = read(PLAN / 'protocol.json')
    seeds = protocol['training_seeds']
    runs, shape_rows, evidence = [], [], []
    by_run = {}
    for arm in protocol['arms']:
        for seed in seeds:
            training = verify_training(arm, seed)
            for dataset, count in [('validation', 192), ('test', 1584), ('absent', 64)]:
                folder = OUTPUT / f'evaluation/{arm}/{seed}/{dataset}'
                completed = read(folder / 'completion.json')
                if completed['status'] != 'complete' or completed['sample_count'] != count:
                    raise ValueError('評価が未完了')
                if completed['checkpoint_sha256'] != training['checkpoint_sha256']:
                    raise ValueError('評価と学習済み重みが不一致')
                if completed['manifest_sha256'] != sha(OUTPUT / dataset / 'manifest.json'):
                    raise ValueError('評価と入力manifestが不一致')
                for field, name in [('records_sha256', 'records.jsonl'), ('results_sha256', 'results.csv')]:
                    if completed[field] != sha(folder / name):
                        raise ValueError('保存評価hash不一致')
                rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
                expected = read(OUTPUT / dataset / 'manifest.json')['samples']
                if len(rows) != count or [r['sample_id'] for r in rows] != [s['sample_id'] for s in expected]:
                    raise ValueError('評価画像対応不正')
                for row in rows:
                    if row['target_present']:
                        success = row['detected'] and row['ellipse_iou'] >= 0.8 and row['contour_p95_px'] <= 2
                        if row['strict_success'] != success or row['missed'] != (not row['detected']):
                            raise ValueError('成功判定が仕様と不一致')
                        if row['wrong_output'] != (row['detected'] and not success):
                            raise ValueError('誤楕円判定が仕様と不一致')
                    elif row['absent_false_positive'] != row['detected']:
                        raise ValueError('対象なし判定が仕様と不一致')
                by_run[arm, seed, dataset] = rows
                grouped = defaultdict(list)
                for row in rows:
                    grouped[row['split'], row['shape_id']].append(row)
                for (split, sid), group in grouped.items():
                    present = group[0]['target_present']
                    distances = [r['contour_p95_px'] for r in group if r['contour_p95_px'] is not None]
                    shape_rows.append({'arm': arm, 'seed': seed, 'dataset': dataset, 'split': split,
                                       'shape_id': sid, 'family': group[0]['family'], 'n': len(group),
                                       'successes': sum(r['strict_success'] for r in group) if present else None,
                                       'missed': sum(r['missed'] for r in group) if present else None,
                                       'wrong_output': sum(r['wrong_output'] for r in group) if present else None,
                                       'false_positives': sum(r['absent_false_positive'] for r in group) if not present else None,
                                       'mean_contour_p95_px_detected': float(np.mean(distances)) if distances else None})
                evidence.append({'path': folder.relative_to(ROOT).as_posix(), **completed})
            runs.append(training)
    contrasts = []
    splits = sorted({r['split'] for r in shape_rows if r['dataset'] == 'test'})
    for split in splits:
        for seed in seeds:
            base_split, _, family = split.partition(':')
            selected = [r for r in shape_rows if r['split'] == base_split and r['seed'] == seed
                        and (not family or r['family'] == family)]
            lookup = {(r['arm'], r['shape_id']): r for r in selected}
            ids = sorted({r['shape_id'] for r in selected})
            differences = [(lookup['random', sid]['successes'] - lookup['single', sid]['successes']) /
                           lookup['single', sid]['n'] * 100 for sid in ids]
            contrasts.append({'split': split, 'seed': seed, 'shape_count': len(ids),
                              'mean_shape_difference_pp': float(np.mean(differences)),
                              'improved_shapes': sum(x > 0 for x in differences),
                              'worsened_shapes': sum(x < 0 for x in differences),
                              'shape_differences_pp': dict(zip(ids, differences))})
    primary = [r['mean_shape_difference_pp'] for r in contrasts if r['split'] == protocol['primary_split']]
    primary_supported = all(x > 0 for x in primary) and np.mean(primary) >= 10
    monitor = [r['mean_shape_difference_pp'] for r in contrasts if r['split'] == 'single_shape_monitor']
    target_absent = []
    for seed in seeds:
        rates = {arm: sum(r['absent_false_positive'] for r in by_run[arm, seed, 'absent']) / 64 * 100
                 for arm in protocol['arms']}
        target_absent.append({'seed': seed, **rates, 'difference_pp': rates['random'] - rates['single']})
    report_dir = OUTPUT / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {'experiment_id': protocol['experiment_id'], 'primary_hypothesis_supported': bool(primary_supported),
              'primary_mean_difference_pp': float(np.mean(primary)),
              'single_shape_monitor_mean_difference_pp': float(np.mean(monitor)),
              'single_shape_monitor_within_5pp_each_seed': all(x >= -5 for x in monitor),
              'absent_false_positive_rates_percent': target_absent,
              'runs': runs, 'contrasts': contrasts, 'evidence': evidence,
              'freeze_sha256': sha(PLAN / 'freeze.json'), 'runtime_versions': runtime_versions()}
    write(report_dir / 'summary.json', result)
    write_csv(report_dir / 'shape_results.csv', shape_rows)
    write_csv(report_dir / 'contrasts.csv', [{k: v for k, v in r.items() if k != 'shape_differences_pp'} for r in contrasts])
    lines = ['# ランダム寸法によるPAF内周検出の独立評価', '', '実験ID：`paf_random_dimensions_v1`。', '',
             '単一寸法と、画像ごとに寸法を抽選した1024形状を比較する。両条件は1024枚・12epoch・768更新・3学習seed。',
             '視点・外観・照明・正解内周投影・CNN・後処理を対応させる。1194MのCAD・画像・重みは学習に使わない。',
             '単一寸法の学習済み3モデルは条件一致とhashを検証して再利用する。固定16形状は主比較に含めない。', '',
             '主指標：IoU≥0.8かつ双方向輪郭距離p95≤2px（256入力）。分母は未検出を含む全画像。',
             '補間12形状を主テストとし、外挿12形状・撮像条件は別集計する。',
             'テストによる学習・閾値・checkpointの選び直しは行わない。', '',
             '|評価区分|seed|単一寸法 成功/全画像|ランダム寸法 成功/全画像|形状平均差（ポイント）|',
             '|---|---:|---:|---:|---:|']
    for contrast in contrasts:
        split, seed = contrast['split'], contrast['seed']
        totals = {}
        for arm in protocol['arms']:
            base_split, _, family = split.partition(':')
            group = [r for r in by_run[arm, seed, 'test'] if r['split'] == base_split
                     and (not family or r['family'] == family)]
            totals[arm] = f'{sum(r["strict_success"] for r in group)}/{len(group)}'
        lines.append(f'|{split}|{seed}|{totals["single"]}|{totals["random"]}|{contrast["mean_shape_difference_pp"]:+.2f}|')
    lines += ['', f'主テストの3 seed平均差：{np.mean(primary):+.2f}ポイント。',
              '事前基準（全seedで正、平均10ポイント以上）：' + ('支持。' if primary_supported else '未達。'),
              f'単一寸法monitorの平均差：{np.mean(monitor):+.2f}ポイント。これは単一寸法側の学習形状1個に限った補足。', '',
              '|seed|対象なし誤検出：単一寸法|対象なし誤検出：ランダム寸法|', '|---|---:|---:|']
    for row in target_absent:
        lines.append(f'|{row["seed"]}|{row["single"]:.2f}%|{row["random"]:.2f}%|')
    lines += ['', '対象なしは黒画像1枚と穴なし円盤63枚であり、64個の独立形状ではない。',
              '形状ごとの成績と改善・悪化はshape_results.csvとsummary.jsonを参照。未検出・誤楕円も全画像を分母に保存する。',
              '学習seedはモデル初期化・学習順序の3反復であり、ランダム寸法データ自体の再抽選3反復ではない。',
              '形状数の必要十分性、十分な収束、実在PAF全般、実画像、背景変化への一般化は未検証。',
              '全原記録・入力manifest・重み・凍結記録の対応はsummary.jsonに保存する。', '']
    (report_dir / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'独立評価集計完了: 主テスト差 {np.mean(primary):+.2f}ポイント', flush=True)


def run_pipeline():
    """固定した2条件の検証を順に実行し、完了した段階は入力一致を確認して再利用する。"""
    logs = OUTPUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    protocol = read(PLAN / 'protocol.json')
    blender = read(SOURCE_PLAN / 'configs/B_20260907.json')['blender']
    script = str(Path(__file__))
    def run(label, arguments, rendering=False):
        command = ([blender, '--background', '--python-exit-code', '1', '--python', script, '--'] if rendering
                   else [sys.executable, '-X', 'utf8', '-B', script]) + arguments
        print(f'Start: {label}', flush=True)
        with (logs / f'{label}.log').open('a', encoding='utf-8') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'Complete: {label}', flush=True)
    if not (PLAN / 'freeze.json').exists():
        run('plan', ['plan'])
        run('render_train', ['render-train'], True)
        run('freeze', ['freeze'])
    else:
        verify_freeze()
        run('check', ['check'])
    run('repro_single', ['repro', '--arm', 'single'])
    for seed in protocol['training_seeds']:
        run(f'train_random_{seed}', ['train', '--seed', str(seed)])
    run('repro_random', ['repro', '--arm', 'random'])
    for arm in protocol['arms']:
        for seed in protocol['training_seeds']:
            run(f'evaluate_{arm}_{seed}_validation', ['evaluate', '--arm', arm, '--seed', str(seed), '--dataset', 'validation'])
    run('render_test', ['render-test'], True)
    run('render_absent', ['render-absent'])
    for arm in protocol['arms']:
        for seed in protocol['training_seeds']:
            for dataset in ('test', 'absent'):
                run(f'evaluate_{arm}_{seed}_{dataset}', ['evaluate', '--arm', arm, '--seed', str(seed), '--dataset', dataset])
    run('summarize', ['summarize'])


def main():
    """現行のD基準データ生成を実行する。"""
    from parametric_paf.random_dataset import main as dataset_main
    dataset_main()


if __name__ == '__main__':
    main()
