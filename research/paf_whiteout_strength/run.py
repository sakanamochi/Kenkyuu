"""検証画像で白飛びの強度範囲を決定し、固定4方式でテスト評価する。"""

import argparse
import hashlib
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import (read_json, write_json, read_image, load_samples,
                                     label_ellipse, scaled_ellipse)
from paf_ring_detection.effects import apply_effect
from paf_ring_detection.evaluate import _geometry
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_whiteout_strength.effect import apply_whiteout

BASE = ROOT/'output/hole_region_comparison_v1/datasets'
OUT = ROOT/'output/paf_whiteout_strength_v1'
PRIOR = ROOT/'output/paf_1194m_no_surface_v1'
CONFIG = Path(__file__).with_name('config.json')
METHODS = ('1194m', 'diversified', 'canny', 'zhang2019')
DATASETS = ('1194m', 'diversified')


def sha(path):
    """入力・実装・重みの対応を確認するためSHA256を保存する。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_paths(seed):
    return {'1194m': PRIOR/f'runs/1194m/{seed}/best.pt',
            'diversified': ROOT/f'output/paf_generalization_v2/runs/boundary_projection_relative/{seed}/best.pt'}


def prepare():
    """強度の選択規則と依存ファイルを評価前に固定する。"""
    config = read_json(CONFIG)
    files = [CONFIG, Path(__file__), Path(__file__).with_name('effect.py'),
             ROOT/'config/experiment.json', ROOT/'research/paf_fair_comparison/no_surface.json',
             ROOT/'paf_ring_detection/effects.py', ROOT/'paf_ring_detection/geometry.py',
             ROOT/'paf_ring_detection/evaluate.py', ROOT/'paf_ring_detection/methods/ransac.py',
             ROOT/'research/paf_generalization/boundary.py']
    files += list(model_paths(config['seed']).values())
    files += [BASE/d/'manifest.json' for d in DATASETS]
    plan = {'config': config, 'files': {str(p.relative_to(ROOT)): sha(p) for p in files},
            'calibration_originals': {d: 105 for d in DATASETS},
            'test_originals': {'1194m': 112, 'diversified': 154},
            'selection_uses_test': False, 'retrained': False}
    path = OUT/'plan.json'
    if path.exists():
        assert read_json(path) == plan, '固定した依存ファイル・設定が変わっています。'
    else:
        write_json(path, plan)
    return plan


def load_models(seed):
    models = {}
    for source, path in model_paths(seed).items():
        assert read_json(path.parent/'complete.json')['checkpoint_sha256'] == sha(path)
        model = TinyUNet(16).cuda().eval()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state'])
        models[source] = model
    return models


def samples_for(dataset, split, severity):
    """テストは既存白飛び行の乱数を使い、validationは最大強度用の固定行を作る。"""
    samples = load_samples(BASE/dataset, split)
    if split == 'test':
        return [s for s in samples if s['conditions']['degradation'] == 'sensor_whiteout'
                and s['conditions']['severity'] == severity]
    result = []
    for original in samples:
        if original['conditions']['degradation'] != 'clean':
            continue
        base_id = original['conditions']['base_sample_id']
        noise_seed = (2026102501 + zlib.crc32(base_id.encode('utf-8'))) % 2**32
        result.append({**original, 'sample_id': base_id+'__whiteout_calibration',
                       'augmentation': {'seed': noise_seed},
                       'conditions': {**original['conditions'], 'degradation': 'sensor_whiteout',
                                      'severity': severity}})
    return result


def inputs(dataset, sample, severity, maximum_strength):
    root = BASE/dataset
    image = read_image(root/sample['image'])
    label = read_json(root/sample['label'])
    seed = sample['augmentation']['seed']
    enhanced = apply_whiteout(image, severity, np.random.default_rng(seed), maximum_strength)
    # 拡張倍率1では元の処理と全画素が一致することを確認する。
    original = apply_effect(image, 'sensor_whiteout', severity, np.random.default_rng(seed))
    equivalent = apply_whiteout(image, severity, np.random.default_rng(seed), 1.)
    assert np.array_equal(original, equivalent)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    shared = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (256, 256), interpolation=cv2.INTER_AREA)
    truth = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    return shared, truth


def evaluate(dataset, split, severity, maximum_strength, models, plan):
    destination = OUT/f'results/{split}/{dataset}_max{maximum_strength:g}_s{severity:g}.json'
    if destination.exists():
        saved = read_json(destination)
        assert saved['plan_sha256'] == sha(OUT/'plan.json')
        return saved
    cnn_config = read_json(ROOT/'research/paf_fair_comparison/no_surface.json')
    geometry_config = read_json(ROOT/'config/experiment.json')
    samples = samples_for(dataset, split, severity)
    rows = []
    for index, sample in enumerate(samples):
        image, truth = inputs(dataset, sample, severity, maximum_strength)
        # 4方式へ同一のモノクロ256画像を渡し、重み・検出閾値は固定する。
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        predictions = {}
        if np.ptp(image) != 0:
            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                probabilities = {m: torch.sigmoid(models[m](tensor))[0, 0].cpu().numpy()
                                 for m in ('1194m', 'diversified')}
            for method in ('1194m', 'diversified'):
                cv2.setRNGSeed(cnn_config['opencv_seed_per_image'])
                predictions[method] = boundary_ellipse(probabilities[method], cnn_config['ransac'])
            random_seed = (geometry_config['evaluation_seed'] + zlib.crc32(sample['sample_id'].encode('utf-8'))) % 2**32
            for method in ('canny', 'zhang2019'):
                predictions[method] = _geometry(image, method, geometry_config, random_seed)
        for method in METHODS:
            ellipse = predictions.get(method)
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'base_sample_id': sample['conditions']['base_sample_id'],
                         'group_id': sample['group_id'], 'method': method, 'ellipse': ellipse,
                         'truth_ellipse': truth, 'ellipse_iou': iou, 'success': iou >= .8,
                         'image_sha256': image_hash, 'noise_seed': sample['augmentation']['seed']})
        if (index+1) % 25 == 0:
            print(f'{split} {dataset} max={maximum_strength:g} s={severity:g} {index+1}/{len(samples)}', flush=True)
    result = {'dataset': dataset, 'split': split, 'severity': severity, 'maximum_strength': maximum_strength,
              'seed': plan['config']['seed'], 'plan_sha256': sha(OUT/'plan.json'), 'rows': rows,
              'summary': {m: {'n': len(samples),
                              'success_percent': 100*np.mean([r['success'] for r in rows if r['method'] == m]),
                              'mean_iou': np.mean([r['ellipse_iou'] for r in rows if r['method'] == m])}
                          for m in METHODS}}
    write_json(destination, result)
    print('DONE', split, dataset, maximum_strength, severity,
          {m: round(v['success_percent'], 2) for m, v in result['summary'].items()}, flush=True)
    return result


def calibrate(plan, models):
    """方式間の順位を使わず、検証集合で十分な劣化が生じる最小強度を選ぶ。"""
    config = plan['config']
    values = []
    for maximum in config['calibration_maximum_strengths']:
        results = [evaluate(d, 'validation', 1., maximum, models, plan) for d in DATASETS]
        pooled = float(np.mean([s['success_percent'] for r in results for s in r['summary'].values()]))
        values.append({'maximum_strength': maximum, 'pooled_success_percent': pooled,
                       'by_dataset': {r['dataset']: r['summary'] for r in results}})
    original = values[0]['pooled_success_percent']
    low, high = config['acceptable_pooled_success_percent']
    eligible = [v for v in values[1:] if low <= v['pooled_success_percent'] <= high
                and original-v['pooled_success_percent'] >= config['minimum_drop_from_original_pp']]
    result = {'calibration': values, 'selection_rule': config['selection_rule'],
              'selected_maximum_strength': eligible[0]['maximum_strength'] if eligible else None,
              'plan_sha256': sha(OUT/'plan.json')}
    write_json(OUT/'calibration.json', result)
    print('CALIBRATION', result, flush=True)
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['calibrate', 'test'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    plan = prepare()
    models = load_models(plan['config']['seed'])
    if args.stage == 'calibrate':
        calibrate(plan, models)
    else:
        calibration = read_json(OUT/'calibration.json')
        assert calibration['plan_sha256'] == sha(OUT/'plan.json')
        maximum = calibration['selected_maximum_strength']
        if maximum is None:
            raise RuntimeError('検証用画像で選択規則を満たす強度がありません。')
        for severity in plan['config']['test_severities']:
            for dataset in DATASETS:
                evaluate(dataset, 'test', severity, maximum, models, plan)


if __name__ == '__main__':
    main()
