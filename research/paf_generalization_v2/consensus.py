"""RANSACから角度被覆の重みを省き、CNN支持量だけで選べるかを比較する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods import ransac
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha, image_and_truth
from research.paf_generalization_v2.fitting import load

OUT = ROOT/'output/paf_generalization_v2/consensus'
BASE_SCORE = ransac._ransac_hypothesis_score


def support_score(consensus_weight, ellipse, coverage, settings):
    """部分弧の角度被覆率による減点を行わず、CNN確率の支持総量を使う。"""
    assert settings['perimeter_power'] == 0
    return float(consensus_weight)


def estimate(probability, settings, method):
    # 実験プロセス内だけで評価式を差し替え、元の実装ファイルは変更しない。
    ransac._ransac_hypothesis_score = BASE_SCORE if method == 'coverage' else support_score
    try:
        cv2.setRNGSeed(20261025)
        return boundary_ellipse(probability, settings)
    finally:
        ransac._ransac_hypothesis_score = BASE_SCORE


def evaluate(model, protocol, root, split, name):
    rows = []
    for sample in load_samples(root, split):
        image, truth = image_and_truth(root, sample)
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
        with torch.inference_mode():
            probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        for method in protocol['methods']:
            ellipse = estimate(probability, protocol['ransac'], method)
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                         'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                         'method': method, 'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
    rates = {method: float(np.mean([r['success'] for r in rows if r['method'] == method])) for method in protocol['methods']}
    write_json(OUT/f'{name}.json', {'protocol': protocol, 'manifest_sha256': sha(root/'manifest.json'), 'rates': rates, 'rows': rows})
    print(name, rates, flush=True)
    return rates


def main(stage):
    model, baseline = load()
    protocol = {'implementation_sha256': sha(Path(__file__)), 'checkpoint_sha256': baseline['checkpoint_sha256'],
                'boundary_sha256': baseline['boundary_sha256'], 'ransac_sha256': baseline['ransac_sha256'],
                'ransac': baseline['ransac'], 'inference_batch_size': 1, 'opencv_seed_per_image': 20261025,
                'methods': ['coverage', 'support'],
                'selection': 'mean source validation success; ties prefer the current coverage score',
                'change': 'Remove angular coverage multiplier; no additional inference stage.'}
    path = OUT/'protocol.json'
    if path.exists():
        assert read_json(path) == protocol
    else:
        write_json(path, protocol)
    if stage == 'validation':
        names = ('original_validation', 'surface_validation')
        scores = [evaluate(model, protocol, root, 'validation', name) for name, root in
                  zip(names, (BASE/'datasets/diversified', PREVIOUS/'surface_counterfactual'))]
        rates = {method: float(np.mean([s[method] for s in scores])) for method in protocol['methods']}
        write_json(OUT/'selection.json', {'rates': rates, 'selected': max(rates, key=rates.get),
                   'input_hashes': [sha(OUT/f'{name}.json') for name in names]})
    else:
        selection = read_json(OUT/'selection.json')
        if selection['selected'] != 'support':
            print('Source validation does not select simplification; retain coverage.', flush=True)
            return
        for source in ('1194m', 'diversified'):
            evaluate(model, protocol, BASE/'datasets'/source, 'test', source)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validation', 'test'])
    main(parser.parse_args().stage)
