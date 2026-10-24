"""境界教師を領域の形態学的帯と解析楕円の距離帯で比較する、学習前の診断。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.ransac import _point_distances
from research.paf_generalization import boundary
from research.paf_generalization.diagnose import BASE, sha, image_and_truth

OUT = ROOT/'output/paf_generalization_v2/analytic_teacher'
METHODS = ('morphological', 'analytic_gaussian')


def analytic_band(ellipse, size=256):
    """楕円近傍の一次距離に標準偏差1pxのGaussianを適用する。幅を探索しない。"""
    y, x = np.mgrid[:size, :size]
    points = np.stack([x.ravel(), y.ravel()], 1)
    distance = _point_distances(points, ellipse).reshape(size, size)
    return np.exp(-.5*distance**2).astype(np.float32)


def protocol():
    paths = [Path(__file__), ROOT/'research/paf_generalization/boundary.py',
             ROOT/'research/paf_generalization/diagnose.py', ROOT/'paf_ring_detection/data.py',
             ROOT/'paf_ring_detection/effects.py', ROOT/'paf_ring_detection/geometry.py',
             ROOT/'paf_ring_detection/methods/ransac.py']
    return {'files': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'ransac': read_json(ROOT/'research/paf_generalization/boundary.json')['ransac'],
        'opencv_seed_per_fit': 20261025, 'sigma_px': 1.,
        'selection': 'original and projection source validation equal mean improvement >= 1 pp, neither clean view degrades more than 1 pp',
        'scope': 'GT境界を推定器へ渡す教師の診断。CNNの学習・性能ではなく厳密な上限でもない。距離は輪郭近傍の一次近似。標準推論は変えない。'}


def evaluate(name, root, split, spec, audit=False):
    destination = OUT/f'{name}.json'
    meta = {'protocol': spec, 'root': str(root.relative_to(ROOT)), 'split': split,
            'manifest_sha256': sha(root/'manifest.json')}
    if destination.exists() and not audit:
        old = read_json(destination)
        assert all(old[k] == value for k, value in meta.items())
        return old
    dataset = boundary.BoundaryDataset(root, split, 256, cache=False)
    rows = []
    for index, sample in enumerate(dataset.samples):
        c = sample['conditions']
        if c['degradation'] not in ('clean', 'black_rectangle'):
            continue
        if split == 'test' and c['degradation'] == 'black_rectangle' and c['severity'] != .5:
            continue
        _, masks, _ = dataset[index]
        _, truth = image_and_truth(root, sample)
        probabilities = [boundary.boundary_target(masks[None])[0, 0].numpy(),
                         analytic_band(truth)*masks[1].numpy()]
        for method, probability in zip(METHODS, probabilities):
            cv2.setRNGSeed(spec['opencv_seed_per_fit'])
            ellipse = boundary.boundary_ellipse(probability, spec['ransac'])
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            encoded = [list(ellipse[0]), list(ellipse[1]), ellipse[2]] if ellipse else None
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                'degradation': c['degradation'], 'severity': c['severity'],
                'method': method, 'ellipse': encoded, 'ellipse_iou': iou, 'success': iou >= .8})
    assert len(rows) == (630 if split == 'validation' else (448 if name == '1194m' else 616))
    summary = []
    for method in METHODS:
        for effect in ('all', 'clean', 'black_rectangle'):
            part = [r for r in rows if r['method'] == method and (effect == 'all' or r['degradation'] == effect)]
            summary.append({'method': method, 'degradation': effect, 'n': len(part),
                            'success_rate': float(np.mean([r['success'] for r in part]))})
    result = {**meta, 'rows': rows, 'summary': summary}
    if audit:
        assert read_json(destination) == result
    else:
        write_json(destination, result)
    print(name, summary, flush=True)
    return result


def main(stage):
    spec = protocol()
    path = OUT/'protocol.json'
    if path.exists():
        assert read_json(path) == spec
    else:
        write_json(path, spec)
    views = [('original_validation', BASE/'datasets/diversified'),
             ('projection_validation', ROOT/'output/paf_generalization_v2/projection_coverage/native_plain')]
    if stage == 'validation':
        results = [evaluate(name, root, 'validation', spec) for name, root in views]
        rates = {m: float(np.mean([next(r['success_rate'] for r in result['summary']
                    if r['method'] == m and r['degradation'] == 'all') for result in results])) for m in METHODS}
        clean_deltas = [next(r['success_rate'] for r in result['summary'] if r['method'] == METHODS[1] and r['degradation'] == 'clean')
                       -next(r['success_rate'] for r in result['summary'] if r['method'] == METHODS[0] and r['degradation'] == 'clean') for result in results]
        selected = rates[METHODS[1]]-rates[METHODS[0]] >= .01 and min(clean_deltas) >= -.01
        write_json(OUT/'selection.json', {'rates': rates, 'clean_deltas': clean_deltas, 'selected': selected,
            'inputs': {name: sha(OUT/f'{name}.json') for name, _ in views}})
        print({'selected': selected, 'rates': rates}, flush=True)
    elif stage == 'test':
        if not read_json(OUT/'selection.json')['selected']:
            print('source validation did not select analytic teacher; test skipped', flush=True)
            return
        for source in ('1194m', 'diversified'):
            evaluate(source, BASE/'datasets'/source, 'test', spec)
    else:
        names = [name for name, _ in views]
        for name in names+['1194m', 'diversified']:
            saved = OUT/f'{name}.json'
            if saved.exists():
                data = read_json(saved)
                evaluate(name, ROOT/data['root'], data['split'], spec, audit=True)
        results = [read_json(OUT/f'{name}.json') for name in names]
        rates = {m: float(np.mean([next(r['success_rate'] for r in result['summary']
                    if r['method'] == m and r['degradation'] == 'all') for result in results])) for m in METHODS}
        selected = read_json(OUT/'selection.json')
        assert selected['rates'] == rates
        deltas = [next(r['success_rate'] for r in result['summary'] if r['method'] == METHODS[1] and r['degradation'] == 'clean')
                  -next(r['success_rate'] for r in result['summary'] if r['method'] == METHODS[0] and r['degradation'] == 'clean') for result in results]
        assert selected['clean_deltas'] == deltas
        assert selected['selected'] == (rates[METHODS[1]]-rates[METHODS[0]] >= .01 and min(deltas) >= -.01)
        assert all(sha(OUT/f'{name}.json') == digest for name, digest in selected['inputs'].items())
        write_json(OUT/'audit.json', {'recomputed': {str(p.relative_to(ROOT)): sha(p) for p in OUT.glob('*.json')
                    if p.name != 'audit.json'}, 'scope': '教師生成・細線化・RANSAC・元GTのIoU・集計・source選択を再計算して照合。'})


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validation', 'test', 'audit'])
    main(parser.parse_args().stage)
