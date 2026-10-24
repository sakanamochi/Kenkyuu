"""RANSACの仮説選択を固定し、インライアの当てはめ式だけを比較する。"""

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
from paf_ring_detection.methods.ransac import fit_ellipse_ransac, _is_valid_ellipse, _point_distances
from research.paf_generalization.boundary import skeleton_points
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha, image_and_truth
from research.paf_generalization_v2.fitting import load

OUT = ROOT/'output/paf_generalization_v2/refinement'


def refine(initial, points, shape, settings, fit):
    """既存の反復数・有効性・インライア更新を保ち、fit関数だけを替える。"""
    if initial is None:
        return None
    ellipse, mask = initial['ellipse'], initial['inlier_mask']
    threshold, minimum = settings['distance_threshold_px'], settings['min_inliers']
    for _ in range(settings['refine_iterations']):
        try:
            candidate = fit(points[mask].reshape(-1, 1, 2))
        except cv2.error:
            break
        if not _is_valid_ellipse(candidate, shape, settings):
            break
        ellipse = candidate
        updated = _point_distances(points, ellipse) <= threshold
        if np.array_equal(updated, mask):
            break
        if np.count_nonzero(updated) < minimum:
            break
        mask = updated
    if np.count_nonzero(_point_distances(points, ellipse) <= threshold) < minimum:
        return initial['ellipse']
    return ellipse


def evaluate(model, settings, root, split, name, protocol):
    rows = []
    reference = None
    if split == 'test':
        saved = read_json(PREVIOUS/f'runs/boundary_surface/20260907/test_{name}.json')
        assert saved['checkpoint_sha256'] == protocol['checkpoint_sha256']
        reference = {r['sample_id']: r for r in saved['rows']}
    samples = load_samples(root, split)
    # 確率値の丸め差が重み付き抽選へ影響するため、基準評価と同じ1枚推論に揃える。
    for start in range(len(samples)):
        batch = samples[start:start+1]
        cases = [image_and_truth(root, sample) for sample in batch]
        x = torch.from_numpy(np.stack([im.transpose(2, 0, 1) for im, _ in cases]).copy()).float().cuda()/127.5-1
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(x))[:, 0].cpu().numpy()
        for sample, (_, truth), probability in zip(batch, cases, probabilities):
            points, weights = skeleton_points(probability)
            # OpenCV内部の縮退点処理にも乱数状態があるため、画像ごとに固定する。
            cv2.setRNGSeed(20261025)
            initial = fit_ellipse_ransac(points, probability.shape, {**settings, 'refine_iterations': 0}, weights=weights) if len(points) >= settings['min_inliers'] else None
            for method, fit in [('standard', cv2.fitEllipse), ('ams', cv2.fitEllipseAMS)]:
                ellipse = refine(initial, points, probability.shape, settings, fit)
                iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                             'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                             'method': method, 'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8,
                             'stored_reference_iou': reference[sample['sample_id']]['ellipse_iou'] if reference is not None else None})
    rates = {method: float(np.mean([r['success'] for r in rows if r['method'] == method])) for method in ('standard', 'ams')}
    standard = [r for r in rows if r['method'] == 'standard' and r['stored_reference_iou'] is not None]
    comparison = {'different_iou': sum(r['ellipse_iou'] != r['stored_reference_iou'] for r in standard),
                  'different_success': sum(r['success'] != (r['stored_reference_iou'] >= .8) for r in standard)}
    write_json(OUT/f'{name}.json', {'protocol': protocol, 'manifest_sha256': sha(root/'manifest.json'),
                                  'rates': rates, 'stored_reference_comparison': comparison, 'rows': rows})
    print(name, rates, flush=True)
    return rates


def main(stage):
    model, base = load()
    settings = base['ransac']
    protocol = {'implementation_sha256': sha(Path(__file__)), 'checkpoint_sha256': base['checkpoint_sha256'],
                'boundary_sha256': base['boundary_sha256'], 'ransac_sha256': base['ransac_sha256'],
                'settings': settings, 'changes': 'Only the inlier refit function: fitEllipse versus fitEllipseAMS',
                'inference_batch_size': 1,
                'opencv_seed_per_image': 20261025,
                'comparison_scope': 'The paired standard and AMS fits share exactly the same RANSAC hypothesis. Stored reference uses an unspecified OpenCV RNG state; numerical and success differences are counted.',
                'selection': 'mean success on original and surface source validation; ties prefer standard',
                'reference': 'https://docs.opencv.org/3.4.15/d3/dc0/group__imgproc__shape.html'}
    path = OUT/'protocol.json'
    if path.exists():
        assert read_json(path) == protocol
    else:
        write_json(path, protocol)
    if stage == 'validation':
        scores = []
        for name, root in [('original_validation', BASE/'datasets/diversified'), ('surface_validation', PREVIOUS/'surface_counterfactual')]:
            scores.append(evaluate(model, settings, root, 'validation', name, protocol))
        average = {method: float(np.mean([r[method] for r in scores])) for method in ('standard', 'ams')}
        write_json(OUT/'selection.json', {'rates': average, 'selected': max(average, key=average.get),
                    'input_hashes': [sha(OUT/f'{name}.json') for name in ('original_validation', 'surface_validation')]})
    else:
        selection = read_json(OUT/'selection.json')
        print('source selected:', selection['selected'], flush=True)
        for source in ('1194m', 'diversified'):
            evaluate(model, settings, BASE/'datasets'/source, 'test', source, protocol)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validation', 'test'])
    main(parser.parse_args().stage)
