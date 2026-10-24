"""画素から中心への投票で楕円仮説を作る、単一RANSACの比較。"""

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
from research.paf_generalization import boundary
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha, image_and_truth
from research.paf_generalization_v2.oriented import OrientedUNet
from research.paf_generalization_v2.center_geometry import fit_center_votes

OUT = ROOT/'output/paf_generalization_v2/center_votes'


def vote_ellipse(probability, directions, settings):
    """3点の位置と方向から仮説を作り、元と同じ境界支持量で選択する。"""
    points, weights = boundary.skeleton_points(probability)
    minimum = settings['min_inliers']
    if len(points) < minimum:
        return None
    vectors = directions[points[:, 1].astype(int), points[:, 0].astype(int)]
    rng = np.random.default_rng(settings['random_seed'])
    probabilities = weights.astype(np.float64)/np.sum(weights, dtype=np.float64)
    best, best_mask, best_score = None, None, -np.inf
    for _ in range(settings['iterations']):
        chosen = rng.choice(len(points), 3, replace=False, p=probabilities)
        ellipse = fit_center_votes(points[chosen], vectors[chosen])
        if ellipse is None or not ransac._is_valid_ellipse(ellipse, probability.shape, settings):
            continue
        inliers = ransac._point_distances(points, ellipse) <= settings['distance_threshold_px']
        if inliers.sum() < minimum:
            continue
        coverage = ransac._angular_coverage(points[inliers], ellipse, settings['angular_bins'])
        score = ransac._ransac_hypothesis_score(float(weights[inliers].sum()), ellipse, coverage, settings)
        if score > best_score:
            best, best_mask, best_score = ellipse, inliers, score
    if best is None:
        return None
    ellipse, inliers = best, best_mask
    # 同じインライアから中心・楕円を再推定する。別の検出器や後段補正は加えない。
    for _ in range(settings['refine_iterations']):
        fitted = fit_center_votes(points[inliers], vectors[inliers], weights[inliers])
        if fitted is None or not ransac._is_valid_ellipse(fitted, probability.shape, settings):
            break
        updated = ransac._point_distances(points, fitted) <= settings['distance_threshold_px']
        if updated.sum() < minimum:
            break
        ellipse = fitted
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    return ellipse


def evaluate(model, protocol, root, split, name, destination):
    path = destination/f'{name}.json'
    if path.exists():
        saved = read_json(path)
        assert saved['protocol'] == protocol and saved['manifest_sha256'] == sha(root/'manifest.json')
        return saved['rates']
    rows = []
    for sample in load_samples(root, split):
        image, truth = image_and_truth(root, sample)
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
        with torch.inference_mode():
            logits = model(x)[0].cpu()
            probability = torch.sigmoid(logits[0]).numpy()
            direction = torch.nn.functional.normalize(logits[1:3], dim=0, eps=1e-6).numpy().transpose(1, 2, 0)
        for method in protocol['methods']:
            cv2.setRNGSeed(20261025)
            ellipse = boundary.boundary_ellipse(probability, protocol['ransac']) if method == 'standard' else vote_ellipse(probability, direction, protocol['ransac'])
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                         'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                         'method': method, 'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
    rates = {m: float(np.mean([r['success'] for r in rows if r['method'] == m])) for m in protocol['methods']}
    write_json(path, {'protocol': protocol, 'manifest_sha256': sha(root/'manifest.json'), 'rates': rates, 'rows': rows})
    print(name, rates, flush=True)
    return rates


def main(stage, seed):
    folder = ROOT/f'output/paf_generalization_v2/runs/boundary_center_aux/{seed}'
    digest = sha(folder/'best.pt')
    assert read_json(folder/'complete.json')['checkpoint_sha256'] == digest
    checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    model = OrientedUNet(16).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])
    views = {'original_validation': BASE/'datasets/diversified', 'surface_validation': PREVIOUS/'surface_counterfactual',
             'projection_plain_validation': ROOT/'output/paf_generalization_v2/projection_coverage/native_plain',
             'projection_surface_validation': ROOT/'output/paf_generalization_v2/projection_coverage/native_surface'}
    protocol = {'implementation_sha256': sha(Path(__file__)), 'checkpoint_sha256': digest,
                'geometry_sha256': sha(Path(__file__).with_name('center_geometry.py')),
                'learning_sha256': sha(Path(__file__).with_name('center_learning.py')),
                'oriented_sha256': sha(Path(__file__).with_name('oriented.py')),
                'boundary_sha256': sha(ROOT/'research/paf_generalization/boundary.py'),
                'ransac_sha256': sha(ROOT/'paf_ring_detection/methods/ransac.py'),
                'seed': seed, 'methods': ['standard', 'center_votes'], 'minimum_gain': .01,
                'ransac': checkpoint['provenance']['config']['ransac'],
                'validation_views': {name: {'root': str(root), 'manifest_sha256': sha(root/'manifest.json')} for name, root in views.items()},
                'selection': 'mean success over four paired source validation views; require 1pp gain; target not used',
                'interpretation': 'four views of105 source validation shapes; not2940 independent shapes',
                'inference_batch_size': 1, 'opencv_seed_per_image': 20261025}
    destination = OUT/str(seed)
    path = destination/'protocol.json'
    if path.exists():
        assert read_json(path) == protocol
    else:
        write_json(path, protocol)
    if stage == 'validation':
        scores = [evaluate(model, protocol, root, 'validation', name, destination) for name, root in views.items()]
        rates = {m: float(np.mean([r[m] for r in scores])) for m in protocol['methods']}
        selected = 'center_votes' if rates['center_votes']-rates['standard'] >= protocol['minimum_gain'] else 'standard'
        write_json(destination/'selection.json', {'rates': rates, 'selected': selected,
                   'input_hashes': [sha(destination/f'{name}.json') for name in views]})
    else:
        if read_json(destination/'selection.json')['selected'] != 'center_votes':
            print('Source validation retains standard RANSAC.', flush=True)
            return
        for source in ('1194m', 'diversified'):
            evaluate(model, protocol, BASE/'datasets'/source, 'test', source, destination)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validation', 'test'])
    parser.add_argument('--seed', type=int, default=20260907)
    args = parser.parse_args()
    main(args.stage, args.seed)
