"""法線の補助学習を固定し、向きの整合を使うRANSACをsourceだけで選択する。"""

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
from paf_ring_detection.methods import ransac
from research.paf_generalization import boundary
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha, image_and_truth
from research.paf_generalization_v2.oriented import OrientedUNet
from research.paf_generalization_v2.normal_geometry import ellipse_normals

OUT = ROOT/'output/paf_generalization_v2/direction_score'
BASE_SCORE = ransac._ransac_hypothesis_score


def estimate(probability, normals, settings, method):
    """法線の内積の正部分二乗で支持量だけを重み付けする。仮説生成は共通。"""
    points, weights = boundary.skeleton_points(probability)
    if len(points) < settings['min_inliers']:
        return None
    predicted = normals[points[:, 1].astype(int), points[:, 0].astype(int)]

    def score(consensus_weight, ellipse, coverage, options):
        if method == 'direction':
            inliers = ransac._point_distances(points, ellipse) <= options['distance_threshold_px']
            cosine = np.sum(predicted*ellipse_normals(points, ellipse), axis=1)
            consensus_weight = float(np.sum(weights[inliers]*np.maximum(cosine[inliers], 0)**2))
        return BASE_SCORE(consensus_weight, ellipse, coverage, options)

    # 元ファイルは変えず、この同期処理内だけ評価関数を差し替える。
    ransac._ransac_hypothesis_score = score
    try:
        cv2.setRNGSeed(20261025)
        fitted = ransac.fit_ellipse_ransac(points, probability.shape, settings, weights=weights)
        return fitted['ellipse'] if fitted else None
    finally:
        ransac._ransac_hypothesis_score = BASE_SCORE


def evaluate(model, protocol, root, split, name, destination):
    """同じ画素出力から2方式を比較し、正解境界上の方向誤差も診断する。"""
    target = destination/f'{name}.json'
    if target.exists():
        saved = read_json(target)
        assert saved['protocol'] == protocol and saved['manifest_sha256'] == sha(root/'manifest.json')
        return saved['rates']
    dataset = boundary.BoundaryDataset(root, split, 256)
    rows, errors = [], []
    for index, sample in enumerate(dataset.samples):
        image, truth = image_and_truth(root, sample)
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
        with torch.inference_mode():
            logits = model(x)[0].cpu()
            probability = torch.sigmoid(logits[0]).numpy()
            normals = torch.nn.functional.normalize(logits[1:3], dim=0, eps=1e-6).numpy().transpose(1, 2, 0)
        # 正解は方向の診断だけに使い、推定・方式選択の入力へ渡さない。
        _, masks, _ = dataset[index]
        known_visible = boundary.boundary_target(masks[None])[0, 0].numpy() > .5
        yy, xx = np.where(known_visible)
        if len(xx):
            ideal = ellipse_normals(np.stack([xx, yy], 1), truth)
            degrees = np.rad2deg(np.arccos(np.clip(np.sum(normals[yy, xx]*ideal, axis=1), -1, 1)))
            errors.append({'sample_id': sample['sample_id'], 'degradation': sample['conditions']['degradation'],
                           'severity': sample['conditions']['severity'], 'n_boundary_pixels': len(xx),
                           'mean_angle_deg': float(degrees.mean()), 'median_angle_deg': float(np.median(degrees)),
                           'p90_angle_deg': float(np.quantile(degrees, .9))})
        for method in protocol['methods']:
            ellipse = estimate(probability, normals, protocol['ransac'], method)
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                         'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                         'method': method, 'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
    rates = {method: float(np.mean([r['success'] for r in rows if r['method'] == method])) for method in protocol['methods']}
    write_json(target, {'protocol': protocol, 'manifest_sha256': sha(root/'manifest.json'),
                        'rates': rates, 'normal_errors': errors, 'rows': rows})
    print(name, rates, flush=True)
    return rates


def main(stage, seed):
    folder = ROOT/f'output/paf_generalization_v2/runs/boundary_normal_aux/{seed}'
    digest = sha(folder/'best.pt')
    assert read_json(folder/'complete.json')['checkpoint_sha256'] == digest
    checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    model = OrientedUNet(16).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])
    protocol = {'implementation_sha256': sha(Path(__file__)), 'checkpoint_sha256': digest,
                'normal_geometry_sha256': sha(Path(__file__).with_name('normal_geometry.py')),
                'oriented_sha256': sha(Path(__file__).with_name('oriented.py')),
                'boundary_sha256': sha(ROOT/'research/paf_generalization/boundary.py'),
                'ransac_sha256': sha(ROOT/'paf_ring_detection/methods/ransac.py'),
                'ransac': checkpoint['provenance']['config']['ransac'], 'inference_batch_size': 1,
                'opencv_seed_per_image': 20261025, 'methods': ['standard', 'direction'],
                'selection': 'direction only if mean source validation success improves by at least 1 percentage point',
                'minimum_gain': .01, 'seed': seed}
    destination = OUT/str(seed)
    path = destination/'protocol.json'
    if path.exists():
        assert read_json(path) == protocol
    else:
        write_json(path, protocol)
    if stage == 'validation':
        names = ('original_validation', 'surface_validation')
        scores = [evaluate(model, protocol, root, 'validation', name, destination) for name, root in
                  zip(names, (BASE/'datasets/diversified', PREVIOUS/'surface_counterfactual'))]
        rates = {method: float(np.mean([s[method] for s in scores])) for method in protocol['methods']}
        selected = 'direction' if rates['direction']-rates['standard'] >= protocol['minimum_gain'] else 'standard'
        write_json(destination/'selection.json', {'rates': rates, 'selected': selected,
                   'input_hashes': [sha(destination/f'{name}.json') for name in names]})
    else:
        if read_json(destination/'selection.json')['selected'] != 'direction':
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
