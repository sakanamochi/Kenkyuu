"""中心方向モデルの誤境界選択と、正解近傍でも残る中心バイアスを切り分ける。"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.ransac import _point_distances
from research.paf_generalization.boundary import skeleton_points
from research.paf_generalization.diagnose import BASE, sha, image_and_truth
from research.paf_generalization_v2.oriented import OrientedUNet
from research.paf_generalization_v2.center_geometry import center_from_rays, fit_centered, fit_center_votes


def direction_errors(points, directions, center):
    """角度の絶対誤差と符号付き円平均を分ける。中心への向きを正解とする。"""
    expected = np.asarray(center)-points
    expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-9)
    cross = expected[:, 0]*directions[:, 1]-expected[:, 1]*directions[:, 0]
    dot = np.sum(expected*directions, axis=1)
    radians = np.arctan2(cross, dot)
    bias = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return {'mean_absolute_angle_deg': float(np.abs(np.rad2deg(radians)).mean()),
            'signed_mean_angle_deg': float(np.rad2deg(bias)),
            'absolute_bias_deg': float(abs(np.rad2deg(bias)))}


def main(seed):
    folder = ROOT/f'output/paf_generalization_v2/runs/boundary_center_aux/{seed}'
    digest = sha(folder/'best.pt')
    assert read_json(folder/'complete.json')['checkpoint_sha256'] == digest
    model = OrientedUNet(16).cuda().eval()
    model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=False)['model_state'])
    rows = []
    for source in ('1194m', 'diversified'):
        root = BASE/'datasets'/source
        for sample in load_samples(root, 'test'):
            condition = sample['conditions']
            if condition['degradation'] != 'clean' and condition['severity'] != .5:
                continue
            image, truth = image_and_truth(root, sample)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                logits = model(x)[0].cpu()
                probability = torch.sigmoid(logits[0]).numpy()
                field = torch.nn.functional.normalize(logits[1:3], dim=0, eps=1e-6).numpy().transpose(1, 2, 0)
            points, weights = skeleton_points(probability)
            near = _point_distances(points, truth) <= 2 if len(points) else np.zeros(0, bool)
            # 正解近傍の抽出と正確な中心は診断専用。運用可能な推定性能とは扱わない。
            for kind, keep in [('all_predicted_points', np.ones(len(points), bool)), ('oracle_near_points', near)]:
                p, w = points[keep], weights[keep]
                vectors = field[p[:, 1].astype(int), p[:, 0].astype(int)]
                center = center_from_rays(p, vectors, w) if len(p) >= 12 else None
                predicted = fit_center_votes(p, vectors, w) if len(p) >= 12 else None
                oracle = fit_centered(p, np.asarray(truth[0]), w) if len(p) >= 12 else None
                row = {'dataset': source, 'sample_id': sample['sample_id'], 'degradation': condition['degradation'],
                       'severity': condition['severity'], 'tilt': condition.get('camera', {}).get('tilt_deg'),
                       'kind': kind, 'n_points': len(p), 'center': center.tolist() if center is not None else None,
                       'predicted_center_ellipse': predicted, 'oracle_center_ellipse': oracle}
                if len(p):
                    row.update(direction_errors(p, vectors, truth[0]))
                if center is not None:
                    delta = center-np.asarray(truth[0])
                    angle = np.deg2rad(truth[2])
                    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                    row['center_error_px'] = float(np.linalg.norm(delta))
                    row['center_error_semiaxis_units'] = float(np.linalg.norm((delta@rotation)/(np.asarray(truth[1])/2)))
                for name, ellipse in [('predicted', predicted), ('oracle', oracle)]:
                    iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                    row[name+'_iou'] = iou
                    row[name+'_success'] = iou >= .8
                rows.append(row)
    summary = []
    for source in ('1194m', 'diversified'):
        for effect in sorted({r['degradation'] for r in rows}):
            for kind in ('all_predicted_points', 'oracle_near_points'):
                selected = [r for r in rows if (r['dataset'], r['degradation'], r['kind']) == (source, effect, kind)]
                for tilt in [None]+([75.] if source == '1194m' else []):
                    part = [r for r in selected if tilt is None or r['tilt'] == tilt]
                    item = {'dataset': source, 'degradation': effect, 'kind': kind, 'tilt': tilt, 'n': len(part)}
                    for field in ('predicted_success', 'oracle_success', 'center_error_px', 'center_error_semiaxis_units', 'mean_absolute_angle_deg', 'signed_mean_angle_deg', 'absolute_bias_deg'):
                        values = [r[field] for r in part if field in r]
                        item[field] = float(np.mean(values)) if values else None
                        item[field+'_n'] = len(values)
                    summary.append(item)
    write_json(ROOT/f'output/paf_generalization_v2/center_transfer_{seed}.json', {
        'implementation_sha256': sha(Path(__file__)), 'checkpoint_sha256': digest,
        'geometry_sha256': sha(Path(__file__).with_name('center_geometry.py')),
        'input_manifests': {source: sha(BASE/f'datasets/{source}/manifest.json') for source in ('1194m', 'diversified')},
        'dependencies': {name: sha(ROOT/name) for name in ('research/paf_generalization/boundary.py',
            'research/paf_generalization/diagnose.py', 'research/paf_generalization_v2/oriented.py')},
        'rows': rows, 'summary': summary,
        'interpretation': '正解近傍点・正解中心を使う事後診断。直接当てはめの比較でありRANSACの運用性能ではない。sourceの方式選択規則を変更せず、平均誤差と系統的偏りを区別する。'})
    for row in summary:
        if row['dataset'] == '1194m' and row['degradation'] == 'black_rectangle' and row['kind'] == 'oracle_near_points':
            print(row, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260907)
    main(parser.parse_args().seed)
