"""内周の証拠選択と楕円推定の誤りを分ける、正解を使う境界点診断。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.effects import apply_effect
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.methods.ransac import _point_distances
from research.paf_generalization.boundary import skeleton_points
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha, image_and_truth
from research.paf_generalization_v2.oriented import OrientedUNet
from research.paf_generalization_v2.group_normalization import GroupUNet

OUT = ROOT/'output/paf_generalization_v2'
CLASSES = {'boundary_surface': TinyUNet, 'boundary_normal_aux': OrientedUNet,
           'boundary_center_aux': OrientedUNet, 'boundary_relative_mask': TinyUNet,
           'boundary_projection_relative': TinyUNet, 'boundary_projection_mix': TinyUNet,
           'boundary_group_norm': GroupUNet}


def ellipse_samples(ellipse, count=256):
    """角度の偏りを避けるため、解析楕円の弧長にほぼ等間隔な点を使う。"""
    center, axes, angle = ellipse
    t = np.linspace(0, 2*np.pi, 4097)
    a = np.deg2rad(angle)
    local = np.stack([axes[0]/2*np.cos(t), axes[1]/2*np.sin(t)], 1)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    points = local@rotation.T+center
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    query = np.arange(count)*length[-1]/count
    return np.stack([np.interp(query, length, points[:, k]) for k in (0, 1)], 1)


def evidence(points, weights, truth, visible_curve):
    """点の正解近傍率と、既知の矩形外にある内周の回収率を別々に測る。"""
    distances = _point_distances(points, truth) if len(points) else np.empty(0)
    near = distances <= 2.
    result = {'n_points': len(points), 'n_near_points': int(near.sum()), 'n_visible_queries': len(visible_curve),
        'point_precision': float(near.mean()) if len(points) else None,
        'weighted_precision': float(weights[near].sum()/weights.sum()) if len(points) else None,
        'near_mean_distance_px': float(distances[near].mean()) if near.any() else None}
    if len(visible_curve):
        nearest = np.min(np.linalg.norm(visible_curve[:, None]-points[None], axis=2), axis=1) if len(points) else np.full(len(visible_curve), np.inf)
        result['visible_recall'] = float(np.mean(nearest <= 2.))
    else:
        result['visible_recall'] = None
    return result


def summarize(rows):
    result = []
    for source in ('1194m', 'diversified'):
        for effect in ('clean', 'black_rectangle', 'sensor_whiteout', 'sensor_black_crush'):
            for success in (None, False, True):
                for tilt in (None, 75.) if source == '1194m' else (None,):
                    part = [r for r in rows if r['dataset'] == source and r['degradation'] == effect
                            and (success is None or r['success'] == success) and (tilt is None or r['tilt'] == tilt)]
                    item = {'dataset': source, 'degradation': effect, 'success': success, 'tilt': tilt, 'n': len(part)}
                    for field in ('n_points', 'n_near_points', 'point_precision', 'weighted_precision', 'visible_recall', 'near_mean_distance_px'):
                        values = [r[field] for r in part if r[field] is not None]
                        item[field+'_n'] = len(values)
                        item[field+'_mean'] = float(np.mean(values)) if values else None
                        item[field+'_median'] = float(np.median(values)) if values else None
                    result.append(item)
    return result


def main(variants, seed):
    paths = [Path(__file__), ROOT/'research/paf_generalization/boundary.py',
             ROOT/'research/paf_generalization/diagnose.py', ROOT/'paf_ring_detection/methods/ransac.py',
             ROOT/'paf_ring_detection/data.py', ROOT/'paf_ring_detection/effects.py',
             ROOT/'paf_ring_detection/methods/cnn.py', Path(__file__).with_name('oriented.py'),
             Path(__file__).with_name('group_normalization.py')]
    dependencies = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for variant in variants:
        folder = (PREVIOUS if variant == 'boundary_surface' else OUT)/f'runs/{variant}/{seed}'
        digest = sha(folder/'best.pt')
        assert read_json(folder/'complete.json')['checkpoint_sha256'] == digest
        destination = OUT/f'boundary_evidence/{variant}/{seed}.json'
        protocol = {'variant': variant, 'seed': seed, 'checkpoint_sha256': digest, 'files': dependencies,
            'test_files': {s: sha(folder/f'test_{s}.json') for s in ('1194m', 'diversified')},
            'input_manifests': {s: sha(BASE/f'datasets/{s}/manifest.json') for s in ('1194m', 'diversified')},
            'threshold_px': 2., 'curve_queries': 256,
            'scope': '正解近傍率と矩形外の投影内周回収率。自己遮蔽・露光での可視性ではなく、運用する信頼度や対象選択へ使用しない。保存済み通常推論の成否で群分けし、当てはめは再実行しない。'}
        if destination.exists():
            assert read_json(destination)['protocol'] == protocol
            continue
        model = CLASSES[variant](16).cuda().eval()
        model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=False)['model_state'])
        rows = []
        for source in ('1194m', 'diversified'):
            root = BASE/'datasets'/source
            predictions = {r['sample_id']: r for r in read_json(folder/f'test_{source}.json')['rows']}
            for sample in load_samples(root, 'test'):
                condition = sample['conditions']
                if condition['degradation'] != 'clean' and condition['severity'] != .5:
                    continue
                image, truth = image_and_truth(root, sample)
                x = torch.from_numpy(image.transpose(2, 0, 1).copy()).cuda().float()[None]/127.5-1
                with torch.inference_mode():
                    probability = torch.sigmoid(model(x)[0, 0]).cpu().numpy()
                points, weights = skeleton_points(probability)
                queries = ellipse_samples(truth)
                inside = np.all((queries >= 0) & (queries < 255), axis=1)
                visibility = np.full((480, 480, 3), 255, np.uint8)
                effect = sample.get('augmentation')
                if effect and effect['name'] == 'black_rectangle':
                    visibility = apply_effect(visibility, effect['name'], effect['severity'], np.random.default_rng(effect['seed']))
                visibility = cv2.resize(visibility[:, :, 0], (256, 256), interpolation=cv2.INTER_AREA)
                xy = np.clip(np.rint(queries).astype(int), 0, 255)
                keep = inside & (visibility[xy[:, 1], xy[:, 0]] >= 128)
                prediction = predictions[sample['sample_id']]
                rows.append({'dataset': source, 'sample_id': sample['sample_id'],
                    'degradation': condition['degradation'], 'severity': condition['severity'],
                    'tilt': condition.get('camera', {}).get('tilt_deg'),
                    'ellipse_iou': prediction['ellipse_iou'], 'success': prediction['success'],
                    **evidence(points, weights, truth, queries[keep])})
        assert len(rows) == 1064
        summary = summarize(rows)
        write_json(destination, {'protocol': protocol, 'rows': rows, 'summary': summary})
        for row in summary:
            if row['dataset'] == '1194m' and row['degradation'] == 'black_rectangle' and row['tilt'] is None:
                print(variant, row, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=list(CLASSES),
                        default=['boundary_surface', 'boundary_normal_aux', 'boundary_relative_mask', 'boundary_center_aux'])
    parser.add_argument('--seed', type=int, default=20260907)
    args = parser.parse_args()
    main(args.variants, args.seed)
