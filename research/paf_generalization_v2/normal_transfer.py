"""法線補助モデルの未知対象への方向精度を、推論方式の選択とは別に診断する。"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.methods.ransac import _point_distances
from research.paf_generalization import boundary
from research.paf_generalization.diagnose import BASE, sha, image_and_truth
from research.paf_generalization_v2.oriented import OrientedUNet
from research.paf_generalization_v2.normal_geometry import ellipse_normals


def angles(vectors, points, ellipse):
    if not len(points):
        return None
    dots = np.sum(vectors*ellipse_normals(points, ellipse), axis=1)
    return float(np.rad2deg(np.arccos(np.clip(dots, -1, 1))).mean())


def main(seed):
    folder = ROOT/f'output/paf_generalization_v2/runs/boundary_normal_aux/{seed}'
    digest = sha(folder/'best.pt')
    assert read_json(folder/'complete.json')['checkpoint_sha256'] == digest
    model = OrientedUNet(16).cuda().eval()
    model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=False)['model_state'])
    rows = []
    for source in ('1194m', 'diversified'):
        data = boundary.BoundaryDataset(BASE/'datasets'/source, 'test', 256)
        for index, sample in enumerate(data.samples):
            condition = sample['conditions']
            if condition['degradation'] != 'clean' and condition['severity'] != .5:
                continue
            image, truth = image_and_truth(data.root, sample)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                logits = model(x)[0].cpu()
                probability = torch.sigmoid(logits[0]).numpy()
                normal = torch.nn.functional.normalize(logits[1:3], dim=0, eps=1e-6).numpy().transpose(1, 2, 0)
            _, mask, _ = data[index]
            yy, xx = np.where(boundary.boundary_target(mask[None])[0, 0].numpy() > .5)
            target_points = np.stack([xx, yy], 1)
            points, weights = boundary.skeleton_points(probability)
            close = _point_distances(points, truth) <= 2 if len(points) else np.zeros(0, bool)
            good = points[close].astype(int)
            row = {'dataset': source, 'sample_id': sample['sample_id'], 'degradation': condition['degradation'],
                   'severity': condition['severity'], 'tilt': condition.get('camera', {}).get('tilt_deg'),
                   'n_target_pixels': len(xx), 'n_predicted_points': len(points), 'n_near_points': int(close.sum()),
                   'target_mean_angle_deg': angles(normal[yy, xx], target_points, truth),
                   'near_point_mean_angle_deg': angles(normal[good[:, 1], good[:, 0]], good, truth),
                   'near_point_precision': float(close.mean()) if len(close) else 0.,
                   'target_band_recall': float((probability[yy, xx] >= .5).mean()) if len(xx) else None}
            rows.append(row)
    summary = []
    for source in ('1194m', 'diversified'):
        for effect in sorted({r['degradation'] for r in rows}):
            selected = [r for r in rows if r['dataset'] == source and r['degradation'] == effect]
            for tilt in [None]+(sorted({r['tilt'] for r in selected}) if source == '1194m' else []):
                part = [r for r in selected if tilt is None or r['tilt'] == tilt]
                item = {'dataset': source, 'degradation': effect, 'tilt': tilt, 'n': len(part)}
                for field in ('target_mean_angle_deg', 'near_point_mean_angle_deg', 'near_point_precision', 'target_band_recall'):
                    values = [r[field] for r in part if r[field] is not None]
                    item[field] = float(np.mean(values)) if values else None
                summary.append(item)
    write_json(ROOT/f'output/paf_generalization_v2/normal_transfer_{seed}.json', {
        'implementation_sha256': sha(Path(__file__)), 'checkpoint_sha256': digest,
        'manifests': {name: sha(BASE/f'datasets/{name}/manifest.json') for name in ('1194m', 'diversified')},
        'rows': rows, 'summary': summary,
        'interpretation': '正解境界での方向・選択精度の事後診断。正解は推定に渡さず、direction_scoreのsource選択規則を変更しない。矩形外を既知の可視領域とし、光学的可視性を推定しない。'})
    for row in summary:
        if row['dataset'] == '1194m' and row['tilt'] in (None, 75):
            print(row, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260907)
    main(parser.parse_args().seed)
