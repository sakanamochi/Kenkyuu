"""相似変換前後の解析楕円を用い、面積モーメント由来の中心方向教師を照合する。"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import region_mask, write_json
from research.paf_generalization import experiment, boundary
from research.paf_generalization_v2.center_learning import center_target
from research.paf_generalization.diagnose import sha


def main():
    torch.set_num_threads(2)
    ellipses = [((127.3, 140.2), (major, major*ratio), angle) for major in (20., 60., 120.)
                for ratio in (.15, .35, .8) for angle in (0., 35., 90.)]
    region = torch.stack([torch.from_numpy(region_mask(256, e)) for e in ellipses])[:, None]
    masks = torch.cat([region, torch.ones_like(region)], 1)
    images = region.expand(-1, 3, -1, -1)*2-1
    captured = []
    original = experiment.F.affine_grid

    def record(theta, size, **kwargs):
        # 実際の学習変換行列を記録し、その逆変換で解析中心の正解を求める。
        captured.append(theta.detach().cpu().numpy())
        return original(theta, size, **kwargs)

    experiment.F.affine_grid = record
    try:
        _, changed = experiment.spatial_augment(images, masks, np.random.default_rng(20261025),
            {'scale_range': [.75, 4.], 'center_range': [.3, .7], 'spatial_probability': 1.})
    finally:
        experiment.F.affine_grid = original
    rows = []
    y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
    for name, targets in [('original', masks), ('transformed', changed)]:
        true_centers = []
        for i, ellipse in enumerate(ellipses):
            center = (np.asarray(ellipse[0])+.5)*2/256-1
            if name == 'transformed':
                theta = captured[0][i]
                center = np.linalg.solve(theta[:, :2], center-theta[:, 2])
            true_centers.append((center+1)*128-.5)
        centers = torch.tensor(np.asarray(true_centers), dtype=torch.float32)
        expected = torch.stack([centers[:, 0, None, None]-x, centers[:, 1, None, None]-y], 1)
        expected = torch.nn.functional.normalize(expected, dim=1, eps=1e-6)
        teacher = center_target(targets)
        angles = torch.rad2deg(torch.acos((expected*teacher).sum(1).clamp(-1, 1)))
        edge = boundary.boundary_target(targets)[:, 0]
        cx, cy, *_ = experiment.moments(targets[:, :1])
        measured = (torch.stack([cx, cy], 1)+1)*128-.5
        for i, ellipse in enumerate(ellipses):
            error = float(torch.linalg.norm(measured[i]-centers[i]))
            average = float((angles[i]*edge[i]).sum()/edge[i].sum())
            assert error < 1 and average < 5
            if name == 'transformed':
                rim = targets[i, 0]
                assert not any(torch.any(part > .5) for part in (rim[0], rim[-1], rim[:, 0], rim[:, -1]))
            rows.append({'stage': name, 'ellipse': ellipse, 'true_center': true_centers[i].tolist(),
                'moment_center_error_px': error, 'boundary_mean_direction_error_deg': average})
    paths = [Path(__file__), ROOT/'research/paf_generalization/experiment.py',
             ROOT/'research/paf_generalization/boundary.py', Path(__file__).with_name('center_learning.py')]
    result = {'files': {str(p.relative_to(ROOT)): sha(p) for p in paths}, 'rng_seed': 20261025,
        'n': len(rows), 'rows': rows, 'max_center_error_px': max(r['moment_center_error_px'] for r in rows),
        'max_direction_error_deg': max(r['boundary_mean_direction_error_deg'] for r in rows),
        'interpretation': '解析楕円の変換前後における教師の数値検証。学習済みモデルの性能ではない。学習の相似変換は開口が画面内に収まるよう倍率を制限する。'}
    write_json(ROOT/'output/paf_generalization_v2/center_teacher_check.json', result)
    print({k: v for k, v in result.items() if k not in ('rows', 'files', 'interpretation')})


if __name__ == '__main__':
    main()
