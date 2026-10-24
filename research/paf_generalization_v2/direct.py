"""小型CNNが中心と正定値行列の5自由度から完全楕円を直接推定する。"""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import RingDataset, read_json, write_json
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import DoubleConv
from research.paf_generalization import experiment, boundary_surface

BASE_FREEZE = boundary_surface.freeze_boundary_surface


class DirectEllipse(nn.Module):
    """U-Netと同じ4段の畳み込みに、4×4集約と小さな全結合層を接続する。"""

    def __init__(self, base_channels=16):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            DoubleConv(3, c), nn.MaxPool2d(2),
            DoubleConv(c, c*2), nn.MaxPool2d(2),
            DoubleConv(c*2, c*4), nn.MaxPool2d(2), DoubleConv(c*4, c*8))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                  nn.Linear(c*8*16, 64), nn.ReLU(), nn.Linear(64, 5))
        # 初期出力は画像中央の円。角度を直接回帰せず円の角度不定性を避ける。
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[2:4] = math.log(math.expm1(.1))

    def forward(self, image):
        return self.head(self.features(image))


def decode(raw):
    """Cholesky因子から、正の半軸を持つ楕円の面積共分散を構成する。"""
    center = torch.tanh(raw[:, :2])
    diagonal = F.softplus(raw[:, 2:4]) + 1e-4
    a, b, off = diagonal[:, 0], diagonal[:, 1], raw[:, 4]
    vx, vy, xy = a.square(), b.square()+off.square(), a*off
    return center[:, 0], center[:, 1], vx, vy, xy


def wasserstein_loss(predicted, target):
    """2次元Gaussian間の二乗Wasserstein距離を教師の大きさで正規化する。"""
    cx, cy, vx, vy, xy = predicted
    tx, ty, tvx, tvy, txy = target
    determinant = (vx*vy-xy.square()).clamp_min(1e-12)
    target_determinant = (tvx*tvy-txy.square()).clamp_min(1e-12)
    # 2×2正定値行列では、行列平方根のtraceを固有値分解なしで計算できる。
    cross_trace = torch.sqrt((vx*tvx+vy*tvy+2*xy*txy
                             + 2*torch.sqrt(determinant*target_determinant)).clamp_min(1e-12))
    shape = (vx+vy+tvx+tvy-2*cross_trace).clamp_min(0)
    distance = (cx-tx).square()+(cy-ty).square()+shape
    return torch.log1p(distance/(tvx+tvy).clamp_min(1e-6)).mean()


def objective(raw, masks, settings, config, bce):
    # 完全穴領域の面積モーメントだけを教師に使い、背景確率の集積は行わない。
    target = experiment.moments(masks[:, 0:1])[:5]
    return wasserstein_loss(decode(raw), target)


def to_ellipse(values, size=256):
    """一様楕円の面積共分散=半軸二乗/4から、OpenCV表現へ戻す。"""
    cx, cy, vx, vy, xy = [float(value) for value in values]
    eigenvalues, eigenvectors = np.linalg.eigh([[vx, xy], [xy, vy]])
    if not np.isfinite(eigenvalues).all() or min(eigenvalues) <= 0:
        return None
    direction = eigenvectors[:, 0]
    return (((cx+1)*size/2-.5, (cy+1)*size/2-.5),
            tuple((2*size*np.sqrt(eigenvalues)).tolist()),
            float(np.rad2deg(np.arctan2(direction[1], direction[0]))))


def freeze(folder, config, variant, seed):
    """直接楕円表現と学習入力のhashを固定する。"""
    config = copy.deepcopy(config)
    config['direct_implementation_sha256'] = experiment.sha(Path(__file__))
    return BASE_FREEZE(folder, config, variant, seed)


def evaluate(variant, seed):
    config = read_json(experiment.CONFIG)
    folder = experiment.OUT/'runs'/variant/str(seed)
    provenance = freeze(folder, config, variant, seed)
    checkpoint = folder/'best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['provenance'] == provenance
    model = DirectEllipse().cuda().eval()
    model.load_state_dict(saved['model_state'])
    for source in ('1194m', 'diversified'):
        path = folder/f'test_{source}.json'
        if path.exists():
            assert read_json(path)['checkpoint_sha256'] == experiment.sha(checkpoint)
            continue
        dataset = RingDataset(experiment.BASE/'datasets'/source, 'test', 256)
        rows = []
        for sample in dataset.samples:
            image, truth = experiment.image_and_truth(dataset.root, sample)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                values = [v[0].cpu() for v in decode(model(x))]
            ellipse = to_ellipse(values)
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                         'base_sample_id': sample['conditions']['base_sample_id'],
                         'degradation': sample['conditions']['degradation'],
                         'severity': sample['conditions']['severity'], 'ellipse': ellipse,
                         'truth_ellipse': truth, 'ellipse_iou': iou, 'success': iou >= .8,
                         'mask_iou': None})
        write_json(path, {'dataset': source, 'variant': variant, 'seed': seed,
                         'checkpoint_sha256': experiment.sha(checkpoint),
                         'provenance': provenance, 'rows': rows})
        print('EVAL', variant, seed, source, sum(r['success'] for r in rows), len(rows), flush=True)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = ROOT/'output/paf_generalization_v2'
    experiment.RingDataset = boundary_surface.SurfaceBoundaryDataset
    experiment.TinyUNet = DirectEllipse
    experiment.objective = objective
    experiment.freeze_run = freeze
    experiment.evaluate_variant = evaluate
    experiment.main()
