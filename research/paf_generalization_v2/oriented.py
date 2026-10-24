"""可視内周の位置と開口から外へ向く法線を、同じ小型CNNで学習する。"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization import experiment, boundary, boundary_surface
from research.paf_generalization_v2.projection import PackedBoundaryDataset

BASE_FREEZE = boundary_surface.freeze_boundary_surface


class OrientedUNet(TinyUNet):
    """出力は境界logit・法線x・法線yの3値。共有畳み込みは元モデルと同じ。"""

    def __init__(self, base_channels=16):
        super().__init__(base_channels)
        original = self.output
        # 共通部分と境界出力の初期値、後続の乱数列を基準と揃える。
        with torch.random.fork_rng(devices=[]):
            self.output = nn.Conv2d(base_channels, 3, 1)
        with torch.no_grad():
            self.output.weight[:1].copy_(original.weight)
            self.output.bias[:1].copy_(original.bias)


class OrientedDataset(PackedBoundaryDataset):
    """元と表面模様の対応画像を基準と同じ乱数・確率で切り替える。"""

    def __init__(self, root, split, input_size, cache=False):
        super().__init__(root, split, input_size, cache)
        self.intervention = None
        if Path(root).name == 'diversified' and split == 'train':
            self.intervention = PackedBoundaryDataset(boundary_surface.DATA, split, input_size, cache)
            assert [s['sample_id'] for s in self.samples] == [s['sample_id'] for s in self.intervention.samples]
            self.rng = np.random.default_rng(torch.initial_seed()+5200)

    def __getitem__(self, index):
        if self.intervention is not None and self.rng.random() < .5:
            return self.intervention[index]
        return super().__getitem__(index)


def normal_target(masks):
    """完全穴の面積共分散から楕円の外向き法線を求める。変換後の教師に適用する。"""
    _, _, vx, vy, xy, dx, dy = experiment.moments(masks[:, :1])
    # 正の共通係数は正規化で消えるため、共分散逆行列の分子だけを用いる。
    nx = vy[:, None, None]*dx - xy[:, None, None]*dy
    ny = vx[:, None, None]*dy - xy[:, None, None]*dx
    return F.normalize(torch.stack([nx, ny], dim=1), dim=1, eps=1e-6)


def objective(logits, masks, settings, config, bce):
    """境界BCE＋Diceに、可視境界だけの法線cosine誤差を加える。"""
    edge = boundary.boundary_target(masks)
    prediction = F.normalize(logits[:, 1:3], dim=1, eps=1e-6)
    cosine = (prediction*normal_target(masks)).sum(1, keepdim=True)
    # 全被覆の画像は法線損失0。白飛び・黒つぶれの可視性は推測しない。
    direction = ((1-cosine)*edge).sum((1, 2, 3))/edge.sum((1, 2, 3)).clamp_min(1e-6)
    return boundary.boundary_objective(logits[:, :1], masks, settings, config, bce) + config['normal_weight']*direction.mean()


def freeze(folder, config, variant, seed):
    """法線教師・モデル・読込実装を固定する。1194Mは学習に用いない。"""
    config = copy.deepcopy(config)
    paths = [Path(__file__), Path(__file__).with_name('projection.py')]
    config['oriented_files'] = {str(p.relative_to(ROOT)): experiment.sha(p) for p in paths}
    return BASE_FREEZE(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = ROOT/'output/paf_generalization_v2'
    experiment.TinyUNet = OrientedUNet
    experiment.RingDataset = OrientedDataset
    experiment.objective = objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    # 補助学習の効果は、方向を推論に使わない基準と同じRANSACで測定する。
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
