"""境界CNNのBatchNormだけを画像内のGroupNormへ置き換える対照実験。"""

import copy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization import experiment, boundary, boundary_surface


class GroupUNet(TinyUNet):
    """8グループの正規化。同じ畳み込み・出力を保ち、バッチの移動統計を持たない。"""

    def __init__(self, base_channels=16):
        super().__init__(base_channels)
        for block in self.children():
            if isinstance(block, nn.Sequential):
                for index, layer in enumerate(block):
                    if isinstance(layer, nn.BatchNorm2d):
                        replacement = nn.GroupNorm(8, layer.num_features, eps=layer.eps, affine=True)
                        # アフィン係数と共有重みを同じ初期値にし、追加の乱数消費を行わない。
                        with torch.no_grad():
                            replacement.weight.copy_(layer.weight)
                            replacement.bias.copy_(layer.bias)
                        block[index] = replacement


def freeze(folder, config, variant, seed):
    """正規化の置換のみを追加固定し、元の境界・表面模様実装を使う。"""
    config = copy.deepcopy(config)
    assert config['normalization_groups'] == 8
    config['normalization_files'] = {str(Path(__file__).relative_to(ROOT)): experiment.sha(Path(__file__))}
    return boundary_surface.freeze_boundary_surface(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = ROOT/'output/paf_generalization_v2'
    experiment.TinyUNet = GroupUNet
    experiment.RingDataset = boundary_surface.SurfaceBoundaryDataset
    experiment.objective = boundary.boundary_objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
