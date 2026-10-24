"""可視内周と、各境界画素から開口中心へ向く方向を共同学習する。"""

import copy
import sys
from pathlib import Path

from torch.nn import functional as F
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_generalization import experiment, boundary
from research.paf_generalization_v2 import oriented


def center_target(masks):
    """完全穴の重心へ向く単位ベクトル。教師の座標は画像と同じ変換を受ける。"""
    *_, dx, dy = experiment.moments(masks[:, :1])
    return F.normalize(torch.stack([-dx, -dy], dim=1), dim=1, eps=1e-6)


def objective(logits, masks, settings, config, bce):
    """位置のBCE＋Diceと、可視境界だけの中心方向cosine損失を最小化する。"""
    edge = boundary.boundary_target(masks)
    direction = F.normalize(logits[:, 1:3], dim=1, eps=1e-6)
    cosine = (direction*center_target(masks)).sum(1, keepdim=True)
    # 全被覆では方向を学習せず、不可視の中心を画素として描く教師にも変えない。
    loss = ((1-cosine)*edge).sum((1, 2, 3))/edge.sum((1, 2, 3)).clamp_min(1e-6)
    return boundary.boundary_objective(logits[:, :1], masks, settings, config, bce)+config['center_direction_weight']*loss.mean()


def freeze(folder, config, variant, seed):
    """方向の意味と共有する3出力CNNの実装を一緒に固定する。"""
    config = copy.deepcopy(config)
    config['center_learning_files'] = {str(Path(__file__).relative_to(ROOT)): experiment.sha(Path(__file__))}
    return oriented.freeze(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = ROOT/'output/paf_generalization_v2'
    experiment.TinyUNet = oriented.OrientedUNet
    experiment.RingDataset = oriented.OrientedDataset
    experiment.objective = objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    # 境界だけを使う成績を共通評価として保存し、中心投票の推論とは分ける。
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
