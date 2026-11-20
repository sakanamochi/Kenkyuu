"""完全穴領域と可視マスクから境界教師を作る。"""

import torch
from torch.nn import functional as F


def boundary_target(masks):
    """幅約2pxの境界から、既知の矩形遮蔽だけを除く。"""
    if masks.ndim != 4 or masks.shape[1] != 2:
        raise ValueError("教師はN×2×H×Wの穴領域・可視マスクにしてください。")
    region = masks[:, :1]
    rim = F.max_pool2d(region, 3, 1, 1) + F.max_pool2d(-region, 3, 1, 1)
    return rim.clamp(0, 1) * masks[:, 1:2]


def boundary_loss(logits, masks, positive_weight=12.0):
    """正例重み付きBCEと、画像ごとのDice損失を加算する。"""
    if not 0 < positive_weight < float("inf"):
        raise ValueError("正例重みは有限の正数にしてください。")
    target = boundary_target(masks)
    if logits.shape != target.shape:
        raise ValueError("logitと境界教師の形状が一致しません。")
    weight = logits.new_tensor([positive_weight])
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = (probability + target).sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce + dice
