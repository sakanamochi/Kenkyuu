"""CNNによる直接楕円パラメータ回帰モデルとProbIoU（2DガウスHellinger距離）損失。

境界マップ＋RANSAC方式（TinyUNet16: 482,737パラメータ）と公平に比較するため、
同規模の軽量エンコーダへCoordConvと空間特徴ヘッドを組み合わせて
(cx, cy, axis_1, axis_2, cos(2θ), sin(2θ)) を直接回帰する。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from core.geometry import Ellipse
from core.model import DoubleConv


class TinyEllipseRegressor(nn.Module):
    """TinyUNet16と同規模（約46万パラメータ）の直接楕円パラメータ回帰CNN。"""

    def __init__(self, base_channels: int = 16, max_axis_ratio: float = 1.4, min_axis_ratio: float = 0.02) -> None:
        super().__init__()
        c = int(base_channels)
        self.base_channels = c
        self.max_axis_ratio = float(max_axis_ratio)
        self.min_axis_ratio = float(min_axis_ratio)

        # TinyUNet16と同一の4段エンコーダ（256 -> 128 -> 64 -> 32）
        self.encoder_1 = DoubleConv(3, c)
        self.encoder_2 = DoubleConv(c, c * 2)
        self.encoder_3 = DoubleConv(c * 2, c * 4)
        self.encoder_4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        # CoordConv（正規化x, y座標2ch）を付与して空間位置を保持しつつ16 -> 8へ集約する
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c * 8 + 2, c * 6, 3, padding=1, bias=False),
            nn.BatchNorm2d(c * 6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 6, c * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 2 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

    @staticmethod
    def _append_coords(feature: torch.Tensor) -> torch.Tensor:
        """特徴マップへ [-1, 1] 正規化 (x, y) 座標チャネルを結合する。"""
        batch, _, height, width = feature.shape
        ys = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        return torch.cat((feature, coords), dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """正規化楕円パラメータ [cx, cy, axis_1, axis_2, cos2t, sin2t]（全て画像サイズ正規化）を返す。"""
        f1 = self.encoder_1(image)
        f2 = self.encoder_2(self.pool(f1))
        f3 = self.encoder_3(self.pool(f2))
        f4 = self.encoder_4(self.pool(f3))
        f5 = self.spatial_conv(self._append_coords(self.pool(f4)))
        raw = self.head(f5)

        center = torch.sigmoid(raw[:, 0:2])
        axes = self.min_axis_ratio + self.max_axis_ratio * torch.sigmoid(raw[:, 2:4])
        orient = F.normalize(raw[:, 4:6], dim=-1, eps=1e-6)
        return torch.cat((center, axes, orient), dim=-1)


def ellipse_to_normalized_vector(ellipse: Ellipse, input_size: int) -> list[float]:
    """OpenCV形式楕円を正規化6次元ベクトル [cx, cy, axis_1, axis_2, cos2t, sin2t] へ変換する。"""
    (cx, cy), (axis_1, axis_2), angle_deg = ellipse
    s = float(input_size)
    rad2 = math.radians(2.0 * float(angle_deg))
    return [
        float(cx) / s,
        float(cy) / s,
        float(axis_1) / s,
        float(axis_2) / s,
        math.cos(rad2),
        math.sin(rad2),
    ]


def normalized_vector_to_ellipse(vector: torch.Tensor | list[float], input_size: int) -> Ellipse:
    """正規化6次元ベクトルをOpenCV形式楕円 ((cx, cy), (axis_1, axis_2), angle_deg) へ変換する。"""
    vals = vector.detach().cpu().tolist() if isinstance(vector, torch.Tensor) else list(vector)
    s = float(input_size)
    cx, cy, a1, a2, cos2t, sin2t = [float(v) for v in vals]
    angle_deg = math.degrees(0.5 * math.atan2(sin2t, cos2t)) % 180.0
    return ((cx * s, cy * s), (max(1.0, a1 * s), max(1.0, a2 * s)), float(angle_deg))


def _vector_to_gaussian(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """正規化楕円ベクトルから2D一様楕円の平均(mu)と共分散成分(sxx, syy, sxy)を解析計算する。

    OpenCV楕円では axis_1 が角度θ方向、axis_2 が θ+90° 方向の全径を表すため、
    半軸 r1 = axis_1/2, r2 = axis_2/2 の一様楕円共分散は
      sxx = (r1^2 + r2^2 + (r1^2 - r2^2)*cos(2θ)) / 8
      syy = (r1^2 + r2^2 - (r1^2 - r2^2)*cos(2θ)) / 8
      sxy = ((r1^2 - r2^2)*sin(2θ)) / 8
    となり、角度の周期不連続・長短軸入れ替わり・真円付近の角度不定性をすべて回避できる。
    """
    mu = vec[:, 0:2]
    r1_sq = (0.5 * vec[:, 2]).square()
    r2_sq = (0.5 * vec[:, 3]).square()
    cos2t = vec[:, 4]
    sin2t = vec[:, 5]

    sum_r = r1_sq + r2_sq
    diff_r = r1_sq - r2_sq
    sxx = 0.125 * (sum_r + diff_r * cos2t) + 1e-6
    syy = 0.125 * (sum_r - diff_r * cos2t) + 1e-6
    sxy = 0.125 * (diff_r * sin2t)
    return mu, sxx, syy, sxy


def prob_iou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    center_weight: float = 0.5,
) -> torch.Tensor:
    """2DガウスHellinger距離（ProbIoU）＋中心補助SmoothL1損失を計算する。"""
    mu1, a1, b1, c1 = _vector_to_gaussian(pred)
    mu2, a2, b2, c2 = _vector_to_gaussian(target)

    # 2x2対称共分散行列の平均 Σ = (Σ1 + Σ2) / 2
    a = 0.5 * (a1 + a2)
    b = 0.5 * (b1 + b2)
    c = 0.5 * (c1 + c2)

    det1 = (a1 * b1 - c1.square()).clamp_min(1e-12)
    det2 = (a2 * b2 - c2.square()).clamp_min(1e-12)
    det = (a * b - c.square()).clamp_min(1e-12)

    dx = mu1[:, 0] - mu2[:, 0]
    dy = mu1[:, 1] - mu2[:, 1]
    quad = (b * dx.square() - 2.0 * c * dx * dy + a * dy.square()) / det
    b1_term = 0.125 * quad
    b2_term = 0.5 * torch.log((det / torch.sqrt(det1 * det2)).clamp_min(1.0))
    bd = (b1_term + b2_term).clamp(0.0, 100.0)

    # Hellinger距離 in [0, 1]（0が完全一致）
    hellinger = torch.sqrt((1.0 - torch.exp(-bd)).clamp_min(1e-8))
    center_l1 = F.smooth_l1_loss(mu1, mu2, reduction="none", beta=0.05).mean(dim=-1)
    per_sample = hellinger + float(center_weight) * center_l1

    if valid_mask is not None:
        mask = valid_mask.to(per_sample.dtype).reshape(-1)
        denom = mask.sum()
        if float(denom.item()) <= 0.0:
            return (per_sample * 0.0).sum()
        return (per_sample * mask).sum() / denom
    return per_sample.mean()


def load_regressor_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    default_base_channels: int = 16,
) -> tuple[TinyEllipseRegressor, dict[str, Any]]:
    """直接楕円回帰モデルのチェックポイントを読み込み、推論モードで返す。"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"直接回帰モデル重みが見つかりません: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    base_channels = int(checkpoint.get("base_channels", default_base_channels))
    max_axis_ratio = float(checkpoint.get("max_axis_ratio", 1.4))
    min_axis_ratio = float(checkpoint.get("min_axis_ratio", 0.02))
    model = TinyEllipseRegressor(
        base_channels=base_channels,
        max_axis_ratio=max_axis_ratio,
        min_axis_ratio=min_axis_ratio,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint
