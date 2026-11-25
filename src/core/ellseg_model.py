"""EllSeg型マルチタスクU-Net（可視内周境界＋Amodal完全穴マスクの2ch出力）と融合推論。

現行TinyUNet(16)（482,737パラメータ）と同一のエンコーダ・デコーダを維持し、
最終1×1畳み込みのみを1ch→2ch（482,754パラメータ、+17増）へ拡張する。
  - Ch 0: 可視内周境界（矩形遮蔽を除いた高精度輪郭）
  - Ch 1: Amodal完全穴マスク（遮蔽・劣化部も含めた内周楕円の内部領域）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.geometry import Ellipse, region_mask
from core.model import DoubleConv
from core.ransac import boundary_ellipse


class TinyEllSegUNet(nn.Module):
    """可視内周境界(Ch0)とAmodal完全穴マスク(Ch1)を同時推定する小型U-Net。"""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = int(base_channels)
        self.base_channels = c
        self.encoder_1 = DoubleConv(3, c)
        self.encoder_2 = DoubleConv(c, c * 2)
        self.encoder_3 = DoubleConv(c * 2, c * 4)
        self.bottleneck = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.up_3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.decoder_3 = DoubleConv(c * 8, c * 4)
        self.up_2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.decoder_2 = DoubleConv(c * 4, c * 2)
        self.up_1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.decoder_1 = DoubleConv(c * 2, c)
        self.output = nn.Conv2d(c, 2, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        f1 = self.encoder_1(image)
        f2 = self.encoder_2(self.pool(f1))
        f3 = self.encoder_3(self.pool(f2))
        b = self.bottleneck(self.pool(f3))
        d3 = self.decoder_3(torch.cat((self.up_3(b), f3), dim=1))
        d2 = self.decoder_2(torch.cat((self.up_2(d3), f2), dim=1))
        d1 = self.decoder_1(torch.cat((self.up_1(d2), f1), dim=1))
        return self.output(d1)


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = (prob + target).sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def _center_of_mass(mask: torch.Tensor) -> torch.Tensor:
    """空間確率マップ [B, 1, H, W] の正規化重心 (cx, cy) in [0, 1]^2 を微分可能に計算する。"""
    _, _, h, w = mask.shape
    ys = torch.linspace(0.0, 1.0, h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w)
    mass = mask.sum(dim=(2, 3)).clamp_min(1e-5)
    cx = (mask * xs).sum(dim=(2, 3)) / mass
    cy = (mask * ys).sum(dim=(2, 3)) / mass
    return torch.cat((cx, cy), dim=1)


def ellseg_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    boundary_bce: nn.Module,
    *,
    hole_weight: float = 1.0,
    com_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """可視内周境界損失＋Amodal完全穴マスク損失＋EllSeg重心一致損失を計算する。"""
    region = masks[:, 0:1]
    visibility = masks[:, 1:2]
    rim = (F.max_pool2d(region, 3, 1, 1) + F.max_pool2d(-region, 3, 1, 1)).clamp(0.0, 1.0)
    visible_rim = rim * visibility

    rim_logits = logits[:, 0:1]
    hole_logits = logits[:, 1:2]

    loss_rim = boundary_bce(rim_logits, visible_rim) + _soft_dice_loss(rim_logits, visible_rim)
    loss_hole = F.binary_cross_entropy_with_logits(hole_logits, region) + _soft_dice_loss(hole_logits, region)

    # 全遮蔽（画像に可視領域がほぼない）サンプルでも真値重心が定義できるため、穴領域があるサンプルで重心損失を計算する
    hole_prob = torch.sigmoid(hole_logits)
    pred_com = _center_of_mass(hole_prob)
    gt_com = _center_of_mass(region)
    loss_com = F.l1_loss(pred_com, gt_com)

    total = loss_rim + float(hole_weight) * loss_hole + float(com_weight) * loss_com
    parts = {
        "rim": float(loss_rim.item()),
        "hole": float(loss_hole.item()),
        "com": float(loss_com.item()),
    }
    return total, parts


def _largest_hole_component(prob_hole: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """穴確率マップから面積最大の連結成分マスク（uint8 0/255）を抽出する。"""
    binary = (prob_hole >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_id = int(np.argmax(areas)) + 1
    if int(stats[best_id, cv2.CC_STAT_AREA]) < 24:
        return np.zeros_like(binary, dtype=np.uint8)
    return ((labels == best_id).astype(np.uint8)) * 255


def hole_mask_to_ellipse(
    prob_hole: np.ndarray,
    *,
    threshold: float = 0.5,
    min_axis_px: float = 6.0,
) -> Ellipse | None:
    """Amodal完全穴マスクの最大連結成分の輪郭から直接楕円を推定する。"""
    comp = _largest_hole_component(prob_hole, threshold=threshold)
    if int(comp.max()) == 0:
        return None
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=len)
    if len(contour) < 5:
        return None
    ellipse = cv2.fitEllipseDirect(contour.astype(np.float32))
    (cx, cy), (a1, a2), angle = ellipse
    if not np.isfinite([cx, cy, a1, a2, angle]).all():
        return None
    if min(a1, a2) < min_axis_px or max(a1, a2) > 2.2 * prob_hole.shape[0]:
        return None
    return ((float(cx), float(cy)), (float(a1), float(a2)), float(angle))


def _ellipse_mask_iou(ellipse: Ellipse, mask_u8: np.ndarray) -> float:
    """推定楕円の塗りつぶし領域と二値穴マスク(0/255)のIoUを計算する。"""
    h, w = mask_u8.shape[:2]
    e_mask = region_mask(h, ellipse) >= 0.5
    h_mask = mask_u8 > 0
    inter = int(np.count_nonzero(e_mask & h_mask))
    union = int(np.count_nonzero(e_mask | h_mask))
    return float(inter / union) if union > 0 else 0.0


def predict_ellseg_ellipse(
    prob_boundary: np.ndarray,
    prob_hole: np.ndarray,
    ransac_cfg: dict[str, Any],
    *,
    mode: str = "fused",
    threshold: float = 0.5,
    gate_radius_px: int = 12,
) -> Ellipse | None:
    """2ch出力（可視境界＋Amodal穴マスク）から指定モードで楕円を推定する。

    mode:
      - 'boundary_only': Ch0のみを現行と同じ確率重み付きRANSACへ渡す
      - 'hole_only'    : Ch1（Amodal完全穴マスク）の輪郭のみから楕円フィットする
      - 'fused'        : Ch1の穴輪郭バンドでCh0の外周誤反応を除去してRANSACし、
                         極限遮蔽等で可視境界が不足・不整合な場合はCh1楕円へフォールバックする
    """
    if mode == "boundary_only":
        return boundary_ellipse(prob_boundary, ransac_cfg, threshold=threshold)

    hole_comp = _largest_hole_component(prob_hole, threshold=threshold)
    hole_ellipse = hole_mask_to_ellipse(prob_hole, threshold=threshold, min_axis_px=float(ransac_cfg["min_axis_px"]))
    if mode == "hole_only":
        return hole_ellipse

    if mode != "fused":
        raise ValueError(f"未対応のmodeです: {mode}")

    # 穴マスクが存在する場合は、その輪郭周辺（±gate_radius_px）に境界候補を限定して外周誤認を防ぐ
    if int(hole_comp.max()) > 0:
        ksize = 2 * int(gate_radius_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(hole_comp, kernel)
        eroded = cv2.erode(hole_comp, kernel)
        rim_band = (dilated > 0) & (eroded == 0)
        gated_prob = np.where(rim_band, prob_boundary, 0.0).astype(np.float32)

        gated_ellipse = boundary_ellipse(gated_prob, ransac_cfg, threshold=threshold)
        if gated_ellipse is not None:
            # ゲート付き境界RANSACが穴マスクと整合していれば高精度な境界推定を採用する
            if _ellipse_mask_iou(gated_ellipse, hole_comp) >= 0.65:
                return gated_ellipse
            # 極限遮蔽等で可視弧が短く境界RANSACが歪んだ場合は、完全穴マスクの楕円を優先する
            if hole_ellipse is not None:
                return hole_ellipse
            return gated_ellipse

        # 可視境界点が枯渇（遮蔽0.75や強黒つぶれ）している場合はAmodal穴マスクから補完する
        if hole_ellipse is not None:
            return hole_ellipse

    # 穴マスク自体が得られない場合の最終フォールバック
    return boundary_ellipse(prob_boundary, ransac_cfg, threshold=threshold)


def load_ellseg_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    default_base_channels: int = 16,
) -> tuple[TinyEllSegUNet, dict[str, Any]]:
    """EllSeg型モデルのチェックポイントを読み込み、推論モードで返す。"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"EllSeg型モデル重みが見つかりません: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    base_channels = int(checkpoint.get("base_channels", default_base_channels))
    model = TinyEllSegUNet(base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint
