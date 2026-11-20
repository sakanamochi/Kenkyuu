"""楕円の正解生成・座標変換・塗りつぶし領域IoU評価・重ね描き描画。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

Ellipse = tuple[tuple[float, float], tuple[float, float], float]


def label_to_ellipse(label: dict[str, Any]) -> Ellipse:
    """投影された内周頂点列からOpenCV形式の正解楕円を求める。"""
    points = np.asarray(label["image_points"], dtype=np.float32).reshape(-1, 1, 2)
    if len(points) < 5:
        raise ValueError("正解楕円の計算には5点以上必要です。")
    return cv2.fitEllipse(points)


def scale_ellipse(ellipse: Ellipse, scale: float) -> Ellipse:
    """楕円の中心・軸長を倍率scaleで変換する。"""
    (cx, cy), (axis_1, axis_2), angle = ellipse
    return (
        (float(cx * scale), float(cy * scale)),
        (float(axis_1 * scale), float(axis_2 * scale)),
        float(angle),
    )


def translate_ellipse(ellipse: Ellipse, dx: float, dy: float) -> Ellipse:
    """楕円の中心を平行移動する。"""
    (cx, cy), (axis_1, axis_2), angle = ellipse
    return ((float(cx + dx), float(cy + dy)), (float(axis_1), float(axis_2)), float(angle))


def ellipse_to_dict(ellipse: Ellipse | None) -> dict[str, float] | None:
    """楕円を可読なJSON辞書へ変換する。"""
    if ellipse is None:
        return None
    (cx, cy), (axis_1, axis_2), angle = ellipse
    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "axis_1": float(axis_1),
        "axis_2": float(axis_2),
        "angle_deg": float(angle),
    }


def ellipse_to_list(ellipse: Ellipse | None) -> list[Any] | None:
    """OpenCV形式の楕円を入れ子リスト形式へ変換する。"""
    if ellipse is None:
        return None
    (cx, cy), (axis_1, axis_2), angle = ellipse
    return [[float(cx), float(cy)], [float(axis_1), float(axis_2)], float(angle)]


def region_mask(size: int, ellipse: Ellipse) -> np.ndarray:
    """アンチエイリアス付きの完全な穴内部マスク（0〜1 float32）を作る。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1, cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def _major_axis_angle(ellipse: Ellipse) -> float:
    _, (axis_1, axis_2), angle = ellipse
    return angle if axis_1 >= axis_2 else (angle + 90.0) % 180.0


def evaluate_ellipses(
    detected: Ellipse,
    ground_truth: Ellipse,
    image_shape: tuple[int, ...],
) -> dict[str, float]:
    """塗りつぶし楕円領域のIoUおよび中心・軸長・角度誤差を計算する。"""
    (cx, cy), detected_axis_pair, _ = detected
    (gt_cx, gt_cy), ground_truth_axis_pair, _ = ground_truth

    detected_axes = sorted(detected_axis_pair)
    ground_truth_axes = sorted(ground_truth_axis_pair)
    raw_angle_error = abs(_major_axis_angle(detected) - _major_axis_angle(ground_truth))

    height, width = image_shape[:2]
    detected_mask = np.zeros((height, width), dtype=np.uint8)
    ground_truth_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(detected_mask, detected, 255, -1)
    cv2.ellipse(ground_truth_mask, ground_truth, 255, -1)

    intersection = np.count_nonzero((detected_mask > 0) & (ground_truth_mask > 0))
    union = np.count_nonzero((detected_mask > 0) | (ground_truth_mask > 0))
    return {
        "center_error_px": float(math.hypot(cx - gt_cx, cy - gt_cy)),
        "minor_axis_error_px": float(abs(detected_axes[0] - ground_truth_axes[0])),
        "major_axis_error_px": float(abs(detected_axes[1] - ground_truth_axes[1])),
        "angle_error_deg": float(min(raw_angle_error, 180.0 - raw_angle_error)),
        "ellipse_iou": float(intersection / union if union else 0.0),
    }


def draw_ellipse_overlay(
    image: np.ndarray,
    detected: Ellipse | None,
    ground_truth: Ellipse | None = None,
) -> np.ndarray:
    """推定楕円（緑）および任意の正解楕円（シアン）を重ね描きする。"""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    thickness = max(2, round(max(canvas.shape[:2]) / 240))
    if ground_truth is not None:
        cv2.ellipse(canvas, ground_truth, (255, 220, 0), thickness, cv2.LINE_AA)
    if detected is not None:
        cv2.ellipse(canvas, detected, (0, 220, 0), thickness, cv2.LINE_AA)
        center = (int(round(detected[0][0])), int(round(detected[0][1])))
        cv2.circle(canvas, center, max(2, thickness), (0, 220, 0), -1, cv2.LINE_AA)
    return canvas
