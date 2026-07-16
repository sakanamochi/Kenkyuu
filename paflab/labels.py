from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


Ellipse = tuple[tuple[float, float], tuple[float, float], float]


def fit_label_ellipse(label: dict) -> Ellipse:
    """Blenderが投影したリング頂点から正解楕円を求める。"""
    points = np.asarray(label["image_points"], dtype=np.float32).reshape(-1, 1, 2)
    if len(points) < 5:
        raise ValueError("楕円フィットには5点以上の正解点が必要です")
    return cv2.fitEllipse(points)


def read_label_ellipse(path: Path) -> tuple[dict, Ellipse]:
    label = json.loads(path.read_text(encoding="utf-8"))
    return label, fit_label_ellipse(label)


def scale_ellipse(ellipse: Ellipse, scale_x: float, scale_y: float) -> Ellipse:
    (cx, cy), (axis_x, axis_y), angle = ellipse
    # 本実験は正方形への等方リサイズを使う。異方リサイズは角度も変えるため拒否する。
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError("正解楕円は等方リサイズのみ対応しています")
    return (
        (float(cx * scale_x), float(cy * scale_y)),
        (float(axis_x * scale_x), float(axis_y * scale_y)),
        float(angle),
    )


def rasterize_ring_mask(
    height: int,
    width: int,
    ellipse: Ellipse,
    *,
    thickness: int = 3,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """隠れた部分を含む幾何学的な全リングを教師マスクとして描く。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, thickness, cv2.LINE_AA)
    result = mask.astype(np.float32) / 255.0
    if blur_sigma > 0:
        result = cv2.GaussianBlur(result, (0, 0), blur_sigma)
        maximum = float(result.max())
        if maximum > 0:
            result /= maximum
    return result
