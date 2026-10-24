"""通常画像へ遮蔽・白飛び・黒つぶれを加える。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from paf_ring_detection.data import Ellipse


def _sector_mask(shape, ellipse: Ellipse, severity: float, start: float) -> np.ndarray:
    """楕円周上の連続区間から画像端までを覆うマスクを作る。"""
    height, width = shape[:2]
    if severity >= 1.0:
        return np.ones((height, width), dtype=bool)

    (cx, cy), (axis_1, axis_2), angle = ellipse
    rows, columns = np.indices((height, width), dtype=np.float32)
    x = columns - cx
    y = rows - cy
    rotation = math.radians(angle)
    local_x = x * math.cos(rotation) + y * math.sin(rotation)
    local_y = -x * math.sin(rotation) + y * math.cos(rotation)
    ellipse_angle = np.degrees(
        np.arctan2(local_y / (axis_2 / 2), local_x / (axis_1 / 2))
    )
    return np.mod(ellipse_angle - start, 360.0) <= severity * 360.0


def training_effect(
    image: np.ndarray,
    name: str,
    severity: float,
    ellipse: Ellipse,
    rng: np.random.Generator,
) -> np.ndarray:
    """学習・検証・評価に共通の撮像効果を適用する。"""
    if name == "clean" or severity == 0:
        return image.copy()

    if name not in {'occlusion', 'whiteout', 'black_crush'}:
        raise ValueError(f'未対応の撮像効果です: {name}')

    normalized = image.astype(np.float32) / 255.0
    if name == "black_crush":
        threshold = 0.58 * severity
        result = np.clip((normalized - threshold) / (1.0 - threshold), 0.0, 1.0)
        return np.rint(result ** (1.0 + 1.8 * severity) * 255).astype(np.uint8)

    start = float(rng.uniform(0, 360))
    mask = _sector_mask(image.shape, ellipse, severity, start)
    if name == "occlusion":
        result = image.copy()
        result[mask] = 0
        return result

    bloom = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2 + 14 * severity)
    exposed = np.clip(normalized * (1 + 1.8 * severity), 0, 1)
    alpha = np.clip(bloom * (0.85 + 0.15 * severity), 0, 1)[..., None]
    return np.rint((exposed * (1 - alpha) + alpha) * 255).astype(np.uint8)
