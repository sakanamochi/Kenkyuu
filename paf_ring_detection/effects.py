"""学習画像と撮像診断画像へ加える効果。"""

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
    """初期学習で使う遮蔽・白飛び・黒つぶれを適用する。"""
    if name == "clean" or severity == 0:
        return image.copy()

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


def black_rectangle(
    image: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """左右どちらかの端から黒矩形を伸ばす。"""
    width = image.shape[1]
    patch_width = round(width * severity)
    left = 0 if int(rng.integers(0, 2)) == 0 else width - patch_width
    result = image.copy()
    result[:, left : left + patch_width] = 0
    return result


def _linearize(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) / 255.0) ** 2.2


def _encode(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def sensor_whiteout(
    image: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """露光増加、飽和、簡単なブルームを加える。"""
    linear = _linearize(image) * 2 ** (5 * severity)
    saturated = np.clip((linear.mean(axis=2) - 0.82) / 0.18, 0, 1)
    bloom = cv2.GaussianBlur(saturated, (0, 0), 1 + 18 * severity)[..., None]
    noise = rng.normal(0, 0.002 + 0.006 * severity, linear.shape)
    linear += noise.astype(np.float32) * np.sqrt(np.maximum(linear, 0))
    return _encode(linear + bloom * (0.15 + 0.85 * severity))


def sensor_black_crush(
    image: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """低露光、黒レベル、低bit量子化を加える。"""
    linear = _linearize(image) * 2 ** (-2.5 * severity**1.25)
    linear += rng.normal(0, 0.0015 + 0.003 * severity, linear.shape)
    black_level = 0.004 + 0.021 * severity**1.5
    linear = np.clip((linear - black_level) / (1 - black_level), 0, 1)
    levels = 2 ** max(6, round(8 - 2 * severity)) - 1
    return _encode(np.rint(linear * levels) / levels)


DIAGNOSTIC_EFFECTS = {
    "black_rectangle": black_rectangle,
    "sensor_whiteout": sensor_whiteout,
    "sensor_black_crush": sensor_black_crush,
}
