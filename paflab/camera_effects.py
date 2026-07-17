from __future__ import annotations

import cv2
import numpy as np

from paflab.labels import Ellipse


def _severity(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"severityは0から1の範囲で指定してください: {value}")
    return value


def _linearize(image: np.ndarray) -> np.ndarray:
    return np.power(image.astype(np.float32) / 255.0, 2.2)


def _encode(linear: np.ndarray) -> np.ndarray:
    encoded = np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.2)
    return np.rint(encoded * 255.0).astype(np.uint8)


def black_rectangle(
    image: np.ndarray,
    ellipse: Ellipse,
    severity: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """左右どちらかの画像端から伸びる黒矩形で単純遮蔽を検査する。"""
    severity = _severity(severity)
    if severity == 0.0:
        return image.copy(), {"severity": 0.0}
    height, width = image.shape[:2]
    patch_width = max(1, round(width * severity))
    side = "left" if int(rng.integers(0, 2)) == 0 else "right"
    left = 0 if side == "left" else width - patch_width
    right = left + patch_width
    result = image.copy()
    result[:, left:right] = 0
    return result, {
        "severity": severity,
        "side": side,
        "rectangle_xyxy": [left, 0, right, height],
        "definition": "左右いずれかの画像端から伸びる黒矩形。幅は画像幅×severity",
    }


def sensor_whiteout(
    image: np.ndarray,
    severity: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """線形露光、full-well飽和、ハイライト拡散を近似する。"""
    severity = _severity(severity)
    if severity == 0.0:
        return image.copy(), {"severity": 0.0}
    exposure_stops = 5.0 * severity
    linear = _linearize(image) * (2.0**exposure_stops)
    luminance = linear.mean(axis=2)
    saturated = np.clip((luminance - 0.82) / 0.18, 0.0, 1.0)
    sigma = 1.0 + 18.0 * severity
    bloom = cv2.GaussianBlur(saturated, (0, 0), sigma)[..., None]
    shot_sigma = 0.002 + 0.006 * severity
    linear += rng.normal(0.0, shot_sigma, linear.shape).astype(np.float32) * np.sqrt(
        np.maximum(linear, 0.0)
    )
    linear = np.clip(linear + bloom * (0.15 + 0.85 * severity), 0.0, 1.0)
    return _encode(linear), {
        "severity": severity,
        "exposure_stops": exposure_stops,
        "bloom_sigma_px": sigma,
        "definition": "線形露光増加、full-well飽和、shot noise、ハイライト拡散の近似",
    }


def sensor_black_crush(
    image: np.ndarray,
    severity: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """低露光、read noise、黒レベル、低bit量子化を近似する。"""
    severity = _severity(severity)
    if severity == 0.0:
        return image.copy(), {"severity": 0.0}
    # 暗い宇宙背景が多い画像では線形な5段低下が50%付近でほぼ全面黒になるため、
    # 後半ほど変化を緩め、最大強度でも明部の形状がわずかに残る範囲へ校正する。
    exposure_stops = -2.5 * severity**1.25
    black_level = 0.004 + 0.021 * severity**1.5
    bits = max(6, round(8 - 2 * severity))
    linear = _linearize(image) * (2.0**exposure_stops)
    linear += rng.normal(0.0, 0.0015 + 0.003 * severity, linear.shape).astype(
        np.float32
    )
    linear = np.clip((linear - black_level) / (1.0 - black_level), 0.0, 1.0)
    levels = 2**bits - 1
    linear = np.rint(linear * levels) / levels
    return _encode(linear), {
        "severity": severity,
        "exposure_stops": exposure_stops,
        "black_level": black_level,
        "quantization_bits": bits,
        "definition": (
            "緩やかな低露光、read noise、黒レベルclipping、"
            "低bit量子化の近似"
        ),
    }


EFFECTS = {
    "black_rectangle": black_rectangle,
    "sensor_whiteout": sensor_whiteout,
    "sensor_black_crush": sensor_black_crush,
}
