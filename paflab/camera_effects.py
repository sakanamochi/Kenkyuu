from __future__ import annotations

import math

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
    """画像全高の黒い矩形帯で、学習方式の有用性を単純に検査する。"""
    severity = _severity(severity)
    if severity == 0.0:
        return image.copy(), {"severity": 0.0}
    height, width = image.shape[:2]
    center_x = float(ellipse[0][0]) + float(rng.uniform(-0.18, 0.18)) * max(
        ellipse[1]
    )
    patch_width = max(1, round(width * severity))
    left = int(np.clip(round(center_x - patch_width / 2), 0, width))
    right = int(np.clip(left + patch_width, 0, width))
    if right - left < patch_width:
        left = max(0, right - patch_width)
    result = image.copy()
    result[:, left:right] = 0
    return result, {
        "severity": severity,
        "rectangle_xyxy": [left, 0, right, height],
        "definition": "画像全高の黒矩形帯。幅は画像幅×severity",
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
    exposure_stops = -5.0 * severity
    black_level = 0.015 + 0.14 * severity
    bits = max(4, round(8 - 4 * severity))
    linear = _linearize(image) * (2.0**exposure_stops)
    linear += rng.normal(0.0, 0.002 + 0.008 * severity, linear.shape).astype(np.float32)
    linear = np.clip((linear - black_level) / (1.0 - black_level), 0.0, 1.0)
    levels = 2**bits - 1
    linear = np.rint(linear * levels) / levels
    return _encode(linear), {
        "severity": severity,
        "exposure_stops": exposure_stops,
        "black_level": black_level,
        "quantization_bits": bits,
        "definition": "低露光、read noise、黒レベルclipping、低bit量子化の近似",
    }


def _sun_uv(conditions: dict) -> tuple[float, float, float]:
    camera = conditions["camera"]
    location = np.asarray(camera["location"], dtype=np.float64)
    target = np.asarray(camera.get("target", [0.0, 0.0, 0.0]), dtype=np.float64)
    forward = target - location
    forward /= np.linalg.norm(forward)
    reference_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, reference_up))) > 0.95:
        reference_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ray = np.asarray(conditions["lighting"]["ray_direction"], dtype=np.float64)
    source = -ray / np.linalg.norm(ray)
    depth = float(np.dot(source, forward))
    lens_mm = float(camera.get("lens_mm", 55.0))
    tangent = 36.0 / (2.0 * lens_mm)
    safe_depth = max(abs(depth), 0.15)
    u = 0.5 + float(np.dot(source, right)) / (2.0 * safe_depth * tangent)
    v = 0.5 - float(np.dot(source, up)) / (2.0 * safe_depth * tangent)
    visibility = 1.0 if depth > 0 else 0.15
    return u, v, visibility


def lens_flare(
    image: np.ndarray,
    conditions: dict,
    severity: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """CGの太陽方向を用いたveiling glare、ghost、bloomの決定論的近似。"""
    severity = _severity(severity)
    if severity == 0.0:
        return image.copy(), {"severity": 0.0}
    height, width = image.shape[:2]
    u, v, visibility = _sun_uv(conditions)
    source = np.array([u * width, v * height], dtype=np.float32)
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    overlay = np.zeros((height, width, 3), dtype=np.float32)

    distance2 = (xx - source[0]) ** 2 + (yy - source[1]) ** 2
    veil_sigma = width * (0.18 + 0.28 * severity)
    veil = np.exp(-distance2 / (2.0 * veil_sigma**2))[..., None]
    overlay += veil * np.array([1.0, 0.82, 0.58], dtype=np.float32) * 0.55

    for fraction, radius, strength in ((0.25, 0.055, 0.24), (0.55, 0.035, 0.18), (0.82, 0.075, 0.12)):
        ghost_center = source + (center - source) * fraction
        ghost_distance = np.sqrt((xx - ghost_center[0]) ** 2 + (yy - ghost_center[1]) ** 2)
        ring = np.exp(-((ghost_distance - width * radius) ** 2) / (2.0 * (width * 0.012) ** 2))
        color = np.array([0.45, 0.72, 1.0], dtype=np.float32)
        overlay += ring[..., None] * color * strength

    linear = _linearize(image)
    highlights = np.clip((linear.mean(axis=2) - 0.65) / 0.35, 0.0, 1.0)
    bloom = cv2.GaussianBlur(highlights, (0, 0), 3.0 + 16.0 * severity)[..., None]
    jitter = float(rng.uniform(0.96, 1.04))
    linear = np.clip(
        linear + (overlay * visibility * severity * jitter) + bloom * 0.35 * severity,
        0.0,
        1.0,
    )
    return _encode(linear), {
        "severity": severity,
        "sun_uv": [u, v],
        "sun_front_visibility": visibility,
        "definition": "CG太陽方向に整合したveiling glare、ghost、highlight bloomの近似",
    }


EFFECTS = {
    "black_rectangle": black_rectangle,
    "sensor_whiteout": sensor_whiteout,
    "sensor_black_crush": sensor_black_crush,
    "lens_flare": lens_flare,
}
