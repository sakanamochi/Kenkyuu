from __future__ import annotations

import math

import cv2
import numpy as np

from paflab.labels import Ellipse


DEGRADATION_TYPES = ("occlusion", "whiteout", "black_crush")


def _validate_severity(severity: float) -> float:
    value = float(severity)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"severityは0から1の範囲で指定してください: {value}")
    return value


def _ellipse_sector_mask(
    image_shape,
    ellipse: Ellipse,
    severity: float,
    start_angle_deg: float,
) -> np.ndarray:
    """楕円角度で指定割合を覆い、画像端まで続くセクタを生成する。"""
    height, width = image_shape[:2]
    (cx, cy), (axis_1, axis_2), ellipse_angle = ellipse
    extent = severity * 360.0
    if extent >= 360.0 - 1e-9:
        return np.full((height, width), 255, dtype=np.uint8)
    rows, columns = np.indices((height, width), dtype=np.float32)
    centered_x = columns - float(cx)
    centered_y = rows - float(cy)
    rotation = math.radians(float(ellipse_angle))
    local_x = centered_x * math.cos(rotation) + centered_y * math.sin(rotation)
    local_y = -centered_x * math.sin(rotation) + centered_y * math.cos(rotation)
    normalized_x = local_x / max(float(axis_1) / 2.0, 1e-6)
    normalized_y = local_y / max(float(axis_2) / 2.0, 1e-6)
    angles = np.mod(np.degrees(np.arctan2(normalized_y, normalized_x)), 360.0)
    relative = np.mod(angles - float(start_angle_deg), 360.0)
    return np.where(relative <= extent, 255, 0).astype(np.uint8)


def apply_degradation(
    image: np.ndarray,
    degradation: str,
    severity: float,
    ellipse: Ellipse,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """制御可能な単一劣化を画像へ適用し、生成条件も返す。"""
    severity = _validate_severity(severity)
    if degradation == "clean" or severity == 0.0:
        return image.copy(), {"degradation": "clean", "severity": 0.0}
    if degradation not in DEGRADATION_TYPES:
        raise ValueError(f"未対応の劣化です: {degradation}")

    normalized = image.astype(np.float32) / 255.0
    metadata = {"degradation": degradation, "severity": severity}

    if degradation == "occlusion":
        start_angle = float(rng.uniform(0.0, 360.0))
        mask = _ellipse_sector_mask(image.shape, ellipse, severity, start_angle)
        result = image.copy()
        result[mask > 0] = 0
        metadata["start_angle_deg"] = start_angle
        metadata["definition"] = "正解楕円周上の連続遮蔽率"
        return result, metadata

    if degradation == "whiteout":
        # 全体露光に加え、リング周辺の連続区間を局所飽和させてブルームを模擬する。
        start_angle = float(rng.uniform(0.0, 360.0))
        sector = _ellipse_sector_mask(image.shape, ellipse, severity, start_angle)
        bloom_sigma = 2.0 + 14.0 * severity
        bloom = cv2.GaussianBlur(
            sector.astype(np.float32) / 255.0,
            (0, 0),
            bloom_sigma,
        )
        gain = 1.0 + 1.8 * severity
        exposed = np.clip(normalized * gain, 0.0, 1.0)
        alpha = np.clip(bloom * (0.85 + 0.15 * severity), 0.0, 1.0)[..., None]
        result = exposed * (1.0 - alpha) + alpha
        metadata["start_angle_deg"] = start_angle
        metadata["exposure_gain"] = gain
        metadata["bloom_sigma_px"] = bloom_sigma
        metadata["definition"] = "全体露光増加と正解楕円周上の局所飽和・ブルーム"
        return np.rint(result * 255.0).astype(np.uint8), metadata

    threshold = 0.58 * severity
    gamma = 1.0 + 1.8 * severity
    result = np.clip((normalized - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    result = result**gamma
    metadata["black_level_threshold"] = threshold
    metadata["gamma"] = gamma
    metadata["definition"] = "黒レベル上昇とガンマによる暗部量子化"
    return np.rint(result * 255.0).astype(np.uint8), metadata
