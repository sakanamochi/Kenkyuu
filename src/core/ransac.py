"""確率重み付きRANSACと形態学的スケルトンによる境界楕円推定。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from core.geometry import Ellipse


def _point_distances(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    """楕円陰関数の一次近似で各点から輪郭までの幾何距離を求める。"""
    (cx, cy), (axis_1, axis_2), angle = ellipse
    semi_axis_1 = axis_1 / 2.0
    semi_axis_2 = axis_2 / 2.0
    centered = points - np.array([cx, cy], dtype=np.float64)
    radians = math.radians(angle)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    local_x = centered[:, 0] * cos_angle + centered[:, 1] * sin_angle
    local_y = -centered[:, 0] * sin_angle + centered[:, 1] * cos_angle

    implicit = local_x**2 / semi_axis_1**2 + local_y**2 / semi_axis_2**2 - 1.0
    gradient = 2.0 * np.sqrt(local_x**2 / semi_axis_1**4 + local_y**2 / semi_axis_2**4)
    return np.abs(implicit) / np.maximum(gradient, 1e-9)


def _angular_coverage(points: np.ndarray, ellipse: Ellipse, bins: int) -> float:
    """インライア点群が楕円全周の角度ビンを覆う割合を返す。"""
    (cx, cy), (axis_1, axis_2), angle = ellipse
    centered = points - np.array([cx, cy], dtype=np.float64)
    radians = math.radians(angle)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    local_x = centered[:, 0] * cos_angle + centered[:, 1] * sin_angle
    local_y = -centered[:, 0] * sin_angle + centered[:, 1] * cos_angle
    angles = np.mod(np.arctan2(local_y / (axis_2 / 2.0), local_x / (axis_1 / 2.0)), 2.0 * np.pi)
    occupied = np.unique((angles / (2.0 * np.pi) * bins).astype(int)).size
    return float(occupied / bins)


def _ellipse_perimeter(ellipse: Ellipse) -> float:
    """Ramanujan近似式で楕円周長を求める。"""
    _, (axis_1, axis_2), _ = ellipse
    semi_axis_1 = axis_1 / 2.0
    semi_axis_2 = axis_2 / 2.0
    return math.pi * (
        3.0 * (semi_axis_1 + semi_axis_2)
        - math.sqrt((3.0 * semi_axis_1 + semi_axis_2) * (semi_axis_1 + 3.0 * semi_axis_2))
    )


def _hypothesis_score(
    consensus_weight: float,
    ellipse: Ellipse,
    coverage: float,
    settings: dict[str, Any],
) -> float:
    """支持量・角度被覆・周長べき乗から仮説スコアを計算する。"""
    perimeter_power = float(settings.get("perimeter_power", 0.0))
    if not 0.0 <= perimeter_power <= 1.0:
        raise ValueError("perimeter_powerは0.0〜1.0で指定してください。")
    perimeter = max(_ellipse_perimeter(ellipse), 1e-9)
    return float(consensus_weight / (perimeter**perimeter_power) * coverage)


def _is_valid_ellipse(ellipse: Ellipse, image_shape: tuple[int, ...], settings: dict[str, Any]) -> bool:
    (cx, cy), (axis_1, axis_2), angle = ellipse
    values = np.array([cx, cy, axis_1, axis_2, angle], dtype=np.float64)
    if not np.isfinite(values).all() or axis_1 <= 0.0 or axis_2 <= 0.0:
        return False
    minor_axis, major_axis = sorted((axis_1, axis_2))
    if minor_axis < float(settings["min_axis_px"]):
        return False
    if minor_axis / major_axis < float(settings["min_axis_ratio"]):
        return False
    height, width = image_shape[:2]
    if major_axis > math.hypot(width, height) * float(settings["max_axis_diagonal_ratio"]):
        return False
    return True


def fit_ellipse_ransac(
    points: np.ndarray,
    image_shape: tuple[int, ...],
    settings: dict[str, Any],
    *,
    weights: np.ndarray | None = None,
    random_seed: int | None = None,
) -> dict[str, Any] | None:
    """点群から確率重み付きRANSACで楕円を推定する。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 5:
        return None

    if weights is None:
        weights = np.ones(len(points), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(points):
            raise ValueError("pointsとweightsの要素数が一致しません。")
        weights = np.maximum(weights, 0.0)

    seed = int(settings["random_seed"] if random_seed is None else random_seed)
    rng = np.random.default_rng(seed)
    probabilities = weights / weights.sum() if weights.sum() > 0 else None
    distance_threshold = float(settings["distance_threshold_px"])
    angular_bins = int(settings["angular_bins"])
    minimum_inliers = int(settings["min_inliers"])
    best = None

    for _ in range(int(settings["iterations"])):
        try:
            sample_indices = rng.choice(len(points), size=5, replace=False, p=probabilities)
            ellipse = cv2.fitEllipseDirect(points[sample_indices].reshape(-1, 1, 2))
        except cv2.error:
            continue
        if not _is_valid_ellipse(ellipse, image_shape, settings):
            continue

        distances = _point_distances(points, ellipse)
        inlier_mask = distances <= distance_threshold
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < minimum_inliers:
            continue
        coverage = _angular_coverage(points[inlier_mask], ellipse, angular_bins)
        consensus_weight = float(weights[inlier_mask].sum())
        score = _hypothesis_score(consensus_weight, ellipse, coverage, settings)
        if best is None or score > best["score"]:
            best = {
                "ellipse": ellipse,
                "inlier_mask": inlier_mask,
                "inlier_count": inlier_count,
                "angular_coverage": coverage,
                "score": score,
            }

    if best is None:
        return None

    ellipse = best["ellipse"]
    inlier_mask = best["inlier_mask"]
    for _ in range(int(settings["refine_iterations"])):
        try:
            refined = cv2.fitEllipse(points[inlier_mask].reshape(-1, 1, 2))
        except cv2.error:
            break
        if not _is_valid_ellipse(refined, image_shape, settings):
            break
        ellipse = refined
        distances = _point_distances(points, ellipse)
        updated_mask = distances <= distance_threshold
        if np.array_equal(updated_mask, inlier_mask) or np.count_nonzero(updated_mask) < minimum_inliers:
            break
        inlier_mask = updated_mask

    distances = _point_distances(points, ellipse)
    inlier_mask = distances <= distance_threshold
    if np.count_nonzero(inlier_mask) < minimum_inliers:
        ellipse = best["ellipse"]
        inlier_mask = best["inlier_mask"]
        distances = _point_distances(points, ellipse)

    coverage = _angular_coverage(points[inlier_mask], ellipse, angular_bins)
    return {
        "ellipse": ellipse,
        "inlier_mask": inlier_mask,
        "inlier_count": int(np.count_nonzero(inlier_mask)),
        "point_count": len(points),
        "angular_coverage": coverage,
        "mean_inlier_distance_px": float(np.mean(distances[inlier_mask])),
        "score": _hypothesis_score(float(weights[inlier_mask].sum()), ellipse, coverage, settings),
    }


def skeleton_points(probability: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """確率マップを二値化して形態学的スケルトンを抽出し、元の確率を重みとする。"""
    remaining = (probability >= threshold).astype(np.uint8)
    skeleton = np.zeros_like(remaining)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(remaining):
        eroded = cv2.erode(remaining, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        opened = cv2.dilate(eroded, kernel)
        skeleton |= remaining & (1 - opened)
        remaining = eroded
    ys, xs = np.where(skeleton > 0)
    return np.stack([xs, ys], axis=1).astype(np.float32), probability[ys, xs]


def boundary_ellipse(
    probability: np.ndarray,
    ransac_settings: dict[str, Any],
    *,
    threshold: float = 0.5,
) -> Ellipse | None:
    """境界確率マップから中心線点群を作り、確率重み付きRANSACで内周楕円を推定する。"""
    points, weights = skeleton_points(probability, threshold=threshold)
    if len(points) < int(ransac_settings["min_inliers"]):
        return None
    fitted = fit_ellipse_ransac(points, probability.shape, ransac_settings, weights=weights)
    return fitted["ellipse"] if fitted is not None else None
