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


def _ellipse_normalized_radius_sq(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    """各点の楕円正規化二乗半径 (x'/a)^2 + (y'/b)^2 を返す（<=1.0 が内部）。"""
    (cx, cy), (axis_1, axis_2), angle = ellipse
    half_1 = max(float(axis_1) * 0.5, 1e-9)
    half_2 = max(float(axis_2) * 0.5, 1e-9)
    centered = np.asarray(points, dtype=np.float64) - np.array([cx, cy], dtype=np.float64)
    radians = math.radians(float(angle))
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    local_x = centered[:, 0] * cos_a + centered[:, 1] * sin_a
    local_y = -centered[:, 0] * sin_a + centered[:, 1] * cos_a
    return (local_x / half_1) ** 2 + (local_y / half_2) ** 2


def _sample_ellipse_points(ellipse: Ellipse, sample_count: int = 72) -> np.ndarray:
    """楕円輪郭上の等角度サンプル点群を返す。"""
    (cx, cy), (axis_1, axis_2), angle = ellipse
    half_1 = float(axis_1) * 0.5
    half_2 = float(axis_2) * 0.5
    theta = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    radians = math.radians(float(angle))
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    lx = half_1 * np.cos(theta)
    ly = half_2 * np.sin(theta)
    x = cx + lx * cos_a - ly * sin_a
    y = cy + lx * sin_a + ly * cos_a
    return np.column_stack((x, y))


def _is_valid_inner_ellipse(
    outer: dict[str, Any],
    inner: dict[str, Any],
    all_points: np.ndarray,
    all_weights: np.ndarray,
    settings: dict[str, Any],
) -> bool:
    """残差RANSACで得た第2候補が、第1候補に内包される本物の内周リングかを検証する。"""
    outer_e = outer["ellipse"]
    inner_e = inner["ellipse"]

    outer_major = max(float(outer_e[1][0]), float(outer_e[1][1]))
    inner_major = max(float(inner_e[1][0]), float(inner_e[1][1]))
    major_ratio = inner_major / max(outer_major, 1e-9)
    min_ratio = float(settings.get("inner_major_ratio_min", 0.30))
    max_ratio = float(settings.get("inner_major_ratio_max", 0.90))
    if not (min_ratio <= major_ratio <= max_ratio):
        return False

    # 第2候補の輪郭サンプル点のほぼ全てが第1候補の内部（上端接近の微小マージン含む）に収まること
    samples = _sample_ellipse_points(inner_e, sample_count=72)
    inside_fraction = float(np.mean(_ellipse_normalized_radius_sq(samples, outer_e) <= 1.06))
    if inside_fraction < float(settings.get("inner_containment_min", 0.95)):
        return False

    # 全スケルトン点群上で第2候補の角度被覆と周長あたり支持密度を再評価し、局所ノイズ円を排除する
    dist_threshold = float(settings["distance_threshold_px"])
    distances = _point_distances(all_points, inner_e)
    inliers = distances <= dist_threshold
    if int(np.count_nonzero(inliers)) < int(settings["min_inliers"]):
        return False

    coverage = _angular_coverage(all_points[inliers], inner_e, int(settings["angular_bins"]))
    outer_coverage = float(outer["angular_coverage"])
    min_cov = float(settings.get("inner_min_coverage", 0.40))
    rel_cov = float(settings.get("inner_relative_coverage_min", 0.70))
    if coverage < max(min_cov, outer_coverage * rel_cov):
        return False

    weight_density = float(all_weights[inliers].sum()) / max(_ellipse_perimeter(inner_e), 1e-9)
    if weight_density < float(settings.get("inner_min_weight_density", 0.45)):
        return False
    return True


def boundary_ellipse(
    probability: np.ndarray,
    ransac_settings: dict[str, Any],
    *,
    threshold: float = 0.5,
) -> Ellipse | None:
    """境界確率マップから中心線点群を作り、確率重み付きRANSACで内周楕円を推定する。

    prefer_inner_ellipse=True の場合、第1候補楕円の内側に残る点群へ残差RANSACを適用し、
    完全内包かつ十分な全周支持を持つ内周リングが存在すれば最内周を選択する。
    """
    points, weights = skeleton_points(probability, threshold=threshold)
    min_inliers = int(ransac_settings["min_inliers"])
    if len(points) < min_inliers:
        return None
    current = fit_ellipse_ransac(points, probability.shape, ransac_settings, weights=weights)
    if current is None:
        return None

    if not bool(ransac_settings.get("prefer_inner_ellipse", False)):
        return current["ellipse"]

    # 外周と内周が同時反応した場合に最内周を選ぶ（最大2段の残差探索）
    base_seed = int(ransac_settings["random_seed"])
    for step in range(1, 3):
        distances = _point_distances(points, current["ellipse"])
        not_on_outer = distances > float(ransac_settings["distance_threshold_px"])
        inside_outer = _ellipse_normalized_radius_sq(points, current["ellipse"]) <= 1.02
        residual_mask = not_on_outer & inside_outer
        if int(np.count_nonzero(residual_mask)) < min_inliers:
            break
        inner = fit_ellipse_ransac(
            points[residual_mask],
            probability.shape,
            ransac_settings,
            weights=weights[residual_mask],
            random_seed=base_seed + step * 1009,
        )
        if inner is None or not _is_valid_inner_ellipse(current, inner, points, weights, ransac_settings):
            break
        # 採用した内周楕円を全点群インライアで最終精緻化する
        all_dist = _point_distances(points, inner["ellipse"])
        all_inliers = all_dist <= float(ransac_settings["distance_threshold_px"])
        if int(np.count_nonzero(all_inliers)) >= min_inliers:
            try:
                refined = cv2.fitEllipse(points[all_inliers].reshape(-1, 1, 2))
                if _is_valid_ellipse(refined, probability.shape, ransac_settings):
                    inner = {
                        **inner,
                        "ellipse": refined,
                        "angular_coverage": _angular_coverage(
                            points[all_inliers], refined, int(ransac_settings["angular_bins"])
                        ),
                    }
            except cv2.error:
                pass
        current = inner

    return current["ellipse"]
