import math

import cv2
import numpy as np


def _point_distances(points: np.ndarray, ellipse) -> np.ndarray:
    """楕円の陰関数から、輪郭近傍での幾何距離を一次近似する。"""
    (cx, cy), (axis_1, axis_2), angle = ellipse
    semi_axis_1 = axis_1 / 2.0
    semi_axis_2 = axis_2 / 2.0
    centered = points - np.array([cx, cy], dtype=np.float64)
    radians = math.radians(angle)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    local_x = centered[:, 0] * cos_angle + centered[:, 1] * sin_angle
    local_y = -centered[:, 0] * sin_angle + centered[:, 1] * cos_angle

    implicit = (
        local_x**2 / semi_axis_1**2
        + local_y**2 / semi_axis_2**2
        - 1.0
    )
    gradient = 2.0 * np.sqrt(
        local_x**2 / semi_axis_1**4 + local_y**2 / semi_axis_2**4
    )
    return np.abs(implicit) / np.maximum(gradient, 1e-9)


def _angular_coverage(points: np.ndarray, ellipse, bins: int) -> float:
    (cx, cy), (axis_1, axis_2), angle = ellipse
    centered = points - np.array([cx, cy], dtype=np.float64)
    radians = math.radians(angle)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    local_x = centered[:, 0] * cos_angle + centered[:, 1] * sin_angle
    local_y = -centered[:, 0] * sin_angle + centered[:, 1] * cos_angle
    angles = np.mod(
        np.arctan2(local_y / (axis_2 / 2), local_x / (axis_1 / 2)),
        2 * np.pi,
    )
    occupied = np.unique((angles / (2 * np.pi) * bins).astype(int)).size
    return float(occupied / bins)


def _ellipse_perimeter(ellipse) -> float:
    """Ramanujanの近似式で楕円周長を求める。"""
    _, (axis_1, axis_2), _ = ellipse
    semi_axis_1 = axis_1 / 2.0
    semi_axis_2 = axis_2 / 2.0
    return math.pi * (
        3.0 * (semi_axis_1 + semi_axis_2)
        - math.sqrt(
            (3.0 * semi_axis_1 + semi_axis_2)
            * (semi_axis_1 + 3.0 * semi_axis_2)
        )
    )


def _ransac_hypothesis_score(
    consensus_weight: float,
    ellipse,
    coverage: float,
    settings: dict,
) -> float:
    """支持量・角度被覆・周長補正から仮説スコアを計算する。

    perimeter_power=1.0 は従来の支持密度、0.0 はCNN尤度の総支持量を使う。
    中間値は小楕円優遇を連続的に弱めるアブレーション用である。
    """
    perimeter_power = float(settings.get("perimeter_power", 1.0))
    if not 0.0 <= perimeter_power <= 1.0:
        raise ValueError("perimeter_powerは0.0から1.0の範囲で指定してください")
    perimeter = max(_ellipse_perimeter(ellipse), 1e-9)
    return float(
        consensus_weight
        / (perimeter**perimeter_power)
        * coverage
    )


def _is_valid_ellipse(ellipse, image_shape, settings: dict) -> bool:
    (cx, cy), (axis_1, axis_2), angle = ellipse
    values = np.array([cx, cy, axis_1, axis_2, angle], dtype=np.float64)
    if not np.isfinite(values).all() or axis_1 <= 0 or axis_2 <= 0:
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
    image_shape,
    settings: dict,
    *,
    weights: np.ndarray | None = None,
    random_seed: int | None = None,
) -> dict | None:
    """点群からRANSACで楕円を推定する。CNN点群ではweightsを確率として渡せる。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 5:
        return None

    if weights is None:
        weights = np.ones(len(points), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(points):
            raise ValueError("pointsとweightsの要素数が一致しません")
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
            sample_indices = rng.choice(
                len(points), size=5, replace=False, p=probabilities
            )
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
        score = _ransac_hypothesis_score(
            consensus_weight, ellipse, coverage, settings
        )
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

    # 最良仮説のインライアだけで再フィットし、インライア判定も更新する。
    ellipse = best["ellipse"]
    inlier_mask = best["inlier_mask"]
    for _ in range(int(settings["refine_iterations"])):
        try:
            refined_ellipse = cv2.fitEllipse(points[inlier_mask].reshape(-1, 1, 2))
        except cv2.error:
            break
        if not _is_valid_ellipse(refined_ellipse, image_shape, settings):
            break
        ellipse = refined_ellipse
        distances = _point_distances(points, ellipse)
        updated_mask = distances <= distance_threshold
        if np.array_equal(updated_mask, inlier_mask):
            break
        if np.count_nonzero(updated_mask) < minimum_inliers:
            break
        inlier_mask = updated_mask

    distances = _point_distances(points, ellipse)
    inlier_mask = distances <= distance_threshold
    if np.count_nonzero(inlier_mask) < minimum_inliers:
        ellipse = best["ellipse"]
        inlier_mask = best["inlier_mask"]
        distances = _point_distances(points, ellipse)
    inlier_points = points[inlier_mask]
    coverage = _angular_coverage(inlier_points, ellipse, angular_bins)
    return {
        "ellipse": ellipse,
        "inlier_mask": inlier_mask,
        "inlier_count": int(np.count_nonzero(inlier_mask)),
        "point_count": len(points),
        "angular_coverage": coverage,
        "mean_inlier_distance_px": float(np.mean(distances[inlier_mask])),
        "score": _ransac_hypothesis_score(
            float(weights[inlier_mask].sum()), ellipse, coverage, settings
        ),
        "perimeter_power": float(settings.get("perimeter_power", 1.0)),
    }


def canny_edge_points(edges: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(edges)
    return np.column_stack((columns, rows)).astype(np.float32)


def fit_contour_ransac_candidates(
    contours,
    image_shape,
    settings: dict,
    *,
    random_seed: int,
) -> list[dict]:
    """Canny輪郭ごとにRANSACを適用し、楕円候補を返す。"""
    candidates = []
    minimum_contour_points = int(settings["per_contour_min_points"])
    for contour_index, contour in enumerate(contours):
        points = contour[:, 0, :].astype(np.float32)
        if len(points) < minimum_contour_points:
            continue

        local_settings = {
            **settings,
            "iterations": int(settings["per_contour_iterations"]),
            "min_inliers": min(
                int(settings["min_inliers"]),
                max(
                    10,
                    math.ceil(
                        len(points) * float(settings["per_contour_min_inlier_ratio"])
                    ),
                ),
            ),
        }
        result = fit_ellipse_ransac(
            points,
            image_shape,
            local_settings,
            random_seed=(random_seed + contour_index) % (2**32),
        )
        if result is None:
            continue
        inlier_ratio = result["inlier_count"] / result["point_count"]
        selection_score = (
            result["angular_coverage"]
            * inlier_ratio
            / (1.0 + result["mean_inlier_distance_px"])
        )
        candidates.append(
            {
                **result,
                "contour_index": contour_index,
                "contour_points": len(points),
                "inlier_ratio": inlier_ratio,
                "selection_score": selection_score,
            }
        )

    candidates.sort(key=lambda item: item["selection_score"], reverse=True)
    return candidates


def _ellipse_area_proxy(ellipse) -> float:
    return float(ellipse[1][0] * ellipse[1][1])


def _ellipse_axis_ratio(ellipse) -> float:
    axis_1, axis_2 = ellipse[1]
    return float(min(axis_1, axis_2) / max(axis_1, axis_2))


def select_paf_inner_candidate(candidates: list[dict], settings: dict) -> dict | None:
    """同心・相似な二重輪郭では小さい側を内周候補として選ぶ。

    幾何品質だけでは内外の意味を区別できないため、これはPAFが同心二重輪郭を
    持つという明示的なCAD事前知識を使う。条件を満たす対が無ければ従来順位へ戻る。
    """
    if not candidates:
        return None
    maximum = int(settings.get("max_candidates", 10))
    considered = candidates[:maximum]
    best_pair = None
    for smaller in considered:
        smaller_area = _ellipse_area_proxy(smaller["ellipse"])
        smaller_center = np.asarray(smaller["ellipse"][0], dtype=np.float64)
        for larger in considered:
            larger_area = _ellipse_area_proxy(larger["ellipse"])
            if larger_area <= smaller_area:
                continue
            area_ratio = larger_area / smaller_area
            if not (
                float(settings["area_ratio_min"])
                <= area_ratio
                <= float(settings["area_ratio_max"])
            ):
                continue
            larger_center = np.asarray(larger["ellipse"][0], dtype=np.float64)
            larger_major = max(larger["ellipse"][1])
            center_ratio = float(np.linalg.norm(smaller_center - larger_center) / larger_major)
            if center_ratio > float(settings["center_distance_major_ratio"]):
                continue
            shape_difference = abs(
                _ellipse_axis_ratio(smaller["ellipse"])
                - _ellipse_axis_ratio(larger["ellipse"])
            )
            if shape_difference > float(settings["axis_ratio_difference"]):
                continue
            quality_ratio = min(
                smaller["selection_score"], larger["selection_score"]
            ) / max(smaller["selection_score"], larger["selection_score"], 1e-9)
            if quality_ratio < float(settings["min_quality_ratio"]):
                continue
            compatibility = (
                1.0 - center_ratio / float(settings["center_distance_major_ratio"])
            ) * (1.0 - shape_difference / float(settings["axis_ratio_difference"]))
            pair_score = (
                min(smaller["selection_score"], larger["selection_score"])
                * compatibility
            )
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (
                    pair_score,
                    smaller,
                    {
                        "area_ratio": area_ratio,
                        "center_distance_major_ratio": center_ratio,
                        "axis_ratio_difference": shape_difference,
                        "quality_ratio": quality_ratio,
                        "pair_score": pair_score,
                    },
                )
    selected = dict(best_pair[1] if best_pair else candidates[0])
    selected["selection_mode"] = "inner_pair_prior" if best_pair else "quality_fallback"
    selected["inner_pair"] = best_pair[2] if best_pair else None
    return selected
