"""円周輪郭を優先するCanny前処理と共通RANSACを接続する。"""

from __future__ import annotations

import cv2
import numpy as np

from paf_ring_detection.methods.ransac import fit_ellipse_ransac


def extract_curve_contours(image: np.ndarray, settings: dict) -> dict:
    """畳み込み勾配から中心方向のエッジだけを残し、輪郭を分離する。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_size = int(settings["blur_kernel_size"])
    blurred = cv2.GaussianBlur(
        gray,
        (blur_size, blur_size),
        float(settings["blur_sigma"]),
    )
    edges = cv2.Canny(
        blurred,
        int(settings["canny_low"]),
        int(settings["canny_high"]),
    )

    # Sobel畳み込みで得た勾配が推定中心を向くエッジを円周候補とする。
    sobel_size = int(settings["sobel_kernel_size"])
    gradient_x = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        1,
        0,
        ksize=sobel_size,
    )
    gradient_y = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        0,
        1,
        ksize=sobel_size,
    )
    center = _estimate_circular_center(
        edges,
        gradient_x,
        gradient_y,
        settings,
    )
    radial_alignment = _radial_gradient_alignment(
        edges,
        gradient_x,
        gradient_y,
        center,
    )
    circular_edges = np.where(
        (edges > 0)
        & (radial_alignment >= float(settings["min_radial_alignment"])),
        255,
        0,
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        circular_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    minimum_points = int(settings["min_contour_points"])
    minimum_spread_ratio = float(settings["min_contour_spread_ratio"])
    curve_contours = []
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        if len(points) < minimum_points:
            continue

        # PCAの短軸/長軸分散比が小さい輪郭は、ほぼ直線のリブとして除外する。
        covariance = np.cov(points, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[-1] <= 1e-9:
            continue
        spread_ratio = float(eigenvalues[0] / eigenvalues[-1])
        if spread_ratio < minimum_spread_ratio:
            continue
        curve_contours.append(
            {
                "contour": contour,
                "point_count": len(points),
                "spread_ratio": spread_ratio,
            }
        )

    curve_contours.sort(key=lambda item: item["point_count"], reverse=True)
    maximum_contours = int(settings["max_contours"])
    return {
        "gray": gray,
        "blurred": blurred,
        "canny_edges": edges,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "estimated_center": center,
        "radial_alignment": radial_alignment,
        "edges": circular_edges,
        "contours": curve_contours[:maximum_contours],
    }


def _radial_gradient_alignment(
    edges: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    center: tuple[float, float],
) -> np.ndarray:
    """勾配と中心方向の平行度を0から1で返す。"""
    height, width = edges.shape
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    radial_x = columns - float(center[0])
    radial_y = rows - float(center[1])
    radial_norm = np.hypot(radial_x, radial_y)
    gradient_norm = np.hypot(gradient_x, gradient_y)
    dot_product = gradient_x * radial_x + gradient_y * radial_y
    alignment = np.abs(dot_product) / np.maximum(
        radial_norm * gradient_norm,
        1e-6,
    )
    alignment[edges == 0] = 0.0
    return alignment


def _estimate_circular_center(
    edges: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    settings: dict,
) -> tuple[float, float]:
    """画像中央付近から、円周方向の勾配支持が最大になる中心を探す。"""
    rows, columns = np.nonzero(edges)
    height, width = edges.shape
    image_center = (width / 2.0, height / 2.0)
    if len(rows) == 0:
        return image_center

    maximum_points = int(settings["center_search_max_points"])
    if len(rows) > maximum_points:
        indices = np.linspace(
            0,
            len(rows) - 1,
            maximum_points,
            dtype=np.int64,
        )
        rows = rows[indices]
        columns = columns[indices]

    point_gradient_x = gradient_x[rows, columns].astype(np.float64)
    point_gradient_y = gradient_y[rows, columns].astype(np.float64)
    gradient_norm = np.hypot(point_gradient_x, point_gradient_y)
    magnitude_scale = max(float(np.percentile(gradient_norm, 95)), 1e-6)
    weights = np.minimum(gradient_norm / magnitude_scale, 1.0)
    power = float(settings["center_alignment_power"])

    radius_x = width * float(settings["center_search_radius_ratio"])
    radius_y = height * float(settings["center_search_radius_ratio"])
    step = int(settings["center_search_step_px"])
    candidate_x = np.arange(
        image_center[0] - radius_x,
        image_center[0] + radius_x + 1,
        step,
    )
    candidate_x = np.unique(np.append(candidate_x, image_center[0]))
    candidate_y = np.arange(
        image_center[1] - radius_y,
        image_center[1] + radius_y + 1,
        step,
    )
    candidate_y = np.unique(np.append(candidate_y, image_center[1]))

    best_center = image_center
    best_score = -1.0
    for center_y in candidate_y:
        radial_y = rows.astype(np.float64) - center_y
        for center_x in candidate_x:
            radial_x = columns.astype(np.float64) - center_x
            radial_norm = np.hypot(radial_x, radial_y)
            dot_product = point_gradient_x * radial_x + point_gradient_y * radial_y
            alignment = np.abs(dot_product) / np.maximum(
                gradient_norm * radial_norm,
                1e-6,
            )
            score = float(np.sum(weights * alignment**power))
            if score > best_score:
                best_score = score
                best_center = (float(center_x), float(center_y))
    return best_center


def _fit_contour_candidates(
    stages: dict,
    image_shape,
    ransac_settings: dict,
    random_seed: int,
) -> list[dict]:
    """分離済み輪郭ごとに、CNN側と共通のRANSACを適用する。"""
    candidates = []
    for contour_index, item in enumerate(stages["contours"]):
        points = item["contour"][:, 0, :].astype(np.float32)
        result = fit_ellipse_ransac(
            points,
            image_shape,
            ransac_settings,
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
                "contour_spread_ratio": item["spread_ratio"],
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


def select_inner_boundary(candidates: list[dict], settings: dict) -> dict | None:
    """同心・相似な二重輪郭の小さい側をPAF内周として選ぶ。"""
    if not candidates:
        return None

    considered = candidates[: int(settings["max_candidates"])]
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
            larger_major_axis = max(larger["ellipse"][1])
            center_ratio = float(
                np.linalg.norm(smaller_center - larger_center)
                / max(larger_major_axis, 1e-9)
            )
            if center_ratio > float(settings["center_distance_major_ratio"]):
                continue

            shape_difference = abs(
                _ellipse_axis_ratio(smaller["ellipse"])
                - _ellipse_axis_ratio(larger["ellipse"])
            )
            if shape_difference > float(settings["axis_ratio_difference"]):
                continue

            quality_ratio = min(
                smaller["selection_score"],
                larger["selection_score"],
            ) / max(
                smaller["selection_score"],
                larger["selection_score"],
                1e-9,
            )
            if quality_ratio < float(settings["min_quality_ratio"]):
                continue

            center_compatibility = 1.0 - center_ratio / max(
                float(settings["center_distance_major_ratio"]),
                1e-9,
            )
            shape_compatibility = 1.0 - shape_difference / max(
                float(settings["axis_ratio_difference"]),
                1e-9,
            )
            pair_score = (
                min(smaller["selection_score"], larger["selection_score"])
                * center_compatibility
                * shape_compatibility
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
                    },
                )

    selected = dict(best_pair[1] if best_pair else considered[0])
    selected["selection_mode"] = (
        "inner_pair_prior" if best_pair else "quality_fallback"
    )
    selected["inner_pair"] = best_pair[2] if best_pair else None
    return selected


def detect_canny_contour_ransac(
    image: np.ndarray,
    settings: dict,
    ransac_settings: dict,
    random_seed: int,
) -> tuple[dict | None, dict]:
    """円周優先Canny、輪郭分離、共通RANSACでPAF内周を検出する。"""
    stages = extract_curve_contours(image, settings["preprocess"])
    candidates = _fit_contour_candidates(
        stages,
        image.shape,
        ransac_settings,
        random_seed,
    )
    selected = select_inner_boundary(candidates, settings["selector"])
    diagnostics = {
        **stages,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    return selected, diagnostics
