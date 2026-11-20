"""幾何比較器：標準Canny・連結輪郭分離・輪郭別RANSAC・同心内周選択。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.geometry import Ellipse
from core.ransac import fit_ellipse_ransac


def extract_canny_contours(image: np.ndarray, settings: dict[str, Any]) -> dict[str, Any]:
    """Gaussian平滑化、Cannyエッジ抽出、連結輪郭の分離を行う。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur_size = int(settings["blur_kernel_size"])
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), float(settings["blur_sigma"]))
    edges = cv2.Canny(blurred, int(settings["canny_low"]), int(settings["canny_high"]))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    minimum_points = int(settings["min_contour_points"])
    separated = [
        {"contour": contour, "point_count": len(contour)}
        for contour in contours
        if len(contour) >= minimum_points
    ]
    separated.sort(key=lambda item: item["point_count"], reverse=True)
    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "contours": separated[: int(settings["max_contours"])],
    }


def _fit_contour_candidates(
    stages: dict[str, Any],
    image_shape: tuple[int, ...],
    ransac_settings: dict[str, Any],
    random_seed: int,
) -> list[dict[str, Any]]:
    """分離した輪郭ごとにRANSAC楕円推定を行う。"""
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
            result["angular_coverage"] * inlier_ratio / (1.0 + result["mean_inlier_distance_px"])
        )
        candidates.append(
            {
                **result,
                "contour_index": contour_index,
                "inlier_ratio": inlier_ratio,
                "selection_score": selection_score,
            }
        )
    candidates.sort(key=lambda item: item["selection_score"], reverse=True)
    return candidates


def _ellipse_area_proxy(ellipse: Ellipse) -> float:
    return float(ellipse[1][0] * ellipse[1][1])


def _ellipse_axis_ratio(ellipse: Ellipse) -> float:
    axis_1, axis_2 = ellipse[1]
    return float(min(axis_1, axis_2) / max(axis_1, axis_2))


def select_inner_boundary(candidates: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any] | None:
    """同心・相似な二重輪郭候補から内周側を選択する。"""
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
            if not (float(settings["area_ratio_min"]) <= area_ratio <= float(settings["area_ratio_max"])):
                continue
            larger_center = np.asarray(larger["ellipse"][0], dtype=np.float64)
            larger_major = max(larger["ellipse"][1])
            center_ratio = float(np.linalg.norm(smaller_center - larger_center) / max(larger_major, 1e-9))
            if center_ratio > float(settings["center_distance_major_ratio"]):
                continue
            shape_diff = abs(_ellipse_axis_ratio(smaller["ellipse"]) - _ellipse_axis_ratio(larger["ellipse"]))
            if shape_diff > float(settings["axis_ratio_difference"]):
                continue
            quality_ratio = min(smaller["selection_score"], larger["selection_score"]) / max(
                smaller["selection_score"], larger["selection_score"], 1e-9
            )
            if quality_ratio < float(settings["min_quality_ratio"]):
                continue

            center_compat = 1.0 - center_ratio / max(float(settings["center_distance_major_ratio"]), 1e-9)
            shape_compat = 1.0 - shape_diff / max(float(settings["axis_ratio_difference"]), 1e-9)
            pair_score = min(smaller["selection_score"], larger["selection_score"]) * center_compat * shape_compat
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (
                    pair_score,
                    smaller,
                    {
                        "area_ratio": area_ratio,
                        "center_distance_major_ratio": center_ratio,
                        "axis_ratio_difference": shape_diff,
                        "quality_ratio": quality_ratio,
                    },
                )

    selected = dict(best_pair[1] if best_pair else considered[0])
    selected["selection_mode"] = "inner_pair_prior" if best_pair else "quality_fallback"
    selected["inner_pair"] = best_pair[2] if best_pair else None
    return selected


def detect_canny_contour_ransac(
    image: np.ndarray,
    canny_settings: dict[str, Any],
    ransac_settings: dict[str, Any],
    random_seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """標準Canny＋輪郭別RANSAC＋同心内周選択を実行する。"""
    stages = extract_canny_contours(image, canny_settings["preprocess"])
    candidates = _fit_contour_candidates(stages, image.shape, ransac_settings, random_seed)
    selected = select_inner_boundary(candidates, canny_settings["selector"])
    return selected, {**stages, "candidate_count": len(candidates), "candidates": candidates}
