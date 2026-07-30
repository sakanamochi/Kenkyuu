import cv2
import numpy as np

from paf_ring_detection.methods.canny_contour_ransac import (
    detect_canny_contour_ransac,
    extract_curve_contours,
)


def _settings() -> tuple[dict, dict]:
    detector = {
        "preprocess": {
            "blur_kernel_size": 5,
            "blur_sigma": 1.2,
            "canny_low": 40,
            "canny_high": 120,
            "sobel_kernel_size": 3,
            "center_search_radius_ratio": 0.15,
            "center_search_step_px": 6,
            "center_search_max_points": 8000,
            "center_alignment_power": 2.0,
            "min_radial_alignment": 0.40,
            "min_contour_points": 20,
            "min_contour_spread_ratio": 0.01,
            "max_contours": 30,
        },
        "selector": {
            "area_ratio_min": 1.2,
            "area_ratio_max": 4.0,
            "center_distance_major_ratio": 0.12,
            "axis_ratio_difference": 0.10,
            "min_quality_ratio": 0.10,
            "max_candidates": 12,
        },
    }
    ransac = {
        "iterations": 300,
        "distance_threshold_px": 2.0,
        "min_inliers": 30,
        "min_axis_px": 6.0,
        "min_axis_ratio": 0.10,
        "max_axis_diagonal_ratio": 2.0,
        "angular_bins": 72,
        "refine_iterations": 3,
        "perimeter_power": 0.0,
        "random_seed": 7,
    }
    return detector, ransac


def _ring_with_radial_ribs() -> np.ndarray:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 110, (120, 120, 120), -1)
    cv2.circle(image, (128, 128), 72, (0, 0, 0), -1)
    for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
        inner = (
            int(round(128 + 76 * np.cos(angle))),
            int(round(128 + 76 * np.sin(angle))),
        )
        outer = (
            int(round(128 + 106 * np.cos(angle))),
            int(round(128 + 106 * np.sin(angle))),
        )
        cv2.line(image, inner, outer, (20, 20, 20), 2, cv2.LINE_AA)
    return image


def test_radial_gradient_filter_reduces_canny_edge_pixels():
    detector, _ = _settings()
    image = _ring_with_radial_ribs()
    stages = extract_curve_contours(image, detector["preprocess"])

    assert np.count_nonzero(stages["edges"]) < np.count_nonzero(
        stages["canny_edges"]
    )
    assert len(stages["contours"]) >= 2


def test_detects_inner_circle_with_shared_ransac():
    detector, ransac = _settings()
    selected, diagnostics = detect_canny_contour_ransac(
        _ring_with_radial_ribs(),
        detector,
        ransac,
        random_seed=7,
    )

    assert selected is not None
    assert diagnostics["candidate_count"] >= 2
    (center_x, center_y), axes, _ = selected["ellipse"]
    assert abs(center_x - 128) < 3
    assert abs(center_y - 128) < 3
    assert abs(min(axes) - 144) < 8
    assert abs(max(axes) - 144) < 8
