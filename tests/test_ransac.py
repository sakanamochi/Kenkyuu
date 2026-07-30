import math

import numpy as np

from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.ransac import (
    _ransac_hypothesis_score,
    fit_ellipse_ransac,
)


SETTINGS = {
    "iterations": 1000,
    "distance_threshold_px": 2.5,
    "min_inliers": 80,
    "min_axis_px": 8.0,
    "min_axis_ratio": 0.1,
    "max_axis_diagonal_ratio": 2.0,
    "angular_bins": 72,
    "refine_iterations": 3,
    "perimeter_power": 0.0,
    "random_seed": 1,
}


def test_ransac_fits_partial_ellipse_with_outliers():
    rng = np.random.default_rng(1234)
    expected = ((210.0, 160.0), (180.0, 90.0), 30.0)
    angles = np.linspace(math.radians(15), math.radians(300), 260)
    local = np.column_stack((90 * np.cos(angles), 45 * np.sin(angles)))
    radians = math.radians(30)
    rotation = np.array(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]]
    )
    ellipse_points = local @ rotation.T + np.array([210, 160])
    ellipse_points += rng.normal(0, 0.6, ellipse_points.shape)
    outliers = rng.uniform([0, 0], [420, 320], size=(260, 2))

    result = fit_ellipse_ransac(
        np.vstack((ellipse_points, outliers)),
        (320, 420),
        SETTINGS,
        random_seed=42,
    )

    assert result is not None
    metrics = evaluate_ellipses(result["ellipse"], expected, (320, 420))
    assert metrics["ellipse_iou"] > 0.85


def test_total_support_does_not_prefer_small_ellipse():
    small = ((100.0, 100.0), (30.0, 12.0), 0.0)
    large = ((100.0, 100.0), (120.0, 48.0), 0.0)
    small_score = _ransac_hypothesis_score(
        100, small, 1.0, {"perimeter_power": 0.0}
    )
    large_score = _ransac_hypothesis_score(
        100, large, 1.0, {"perimeter_power": 0.0}
    )
    assert small_score == large_score
