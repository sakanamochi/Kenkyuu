"""CNNリング尤度をweighted RANSACで楕円へ変換する。"""

from __future__ import annotations

import numpy as np

from paf_ring_detection.methods.ransac import fit_ellipse_ransac


def detect_from_probability(
    probability: np.ndarray,
    settings: dict,
    random_seed: int,
) -> dict | None:
    """閾値以上の画素を、確率付きのRANSAC点群として使う。"""
    rows, columns = np.nonzero(
        probability >= settings["probability_threshold"]
    )
    if len(rows) < 5:
        return None

    points = np.column_stack((columns, rows)).astype(np.float32)
    weights = probability[rows, columns].astype(np.float64)

    if len(points) > settings["max_points"]:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(
            len(points),
            size=settings["max_points"],
            replace=False,
            p=weights / weights.sum(),
        )
        points = points[indices]
        weights = weights[indices]

    ransac_settings = {
        key: value
        for key, value in settings.items()
        if key not in {"probability_threshold", "max_points"}
    }
    return fit_ellipse_ransac(
        points,
        probability.shape,
        ransac_settings,
        weights=weights,
        random_seed=random_seed,
    )
