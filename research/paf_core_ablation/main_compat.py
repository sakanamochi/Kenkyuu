"""main 088c4b6のリング教師と推論関数を保持する互換部。関数本体は参照と一致する。"""
from __future__ import annotations
import cv2
import numpy as np
from paf_ring_detection.methods.ransac import fit_ellipse_ransac
Ellipse = tuple[tuple[float, float], tuple[float, float], float]

def rasterize_ring_mask(
    height: int,
    width: int,
    ellipse: Ellipse,
    *,
    thickness: int = 3,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """隠れた部分を含む幾何学的な全リングを教師マスクとして描く。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, thickness, cv2.LINE_AA)
    result = mask.astype(np.float32) / 255.0
    if blur_sigma > 0:
        result = cv2.GaussianBlur(result, (0, 0), blur_sigma)
        maximum = float(result.max())
        if maximum > 0:
            result /= maximum
    return result


def probability_ellipse(
    probability: np.ndarray,
    settings: dict,
    *,
    random_seed: int,
):
    threshold = float(settings["probability_threshold"])
    rows, columns = np.nonzero(probability >= threshold)
    if len(rows) < 5:
        return None
    points = np.column_stack((columns, rows)).astype(np.float32)
    weights = probability[rows, columns].astype(np.float64)
    maximum = int(settings["max_points"])
    if len(points) > maximum:
        rng = np.random.default_rng(random_seed)
        distribution = weights / weights.sum()
        indices = rng.choice(len(points), size=maximum, replace=False, p=distribution)
        points = points[indices]
        weights = weights[indices]
    ransac_settings = {
        key: value
        for key, value in settings.items()
        if key not in ("probability_threshold", "max_points")
    }
    return fit_ellipse_ransac(
        points,
        probability.shape,
        ransac_settings,
        weights=weights,
        random_seed=random_seed,
    )
