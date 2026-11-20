"""境界の中心線と確率重み付きRANSACによる楕円推定。"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


Ellipse = tuple[tuple[float, float], tuple[float, float], float]


@dataclass(frozen=True)
class EllipseSettings:
    """採用構成の点群化・幾何制約・乱数を明示する。"""

    probability_threshold: float = 0.5
    iterations: int = 450
    distance_threshold_px: float = 2.0
    min_inliers: int = 12
    min_axis_px: float = 6.0
    min_axis_ratio: float = 0.1
    max_axis_diagonal_ratio: float = 2.0
    angular_bins: int = 72
    refine_iterations: int = 3
    random_seed: int = 20260716
    opencv_seed: int = 20261025

    def __post_init__(self):
        if not 0 < self.probability_threshold <= 1:
            raise ValueError("境界確率の閾値は0より大きく1以下にしてください。")
        if self.iterations < 1 or self.min_inliers < 5 or self.angular_bins < 1 or self.refine_iterations < 0:
            raise ValueError("仮説数・支持点数・角度bin数・再推定回数が不正です。")
        if not (self.distance_threshold_px > 0 and self.min_axis_px > 0
                and 0 < self.min_axis_ratio <= 1 and self.max_axis_diagonal_ratio > 0):
            raise ValueError("楕円の距離・軸長・軸比の制約が不正です。")


def scale_ellipse(ellipse: Ellipse, scale: float) -> Ellipse:
    """中心と軸の全長を拡大縮小し、OpenCVの角度を維持する。"""
    center, axes, angle = ellipse
    return ((float(center[0] * scale), float(center[1] * scale)),
            (float(axes[0] * scale), float(axes[1] * scale)), float(angle))


def label_ellipse(label) -> Ellipse:
    """投影済み内周頂点から正解楕円を求める。"""
    points = np.asarray(label["image_points"], dtype=np.float32).reshape(-1, 1, 2)
    if len(points) < 5 or not np.isfinite(points).all():
        raise ValueError("正解内周には5点以上の有限な座標が必要です。")
    return cv2.fitEllipse(points)


def skeleton_points(probability: np.ndarray, threshold: float = 0.5):
    """二値境界を形態学的スケルトンへ変換し、元確率を重みにする。"""
    remaining = (probability >= threshold).astype(np.uint8)
    skeleton = np.zeros_like(remaining)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while remaining.any():
        eroded = cv2.erode(remaining, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        opened = cv2.dilate(eroded, cross)
        skeleton |= remaining & (1 - opened)
        remaining = eroded
    ys, xs = np.where(skeleton > 0)
    points = np.stack((xs, ys), axis=1).astype(np.float32)
    return points, probability[ys, xs]


def local_coordinates(points: np.ndarray, ellipse: Ellipse):
    """中心を引き、楕円の軸方向へ座標を回転する。"""
    center, _, angle = ellipse
    centered = points - np.array(center, dtype=np.float64)
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    return (centered[:, 0] * cosine + centered[:, 1] * sine,
            -centered[:, 0] * sine + centered[:, 1] * cosine)


def point_distances(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    """楕円の陰関数と勾配から輪郭までの距離を一次近似する。"""
    x, y = local_coordinates(points, ellipse)
    a, b = ellipse[1][0] / 2.0, ellipse[1][1] / 2.0
    implicit = x**2 / a**2 + y**2 / b**2 - 1.0
    gradient = 2.0 * np.sqrt(x**2 / a**4 + y**2 / b**4)
    return np.abs(implicit) / np.maximum(gradient, 1e-9)


def angular_coverage(points: np.ndarray, ellipse: Ellipse, bins: int) -> float:
    """楕円周上の角度binのうち支持点が占める割合を求める。"""
    x, y = local_coordinates(points, ellipse)
    angles = np.mod(np.arctan2(y / (ellipse[1][1] / 2), x / (ellipse[1][0] / 2)), 2 * np.pi)
    return float(np.unique((angles / (2 * np.pi) * bins).astype(int)).size / bins)


def valid_ellipse(ellipse: Ellipse, shape, settings: EllipseSettings) -> bool:
    center, axes, angle = ellipse
    if not np.isfinite([*center, *axes, angle]).all() or min(axes) <= 0:
        return False
    minor, major = sorted(axes)
    return (minor >= settings.min_axis_px and minor / major >= settings.min_axis_ratio
            and major <= math.hypot(shape[1], shape[0]) * settings.max_axis_diagonal_ratio)


def fit_weighted_ellipse(points, weights, shape, settings: EllipseSettings) -> Ellipse | None:
    """確率比例の5点抽選と、確率支持量×角度被覆で最良仮説を選ぶ。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(points) != len(weights) or not np.isfinite(points).all() or not np.isfinite(weights).all():
        raise ValueError("点群と重みは同じ要素数の有限値にしてください。")
    if len(points) < settings.min_inliers:
        return None
    weights = np.maximum(weights, 0.0)
    if np.count_nonzero(weights) < 5:
        return None
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(settings.random_seed)
    best_ellipse = best_mask = None
    best_score = -math.inf
    for _ in range(settings.iterations):
        indices = rng.choice(len(points), size=5, replace=False, p=probabilities)
        try:
            candidate = cv2.fitEllipseDirect(points[indices].reshape(-1, 1, 2))
        except cv2.error:
            continue
        if not valid_ellipse(candidate, shape, settings):
            continue
        mask = point_distances(points, candidate) <= settings.distance_threshold_px
        if np.count_nonzero(mask) < settings.min_inliers:
            continue
        coverage = angular_coverage(points[mask], candidate, settings.angular_bins)
        # 採用方式は周長補正0。支持量と角度被覆だけで順位を決める。
        score = float(weights[mask].sum()) * coverage
        if score > best_score:
            best_ellipse, best_mask, best_score = candidate, mask, score
    if best_ellipse is None:
        return None

    ellipse, mask = best_ellipse, best_mask
    for _ in range(settings.refine_iterations):
        try:
            refined = cv2.fitEllipse(points[mask].reshape(-1, 1, 2))
        except cv2.error:
            break
        if not valid_ellipse(refined, shape, settings):
            break
        ellipse = refined
        updated = point_distances(points, ellipse) <= settings.distance_threshold_px
        if np.array_equal(updated, mask) or np.count_nonzero(updated) < settings.min_inliers:
            break
        mask = updated
    # 再推定で支持点が不足した場合は、最良の初期仮説を採用する。
    final_mask = point_distances(points, ellipse) <= settings.distance_threshold_px
    return best_ellipse if np.count_nonzero(final_mask) < settings.min_inliers else ellipse


def boundary_ellipse(probability: np.ndarray, settings: EllipseSettings) -> Ellipse | None:
    """画像ごとの乱数を固定し、境界確率から推定楕円を返す。"""
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("境界確率は有限値の2次元配列にしてください。")
    cv2.setRNGSeed(settings.opencv_seed)
    points, weights = skeleton_points(probability, settings.probability_threshold)
    return fit_weighted_ellipse(points, weights, probability.shape, settings)
