"""内周輪郭の距離と確率支持を測る、学習を伴わない診断指標。"""

import cv2
import numpy as np


def ellipse_points(ellipse, count=720):
    """楕円パラメータ角を等分し、画像座標の輪郭点を返す。"""
    center, axes, angle = ellipse
    t = np.arange(count) * (2 * np.pi / count)
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = np.column_stack((np.cos(t) * axes[0] / 2, np.sin(t) * axes[1] / 2))
    return local @ rotation.T + np.asarray(center)


def nearest_distances(points, contour):
    """輪郭の離散標本への最近傍距離。大きな点群は分割して計算する。"""
    points, contour = np.asarray(points), np.asarray(contour)
    if not len(points):
        return np.empty(0)
    if not len(contour):
        return np.full(len(points), np.inf)
    return np.concatenate([
        np.sqrt(((points[i:i+256, None] - contour[None]) ** 2).sum(axis=2).min(axis=1))
        for i in range(0, len(points), 256)
    ])


def contour_metrics(ellipse, truth):
    """双方向輪郭距離の平均と95百分位を画素および正解長径比で返す。"""
    if ellipse is None:
        return {k: None for k in ('contour_mean_px', 'contour_p95_px', 'contour_p95_ratio')}
    a, b = ellipse_points(ellipse), ellipse_points(truth)
    distances = np.concatenate((nearest_distances(a, b), nearest_distances(b, a)))
    p95 = float(np.percentile(distances, 95))
    return {'contour_mean_px': float(distances.mean()), 'contour_p95_px': p95,
            'contour_p95_ratio': p95 / max(truth[1])}


def probability_support(probability, truth, threshold=0.35, tolerance=2.0):
    """完全内周に対する閾値点の被覆と適合率。可視部だけの指標とは区別する。"""
    y, x = np.nonzero(probability >= threshold)
    points = np.column_stack((x, y)).astype(np.float32)
    contour = ellipse_points(truth)
    distances = nearest_distances(points, contour)
    support = nearest_distances(contour, points) <= tolerance
    values = cv2.remap(probability, contour[:, 0].astype(np.float32)[None],
                       contour[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
    return points, distances, {
        'threshold_point_count': len(points),
        'truth_band_point_count': int((distances <= tolerance).sum()),
        'full_contour_coverage': float(support.mean()),
        'truth_band_precision': float((distances <= tolerance).mean()) if len(points) else 0.0,
        'truth_probability_mean': float(values.mean()),
    }
