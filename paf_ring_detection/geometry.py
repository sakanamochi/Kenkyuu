"""楕円の変換・評価に使う共通処理。"""

import math

import cv2
import numpy as np


def ellipse_points(ellipse, count=720):
    """楕円の輪郭をパラメータ角で等間隔に標本化する。"""
    center, axes, angle = ellipse
    t = np.arange(count) * (2 * np.pi / count)
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = np.column_stack((np.cos(t) * axes[0] / 2, np.sin(t) * axes[1] / 2))
    return local @ rotation.T + np.asarray(center)


def contour_metrics(ellipse, truth):
    """各楕円720点間の双方向最近傍距離を従来と同じ定義で返す。"""
    if ellipse is None:
        return {k: None for k in ('contour_mean_px', 'contour_p95_px', 'contour_p95_ratio')}
    a, b = ellipse_points(ellipse), ellipse_points(truth)
    def nearest(points, contour):
        return np.concatenate([
            np.sqrt(((points[i:i+256, None] - contour[None]) ** 2).sum(axis=2).min(axis=1))
            for i in range(0, len(points), 256)])
    distances = np.concatenate((nearest(a, b), nearest(b, a)))
    p95 = float(np.percentile(distances, 95))
    return {'contour_mean_px': float(distances.mean()), 'contour_p95_px': p95,
            'contour_p95_ratio': p95 / max(truth[1])}


def fit_ground_truth(image_points: list[list[float]]):
    points = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) < 5:
        raise ValueError("正解楕円の計算には5点以上必要です")
    return cv2.fitEllipse(points)


def ellipse_to_dict(ellipse) -> dict:
    (cx, cy), (axis_1, axis_2), angle = ellipse
    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "axis_1": float(axis_1),
        "axis_2": float(axis_2),
        "angle_deg": float(angle),
    }


def _major_axis_angle(ellipse) -> float:
    _, (axis_1, axis_2), angle = ellipse
    return angle if axis_1 >= axis_2 else (angle + 90.0) % 180.0


def evaluate_ellipses(detected, ground_truth, image_shape) -> dict:
    (cx, cy), detected_axis_pair, _ = detected
    (gt_cx, gt_cy), ground_truth_axis_pair, _ = ground_truth

    detected_axes = sorted(detected_axis_pair)
    ground_truth_axes = sorted(ground_truth_axis_pair)
    raw_angle_error = abs(_major_axis_angle(detected) - _major_axis_angle(ground_truth))

    height, width = image_shape[:2]
    detected_mask = np.zeros((height, width), dtype=np.uint8)
    ground_truth_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(detected_mask, detected, 255, -1)
    cv2.ellipse(ground_truth_mask, ground_truth, 255, -1)
    intersection = np.count_nonzero(
        (detected_mask > 0) & (ground_truth_mask > 0)
    )
    union = np.count_nonzero((detected_mask > 0) | (ground_truth_mask > 0))

    return {
        "center_error_px": float(math.hypot(cx - gt_cx, cy - gt_cy)),
        "minor_axis_error_px": float(abs(detected_axes[0] - ground_truth_axes[0])),
        "major_axis_error_px": float(abs(detected_axes[1] - ground_truth_axes[1])),
        "angle_error_deg": float(min(raw_angle_error, 180.0 - raw_angle_error)),
        "ellipse_iou": float(intersection / union if union else 0.0),
    }
