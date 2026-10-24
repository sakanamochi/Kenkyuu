"""楕円の変換・評価に使う共通処理。"""

import math

import cv2
import numpy as np


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
