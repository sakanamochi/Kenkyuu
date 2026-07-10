import math

import cv2
import numpy as np


def preprocess_image(image: np.ndarray, settings: dict) -> dict:
    """楕円候補抽出前の各画像処理段階を返す。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = int(settings["blur_kernel_size"])
    blurred = cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        float(settings["blur_sigma"]),
    )
    edges = cv2.Canny(
        blurred,
        int(settings["canny_low"]),
        int(settings["canny_high"]),
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "contours": contours,
    }


def detect_candidates(image: np.ndarray, settings: dict) -> list[dict]:
    """画像全体から楕円候補を抽出し、PAF固有の位置・大きさを使わず順位付けする。"""
    stages = preprocess_image(image, settings)
    contours = stages["contours"]

    angular_bins = int(settings["angular_bins"])
    radial_weight = float(settings["radial_error_weight"])
    min_points = int(settings["min_contour_points"])
    candidates = []

    for contour in contours:
        if len(contour) < min_points:
            continue

        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis_1, axis_2), angle = ellipse
        if axis_1 <= 0 or axis_2 <= 0:
            continue

        points = contour[:, 0, :].astype(np.float64)
        centered = points - np.array([cx, cy])
        radians = math.radians(angle)
        cos_angle = math.cos(radians)
        sin_angle = math.sin(radians)
        local_x = centered[:, 0] * cos_angle + centered[:, 1] * sin_angle
        local_y = -centered[:, 0] * sin_angle + centered[:, 1] * cos_angle
        normalized_x = local_x / (axis_1 / 2)
        normalized_y = local_y / (axis_2 / 2)

        radial_error = float(
            np.mean(np.abs(np.sqrt(normalized_x**2 + normalized_y**2) - 1.0))
        )
        point_angles = np.mod(np.arctan2(normalized_y, normalized_x), 2 * np.pi)
        occupied_bins = np.unique(
            (point_angles / (2 * np.pi) * angular_bins).astype(int)
        ).size
        angular_coverage = occupied_bins / angular_bins
        score = angular_coverage / (1.0 + radial_weight * radial_error)

        candidates.append(
            {
                "ellipse": ellipse,
                "contour_points": len(contour),
                "radial_error": radial_error,
                "angular_coverage": angular_coverage,
                "score": score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


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


def candidate_to_dict(candidate: dict, rank: int) -> dict:
    return {
        "rank": rank,
        **ellipse_to_dict(candidate["ellipse"]),
        "score": float(candidate["score"]),
        "radial_error": float(candidate["radial_error"]),
        "angular_coverage": float(candidate["angular_coverage"]),
        "contour_points": int(candidate["contour_points"]),
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


def draw_evaluation(image: np.ndarray, detected, ground_truth) -> np.ndarray:
    output = image.copy()
    cv2.ellipse(output, ground_truth, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.ellipse(output, detected, (0, 0, 255), 1, cv2.LINE_AA)
    return output


def draw_candidates(image: np.ndarray, candidates: list[dict]) -> np.ndarray:
    output = image.copy()
    for rank, candidate in enumerate(candidates, start=1):
        color = (0, 0, 255) if rank == 1 else (255, 160, 0)
        cv2.ellipse(output, candidate["ellipse"], color, 1, cv2.LINE_AA)
    return output
