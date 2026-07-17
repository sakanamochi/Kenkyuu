"""Kojima et al. (2021) が採用したFornaciari楕円検出法の再現実装。

Fornaciari et al. (2014) の処理構成である、Cannyエッジ、勾配方向と
凸性による4種類の弧分類、3弧の組合せ、楕円方程式による候補検証、
重複候補の統合を再現する。原論文の分解1次元Hough投票は、公開された
著者実装を確認できなかったためOpenCVの直接最小二乗楕円に置換した。

References:
- M. Fornaciari, A. Prati, R. Cucchiara, Pattern Recognition 47 (2014),
  https://doi.org/10.1016/j.patcog.2014.05.012
- Y. Kojima et al., Transactions of the JSME 87 (2021),
  https://doi.org/10.1299/transjsme.21-00089
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np


KOJIMA_2021_FORNACIARI_REFERENCE = {
    "method_label": "Kojima 2021採用法（Fornaciari 2014再現実装）",
    "kojima_reference": {
        "title": "画像情報を用いた宇宙機の軌道上組立における位置・姿勢制御",
        "year": 2021,
        "doi": "10.1299/transjsme.21-00089",
        "url": "https://doi.org/10.1299/transjsme.21-00089",
    },
    "ellipse_detector_reference": {
        "title": "A fast and effective ellipse detector for embedded vision applications",
        "authors": ["Michele Fornaciari", "Andrea Prati", "Rita Cucchiara"],
        "journal": "Pattern Recognition",
        "year": 2014,
        "volume": 47,
        "issue": 11,
        "pages": "3693-3708",
        "doi": "10.1016/j.patcog.2014.05.012",
        "url": "https://doi.org/10.1016/j.patcog.2014.05.012",
    },
    "implementation_relation": (
        "論文の弧分類・3弧組合せ・候補検証・クラスタリングを参考にした再現実装。"
        "原論文の分解1次元Hough投票はOpenCV fitEllipseDirectに置換した"
    ),
    "paf_selection_addition": (
        "複数楕円から同心・相似な小さい側をPAF内周として選ぶ処理は、"
        "本評価のために追加したCAD事前知識でありKojima論文の記載処理ではない"
    ),
}


def _automatic_canny(blurred: np.ndarray, settings: dict) -> tuple[np.ndarray, int, int]:
    """非黒画素の中央値から、背景が黒いCGでも退化しない閾値を求める。"""
    nonzero = blurred[blurred > 0]
    median = float(np.median(nonzero)) if nonzero.size else 0.0
    sigma = float(settings["canny_sigma"])
    low = max(int(settings["canny_low_floor"]), int((1.0 - sigma) * median))
    high = max(int(settings["canny_high_floor"]), int((1.0 + sigma) * median))
    high = max(high, low + 1)
    return cv2.Canny(blurred, low, high), low, high


def _arc_convexity(points: np.ndarray) -> int:
    """弧の端点を結ぶ弦に対して、弧がどちら側に膨らむかを返す。"""
    if len(points) < 3:
        return 0
    chord = points[-1].astype(np.float64) - points[0]
    offsets = points[1:-1].astype(np.float64) - points[0]
    cross = chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    median = float(np.median(cross)) if len(cross) else 0.0
    return 1 if median > 0 else (-1 if median < 0 else 0)


def _quadrant(direction: int, convexity: int) -> int:
    """勾配方向2群と凸性2群を、論文の4種類の弧へ対応付ける。"""
    if direction == 0:
        return 0 if convexity > 0 else 2
    return 1 if convexity > 0 else 3


def extract_fornaciari_arcs(image: np.ndarray, detector: dict, settings: dict) -> dict:
    """同一勾配方向の8近傍連結エッジを、凸性つき弧へ変換する。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = int(detector["blur_kernel_size"])
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), float(detector["blur_sigma"]))
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edges, canny_low, canny_high = _automatic_canny(blurred, settings)

    product = gradient_x * gradient_y
    valid_gradient = (gradient_x != 0) & (gradient_y != 0)
    minimum_points = int(settings["min_arc_points"])
    minimum_short_side = float(settings["min_bounding_box_short_side_px"])
    arcs = []
    for direction, direction_mask in enumerate((product >= 0, product < 0)):
        mask = np.where((edges > 0) & valid_gradient & direction_mask, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour_index, contour in enumerate(contours):
            points = contour[:, 0, :]
            if len(points) < minimum_points:
                continue
            rectangle = cv2.minAreaRect(points.astype(np.float32))
            if min(rectangle[1]) < minimum_short_side:
                continue
            convexity = _arc_convexity(points)
            if convexity == 0:
                continue
            arcs.append(
                {
                    "points": points.astype(np.float32),
                    "direction": direction,
                    "convexity": convexity,
                    "quadrant": _quadrant(direction, convexity),
                    "contour_index": contour_index,
                    "length": int(len(points)),
                }
            )
    arcs.sort(key=lambda arc: arc["length"], reverse=True)
    arcs = arcs[: int(settings["max_arcs"])]
    for index, arc in enumerate(arcs):
        arc["arc_index"] = index
    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "arcs": arcs,
    }


def _ellipse_geometry_valid(ellipse, image_shape, settings: dict) -> bool:
    (center_x, center_y), (axis_1, axis_2), _ = ellipse
    height, width = image_shape[:2]
    diagonal = math.hypot(width, height)
    minor, major = sorted((float(axis_1), float(axis_2)))
    margin = float(settings["center_margin_ratio"]) * max(width, height)
    return (
        minor >= float(settings["min_axis_px"])
        and major <= float(settings["max_axis_diagonal_ratio"]) * diagonal
        and minor / max(major, 1e-9) >= float(settings["min_axis_ratio"])
        and -margin <= center_x < width + margin
        and -margin <= center_y < height + margin
    )


def _ellipse_equation_residual(points: np.ndarray, ellipse) -> np.ndarray:
    """正規化楕円方程式 |x^2/a^2 + y^2/b^2 - 1| を計算する。"""
    (center_x, center_y), (axis_1, axis_2), angle = ellipse
    centered = points.astype(np.float64) - np.array([center_x, center_y])
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    local_x = centered[:, 0] * cosine + centered[:, 1] * sine
    local_y = -centered[:, 0] * sine + centered[:, 1] * cosine
    value = (local_x / max(axis_1 * 0.5, 1e-9)) ** 2 + (
        local_y / max(axis_2 * 0.5, 1e-9)
    ) ** 2
    return np.abs(value - 1.0)


def _candidate_groups(arcs: list[dict], settings: dict):
    """異なる3象限の弧を、長さの釣合いがよい順に列挙する。"""
    combinations = []
    for group in itertools.combinations(range(len(arcs)), 3):
        quadrants = {arcs[index]["quadrant"] for index in group}
        if len(quadrants) != 3:
            continue
        lengths = [arcs[index]["length"] for index in group]
        balance = min(lengths) / max(lengths)
        combinations.append(((balance, sum(lengths)), group))
    combinations.sort(reverse=True)
    for _, group in combinations[: int(settings["max_arc_combinations"])]:
        yield group


def _is_duplicate(ellipse, accepted: list[dict], settings: dict) -> bool:
    center = np.asarray(ellipse[0], dtype=np.float64)
    axes = np.sort(np.asarray(ellipse[1], dtype=np.float64))
    for candidate in accepted:
        other = candidate["ellipse"]
        other_center = np.asarray(other[0], dtype=np.float64)
        other_axes = np.sort(np.asarray(other[1], dtype=np.float64))
        scale = max(float(axes[1]), float(other_axes[1]), 1.0)
        if (
            np.linalg.norm(center - other_center) / scale
            <= float(settings["duplicate_center_ratio"])
            and np.max(np.abs(axes - other_axes)) / scale
            <= float(settings["duplicate_axis_ratio"])
        ):
            return True
    return False


def detect_fornaciari_candidates(image: np.ndarray, detector: dict, settings: dict):
    """Fornaciari型の3弧候補を生成し、楕円方程式適合率で順位付けする。"""
    stages = extract_fornaciari_arcs(image, detector, settings)
    arcs = stages["arcs"]
    candidates = []
    for group in _candidate_groups(arcs, settings):
        points = np.concatenate([arcs[index]["points"] for index in group], axis=0)
        maximum_points = int(settings["max_fit_points"])
        fit_points = points
        if len(points) > maximum_points:
            indexes = np.linspace(0, len(points) - 1, maximum_points).astype(int)
            fit_points = points[indexes]
        try:
            ellipse = cv2.fitEllipseDirect(fit_points.reshape(-1, 1, 2))
        except cv2.error:
            continue
        if not _ellipse_geometry_valid(ellipse, image.shape, settings):
            continue
        residuals = _ellipse_equation_residual(points, ellipse)
        fit_ratio = float(np.mean(residuals < float(settings["equation_tolerance"])))
        if fit_ratio < float(settings["min_score"]):
            continue
        mean_residual = float(np.mean(np.minimum(residuals, 1.0)))
        candidates.append(
            {
                "ellipse": ellipse,
                "arc_indices": [int(index) for index in group],
                "arc_count": 3,
                "point_count": int(len(points)),
                "equation_fit_ratio": fit_ratio,
                "mean_equation_residual": mean_residual,
                "selection_score": fit_ratio / (1.0 + mean_residual),
            }
        )
    candidates.sort(key=lambda candidate: candidate["selection_score"], reverse=True)
    unique = []
    for candidate in candidates:
        if not _is_duplicate(candidate["ellipse"], unique, settings):
            unique.append(candidate)
        if len(unique) >= int(settings["max_candidates"]):
            break
    return unique, stages


def draw_fornaciari_arcs(image: np.ndarray, stages: dict, candidates: list[dict]) -> np.ndarray:
    visualization = image.copy()
    colors = ((255, 90, 40), (50, 210, 80), (30, 150, 255), (210, 80, 220))
    for arc in stages["arcs"]:
        points = np.rint(arc["points"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            visualization, [points], False, colors[arc["quadrant"]], 1, cv2.LINE_AA
        )
    for rank, candidate in enumerate(candidates[:5], start=1):
        cv2.ellipse(
            visualization,
            candidate["ellipse"],
            (0, 0, 255) if rank == 1 else (0, 200, 255),
            2 if rank == 1 else 1,
            cv2.LINE_AA,
        )
    return visualization
