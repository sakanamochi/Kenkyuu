"""Zhang et al. (2019)を参考にした弧統合検出器。

論文の処理構成（Canny、勾配象限と凸性による弧選別、隣接象限の
幾何制約、三弧の中心整合、最小二乗楕円、弧上点による適合度検証）を
再現可能なPython実装にしたもの。著者コードの移植ではないため、
成果物では必ず「Zhang 2019型（再現実装）」と表記する。

Reference:
L. Zhang, W. Pan, X. Ma, "Real-Time Docking Ring Detection Based on the
Geometrical Shape for an On-Orbit Spacecraft", Sensors 19(23), 5243, 2019.
https://doi.org/10.3390/s19235243
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np


ZHANG_2019_REFERENCE = {
    "title": (
        "Real-Time Docking Ring Detection Based on the Geometrical Shape "
        "for an On-Orbit Spacecraft"
    ),
    "authors": ["Limin Zhang", "Wang Pan", "Xianghua Ma"],
    "journal": "Sensors",
    "year": 2019,
    "volume": 19,
    "issue": 23,
    "article": 5243,
    "doi": "10.3390/s19235243",
    "url": "https://doi.org/10.3390/s19235243",
    "implementation_relation": (
        "論文の処理構成を参考にした再現実装。著者コードの移植・完全再現ではない"
    ),
}

# Zhang et al. (2019)は、以下の閾値が必要であることと記号だけを示しており、
# 具体的な数値は掲載していない。中心間距離についても「所定距離内」とだけ
# 記述され、記号・数値ともに示されていない。
# したがって、以下は論文値ではなく本再現実装で採用した経験的な設定値である。
# 値を変更して比較する場合は、このブロックだけを編集する。
# T_p相当: これ未満の画素数しかない短い円弧を除外する。
MIN_ARC_POINTS = 8

# 論文中で数値未記載: 二組の円弧から推定した中心間の許容距離[pixel]。
CENTER_CONSISTENCY_THRESHOLD_PX = 12.0

# T_d相当: 弧点を候補楕円上の適合点と数える距離[pixel]。
EDGE_DISTANCE_THRESHOLD_PX = 2.5

# T_s相当: 三円弧上の全点に占める適合点の割合の採用閾値。
MIN_GROUP_FIT_RATIO = 0.55

# 論文の象限番号を0始まりで表す: I, II, III, IV。
QUADRANT_NAMES = ("I", "II", "III", "IV")
EXPECTED_CONVEXITY = (1, 1, -1, -1)


def _gradient_quadrants(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """式(1)どおりに勾配を象限I～IVへ分類する。"""
    x = np.clip(points[:, 0], 0, gradient_x.shape[1] - 1)
    y = np.clip(points[:, 1], 0, gradient_x.shape[0] - 1)
    dx = gradient_x[y, x]
    dy = gradient_y[y, x]
    labels = np.full(len(points), -1, dtype=np.int8)
    labels[(dx > 0) & (dy > 0)] = 0
    labels[(dx < 0) & (dy > 0)] = 1
    labels[(dx < 0) & (dy < 0)] = 2
    labels[(dx > 0) & (dy < 0)] = 3
    return labels


def _split_runs(
    points: np.ndarray,
    labels: np.ndarray,
    minimum: int,
) -> list[tuple[np.ndarray, int]]:
    """順序付き輪郭を勾配象限が一定な連続弧へ分割する。"""
    if len(points) == 0:
        return []
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    runs = np.split(np.arange(len(points)), boundaries)
    if len(runs) > 1 and labels[runs[0][0]] == labels[runs[-1][0]]:
        runs = [np.concatenate((runs[-1], runs[0]))] + runs[1:-1]
    return [
        (points[indexes].astype(np.float32), int(labels[indexes[0]]))
        for indexes in runs
        if len(indexes) >= minimum and labels[indexes[0]] >= 0
    ]


def _left_to_right(points: np.ndarray) -> np.ndarray:
    """論文の左端点P^Lから右端点P^Rへ弧の順序をそろえる。"""
    if points[0, 0] > points[-1, 0]:
        return points[::-1].copy()
    return points


def _convexity(points: np.ndarray) -> int:
    """式(2)の外積符号から凸性を求める。"""
    if len(points) < 3:
        return 0
    chord = points[-1] - points[0]
    offsets = points[1:-1] - points[0]
    cross = chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    median = float(np.median(cross)) if len(cross) else 0.0
    return 1 if median > 0 else (-1 if median < 0 else 0)


def _sample_gradient(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    x = int(np.clip(round(float(point[0])), 0, gradient_x.shape[1] - 1))
    y = int(np.clip(round(float(point[1])), 0, gradient_x.shape[0] - 1))
    return np.asarray([gradient_x[y, x], gradient_y[y, x]], dtype=np.float64)


def extract_zhang_arcs(image: np.ndarray, detector: dict, settings: dict) -> dict:
    """Canny輪郭から式(3)を満たす象限別の楕円弧を抽出する。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = int(detector["blur_kernel_size"])
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), float(detector["blur_sigma"]))
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.Canny(
        blurred,
        int(detector["canny_low"]),
        int(detector["canny_high"]),
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    minimum = MIN_ARC_POINTS
    arcs = []
    for contour_index, contour in enumerate(contours):
        points = contour[:, 0, :]
        if len(points) < minimum:
            continue
        quadrants = _gradient_quadrants(gradient_x, gradient_y, points)
        for points_run, quadrant in _split_runs(points, quadrants, minimum):
            points_run = _left_to_right(points_run)
            convexity = _convexity(points_run)
            # 式(3): I・IIは正、III・IVは負の凸性だけを残す。
            if convexity != EXPECTED_CONVEXITY[quadrant]:
                continue
            middle_index = len(points_run) // 2
            arcs.append(
                {
                    "points": points_run,
                    "quadrant": quadrant,
                    "quadrant_name": QUADRANT_NAMES[quadrant],
                    "convexity": convexity,
                    "left_point": points_run[0].astype(np.float64),
                    "middle_point": points_run[middle_index].astype(np.float64),
                    "right_point": points_run[-1].astype(np.float64),
                    "left_gradient": _sample_gradient(
                        gradient_x, gradient_y, points_run[0]
                    ),
                    "right_gradient": _sample_gradient(
                        gradient_x, gradient_y, points_run[-1]
                    ),
                    "contour_index": contour_index,
                    "length": int(len(points_run)),
                }
            )
    arcs.sort(key=lambda arc: arc["length"], reverse=True)
    arcs = arcs[: int(settings["max_arcs"])]
    for index, arc in enumerate(arcs):
        arc["arc_index"] = index
    return {
        "gray": gray,
        "blurred": blurred,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "edges": edges,
        "arcs": arcs,
    }


def _ellipse_geometry_valid(ellipse, image_shape, settings: dict) -> bool:
    (center_x, center_y), (axis_1, axis_2), _ = ellipse
    height, width = image_shape[:2]
    diagonal = math.hypot(width, height)
    minor = min(axis_1, axis_2)
    major = max(axis_1, axis_2)
    margin = float(settings["center_margin_ratio"]) * max(width, height)
    return (
        minor >= float(settings["min_axis_px"])
        and major <= float(settings["max_axis_diagonal_ratio"]) * diagonal
        and minor / max(major, 1e-9) >= float(settings["min_axis_ratio"])
        and -margin <= center_x < width + margin
        and -margin <= center_y < height + margin
    )


def _ellipse_samples(ellipse, count: int) -> tuple[np.ndarray, np.ndarray]:
    (center_x, center_y), (axis_1, axis_2), angle = ellipse
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local = np.column_stack(
        (axis_1 * 0.5 * np.cos(theta), axis_2 * 0.5 * np.sin(theta))
    )
    radians = math.radians(angle)
    rotation = np.array(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float64,
    )
    points = local @ rotation.T + np.array([center_x, center_y])
    return points, theta


def _radial_distance(points: np.ndarray, ellipse) -> np.ndarray:
    """式(10)の中心光線と楕円の交点までの絶対距離を求める。"""
    (center_x, center_y), (axis_1, axis_2), angle = ellipse
    centered = points.astype(np.float64) - np.array([center_x, center_y])
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    local_x = centered[:, 0] * cosine + centered[:, 1] * sine
    local_y = -centered[:, 0] * sine + centered[:, 1] * cosine
    normalized_radius = np.sqrt(
        (local_x / max(axis_1 * 0.5, 1e-9)) ** 2
        + (local_y / max(axis_2 * 0.5, 1e-9)) ** 2
    )
    center_distance = np.linalg.norm(centered, axis=1)
    distance = np.full(len(points), np.inf, dtype=np.float64)
    valid = normalized_radius > 1e-12
    # 同じ中心光線上の楕円交点は、中心から1/rho倍した位置にある。
    distance[valid] = center_distance[valid] * np.abs(
        1.0 - 1.0 / normalized_radius[valid]
    )
    return distance


def _paper_fitness(ellipse, group_points: np.ndarray) -> dict | None:
    """式(12)の、三弧上の全点に対する適合点率を計算する。"""
    distances = _radial_distance(group_points, ellipse)
    fit_ratio = float(np.mean(distances < EDGE_DISTANCE_THRESHOLD_PX))
    if fit_ratio <= MIN_GROUP_FIT_RATIO:
        return None
    return {
        "group_fit_ratio": fit_ratio,
        "fitness_score": fit_ratio,
        "mean_arc_distance_px": float(np.mean(distances)),
    }


def _endpoint(arc: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    return arc[f"{side}_point"], arc[f"{side}_gradient"]


def _tangent_y(arc: dict, side: str, x: float) -> float:
    """端点の勾配を法線として接線y=f(x)を評価する。"""
    point, gradient = _endpoint(arc, side)
    return float(point[1] - gradient[0] / gradient[1] * (x - point[0]))


def _tangent_x(arc: dict, side: str, y: float) -> float:
    """端点の勾配を法線として逆接線x=g(y)を評価する。"""
    point, gradient = _endpoint(arc, side)
    return float(point[0] - gradient[1] / gradient[0] * (y - point[1]))


def _pair_constraints(first: dict, second: dict) -> bool:
    """隣接象限の二弧について論文の式(4)～(6)を検査する。"""
    quadrant_pair = (first["quadrant"], second["quadrant"])
    if quadrant_pair not in ((0, 1), (1, 2), (2, 3), (3, 0)):
        return False

    first_left = first["left_point"]
    first_right = first["right_point"]
    second_left = second["left_point"]
    second_right = second["right_point"]
    if quadrant_pair == (0, 1):
        position = first_left[0] > second_right[0]
        tangent = (
            second_right[1] < _tangent_y(first, "left", second_right[0])
            and first_left[1] < _tangent_y(second, "right", first_left[0])
        )
    elif quadrant_pair == (1, 2):
        position = first_left[1] > second_left[1]
        tangent = (
            second_left[0] > _tangent_x(first, "left", second_left[1])
            and first_left[0] > _tangent_x(second, "left", first_left[1])
        )
    elif quadrant_pair == (2, 3):
        position = first_right[0] < second_left[0]
        tangent = (
            second_left[1] > _tangent_y(first, "right", second_left[0])
            and first_right[1] > _tangent_y(second, "left", first_right[0])
        )
    else:
        position = first_right[1] < second_right[1]
        tangent = (
            second_right[0] < _tangent_x(first, "right", second_right[1])
            and first_right[0] < _tangent_x(second, "right", first_right[1])
        )
    return bool(position and tangent)


def _midpoint_line(
    first_points: np.ndarray,
    second_points: np.ndarray,
    chord_direction: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """平行弦の中点群を作り、その中点直線を推定する。"""
    length = float(np.linalg.norm(chord_direction))
    if length <= 1e-9:
        return None
    direction = chord_direction / length
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    first_projection = first_points @ normal
    second_projection = second_points @ normal
    lower = max(float(np.min(first_projection)), float(np.min(second_projection)))
    upper = min(float(np.max(first_projection)), float(np.max(second_projection)))
    if upper - lower <= 1.0:
        return None

    offsets = np.linspace(lower, upper, sample_count + 2)[1:-1]

    def interpolate(points: np.ndarray, projection: np.ndarray) -> np.ndarray:
        order = np.argsort(projection)
        sorted_projection = projection[order]
        sorted_points = points[order]
        unique_projection, unique_index = np.unique(
            sorted_projection, return_index=True
        )
        unique_points = sorted_points[unique_index]
        x = np.interp(offsets, unique_projection, unique_points[:, 0])
        y = np.interp(offsets, unique_projection, unique_points[:, 1])
        return np.column_stack((x, y))

    first_intersections = interpolate(first_points, first_projection)
    second_intersections = interpolate(second_points, second_projection)
    midpoints = (first_intersections + second_intersections) * 0.5
    if len(midpoints) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(
        midpoints.astype(np.float32),
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    return (
        np.asarray([x0, y0], dtype=np.float64),
        np.asarray([vx, vy], dtype=np.float64),
    )


def _line_intersection(
    first_line: tuple[np.ndarray, np.ndarray],
    second_line: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    first_point, first_direction = first_line
    second_point, second_direction = second_line
    cross = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(float(cross)) <= 1e-6:
        return None
    difference = second_point - first_point
    scale = (
        difference[0] * second_direction[1]
        - difference[1] * second_direction[0]
    ) / cross
    return first_point + scale * first_direction


def _estimate_pair_center(
    first: dict,
    second: dict,
    sample_count: int,
) -> np.ndarray | None:
    """Figure 4の平行弦中点直線2本から二弧の楕円中心を求める。"""
    first_points = first["points"].astype(np.float64)
    second_points = second["points"].astype(np.float64)
    first_line = _midpoint_line(
        first_points,
        second_points,
        second["middle_point"] - first["left_point"],
        sample_count,
    )
    # 上下に並ぶ象限対では左端側、左右に並ぶ象限対では右端側を使い、
    # 1本目と異なる方向の平行弦群を作る。
    second_endpoint = (
        second["right_point"]
        if first["quadrant"] in (0, 2)
        else second["left_point"]
    )
    second_line = _midpoint_line(
        first_points,
        second_points,
        second_endpoint - first["middle_point"],
        sample_count,
    )
    if first_line is None or second_line is None:
        return None
    return _line_intersection(first_line, second_line)


def _three_arc_geometry(
    arcs: list[dict],
    group: tuple[int, ...],
    settings: dict,
) -> dict | None:
    """三弧の隣接ペア制約と、二つの推定中心の整合性を検査する。"""
    if len(group) != 3:
        return None
    by_quadrant = {arcs[index]["quadrant"]: arcs[index] for index in group}
    if len(by_quadrant) != 3:
        return None
    adjacent_pairs = [
        (quadrant, (quadrant + 1) % 4)
        for quadrant in range(4)
        if quadrant in by_quadrant and (quadrant + 1) % 4 in by_quadrant
    ]
    if len(adjacent_pairs) != 2:
        return None

    centers = []
    for first_quadrant, second_quadrant in adjacent_pairs:
        first = by_quadrant[first_quadrant]
        second = by_quadrant[second_quadrant]
        if not _pair_constraints(first, second):
            return None
        center = _estimate_pair_center(
            first,
            second,
            int(settings["center_chord_samples"]),
        )
        if center is None or not np.all(np.isfinite(center)):
            return None
        centers.append(center)

    difference = float(np.linalg.norm(centers[0] - centers[1]))
    if difference > CENTER_CONSISTENCY_THRESHOLD_PX:
        return None
    return {
        "pair_centers": [center.tolist() for center in centers],
        "pair_center_difference_px": difference,
    }


def _candidate_groups(arcs: list[dict], settings: dict):
    """計算量上位の組合せから、論文の三弧選択規則を満たすものを返す。"""
    combinations = []
    for group in itertools.combinations(range(len(arcs)), 3):
        quadrants = {arcs[index]["quadrant"] for index in group}
        if len(quadrants) != 3:
            continue
        lengths = [arcs[index]["length"] for index in group]
        point_count = sum(lengths)
        distinct_contours = len(
            {arcs[index]["contour_index"] for index in group}
        )
        length_balance = min(lengths) / max(lengths)
        priority = (-distinct_contours, length_balance, point_count)
        combinations.append((priority, group))
    combinations.sort(reverse=True)
    for _, group in combinations[: int(settings["max_arc_combinations"])]:
        geometry = _three_arc_geometry(arcs, group, settings)
        if geometry is not None:
            yield group, geometry


def _is_duplicate(ellipse, accepted: list[dict], settings: dict) -> bool:
    center = np.asarray(ellipse[0], dtype=np.float64)
    axes = np.sort(np.asarray(ellipse[1], dtype=np.float64))
    for candidate in accepted:
        other = candidate["ellipse"]
        other_axes = np.sort(np.asarray(other[1], dtype=np.float64))
        scale = max(float(np.max(axes)), float(np.max(other_axes)), 1.0)
        center_difference = np.linalg.norm(center - np.asarray(other[0])) / scale
        axes_difference = float(np.max(np.abs(axes - other_axes)) / scale)
        if (
            center_difference <= float(settings["duplicate_center_ratio"])
            and axes_difference <= float(settings["duplicate_axis_ratio"])
        ):
            return True
    return False


def detect_zhang_arc_candidates(
    image: np.ndarray,
    detector: dict,
    settings: dict,
):
    """論文コアの三弧統合と適合度検証までを実行する。"""
    stages = extract_zhang_arcs(image, detector, settings)
    arcs = stages["arcs"]
    if not arcs:
        return [], stages
    candidates = []
    for group, geometry in _candidate_groups(arcs, settings):
        points = np.concatenate(
            [arcs[index]["points"] for index in group],
            axis=0,
        )
        if len(points) < 5:
            continue
        maximum_points = int(settings["max_fit_points"])
        if len(points) > maximum_points:
            indexes = np.linspace(0, len(points) - 1, maximum_points).astype(int)
            fit_points = points[indexes]
        else:
            fit_points = points
        try:
            ellipse = cv2.fitEllipseDirect(fit_points.reshape(-1, 1, 2))
        except cv2.error:
            continue
        if not _ellipse_geometry_valid(ellipse, image.shape, settings):
            continue
        metrics = _paper_fitness(ellipse, points)
        if metrics is None:
            continue
        candidates.append(
            {
                "ellipse": ellipse,
                "arc_indices": [int(index) for index in group],
                "arc_quadrants": [
                    arcs[index]["quadrant_name"] for index in group
                ],
                "arc_count": len(group),
                "point_count": int(len(points)),
                **geometry,
                **metrics,
            }
        )

    candidates.sort(
        key=lambda candidate: candidate["fitness_score"],
        reverse=True,
    )
    unique = []
    for candidate in candidates:
        if not _is_duplicate(candidate["ellipse"], unique, settings):
            unique.append(candidate)
        if len(unique) >= int(settings["max_candidates"]):
            break
    return unique, stages
