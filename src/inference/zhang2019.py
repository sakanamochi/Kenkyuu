"""幾何比較器：Zhang 2019型（再現実装）とPAF固有の後処理。

論文の処理構成（Canny、勾配象限・凸性による弧選別、隣接象限の幾何制約、
三弧の中心整合、最小二乗楕円、弧上点による適合度検証）を参考にした再現実装。
著者コードの移植ではなく、PAF固有の全周エッジ支持・内包内周選択を追加している。
成果物では必ず「Zhang 2019型（再現実装）」と表記する。

出典：
Limin Zhang, Wang Pan, Xianghua Ma,
"Real-Time Docking Ring Detection Based on the Geometrical Shape for an On-Orbit Spacecraft,"
Sensors, 19(23), 5243, 2019. DOI: 10.3390/s19235243
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import cv2
import numpy as np

from core.geometry import Ellipse

ZHANG_2019_REFERENCE = {
    "label": "Zhang 2019型（再現実装）",
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
        "論文の処理構成を参考にしたPython再現実装。著者コードの移植・完全再現ではない。"
    ),
}

MIN_ARC_POINTS = 8
CENTER_CONSISTENCY_THRESHOLD_PX = 12.0
EDGE_DISTANCE_THRESHOLD_PX = 2.5
MIN_GROUP_FIT_RATIO = 0.55

QUADRANT_NAMES = ("I", "II", "III", "IV")
EXPECTED_CONVEXITY = (1, 1, -1, -1)


def _gradient_quadrants(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
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
    return points[::-1].copy() if points[0, 0] > points[-1, 0] else points


def _convexity(points: np.ndarray) -> int:
    if len(points) < 3:
        return 0
    chord = points[-1] - points[0]
    offsets = points[1:-1] - points[0]
    cross = chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    median = float(np.median(cross)) if len(cross) else 0.0
    return 1 if median > 0 else (-1 if median < 0 else 0)


def _sample_gradient(gradient_x: np.ndarray, gradient_y: np.ndarray, point: np.ndarray) -> np.ndarray:
    x = int(np.clip(round(float(point[0])), 0, gradient_x.shape[1] - 1))
    y = int(np.clip(round(float(point[1])), 0, gradient_x.shape[0] - 1))
    return np.asarray([gradient_x[y, x], gradient_y[y, x]], dtype=np.float64)


def extract_zhang_arcs(
    image: np.ndarray,
    preprocess: dict[str, Any],
    detector: dict[str, Any],
) -> dict[str, Any]:
    """Canny輪郭から象限と凸性の条件を満たす楕円弧を抽出する。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kernel = int(preprocess["blur_kernel_size"])
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), float(preprocess["blur_sigma"]))
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.Canny(blurred, int(preprocess["canny_low"]), int(preprocess["canny_high"]))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    arcs = []
    for contour_index, contour in enumerate(contours):
        points = contour[:, 0, :]
        if len(points) < MIN_ARC_POINTS:
            continue
        quadrants = _gradient_quadrants(gradient_x, gradient_y, points)
        for points_run, quadrant in _split_runs(points, quadrants, MIN_ARC_POINTS):
            points_run = _left_to_right(points_run)
            convexity = _convexity(points_run)
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
                    "left_gradient": _sample_gradient(gradient_x, gradient_y, points_run[0]),
                    "right_gradient": _sample_gradient(gradient_x, gradient_y, points_run[-1]),
                    "contour_index": contour_index,
                    "length": int(len(points_run)),
                }
            )
    arcs.sort(key=lambda arc: arc["length"], reverse=True)
    arcs = arcs[: int(detector["max_arcs"])]
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


def _ellipse_geometry_valid(ellipse: Ellipse, image_shape: tuple[int, ...], settings: dict[str, Any]) -> bool:
    (center_x, center_y), (axis_1, axis_2), _ = ellipse
    height, width = image_shape[:2]
    diagonal = math.hypot(width, height)
    minor, major = min(axis_1, axis_2), max(axis_1, axis_2)
    margin = float(settings["center_margin_ratio"]) * max(width, height)
    return (
        minor >= float(settings["min_axis_px"])
        and major <= float(settings["max_axis_diagonal_ratio"]) * diagonal
        and minor / max(major, 1e-9) >= float(settings["min_axis_ratio"])
        and -margin <= center_x < width + margin
        and -margin <= center_y < height + margin
    )


def _ellipse_samples(ellipse: Ellipse, count: int) -> tuple[np.ndarray, np.ndarray]:
    (center_x, center_y), (axis_1, axis_2), angle = ellipse
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local = np.column_stack((axis_1 * 0.5 * np.cos(theta), axis_2 * 0.5 * np.sin(theta)))
    radians = math.radians(angle)
    rotation = np.array(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.array([center_x, center_y]), theta


def _radial_distance(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    (center_x, center_y), (axis_1, axis_2), angle = ellipse
    centered = points.astype(np.float64) - np.array([center_x, center_y])
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    local_x = centered[:, 0] * cosine + centered[:, 1] * sine
    local_y = -centered[:, 0] * sine + centered[:, 1] * cosine
    normalized_radius = np.sqrt(
        (local_x / max(axis_1 * 0.5, 1e-9)) ** 2 + (local_y / max(axis_2 * 0.5, 1e-9)) ** 2
    )
    center_distance = np.linalg.norm(centered, axis=1)
    distance = np.full(len(points), np.inf, dtype=np.float64)
    valid = normalized_radius > 1e-12
    distance[valid] = center_distance[valid] * np.abs(1.0 - 1.0 / normalized_radius[valid])
    return distance


def _paper_fitness(ellipse: Ellipse, group_points: np.ndarray) -> dict[str, float] | None:
    distances = _radial_distance(group_points, ellipse)
    fit_ratio = float(np.mean(distances < EDGE_DISTANCE_THRESHOLD_PX))
    if fit_ratio <= MIN_GROUP_FIT_RATIO:
        return None
    return {
        "group_fit_ratio": fit_ratio,
        "fitness_score": fit_ratio,
        "mean_arc_distance_px": float(np.mean(distances)),
    }


def _tangent_y(arc: dict[str, Any], side: str, x: float) -> float:
    point, gradient = arc[f"{side}_point"], arc[f"{side}_gradient"]
    return float(point[1] - gradient[0] / gradient[1] * (x - point[0]))


def _tangent_x(arc: dict[str, Any], side: str, y: float) -> float:
    point, gradient = arc[f"{side}_point"], arc[f"{side}_gradient"]
    return float(point[0] - gradient[1] / gradient[0] * (y - point[1]))


def _pair_constraints(first: dict[str, Any], second: dict[str, Any]) -> bool:
    quadrant_pair = (first["quadrant"], second["quadrant"])
    if quadrant_pair not in ((0, 1), (1, 2), (2, 3), (3, 0)):
        return False
    first_left, first_right = first["left_point"], first["right_point"]
    second_left, second_right = second["left_point"], second["right_point"]
    if quadrant_pair == (0, 1):
        position = first_left[0] > second_right[0]
        tangent = second_right[1] < _tangent_y(first, "left", second_right[0]) and first_left[1] < _tangent_y(
            second, "right", first_left[0]
        )
    elif quadrant_pair == (1, 2):
        position = first_left[1] > second_left[1]
        tangent = second_left[0] > _tangent_x(first, "left", second_left[1]) and first_left[0] > _tangent_x(
            second, "left", first_left[1]
        )
    elif quadrant_pair == (2, 3):
        position = first_right[0] < second_left[0]
        tangent = second_left[1] > _tangent_y(first, "right", second_left[0]) and first_right[1] > _tangent_y(
            second, "left", first_right[0]
        )
    else:
        position = first_right[1] < second_right[1]
        tangent = second_right[0] < _tangent_x(first, "right", second_right[1]) and first_right[0] < _tangent_x(
            second, "right", first_right[1]
        )
    return bool(position and tangent)


def _midpoint_line(
    first_points: np.ndarray,
    second_points: np.ndarray,
    chord_direction: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    length = float(np.linalg.norm(chord_direction))
    if length <= 1e-9:
        return None
    direction = chord_direction / length
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    first_proj = first_points @ normal
    second_proj = second_points @ normal
    lower = max(float(np.min(first_proj)), float(np.min(second_proj)))
    upper = min(float(np.max(first_proj)), float(np.max(second_proj)))
    if upper - lower <= 1.0:
        return None

    offsets = np.linspace(lower, upper, sample_count + 2)[1:-1]

    def interpolate(points: np.ndarray, projection: np.ndarray) -> np.ndarray:
        order = np.argsort(projection)
        sorted_proj = projection[order]
        sorted_pts = points[order]
        unique_proj, unique_idx = np.unique(sorted_proj, return_index=True)
        unique_pts = sorted_pts[unique_idx]
        return np.column_stack(
            (np.interp(offsets, unique_proj, unique_pts[:, 0]), np.interp(offsets, unique_proj, unique_pts[:, 1]))
        )

    midpoints = (interpolate(first_points, first_proj) + interpolate(second_points, second_proj)) * 0.5
    if len(midpoints) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(midpoints.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    return np.asarray([x0, y0], dtype=np.float64), np.asarray([vx, vy], dtype=np.float64)


def _line_intersection(
    first_line: tuple[np.ndarray, np.ndarray],
    second_line: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    p1, d1 = first_line
    p2, d2 = second_line
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(float(cross)) <= 1e-6:
        return None
    diff = p2 - p1
    scale = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
    return p1 + scale * d1


def _estimate_pair_center(first: dict[str, Any], second: dict[str, Any], sample_count: int) -> np.ndarray | None:
    first_points = first["points"].astype(np.float64)
    second_points = second["points"].astype(np.float64)
    first_line = _midpoint_line(
        first_points, second_points, second["middle_point"] - first["left_point"], sample_count
    )
    second_endpoint = second["right_point"] if first["quadrant"] in (0, 2) else second["left_point"]
    second_line = _midpoint_line(
        first_points, second_points, second_endpoint - first["middle_point"], sample_count
    )
    if first_line is None or second_line is None:
        return None
    return _line_intersection(first_line, second_line)


def _three_arc_geometry(
    arcs: list[dict[str, Any]],
    group: tuple[int, ...],
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    if len(group) != 3:
        return None
    by_quadrant = {arcs[index]["quadrant"]: arcs[index] for index in group}
    if len(by_quadrant) != 3:
        return None
    adjacent_pairs = [
        (q, (q + 1) % 4) for q in range(4) if q in by_quadrant and (q + 1) % 4 in by_quadrant
    ]
    if len(adjacent_pairs) != 2:
        return None
    centers = []
    for q1, q2 in adjacent_pairs:
        first, second = by_quadrant[q1], by_quadrant[q2]
        if not _pair_constraints(first, second):
            return None
        center = _estimate_pair_center(first, second, int(settings["center_chord_samples"]))
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


def _candidate_groups(arcs: list[dict[str, Any]], settings: dict[str, Any]):
    combinations = []
    for group in itertools.combinations(range(len(arcs)), 3):
        quadrants = {arcs[index]["quadrant"] for index in group}
        if len(quadrants) != 3:
            continue
        lengths = [arcs[index]["length"] for index in group]
        distinct_contours = len({arcs[index]["contour_index"] for index in group})
        priority = (-distinct_contours, min(lengths) / max(lengths), sum(lengths))
        combinations.append((priority, group))
    combinations.sort(reverse=True)
    for _, group in combinations[: int(settings["max_arc_combinations"])]:
        geometry = _three_arc_geometry(arcs, group, settings)
        if geometry is not None:
            yield group, geometry


def _is_duplicate(ellipse: Ellipse, accepted: list[dict[str, Any]], settings: dict[str, Any]) -> bool:
    center = np.asarray(ellipse[0], dtype=np.float64)
    axes = np.sort(np.asarray(ellipse[1], dtype=np.float64))
    for candidate in accepted:
        other = candidate["ellipse"]
        other_axes = np.sort(np.asarray(other[1], dtype=np.float64))
        scale = max(float(np.max(axes)), float(np.max(other_axes)), 1.0)
        center_diff = np.linalg.norm(center - np.asarray(other[0])) / scale
        axes_diff = float(np.max(np.abs(axes - other_axes)) / scale)
        if (
            center_diff <= float(settings["duplicate_center_ratio"])
            and axes_diff <= float(settings["duplicate_axis_ratio"])
        ):
            return True
    return False


def detect_zhang_arc_candidates(
    image: np.ndarray,
    preprocess: dict[str, Any],
    detector: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Zhang 2019型（再現実装）の三弧統合コアによる楕円候補生成。"""
    stages = extract_zhang_arcs(image, preprocess, detector)
    arcs = stages["arcs"]
    if not arcs:
        return [], stages

    candidates = []
    for group, geometry in _candidate_groups(arcs, detector):
        points = np.concatenate([arcs[index]["points"] for index in group], axis=0)
        if len(points) < 5:
            continue
        max_points = int(detector["max_fit_points"])
        fit_points = (
            points[np.linspace(0, len(points) - 1, max_points).astype(int)]
            if len(points) > max_points
            else points
        )
        try:
            ellipse = cv2.fitEllipseDirect(fit_points.reshape(-1, 1, 2))
        except cv2.error:
            continue
        if not _ellipse_geometry_valid(ellipse, image.shape, detector):
            continue
        metrics = _paper_fitness(ellipse, points)
        if metrics is None:
            continue
        candidates.append(
            {
                "ellipse": ellipse,
                "arc_indices": [int(index) for index in group],
                "arc_quadrants": [arcs[index]["quadrant_name"] for index in group],
                "arc_count": len(group),
                "point_count": int(len(points)),
                **geometry,
                **metrics,
            }
        )

    candidates.sort(key=lambda item: item["fitness_score"], reverse=True)
    unique = []
    for candidate in candidates:
        if not _is_duplicate(candidate["ellipse"], unique, detector):
            unique.append(candidate)
        if len(unique) >= int(detector["max_candidates"]):
            break
    return unique, stages


def score_zhang_candidates_for_paf(
    candidates: list[dict[str, Any]],
    stages: dict[str, Any],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """PAF固有後処理：候補楕円の全周エッジ支持率と角度被覆を検証する。"""
    if not candidates:
        return []
    distance_map = cv2.distanceTransform(255 - stages["edges"], cv2.DIST_L2, 3)
    threshold = float(settings["edge_distance_threshold_px"])
    sample_count = int(settings["validation_samples"])
    height, width = distance_map.shape
    scored = []
    for source_candidate in candidates:
        candidate = dict(source_candidate)
        perimeter, _ = _ellipse_samples(candidate["ellipse"], sample_count)
        pixels = np.rint(perimeter).astype(int)
        valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        distances = np.full(sample_count, np.inf, dtype=np.float32)
        distances[valid] = distance_map[pixels[valid, 1], pixels[valid, 0]]
        supported = distances <= threshold
        edge_density = float(np.mean(supported))
        bins = int(settings["angular_bins"])
        coverage = float(
            np.count_nonzero([np.any(supported[indexes]) for indexes in np.array_split(np.arange(sample_count), bins)])
            / bins
        )
        if edge_density < float(settings["min_edge_density"]) or coverage < float(settings["min_angular_coverage"]):
            continue
        finite = distances[np.isfinite(distances)]
        mean_distance = float(np.mean(np.minimum(finite, threshold * 3.0))) if len(finite) else threshold * 3.0
        candidate.update(
            {
                "edge_density": edge_density,
                "angular_coverage": coverage,
                "mean_edge_distance_px": mean_distance,
                "selection_score": edge_density,
            }
        )
        scored.append(candidate)
    scored.sort(key=lambda item: item["edge_density"], reverse=True)
    return scored


def _ellipse_contains(outer: Ellipse, inner: Ellipse, sample_count: int = 180) -> bool:
    points, _ = _ellipse_samples(inner, sample_count)
    (center_x, center_y), (axis_1, axis_2), angle = outer
    half_1 = max(float(axis_1) * 0.5, 1e-9)
    half_2 = max(float(axis_2) * 0.5, 1e-9)
    centered = points - np.asarray([center_x, center_y], dtype=np.float64)
    radians = math.radians(float(angle))
    cosine, sine = math.cos(radians), math.sin(radians)
    local_x = centered[:, 0] * cosine + centered[:, 1] * sine
    local_y = -centered[:, 0] * sine + centered[:, 1] * cosine
    return bool(np.all((local_x / half_1) ** 2 + (local_y / half_2) ** 2 <= 1.0 + 1e-6))


def select_zhang_inner_boundary(
    candidates: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """PAF固有後処理：内包関係があれば最内周を、なければ全周支持率最大を選ぶ。"""
    if not candidates:
        return None
    considered = candidates[: int(settings["max_considered_candidates"])]
    containment_depth = [0] * len(considered)
    for inner_idx, inner in enumerate(considered):
        for outer_idx, outer in enumerate(considered):
            if inner_idx != outer_idx and _ellipse_contains(outer["ellipse"], inner["ellipse"]):
                containment_depth[inner_idx] += 1
    max_depth = max(containment_depth)
    if max_depth > 0:
        selectable = [idx for idx, depth in enumerate(containment_depth) if depth == max_depth]
        selected_idx = max(selectable, key=lambda idx: float(considered[idx]["edge_density"]))
        selection_mode = "contained_inner_boundary"
    else:
        selected_idx = max(range(len(considered)), key=lambda idx: float(considered[idx]["edge_density"]))
        selection_mode = "full_perimeter_support"
    selected = dict(considered[selected_idx])
    selected["selection_mode"] = selection_mode
    selected["containment_depth"] = containment_depth[selected_idx]
    return selected


def detect_zhang2019(
    image: np.ndarray,
    settings: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Zhang 2019型（再現実装）＋PAF固有後処理で内周楕円を検出する。"""
    candidates, stages = detect_zhang_arc_candidates(image, settings["preprocess"], settings["detector"])
    post = settings["paf_postprocess"]
    scored = score_zhang_candidates_for_paf(candidates, stages, post["candidate_validation"])
    selected = select_zhang_inner_boundary(scored, post["selector"])
    return selected, {**stages, "raw_candidates": candidates, "scored_candidates": scored}
