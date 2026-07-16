"""Zhang et al. (2019) を参考にしたドッキングリング用の弧統合検出器。

論文の処理順（Canny、勾配方向による弧分類、複数弧の統合、直接最小
二乗楕円、実エッジによる検証）を再現可能なPython実装にしたもの。
著者実装の逐語的移植ではないため、結果では ``zhang2019_arc_reproduction``
と明記する。

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
        "論文の処理順を参考にした再現実装。著者コードの移植・完全再現ではない"
    ),
}


def _gradient_quadrants(gradient_x: np.ndarray, gradient_y: np.ndarray, points: np.ndarray):
    x = np.clip(points[:, 0], 0, gradient_x.shape[1] - 1)
    y = np.clip(points[:, 1], 0, gradient_x.shape[0] - 1)
    positive_x = gradient_x[y, x] >= 0
    positive_y = gradient_y[y, x] >= 0
    return positive_x.astype(np.int8) + positive_y.astype(np.int8) * 2


def _split_runs(points: np.ndarray, labels: np.ndarray, minimum: int) -> list[tuple[np.ndarray, int]]:
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
        if len(indexes) >= minimum
    ]


def _convexity(points: np.ndarray) -> int:
    if len(points) < 3:
        return 0
    chord = points[-1] - points[0]
    offsets = points[1:-1] - points[0]
    cross = chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    median = float(np.median(cross)) if len(cross) else 0.0
    return 1 if median > 0 else (-1 if median < 0 else 0)


def extract_zhang_arcs(image: np.ndarray, detector: dict, settings: dict) -> dict:
    """Canny輪郭を勾配方向と凸性つきの弧へ変換する。"""
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
    minimum = int(settings["min_arc_points"])
    arcs = []
    for contour_index, contour in enumerate(contours):
        points = contour[:, 0, :]
        if len(points) < minimum:
            continue
        quadrants = _gradient_quadrants(gradient_x, gradient_y, points)
        for points_run, quadrant in _split_runs(points, quadrants, minimum):
            arcs.append(
                {
                    "points": points_run,
                    "quadrant": quadrant,
                    "convexity": _convexity(points_run),
                    "contour_index": contour_index,
                    "length": int(len(points_run)),
                }
            )
    # Zhang法は凸性で楕円弧を選別する。ほぼ直線（convexity=0）を先に除くことで、
    # 長い機体エッジや画像端が候補枠を占有するのを防ぐ。
    arcs = [arc for arc in arcs if arc["convexity"] != 0]
    arcs.sort(key=lambda arc: arc["length"], reverse=True)
    arcs = arcs[: int(settings["max_arcs"])]
    for index, arc in enumerate(arcs):
        arc["arc_index"] = index
    return {
        "gray": gray,
        "blurred": blurred,
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
    local = np.column_stack((axis_1 * 0.5 * np.cos(theta), axis_2 * 0.5 * np.sin(theta)))
    radians = math.radians(angle)
    rotation = np.array(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=np.float64,
    )
    points = local @ rotation.T + np.array([center_x, center_y])
    return points, theta


def _radial_distance(points: np.ndarray, ellipse) -> np.ndarray:
    (center_x, center_y), (axis_1, axis_2), angle = ellipse
    centered = points.astype(np.float64) - np.array([center_x, center_y])
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    local_x = centered[:, 0] * cosine + centered[:, 1] * sine
    local_y = -centered[:, 0] * sine + centered[:, 1] * cosine
    radius = np.sqrt((local_x / (axis_1 * 0.5)) ** 2 + (local_y / (axis_2 * 0.5)) ** 2)
    return np.abs(radius - 1.0) * min(axis_1, axis_2) * 0.5


def _polarity_score(gray: np.ndarray, ellipse, sample_count: int, offset_ratio: float) -> float:
    """内側が暗く外側が明るいというZhang法の反射特性を軟らかい得点にする。"""
    boundary, _ = _ellipse_samples(ellipse, sample_count)
    center = np.asarray(ellipse[0], dtype=np.float64)
    inner = center + (boundary - center) * (1.0 - offset_ratio)
    outer = center + (boundary - center) * (1.0 + offset_ratio)
    height, width = gray.shape
    inner_xy = np.rint(inner).astype(int)
    outer_xy = np.rint(outer).astype(int)
    valid = (
        (inner_xy[:, 0] >= 0)
        & (inner_xy[:, 0] < width)
        & (inner_xy[:, 1] >= 0)
        & (inner_xy[:, 1] < height)
        & (outer_xy[:, 0] >= 0)
        & (outer_xy[:, 0] < width)
        & (outer_xy[:, 1] >= 0)
        & (outer_xy[:, 1] < height)
    )
    if not np.any(valid):
        return 0.5
    difference = (
        gray[outer_xy[valid, 1], outer_xy[valid, 0]].astype(np.float32)
        - gray[inner_xy[valid, 1], inner_xy[valid, 0]].astype(np.float32)
    )
    return float(np.mean(1.0 / (1.0 + np.exp(-difference / 12.0))))


def _score_candidate(
    ellipse,
    group_points: np.ndarray,
    distance_map: np.ndarray,
    gray: np.ndarray,
    settings: dict,
) -> dict | None:
    threshold = float(settings["edge_distance_threshold_px"])
    group_distances = _radial_distance(group_points, ellipse)
    group_fit_ratio = float(np.mean(group_distances <= threshold))
    if group_fit_ratio < float(settings["min_group_fit_ratio"]):
        return None

    sample_count = int(settings["validation_samples"])
    perimeter, _ = _ellipse_samples(ellipse, sample_count)
    pixels = np.rint(perimeter).astype(int)
    height, width = distance_map.shape
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    distances = np.full(sample_count, np.inf, dtype=np.float32)
    distances[valid] = distance_map[pixels[valid, 1], pixels[valid, 0]]
    supported = distances <= threshold
    edge_density = float(np.mean(supported))
    bins = int(settings["angular_bins"])
    coverage = float(
        np.count_nonzero(
            [np.any(supported[indexes]) for indexes in np.array_split(np.arange(sample_count), bins)]
        )
        / bins
    )
    if edge_density < float(settings["min_edge_density"]):
        return None
    if coverage < float(settings["min_angular_coverage"]):
        return None

    finite = distances[np.isfinite(distances)]
    mean_distance = float(np.mean(np.minimum(finite, threshold * 3.0))) if len(finite) else threshold * 3.0
    polarity = _polarity_score(
        gray,
        ellipse,
        int(settings["polarity_samples"]),
        float(settings["polarity_offset_ratio"]),
    )
    polarity_weight = float(settings["polarity_weight"])
    selection_score = (
        edge_density
        * coverage
        * math.sqrt(max(group_fit_ratio, 1e-9))
        * ((1.0 - polarity_weight) + polarity_weight * polarity)
        / (1.0 + mean_distance)
    )
    return {
        "edge_density": edge_density,
        "angular_coverage": coverage,
        "group_fit_ratio": group_fit_ratio,
        "mean_edge_distance_px": mean_distance,
        "polarity_score": polarity,
        "selection_score": selection_score,
    }


def _candidate_groups(arcs: list[dict], settings: dict):
    """異なる勾配象限の3弧を優先し、計算量を設定値で固定する。"""
    combinations = []
    for size in tuple(int(value) for value in settings["group_sizes"]):
        for group in itertools.combinations(range(len(arcs)), size):
            quadrants = {arcs[index]["quadrant"] for index in group}
            if len(quadrants) < int(settings["min_distinct_quadrants"]):
                continue
            lengths = [arcs[index]["length"] for index in group]
            point_count = sum(lengths)
            distinct_contours = len({arcs[index]["contour_index"] for index in group})
            # 同一楕円を象限で分けた弧は長さの桁が近いことが多い。長い背景輪郭
            # だけを優先せず、各象限から釣り合った弧を組み合わせる。
            length_balance = min(lengths) / max(lengths)
            # 1本の連結輪郭を勾配象限で分割した弧を最優先する。これは論文の
            # 8近傍連結→象限分類に対応し、無関係な輪郭を混ぜる組合せを抑える。
            priority = (
                len(quadrants),
                -distinct_contours,
                length_balance,
                point_count,
            )
            combinations.append((priority, group))
    combinations.sort(reverse=True)
    for _, group in combinations[: int(settings["max_arc_combinations"])]:
        yield group


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


def detect_zhang_arc_candidates(image: np.ndarray, detector: dict, settings: dict):
    stages = extract_zhang_arcs(image, detector, settings)
    arcs = stages["arcs"]
    if not arcs:
        return [], stages
    distance_map = cv2.distanceTransform(255 - stages["edges"], cv2.DIST_L2, 3)
    candidates = []
    for group in _candidate_groups(arcs, settings):
        points = np.concatenate([arcs[index]["points"] for index in group], axis=0)
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
        metrics = _score_candidate(
            ellipse,
            points,
            distance_map,
            stages["gray"],
            settings,
        )
        if metrics is None:
            continue
        candidate = {
            "ellipse": ellipse,
            "arc_indices": [int(index) for index in group],
            "arc_count": len(group),
            "point_count": int(len(points)),
            **metrics,
        }
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate["selection_score"], reverse=True)
    unique = []
    for candidate in candidates:
        if not _is_duplicate(candidate["ellipse"], unique, settings):
            unique.append(candidate)
        if len(unique) >= int(settings["max_candidates"]):
            break
    return unique, stages


def draw_zhang_arcs(image: np.ndarray, stages: dict, candidates: list[dict]) -> np.ndarray:
    visualization = image.copy()
    colors = ((255, 90, 40), (50, 210, 80), (30, 150, 255), (210, 80, 220))
    for arc in stages["arcs"]:
        color = colors[int(arc["quadrant"]) % len(colors)]
        points = np.rint(arc["points"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(visualization, [points], False, color, 1, cv2.LINE_AA)
    for rank, candidate in enumerate(candidates[:5], start=1):
        cv2.ellipse(
            visualization,
            candidate["ellipse"],
            (0, 0, 255) if rank == 1 else (0, 200, 255),
            2 if rank == 1 else 1,
            cv2.LINE_AA,
        )
    return visualization


def select_zhang_inner_boundary(candidates: list[dict], settings: dict) -> dict | None:
    """同心に近い輪郭階層から内側を選ぶ、ドッキングリング対象の最終段。

    一律に小さい楕円を優遇せず、中心・軸比が一致する複数候補が実際に存在する
    場合だけ、その連結成分の内側候補を選ぶ。対応する階層がなければ純粋な
    エッジ支持順位へ戻る。
    """
    if not candidates:
        return None
    considered = candidates[: int(settings["max_considered_candidates"])]
    links = {index: set() for index in range(len(considered))}
    for first_index, first in enumerate(considered):
        first_axes = np.sort(np.asarray(first["ellipse"][1], dtype=np.float64))
        first_center = np.asarray(first["ellipse"][0], dtype=np.float64)
        first_area = float(np.prod(first_axes))
        for second_index in range(first_index + 1, len(considered)):
            second = considered[second_index]
            second_axes = np.sort(np.asarray(second["ellipse"][1], dtype=np.float64))
            second_center = np.asarray(second["ellipse"][0], dtype=np.float64)
            second_area = float(np.prod(second_axes))
            larger_major = max(first_axes[1], second_axes[1], 1.0)
            center_ratio = float(np.linalg.norm(first_center - second_center) / larger_major)
            axis_ratio_difference = abs(
                float(first_axes[0] / first_axes[1] - second_axes[0] / second_axes[1])
            )
            area_ratio = max(first_area, second_area) / max(min(first_area, second_area), 1e-9)
            quality_ratio = min(first["selection_score"], second["selection_score"]) / max(
                first["selection_score"], second["selection_score"], 1e-9
            )
            if (
                center_ratio <= float(settings["center_distance_major_ratio"])
                and axis_ratio_difference <= float(settings["axis_ratio_difference"])
                and float(settings["area_ratio_min"]) <= area_ratio <= float(settings["area_ratio_max"])
                and quality_ratio >= float(settings["min_quality_ratio"])
            ):
                links[first_index].add(second_index)
                links[second_index].add(first_index)

    best_pair = None
    for first_index, neighbors in links.items():
        for second_index in neighbors:
            if second_index <= first_index:
                continue
            first = considered[first_index]
            second = considered[second_index]
            first_axes = np.sort(np.asarray(first["ellipse"][1], dtype=np.float64))
            second_axes = np.sort(np.asarray(second["ellipse"][1], dtype=np.float64))
            larger_major = max(first_axes[1], second_axes[1], 1.0)
            center_ratio = float(
                np.linalg.norm(
                    np.asarray(first["ellipse"][0]) - np.asarray(second["ellipse"][0])
                )
                / larger_major
            )
            shape_difference = abs(
                float(first_axes[0] / first_axes[1] - second_axes[0] / second_axes[1])
            )
            compatibility = (
                1.0 - center_ratio / float(settings["center_distance_major_ratio"])
            ) * (
                1.0 - shape_difference / float(settings["axis_ratio_difference"])
            )
            pair_score = min(first["selection_score"], second["selection_score"]) * max(
                compatibility, 0.0
            )
            smaller_index = (
                first_index
                if float(np.prod(first_axes)) < float(np.prod(second_axes))
                else second_index
            )
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (pair_score, smaller_index, first_index, second_index)

    if best_pair is None or best_pair[0] < considered[0]["selection_score"] * float(
        settings["min_pair_to_top_score"]
    ):
        selected = dict(candidates[0])
        selected["selection_mode"] = "edge_support_fallback"
        return selected

    selected = dict(considered[best_pair[1]])
    selected["selection_mode"] = "nested_inner_boundary"
    selected["nested_candidate_ranks"] = [best_pair[2] + 1, best_pair[3] + 1]
    selected["nested_pair_score"] = best_pair[0]
    return selected
