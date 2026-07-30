"""Zhang 2019型の候補に加えるPAF固有の後処理。

このファイルの全周エッジ支持、内外輝度極性、同心候補対からの内周選択は
Zhang et al. (2019)の三弧検出コアそのものではなく、本研究のPAF画像へ
適用するために追加した処理である。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from paf_ring_detection.methods.zhang2019 import _ellipse_samples


def _polarity_score(
    gray: np.ndarray,
    ellipse,
    sample_count: int,
    offset_ratio: float,
) -> float:
    """内側が暗く外側が明るいPAF内周らしさを軟らかい得点にする。"""
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


def score_zhang_candidates_for_paf(
    candidates: list[dict],
    stages: dict,
    settings: dict,
) -> list[dict]:
    """論文コアの候補をPAFの全周支持と輝度極性で再評価する。"""
    if not candidates:
        return []
    distance_map = cv2.distanceTransform(
        255 - stages["edges"],
        cv2.DIST_L2,
        3,
    )
    threshold = float(settings["edge_distance_threshold_px"])
    sample_count = int(settings["validation_samples"])
    height, width = distance_map.shape
    scored = []
    for source_candidate in candidates:
        candidate = dict(source_candidate)
        perimeter, _ = _ellipse_samples(candidate["ellipse"], sample_count)
        pixels = np.rint(perimeter).astype(int)
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
                [
                    np.any(supported[indexes])
                    for indexes in np.array_split(
                        np.arange(sample_count),
                        bins,
                    )
                ]
            )
            / bins
        )
        if edge_density < float(settings["min_edge_density"]):
            continue
        if coverage < float(settings["min_angular_coverage"]):
            continue

        finite = distances[np.isfinite(distances)]
        mean_distance = (
            float(np.mean(np.minimum(finite, threshold * 3.0)))
            if len(finite)
            else threshold * 3.0
        )
        polarity = _polarity_score(
            stages["gray"],
            candidate["ellipse"],
            int(settings["polarity_samples"]),
            float(settings["polarity_offset_ratio"]),
        )
        polarity_weight = float(settings["polarity_weight"])
        selection_score = (
            edge_density
            * coverage
            * math.sqrt(max(candidate["fitness_score"], 1e-9))
            * ((1.0 - polarity_weight) + polarity_weight * polarity)
            / (1.0 + mean_distance)
        )
        candidate.update(
            {
                "edge_density": edge_density,
                "angular_coverage": coverage,
                "mean_edge_distance_px": mean_distance,
                "polarity_score": polarity,
                "selection_score": selection_score,
            }
        )
        scored.append(candidate)
    scored.sort(
        key=lambda candidate: candidate["selection_score"],
        reverse=True,
    )
    return scored


def select_zhang_inner_boundary(
    candidates: list[dict],
    settings: dict,
) -> dict | None:
    """PAFの同心に近い輪郭候補対から内側候補を選ぶ。"""
    if not candidates:
        return None
    considered = candidates[: int(settings["max_considered_candidates"])]
    links = {index: set() for index in range(len(considered))}
    for first_index, first in enumerate(considered):
        first_axes = np.sort(
            np.asarray(first["ellipse"][1], dtype=np.float64)
        )
        first_center = np.asarray(first["ellipse"][0], dtype=np.float64)
        first_area = float(np.prod(first_axes))
        for second_index in range(first_index + 1, len(considered)):
            second = considered[second_index]
            second_axes = np.sort(
                np.asarray(second["ellipse"][1], dtype=np.float64)
            )
            second_center = np.asarray(
                second["ellipse"][0],
                dtype=np.float64,
            )
            second_area = float(np.prod(second_axes))
            larger_major = max(first_axes[1], second_axes[1], 1.0)
            center_ratio = float(
                np.linalg.norm(first_center - second_center) / larger_major
            )
            axis_ratio_difference = abs(
                float(
                    first_axes[0] / first_axes[1]
                    - second_axes[0] / second_axes[1]
                )
            )
            area_ratio = max(first_area, second_area) / max(
                min(first_area, second_area),
                1e-9,
            )
            quality_ratio = min(
                first["selection_score"],
                second["selection_score"],
            ) / max(
                first["selection_score"],
                second["selection_score"],
                1e-9,
            )
            if (
                center_ratio <= float(settings["center_distance_major_ratio"])
                and axis_ratio_difference
                <= float(settings["axis_ratio_difference"])
                and float(settings["area_ratio_min"])
                <= area_ratio
                <= float(settings["area_ratio_max"])
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
            first_axes = np.sort(
                np.asarray(first["ellipse"][1], dtype=np.float64)
            )
            second_axes = np.sort(
                np.asarray(second["ellipse"][1], dtype=np.float64)
            )
            larger_major = max(first_axes[1], second_axes[1], 1.0)
            center_ratio = float(
                np.linalg.norm(
                    np.asarray(first["ellipse"][0])
                    - np.asarray(second["ellipse"][0])
                )
                / larger_major
            )
            shape_difference = abs(
                float(
                    first_axes[0] / first_axes[1]
                    - second_axes[0] / second_axes[1]
                )
            )
            compatibility = (
                1.0
                - center_ratio
                / float(settings["center_distance_major_ratio"])
            ) * (
                1.0
                - shape_difference
                / float(settings["axis_ratio_difference"])
            )
            pair_score = min(
                first["selection_score"],
                second["selection_score"],
            ) * max(compatibility, 0.0)
            smaller_index = (
                first_index
                if float(np.prod(first_axes)) < float(np.prod(second_axes))
                else second_index
            )
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (
                    pair_score,
                    smaller_index,
                    first_index,
                    second_index,
                )

    if best_pair is None or best_pair[0] < considered[0][
        "selection_score"
    ] * float(settings["min_pair_to_top_score"]):
        selected = dict(candidates[0])
        selected["selection_mode"] = "edge_support_fallback"
        return selected

    selected = dict(considered[best_pair[1]])
    selected["selection_mode"] = "nested_inner_boundary"
    selected["nested_candidate_ranks"] = [
        best_pair[2] + 1,
        best_pair[3] + 1,
    ]
    selected["nested_pair_score"] = best_pair[0]
    return selected
