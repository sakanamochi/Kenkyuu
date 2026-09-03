"""Zhang 2019型の候補に加えるPAF固有の後処理。

このファイルの全周エッジ支持と内包関係に基づく内周選択は
Zhang et al. (2019)の三弧検出コアそのものではなく、本研究のPAF画像へ
適用するために追加した処理である。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from paf_ring_detection.methods.zhang2019 import _ellipse_samples


def score_zhang_candidates_for_paf(
    candidates: list[dict],
    stages: dict,
    settings: dict,
) -> list[dict]:
    """論文コアの候補をPAFの全周エッジ支持で検証する。"""
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
        candidate.update(
            {
                "edge_density": edge_density,
                "angular_coverage": coverage,
                "mean_edge_distance_px": mean_distance,
                # 既存の結果出力との互換性を保ちつつ、得点は全周支持率だけにする。
                "selection_score": edge_density,
            }
        )
        scored.append(candidate)
    scored.sort(
        key=lambda candidate: candidate["edge_density"],
        reverse=True,
    )
    return scored


def _ellipse_contains(outer, inner, sample_count: int = 180) -> bool:
    """内側候補の周上点がすべて外側楕円内にあるかを判定する。"""
    points, _ = _ellipse_samples(inner, sample_count)
    (center_x, center_y), (axis_1, axis_2), angle = outer
    half_axis_1 = max(float(axis_1) * 0.5, 1e-9)
    half_axis_2 = max(float(axis_2) * 0.5, 1e-9)
    centered = points - np.asarray([center_x, center_y], dtype=np.float64)
    radians = math.radians(float(angle))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    local_x = centered[:, 0] * cosine + centered[:, 1] * sine
    local_y = -centered[:, 0] * sine + centered[:, 1] * cosine
    normalized_radius_squared = (
        (local_x / half_axis_1) ** 2 + (local_y / half_axis_2) ** 2
    )
    return bool(np.all(normalized_radius_squared <= 1.0 + 1e-6))


def select_zhang_inner_boundary(
    candidates: list[dict],
    settings: dict,
) -> dict | None:
    """内包候補では内側を、内包関係がなければ全周支持率最大を選ぶ。"""
    if not candidates:
        return None
    considered = candidates[: int(settings["max_considered_candidates"])]

    # 複数の同心候補がある場合にも最も内側を優先できるよう、
    # 各候補を内包する別候補の数を数える。
    containment_depth = [0] * len(considered)
    for inner_index, inner in enumerate(considered):
        for outer_index, outer in enumerate(considered):
            if inner_index == outer_index:
                continue
            if _ellipse_contains(outer["ellipse"], inner["ellipse"]):
                containment_depth[inner_index] += 1

    maximum_depth = max(containment_depth)
    if maximum_depth > 0:
        selectable = [
            index
            for index, depth in enumerate(containment_depth)
            if depth == maximum_depth
        ]
        selected_index = max(
            selectable,
            key=lambda index: float(considered[index]["edge_density"]),
        )
        selection_mode = "contained_inner_boundary"
    else:
        selected_index = max(
            range(len(considered)),
            key=lambda index: float(considered[index]["edge_density"]),
        )
        selection_mode = "full_perimeter_support"

    selected = dict(considered[selected_index])
    selected["selection_mode"] = selection_mode
    selected["containment_depth"] = containment_depth[selected_index]
    return selected
