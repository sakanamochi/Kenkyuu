from pathlib import Path

import cv2
import numpy as np

from paf_ring_detection.data import read_json
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.zhang2019 import (
    EXPECTED_CONVEXITY,
    ZHANG_2019_REFERENCE,
    _estimate_pair_center,
    _pair_constraints,
    _radial_distance,
    detect_zhang_arc_candidates,
    extract_zhang_arcs,
)
from paf_ring_detection.methods.zhang2019_paf import (
    score_zhang_candidates_for_paf,
    select_zhang_inner_boundary,
)


ROOT = Path(__file__).resolve().parents[1]


def _circle_arcs():
    """論文の象限I～IVに対応する理想円弧を作る。"""
    center = np.asarray([200.0, 150.0])
    angle_ranges = (
        np.linspace(80.0, 10.0, 71),
        np.linspace(170.0, 100.0, 71),
        np.linspace(190.0, 260.0, 71),
        np.linspace(280.0, 350.0, 71),
    )
    arcs = []
    for quadrant, angles in enumerate(angle_ranges):
        radians = np.deg2rad(angles)
        points = center + 100.0 * np.column_stack(
            (np.cos(radians), np.sin(radians))
        )
        arcs.append(
            {
                "points": points.astype(np.float32),
                "quadrant": quadrant,
                "left_point": points[0],
                "middle_point": points[len(points) // 2],
                "right_point": points[-1],
                "left_gradient": points[0] - center,
                "right_gradient": points[-1] - center,
            }
        )
    return center, arcs


def test_zhang2019_combines_disconnected_arcs():
    config = read_json(ROOT / "config/experiment.json")["zhang2019"]
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    expected = ((200.0, 150.0), (220.0, 100.0), 18.0)
    for start, end in ((10, 95), (135, 225), (265, 345)):
        cv2.ellipse(
            image,
            (200, 150),
            (110, 50),
            18,
            start,
            end,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    candidates, _ = detect_zhang_arc_candidates(
        image,
        config["preprocess"],
        config["detector"],
    )
    metrics = evaluate_ellipses(candidates[0]["ellipse"], expected, image.shape)
    assert metrics["ellipse_iou"] > 0.85
    assert candidates[0]["arc_count"] == 3
    assert len(candidates[0]["pair_centers"]) == 2
    assert "selection_score" not in candidates[0]


def test_zhang2019_arc_selection_matches_quadrant_convexity_rule():
    config = read_json(ROOT / "config/experiment.json")["zhang2019"]
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.ellipse(
        image,
        (200, 150),
        (110, 50),
        18,
        0,
        360,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    stages = extract_zhang_arcs(
        image,
        config["preprocess"],
        config["detector"],
    )
    assert stages["arcs"]
    assert all(
        arc["convexity"] == EXPECTED_CONVEXITY[arc["quadrant"]]
        for arc in stages["arcs"]
    )


def test_zhang2019_pair_constraints_and_centers_on_ideal_arcs():
    expected_center, arcs = _circle_arcs()
    for quadrant in range(4):
        first = arcs[quadrant]
        second = arcs[(quadrant + 1) % 4]
        assert _pair_constraints(first, second)
        estimated = _estimate_pair_center(first, second, sample_count=12)
        assert np.linalg.norm(estimated - expected_center) < 0.01


def test_zhang2019_radial_distance_uses_center_ray_intersection():
    ellipse = ((0.0, 0.0), (200.0, 100.0), 0.0)
    points = np.asarray([[105.0, 0.0], [0.0, 55.0]], dtype=np.float32)
    np.testing.assert_allclose(
        _radial_distance(points, ellipse),
        [5.0, 5.0],
        atol=1e-6,
    )


def test_paf_specific_edge_support_is_applied_outside_zhang_core():
    config = read_json(ROOT / "config/experiment.json")["zhang2019"]
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    for start, end in ((10, 95), (135, 225), (265, 345)):
        cv2.ellipse(
            image,
            (200, 150),
            (110, 50),
            18,
            start,
            end,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    candidates, stages = detect_zhang_arc_candidates(
        image,
        config["preprocess"],
        config["detector"],
    )
    scored = score_zhang_candidates_for_paf(
        candidates,
        stages,
        config["paf_postprocess"]["candidate_validation"],
    )
    assert scored
    assert scored[0]["selection_score"] == scored[0]["edge_density"]
    assert "polarity_score" not in scored[0]


def test_paf_selector_prefers_contained_inner_candidate():
    """全周支持率が低くても、内包される候補を内周として選ぶ。"""
    outer = {
        "ellipse": ((200.0, 150.0), (220.0, 120.0), 15.0),
        "edge_density": 0.90,
    }
    inner = {
        "ellipse": ((200.0, 150.0), (170.0, 80.0), 15.0),
        "edge_density": 0.60,
    }
    selected = select_zhang_inner_boundary(
        [outer, inner],
        {"max_considered_candidates": 20},
    )
    assert selected["ellipse"] == inner["ellipse"]
    assert selected["selection_mode"] == "contained_inner_boundary"


def test_paf_selector_uses_full_perimeter_support_without_containment():
    """別位置の候補では全周支持率が高い方を選ぶ。"""
    lower_support = {
        "ellipse": ((90.0, 150.0), (100.0, 60.0), 0.0),
        "edge_density": 0.55,
    }
    higher_support = {
        "ellipse": ((310.0, 150.0), (100.0, 60.0), 0.0),
        "edge_density": 0.82,
    }
    selected = select_zhang_inner_boundary(
        [lower_support, higher_support],
        {"max_considered_candidates": 20},
    )
    assert selected["ellipse"] == higher_support["ellipse"]
    assert selected["selection_mode"] == "full_perimeter_support"


def test_zhang2019_reference_is_not_author_code():
    assert ZHANG_2019_REFERENCE["doi"] == "10.3390/s19235243"
    assert "再現実装" in ZHANG_2019_REFERENCE["implementation_relation"]
