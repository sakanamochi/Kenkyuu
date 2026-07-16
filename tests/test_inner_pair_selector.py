from analysis.ellipse_ransac import select_paf_inner_candidate


SETTINGS = {
    "area_ratio_min": 1.5,
    "area_ratio_max": 4.0,
    "center_distance_major_ratio": 0.1,
    "axis_ratio_difference": 0.05,
    "min_quality_ratio": 0.25,
    "max_candidates": 10,
}


def candidate(axes, score, center=(100.0, 100.0)):
    return {
        "ellipse": (center, axes, 0.0),
        "selection_score": score,
    }


def test_inner_pair_prior_selects_smaller_concentric_candidate():
    outer = candidate((120.0, 80.0), 0.9)
    inner = candidate((70.0, 47.0), 0.7)
    selected = select_paf_inner_candidate([outer, inner], SETTINGS)
    assert selected["ellipse"] == inner["ellipse"]
    assert selected["selection_mode"] == "inner_pair_prior"


def test_selector_falls_back_to_quality_when_pair_is_not_concentric():
    first = candidate((120.0, 80.0), 0.9)
    far = candidate((70.0, 47.0), 0.7, center=(200.0, 200.0))
    selected = select_paf_inner_candidate([first, far], SETTINGS)
    assert selected["ellipse"] == first["ellipse"]
    assert selected["selection_mode"] == "quality_fallback"
