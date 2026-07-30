from paf_ring_detection.geometry import evaluate_ellipses


def test_same_ellipse_has_perfect_iou():
    ellipse = ((100.0, 100.0), (120.0, 60.0), 20.0)
    result = evaluate_ellipses(ellipse, ellipse, (240, 240))
    assert result["ellipse_iou"] == 1.0
    assert result["center_error_px"] == 0.0


def test_shifted_ellipse_has_lower_iou():
    truth = ((100.0, 100.0), (120.0, 60.0), 20.0)
    shifted = ((130.0, 100.0), (120.0, 60.0), 20.0)
    result = evaluate_ellipses(shifted, truth, (240, 240))
    assert 0.0 < result["ellipse_iou"] < 1.0
