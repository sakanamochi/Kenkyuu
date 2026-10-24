import numpy as np

from paf_ring_detection.effects import (
    training_effect,
)


ELLIPSE = ((64.0, 64.0), (90.0, 54.0), 20.0)


def test_severity_zero_keeps_training_image():
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    for name in ("occlusion", "whiteout", "black_crush"):
        result = training_effect(
            image, name, 0.0, ELLIPSE, np.random.default_rng(1)
        )
        np.testing.assert_array_equal(result, image)


def test_full_occlusion_is_black():
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    result = training_effect(
        image, "occlusion", 1.0, ELLIPSE, np.random.default_rng(1)
    )
    assert np.all(result == 0)
