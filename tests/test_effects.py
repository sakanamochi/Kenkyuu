import numpy as np

from paf_ring_detection.effects import (
    black_rectangle,
    sensor_black_crush,
    sensor_whiteout,
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


def test_black_rectangle_width_matches_severity():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    result = black_rectangle(image, 0.25, np.random.default_rng(1))
    assert np.count_nonzero(np.all(result == 0, axis=2)) == 2500


def test_sensor_effects_change_brightness_in_expected_direction():
    image = np.full((64, 64, 3), 140, dtype=np.uint8)
    white = sensor_whiteout(image, 0.5, np.random.default_rng(1))
    black = sensor_black_crush(image, 0.5, np.random.default_rng(1))
    assert white.mean() > image.mean()
    assert black.mean() < image.mean()
