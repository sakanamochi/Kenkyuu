"""矩形遮蔽と露光変換の方向・強度を確認する。"""

import numpy as np

from paf_ring_detection.effects import apply_effect


def test_severity_zero_keeps_image():
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    for name in ('black_rectangle', 'sensor_whiteout', 'sensor_black_crush'):
        result = apply_effect(image, name, 0, np.random.default_rng(1))
        np.testing.assert_array_equal(result, image)


def test_rectangle_width_and_direction_follow_severity():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    quarter = apply_effect(image, 'black_rectangle', 0.25, np.random.default_rng(1))
    half = apply_effect(image, 'black_rectangle', 0.5, np.random.default_rng(1))
    assert np.count_nonzero(np.all(quarter == 0, axis=2)) == 2500
    assert np.count_nonzero(np.all(half == 0, axis=2)) == 5000
    assert np.all(half[np.all(quarter == 0, axis=2)] == 0)
    assert np.all(apply_effect(image, 'black_rectangle', 1, np.random.default_rng(1)) == 0)


def test_sensor_effects_change_brightness_in_expected_direction():
    image = np.full((64, 64, 3), 140, dtype=np.uint8)
    white = apply_effect(image, 'sensor_whiteout', 0.5, np.random.default_rng(1))
    black = apply_effect(image, 'sensor_black_crush', 0.5, np.random.default_rng(1))
    assert white.mean() > image.mean()
    assert black.mean() < image.mean()
