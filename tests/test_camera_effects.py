import numpy as np

from paflab.camera_effects import black_rectangle, sensor_black_crush, sensor_whiteout


ELLIPSE = ((48.0, 48.0), (50.0, 70.0), 20.0)


def test_black_rectangle_width_matches_severity():
    image = np.full((96, 96, 3), 200, dtype=np.uint8)
    result, metadata = black_rectangle(
        image, ELLIPSE, 0.25, rng=np.random.default_rng(1)
    )
    left, _, right, _ = metadata["rectangle_xyxy"]
    assert right - left == 24
    assert np.all(result[:, left:right] == 0)


def test_sensor_whiteout_increases_brightness_monotonically():
    image = np.full((96, 96, 3), 80, dtype=np.uint8)
    means = [
        sensor_whiteout(image, severity, rng=np.random.default_rng(2))[0].mean()
        for severity in (0.25, 0.5, 0.75, 1.0)
    ]
    assert means == sorted(means)


def test_sensor_black_crush_decreases_brightness_monotonically():
    image = np.full((96, 96, 3), 160, dtype=np.uint8)
    means = [
        sensor_black_crush(image, severity, rng=np.random.default_rng(3))[0].mean()
        for severity in (0.25, 0.5, 0.75, 1.0)
    ]
    assert means == sorted(means, reverse=True)
