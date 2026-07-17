import numpy as np

from paflab.camera_effects import black_rectangle, sensor_black_crush, sensor_whiteout
from paflab.prepare_diagnostic_dataset import diagnostic_seed


ELLIPSE = ((48.0, 48.0), (50.0, 70.0), 20.0)


def test_black_rectangle_width_matches_severity():
    image = np.full((96, 96, 3), 200, dtype=np.uint8)
    result, metadata = black_rectangle(
        image, ELLIPSE, 0.25, rng=np.random.default_rng(1)
    )
    left, _, right, _ = metadata["rectangle_xyxy"]
    assert right - left == 24
    assert left == 0 or right == image.shape[1]
    assert np.all(result[:, left:right] == 0)


def test_black_rectangle_starts_at_selected_image_edge():
    image = np.full((96, 96, 3), 200, dtype=np.uint8)
    sides = {
        black_rectangle(image, ELLIPSE, 0.25, rng=np.random.default_rng(seed))[1][
            "side"
        ]
        for seed in range(10)
    }
    assert sides == {"left", "right"}


def test_black_rectangle_uses_same_side_seed_across_severities():
    seeds = {
        diagnostic_seed("experiment", "sample", "black_rectangle", severity)
        for severity in (0.25, 0.5, 0.75, 1.0)
    }
    assert len(seeds) == 1


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


def test_sensor_black_crush_keeps_gradual_brightness_after_half_severity():
    image = np.full((96, 96, 3), 160, dtype=np.uint8)
    quarter, _ = sensor_black_crush(
        image, 0.25, rng=np.random.default_rng(4)
    )
    half, metadata = sensor_black_crush(
        image, 0.5, rng=np.random.default_rng(4)
    )
    full, _ = sensor_black_crush(
        image, 1.0, rng=np.random.default_rng(4)
    )
    assert half.mean() > quarter.mean() * 0.4
    assert full.mean() > 0
    assert metadata["exposure_stops"] > -1.1


def test_sensor_black_crush_maximum_exposure_drop_is_two_point_five_stops():
    image = np.full((96, 96, 3), 160, dtype=np.uint8)
    _, metadata = sensor_black_crush(
        image, 1.0, rng=np.random.default_rng(5)
    )
    assert metadata["exposure_stops"] == -2.5
