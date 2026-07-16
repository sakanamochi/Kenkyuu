import unittest

import numpy as np

from paflab.degradations import apply_degradation


class DegradationTest(unittest.TestCase):
    def setUp(self) -> None:
        gradient = np.linspace(0, 255, 128, dtype=np.uint8)
        self.image = np.repeat(gradient[None, :, None], 128, axis=0)
        self.image = np.repeat(self.image, 3, axis=2)
        self.ellipse = ((64.0, 64.0), (90.0, 54.0), 20.0)

    def test_zero_severity_is_exactly_clean(self) -> None:
        for degradation in ("occlusion", "whiteout", "black_crush"):
            result, _ = apply_degradation(
                self.image,
                degradation,
                0.0,
                self.ellipse,
                rng=np.random.default_rng(1),
            )
            np.testing.assert_array_equal(result, self.image)

    def test_higher_severity_increases_white_clipping(self) -> None:
        low, _ = apply_degradation(
            self.image, "whiteout", 0.2, self.ellipse, rng=np.random.default_rng(2)
        )
        high, _ = apply_degradation(
            self.image, "whiteout", 0.8, self.ellipse, rng=np.random.default_rng(2)
        )
        self.assertGreater(np.count_nonzero(high == 255), np.count_nonzero(low == 255))

    def test_higher_severity_increases_black_crush(self) -> None:
        low, _ = apply_degradation(
            self.image, "black_crush", 0.2, self.ellipse, rng=np.random.default_rng(3)
        )
        high, _ = apply_degradation(
            self.image, "black_crush", 0.8, self.ellipse, rng=np.random.default_rng(3)
        )
        self.assertGreater(np.count_nonzero(high == 0), np.count_nonzero(low == 0))

    def test_full_occlusion_contains_no_image_information(self) -> None:
        result, _ = apply_degradation(
            self.image, "occlusion", 1.0, self.ellipse, rng=np.random.default_rng(4)
        )
        self.assertTrue(np.all(result == 0))

    def test_full_whiteout_contains_no_image_information(self) -> None:
        result, _ = apply_degradation(
            self.image, "whiteout", 1.0, self.ellipse, rng=np.random.default_rng(5)
        )
        self.assertTrue(np.all(result == 255))


if __name__ == "__main__":
    unittest.main()
