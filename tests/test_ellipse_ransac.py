import json
import math
import unittest
from pathlib import Path

import numpy as np

from analysis.ellipse_baseline import evaluate_ellipses
from analysis.ellipse_ransac import fit_ellipse_ransac


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EllipseRansacTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(
            (PROJECT_ROOT / "config" / "baseline.json").read_text(encoding="utf-8")
        )
        cls.settings = {**config["ransac"], "iterations": 1500, "min_inliers": 80}

    def test_fits_partial_ellipse_with_outliers(self) -> None:
        rng = np.random.default_rng(1234)
        expected = ((210.0, 160.0), (180.0, 90.0), 30.0)
        angles = np.linspace(math.radians(15), math.radians(300), 260)
        local = np.column_stack((90.0 * np.cos(angles), 45.0 * np.sin(angles)))
        rotation = np.array(
            [
                [math.cos(math.radians(30)), -math.sin(math.radians(30))],
                [math.sin(math.radians(30)), math.cos(math.radians(30))],
            ]
        )
        ellipse_points = local @ rotation.T + np.array([210.0, 160.0])
        ellipse_points += rng.normal(0.0, 0.6, ellipse_points.shape)
        outliers = rng.uniform([0.0, 0.0], [420.0, 320.0], size=(260, 2))
        points = np.vstack((ellipse_points, outliers)).astype(np.float32)

        result = fit_ellipse_ransac(
            points,
            (320, 420, 3),
            self.settings,
            random_seed=42,
        )

        self.assertIsNotNone(result)
        evaluation = evaluate_ellipses(result["ellipse"], expected, (320, 420, 3))
        self.assertGreater(evaluation["ellipse_iou"], 0.85)
        self.assertLess(evaluation["center_error_px"], 5.0)
        self.assertGreater(result["inlier_count"], 200)

    def test_probability_weights_select_cnn_target_points(self) -> None:
        rng = np.random.default_rng(5678)

        def ellipse_points(center, axes, angle_deg, count, start=0.0, end=2 * math.pi):
            angles = np.linspace(start, end, count, endpoint=False)
            local = np.column_stack(
                ((axes[0] / 2) * np.cos(angles), (axes[1] / 2) * np.sin(angles))
            )
            radians = math.radians(angle_deg)
            rotation = np.array(
                [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]]
            )
            return local @ rotation.T + np.asarray(center)

        target = ellipse_points(
            (145.0, 190.0),
            (150.0, 82.0),
            18.0,
            220,
            math.radians(20),
            math.radians(315),
        )
        distractor = ellipse_points((310.0, 120.0), (210.0, 130.0), -12.0, 320)
        target += rng.normal(0.0, 0.5, target.shape)
        distractor += rng.normal(0.0, 0.5, distractor.shape)
        points = np.vstack((target, distractor)).astype(np.float32)
        weights = np.concatenate((np.ones(len(target)), np.full(len(distractor), 0.03)))

        result = fit_ellipse_ransac(
            points,
            (320, 480, 3),
            self.settings,
            weights=weights,
            random_seed=84,
        )

        self.assertIsNotNone(result)
        expected = ((145.0, 190.0), (150.0, 82.0), 18.0)
        evaluation = evaluate_ellipses(result["ellipse"], expected, (320, 480, 3))
        self.assertGreater(evaluation["ellipse_iou"], 0.85)


if __name__ == "__main__":
    unittest.main()
