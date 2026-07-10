import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from analysis.ellipse_baseline import detect_candidates, evaluate_ellipses


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EllipseBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(
            (PROJECT_ROOT / "config" / "baseline.json").read_text(encoding="utf-8")
        )
        cls.settings = config["detector"]

    def test_detects_off_center_ellipse_without_position_or_area_prior(self) -> None:
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        expected = ((72.0, 218.0), (124.0, 66.0), 27.0)
        cv2.ellipse(image, expected, (255, 255, 255), 2, cv2.LINE_AA)

        candidates = detect_candidates(image, self.settings)

        self.assertGreater(len(candidates), 0)
        evaluation = evaluate_ellipses(
            candidates[0]["ellipse"], expected, image.shape
        )
        self.assertGreater(evaluation["ellipse_iou"], 0.90)
        self.assertLess(evaluation["center_error_px"], 2.0)


if __name__ == "__main__":
    unittest.main()
