import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from analysis.ellipse_baseline import evaluate_ellipses
from analysis.zhang_arc_detector import (
    ZHANG_2019_REFERENCE,
    detect_zhang_arc_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ZhangArcDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads(
            (PROJECT_ROOT / "config" / "baseline.json").read_text(encoding="utf-8")
        )
        cls.detector = config["detector"]
        cls.settings = config["zhang2019_arc"]

    def test_combines_disconnected_arcs_into_one_ellipse(self) -> None:
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        expected = ((200.0, 150.0), (220.0, 100.0), 18.0)
        for start, end in ((10, 95), (135, 225), (265, 345)):
            cv2.ellipse(
                image,
                (200, 150),
                (110, 50),
                18,
                start,
                end,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        candidates, stages = detect_zhang_arc_candidates(
            image, self.detector, self.settings
        )

        self.assertGreaterEqual(len(stages["arcs"]), 3)
        self.assertGreater(len(candidates), 0)
        self.assertEqual(candidates[0]["arc_count"], 3)
        evaluation = evaluate_ellipses(candidates[0]["ellipse"], expected, image.shape)
        self.assertGreater(evaluation["ellipse_iou"], 0.85)

    def test_reference_metadata_points_to_docking_ring_paper(self) -> None:
        self.assertEqual(ZHANG_2019_REFERENCE["doi"], "10.3390/s19235243")
        self.assertIn("Docking Ring Detection", ZHANG_2019_REFERENCE["title"])
        self.assertIn("再現実装", ZHANG_2019_REFERENCE["implementation_relation"])


if __name__ == "__main__":
    unittest.main()
