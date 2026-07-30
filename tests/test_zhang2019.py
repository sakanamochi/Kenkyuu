from pathlib import Path

import cv2
import numpy as np

from paf_ring_detection.data import read_json
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.zhang2019 import (
    ZHANG_2019_REFERENCE,
    detect_zhang_arc_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_zhang2019_combines_disconnected_arcs():
    config = read_json(ROOT / "config/experiment.json")["zhang2019"]
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

    candidates, _ = detect_zhang_arc_candidates(
        image,
        config["preprocess"],
        config["detector"],
    )
    metrics = evaluate_ellipses(candidates[0]["ellipse"], expected, image.shape)
    assert metrics["ellipse_iou"] > 0.85


def test_zhang2019_reference_is_not_author_code():
    assert ZHANG_2019_REFERENCE["doi"] == "10.3390/s19235243"
    assert "再現実装" in ZHANG_2019_REFERENCE["implementation_relation"]
