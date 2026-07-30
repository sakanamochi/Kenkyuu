import cv2
import numpy as np
import torch

from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.methods.cnn_ransac import detect_from_probability


def test_cnn_outputs_one_probability_map():
    model = TinyUNet(base_channels=2)
    output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)


def test_probability_map_is_converted_to_ellipse():
    probability = np.zeros((160, 160), dtype=np.float32)
    cv2.ellipse(
        probability,
        ((80, 80), (100, 50), 20),
        1.0,
        2,
        cv2.LINE_AA,
    )
    settings = {
        "probability_threshold": 0.35,
        "max_points": 6000,
        "iterations": 300,
        "distance_threshold_px": 2.5,
        "min_inliers": 40,
        "min_axis_px": 6.0,
        "min_axis_ratio": 0.1,
        "max_axis_diagonal_ratio": 2.0,
        "angular_bins": 72,
        "refine_iterations": 3,
        "perimeter_power": 0.0,
        "random_seed": 1,
    }
    assert detect_from_probability(probability, settings, random_seed=1) is not None
