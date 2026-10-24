import cv2
import numpy as np
import torch

from paf_ring_detection.data import region_mask
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.methods.cnn_region import region_ellipse


def test_cnn_outputs_one_probability_map():
    model = TinyUNet(base_channels=2)
    output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)


def test_region_ellipse_follows_largest_region_and_includes_hole_center():
    # 穴内部を塗り、離れた小領域や閾値未満の雑音で楕円が切り替わらないことを確認する。
    truth = ((80.0, 80.0), (100.0, 50.0), 20.0)
    probability = region_mask(160, truth)
    assert probability[80, 80] == 1.0
    cv2.circle(probability, (12, 12), 5, 1.0, -1)
    probability[140:150, 140:150] = 0.49
    ellipse = region_ellipse(probability)
    assert evaluate_ellipses(ellipse, truth, probability.shape)['ellipse_iou'] > 0.97
    assert region_ellipse(np.zeros((32, 32), np.float32)) is None
