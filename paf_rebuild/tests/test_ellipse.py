"""欠損・外れ点・情報不足を含む楕円推定の動作確認。"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from core.ellipse import (EllipseSettings, boundary_ellipse, fit_weighted_ellipse,
                          point_distances, scale_ellipse, skeleton_points)
from evaluation.metrics import ellipse_iou


class EllipseTests(unittest.TestCase):
    def test_zero_probability_has_no_detection(self):
        self.assertIsNone(boundary_ellipse(np.zeros((256, 256), np.float32), EllipseSettings()))

    def test_too_few_points_has_no_detection(self):
        probability = np.zeros((256, 256), np.float32)
        probability[20, 20:24] = 1
        self.assertIsNone(boundary_ellipse(probability, EllipseSettings()))

    def test_skeleton_weights_and_image_edges(self):
        probability = np.zeros((16, 16), np.float32)
        probability[0:3, :] = 0.75
        points, weights = skeleton_points(probability)
        self.assertGreater(len(points), 0)
        self.assertTrue(np.all(weights == 0.75))
        self.assertTrue(np.all(points >= 0) and np.all(points < 16))

    def test_partial_ellipse_with_outliers(self):
        angles = np.linspace(0, np.pi * 1.25, 160)
        truth = ((128.0, 100.0), (120.0, 60.0), 0.0)
        points = np.column_stack((128 + 60 * np.cos(angles), 100 + 30 * np.sin(angles)))
        noise = np.random.default_rng(8).uniform(0, 256, (50, 2))
        points = np.concatenate((points, noise)).astype(np.float32)
        weights = np.r_[np.full(160, 0.9), np.full(50, 0.1)]
        settings = EllipseSettings()
        cv2.setRNGSeed(settings.opencv_seed)
        fitted = fit_weighted_ellipse(points, weights, (256, 256), settings)
        self.assertIsNotNone(fitted)
        self.assertGreater(ellipse_iou(fitted, truth), 0.98)
        self.assertLess(float(point_distances(points[:160], fitted).max()), 1)

    def test_same_input_resets_opencv_random_state(self):
        probability = np.zeros((256, 256), np.float32)
        cv2.ellipse(probability, ((128., 100.), (120., 60.), 30.), 0.9, 2)
        a = boundary_ellipse(probability, EllipseSettings())
        cv2.setRNGSeed(123)
        b = boundary_ellipse(probability, EllipseSettings())
        self.assertIsNotNone(a)
        self.assertEqual(a, b)

    def test_scaling_preserves_angle(self):
        self.assertEqual(scale_ellipse(((10., 20.), (30., 40.), 50.), 2),
                         ((20., 40.), (60., 80.), 50.))

    def test_invalid_settings_and_probabilities(self):
        with self.assertRaises(ValueError):
            EllipseSettings(min_inliers=4)
        with self.assertRaises(ValueError):
            boundary_ellipse(np.full((16, 16), np.nan), EllipseSettings())


if __name__ == "__main__":
    unittest.main()
