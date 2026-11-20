"""可視境界教師、遮蔽の対応、学習用露光の検証。"""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core.effects import apply_effect
from training.data import masked_view
from training.objective import boundary_loss, boundary_target


class TrainingTests(unittest.TestCase):
    def test_occlusion_removes_boundary_without_creating_rectangle_edge(self):
        masks = torch.ones(1, 2, 32, 32)
        masks[:, 0] = 0
        masks[:, 0, 8:24, 8:24] = 1
        masks[:, 1, :, :16] = 0
        target = boundary_target(masks)
        self.assertEqual(float(target[..., :16].sum()), 0)
        self.assertEqual(float(target[0, 0, 16, 16]), 0)
        self.assertEqual(float(target[0, 0, 16, 23]), 1)

    def test_all_hidden_target_and_loss_remain_finite(self):
        masks = torch.zeros(2, 2, 32, 32)
        masks[:, 0, 8:24, 8:24] = 1
        logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
        loss = boundary_loss(logits, masks)
        loss.backward()
        self.assertEqual(float(boundary_target(masks).sum()), 0)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.sum()), 0)

    def test_opposite_masks_cover_image_without_gap(self):
        gray = np.full((480, 480), 255, np.uint8)
        ellipse = ((240., 240.), (200., 100.), 27.)
        left, lm = masked_view(gray, ellipse, .5, True)
        right, rm = masked_view(gray, ellipse, .5, False)
        self.assertTrue(torch.equal(lm + rm, torch.ones_like(lm)))
        self.assertTrue(torch.equal((left[:1] + 1) / 2, lm))
        self.assertTrue(torch.equal((right[:1] + 1) / 2, rm))

    def test_relative_mask_includes_fully_visible_and_hidden_openings(self):
        gray = np.full((480, 480), 255, np.uint8)
        ellipse = ((240., 240.), (120., 120.), 0.)
        _, visible = masked_view(gray, ellipse, 0., True)
        _, hidden = masked_view(gray, ellipse, 1., True)
        self.assertEqual(float(visible[0, 128, 128]), 1)
        self.assertEqual(float(hidden[0, 128, 128]), 0)

    def test_whiteout_training_and_evaluation_are_distinct(self):
        image = np.full((32, 32, 3), 12, np.uint8)
        trained = apply_effect(image, "sensor_whiteout", .5, 7, whiteout_multiplier=1)
        evaluated = apply_effect(image, "sensor_whiteout", .5, 7)
        self.assertGreater(float(evaluated.mean()), float(trained.mean()))
        np.testing.assert_array_equal(evaluated, apply_effect(image, "sensor_whiteout", .5, 7, whiteout_multiplier=2))

    def test_invalid_target_and_weight_are_rejected(self):
        with self.assertRaises(ValueError):
            boundary_target(torch.ones(1, 1, 32, 32))
        for weight in (0, -1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                boundary_loss(torch.zeros(1, 1, 32, 32), torch.ones(1, 2, 32, 32), weight)


if __name__ == "__main__":
    unittest.main()
