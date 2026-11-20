"""入力仕様と保存重みに対応するネットワーク構造の検証。"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch

from paf.core.images import image_tensor, prepare_image
from paf.core.model import TinyUNet


class InputTests(unittest.TestCase):
    def test_color_conversion_and_resize(self):
        image = np.random.default_rng(42).integers(0, 256, (480, 480, 3), dtype=np.uint8)
        expected = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (256, 256),
                              interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(prepare_image(image), expected)

    def test_gray_and_identical_color_channels_agree(self):
        gray = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
        color = np.repeat(gray[:, :, None], 3, axis=2)
        np.testing.assert_array_equal(prepare_image(gray), prepare_image(color))

    def test_tensor_range_and_channels(self):
        gray = np.zeros((256, 256), dtype=np.uint8)
        gray[:, 128:] = 255
        result = image_tensor(gray, torch.device("cpu"))
        self.assertEqual(tuple(result.shape), (1, 3, 256, 256))
        self.assertEqual(float(result.min()), -1)
        self.assertEqual(float(result.max()), 1)
        self.assertTrue(torch.equal(result[:, 0], result[:, 2]))

    def test_invalid_input_is_rejected(self):
        for image in (np.zeros((2, 3), np.uint8), np.zeros((0, 0), np.uint8),
                      np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2), np.float32), None):
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(ValueError):
                    prepare_image(image)


class ModelTests(unittest.TestCase):
    def test_parameter_count_and_output(self):
        torch.set_num_threads(2)
        model = TinyUNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 482737)
        with torch.inference_mode():
            output = model(torch.zeros((1, 3, 256, 256)))
        self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
        self.assertTrue(bool(torch.isfinite(output).all()))


if __name__ == "__main__":
    unittest.main()
