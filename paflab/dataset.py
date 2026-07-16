from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from paflab.image_io import imread
from paflab.labels import fit_label_ellipse, rasterize_ring_mask, scale_ellipse


class StressDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        *,
        split: str,
        input_size: int,
        return_index: bool = False,
        cache_in_memory: bool = False,
    ) -> None:
        self.dataset_dir = dataset_dir
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        self.samples = [sample for sample in manifest["samples"] if sample["split"] == split]
        self.input_size = int(input_size)
        self.return_index = return_index
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if index in self._cache:
            image_tensor, mask_tensor = self._cache[index]
            if self.return_index:
                return image_tensor, mask_tensor, index
            return image_tensor, mask_tensor
        sample = self.samples[index]
        image = imread(self.dataset_dir / sample["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.dataset_dir / sample["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if height != width:
            raise ValueError("研究データは正方形レンダリングを前提としています")

        label = json.loads((self.dataset_dir / sample["label"]).read_text(encoding="utf-8"))
        ellipse = fit_label_ellipse(label)
        scale = self.input_size / float(width)
        scaled_ellipse = scale_ellipse(ellipse, scale, scale)
        mask = rasterize_ring_mask(
            self.input_size,
            self.input_size,
            scaled_ellipse,
            thickness=max(2, round(self.input_size / 96)),
            blur_sigma=0.8,
        )

        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        image_tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float()
        image_tensor = image_tensor / 127.5 - 1.0
        mask_tensor = torch.from_numpy(mask[None, ...]).float()
        if self.cache_in_memory:
            self._cache[index] = (image_tensor, mask_tensor)
        if self.return_index:
            return image_tensor, mask_tensor, index
        return image_tensor, mask_tensor
