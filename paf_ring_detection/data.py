"""データセット、正解楕円、画像入出力をまとめた小さな共通モジュール。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


Ellipse = tuple[tuple[float, float], tuple[float, float], float]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR):
    """日本語を含むWindowsパスでも画像を読めるようにする。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def write_image(path: Path, image: np.ndarray) -> None:
    """日本語を含むWindowsパスへ画像を書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    _, encoded = cv2.imencode(path.suffix or ".png", image)
    encoded.tofile(path)


def label_ellipse(label: dict) -> Ellipse:
    """Blenderが投影した内周頂点から正解楕円を求める。"""
    points = np.asarray(label["image_points"], dtype=np.float32).reshape(-1, 1, 2)
    return cv2.fitEllipse(points)


def scaled_ellipse(ellipse: Ellipse, scale: float) -> Ellipse:
    (cx, cy), (axis_1, axis_2), angle = ellipse
    return (
        (float(cx * scale), float(cy * scale)),
        (float(axis_1 * scale), float(axis_2 * scale)),
        float(angle),
    )


def ring_mask(size: int, ellipse: Ellipse) -> np.ndarray:
    """遮蔽部分も含む完全な内周リングをCNN教師マスクにする。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    thickness = max(2, round(size / 96))
    cv2.ellipse(mask, ellipse, 255, thickness, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.8)
    return mask / max(float(mask.max()), 1e-6)


def load_samples(dataset_dir: Path, split: str) -> list[dict]:
    manifest = read_json(dataset_dir / "manifest.json")
    return [sample for sample in manifest["samples"] if sample["split"] == split]


class RingDataset(Dataset):
    """CNN学習・推論で共通に使うPAFリングデータセット。"""

    def __init__(self, dataset_dir: Path, split: str, input_size: int) -> None:
        self.root = Path(dataset_dir)
        self.samples = load_samples(self.root, split)
        self.input_size = int(input_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = read_image(self.root / sample["image"])
        label = read_json(self.root / sample["label"])
        ellipse = label_ellipse(label)
        scale = self.input_size / float(label["image_width"])

        image = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1).copy()
        image_tensor = torch.from_numpy(image).float() / 127.5 - 1.0

        mask = ring_mask(self.input_size, scaled_ellipse(ellipse, scale))
        mask_tensor = torch.from_numpy(mask[None]).float()
        return image_tensor, mask_tensor, index
