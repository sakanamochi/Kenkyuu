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


def region_mask(size: int, ellipse: Ellipse) -> np.ndarray:
    """不可視部分も含む完全な穴内部をCNN教師にする。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1, cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def sample_image(dataset_dir: Path, sample: dict) -> np.ndarray:
    """固定seedの撮像効果を原画像に適用し、全方式へ同じ画像を渡す。"""
    from paf_ring_detection.effects import training_effect

    image = read_image(dataset_dir / sample["image"])
    if image is None:
        raise ValueError(f"画像を読めません: {sample['image']}")
    effect = sample.get("augmentation")
    if effect:
        label = read_json(dataset_dir / sample["label"])
        image = training_effect(image, effect["name"], effect["severity"],
                                label_ellipse(label), np.random.default_rng(effect["seed"]))
    return image


def load_samples(dataset_dir: Path, split: str) -> list[dict]:
    manifest = read_json(dataset_dir / "manifest.json")
    return [sample for sample in manifest["samples"] if sample["split"] == split]


class RingDataset(Dataset):
    """CNN学習・推論で共通に使う穴領域データセット。"""

    def __init__(self, dataset_dir: Path, split: str, input_size: int, cache: bool = False) -> None:
        self.root = Path(dataset_dir)
        self.samples = load_samples(self.root, split)
        self.input_size = int(input_size)
        # 固定劣化の学習入力はRAMに保持し、epochごとの同じ変換を省く。
        self.cache = {} if cache else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if self.cache is not None and index in self.cache:
            return self.cache[index]
        sample = self.samples[index]
        image = sample_image(self.root, sample)
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

        mask = region_mask(self.input_size, scaled_ellipse(ellipse, scale))
        mask_tensor = torch.from_numpy(mask[None]).float()
        result = image_tensor, mask_tensor, index
        if self.cache is not None:
            self.cache[index] = result
        return result
