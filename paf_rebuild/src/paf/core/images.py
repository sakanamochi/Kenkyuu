"""日本語パスの画像読込と、CNNに共通する入力変換。"""

from pathlib import Path

import cv2
import numpy as np
import torch


INPUT_SIZE = 256


def read_image(path: Path) -> np.ndarray:
    """画像をグレーまたはBGRの元の配列として読み込む。"""
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) if raw.size else None
    if image is None:
        raise ValueError(f"画像を読み込めません: {path}")
    return image


def prepare_image(image: np.ndarray) -> np.ndarray:
    """正方形uint8画像をモノクロ256pxにする。"""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("入力はuint8の画像にしてください。")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("入力はグレースケールまたは3ch BGR画像にしてください。")
    height, width = image.shape[:2]
    if height == 0 or height != width:
        raise ValueError("入力は空でない正方形画像にしてください。")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


def image_tensor(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    """同一輝度を3chに複製し、batch軸を付けて[-1,1]へ変換する。"""
    channels = np.repeat(gray[None, None], 3, axis=1).copy()
    return torch.from_numpy(channels).float().to(device) / 127.5 - 1
