"""撮像劣化（矩形遮蔽・白飛び・黒つぶれ）と共通256pxモノクロ前処理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from core.geometry import Ellipse, label_to_ellipse, scale_ellipse
from core.io import read_image, read_json

VALID_EFFECTS = frozenset({"clean", "black_rectangle", "sensor_whiteout", "sensor_black_crush"})


def _encode_gamma(linear: np.ndarray) -> np.ndarray:
    """線形輝度を[0, 1]へクリップし、ガンマ2.2符号化のuint8画像へ戻す。"""
    return np.rint(np.clip(linear, 0.0, 1.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8)


def apply_effect(
    image: np.ndarray,
    name: str,
    severity: float,
    rng: np.random.Generator,
    *,
    whiteout_maximum_strength: float = 1.0,
) -> np.ndarray:
    """正解形状を使わずに撮像劣化を適用する。"""
    severity = float(severity)
    if not 0.0 <= severity <= 1.0:
        raise ValueError("劣化強度は0〜1で指定してください。")
    if name not in VALID_EFFECTS:
        raise ValueError(f"未対応の撮像効果です: {name}")
    if name == "clean" or severity == 0.0:
        return image.copy()

    if name == "black_rectangle":
        width = image.shape[1]
        patch_width = max(1, round(width * severity))
        left = 0 if int(rng.integers(0, 2)) == 0 else width - patch_width
        result = image.copy()
        result[:, left : left + patch_width] = 0
        return result

    linear = (image.astype(np.float32) / 255.0) ** 2.2
    if name == "sensor_whiteout":
        maximum_strength = float(whiteout_maximum_strength)
        if not 1.0 <= maximum_strength <= 4.0:
            raise ValueError("whiteout_maximum_strengthは1〜4で指定してください。")
        strength = severity * maximum_strength
        linear *= 2.0 ** (5.0 * strength)
        saturated = np.clip((linear.mean(axis=2) - 0.82) / 0.18, 0.0, 1.0)
        bloom = cv2.GaussianBlur(saturated, (0, 0), 1.0 + 18.0 * strength)[..., None]
        noise = rng.normal(0.0, 0.002 + 0.006 * strength, linear.shape).astype(np.float32)
        linear += noise * np.sqrt(np.maximum(linear, 0.0))
        return _encode_gamma(linear + bloom * (0.15 + 0.85 * strength))

    # 低露光・読み出しノイズ・黒レベル・6〜8bit量子化を近似する。
    linear *= 2.0 ** (-2.5 * severity**1.25)
    linear += rng.normal(0.0, 0.0015 + 0.003 * severity, linear.shape).astype(np.float32)
    black_level = 0.004 + 0.021 * severity**1.5
    linear = np.clip((linear - black_level) / (1.0 - black_level), 0.0, 1.0)
    levels = 2 ** max(6, round(8 - 2 * severity)) - 1
    return _encode_gamma(np.rint(linear * levels) / levels)


def to_grayscale_3ch(image: np.ndarray) -> np.ndarray:
    """画像をグレースケール化し、同一輝度の3チャンネル配列へ複製する。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def image_to_tensor(image_3ch: np.ndarray) -> torch.Tensor:
    """uint8の3ch画像(H, W, 3)を[-1, 1]正規化済みのfloatテンソル(3, H, W)へ変換する。"""
    return torch.from_numpy(image_3ch.transpose(2, 0, 1).copy()).float() / 127.5 - 1.0


def render_sample_image(
    dataset_dir: Path,
    sample: dict[str, Any],
    *,
    input_size: int = 256,
    whiteout_maximum_strength: float = 1.0,
) -> np.ndarray:
    """原解像度で劣化を適用し、グレースケール3ch・指定解像度へ縮小する。"""
    image = read_image(Path(dataset_dir) / sample["image"])
    effect = sample.get("augmentation")
    if effect:
        image = apply_effect(
            image,
            effect["name"],
            effect["severity"],
            np.random.default_rng(effect["seed"]),
            whiteout_maximum_strength=whiteout_maximum_strength,
        )
    shared = to_grayscale_3ch(image)
    if shared.shape[0] != input_size or shared.shape[1] != input_size:
        shared = cv2.resize(shared, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return shared


def load_sample_and_truth(
    dataset_dir: Path,
    sample: dict[str, Any],
    *,
    input_size: int = 256,
    whiteout_maximum_strength: float = 1.0,
) -> tuple[np.ndarray, Ellipse]:
    """共通256pxモノクロ画像と、同解像度へ変換した正解楕円を返す。"""
    image = render_sample_image(
        dataset_dir,
        sample,
        input_size=input_size,
        whiteout_maximum_strength=whiteout_maximum_strength,
    )
    label = read_json(Path(dataset_dir) / sample["label"])
    truth = scale_ellipse(label_to_ellipse(label), input_size / float(label["image_width"]))
    return image, truth
