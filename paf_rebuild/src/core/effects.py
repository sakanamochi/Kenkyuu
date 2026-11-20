"""固定条件の矩形遮蔽・白飛び・黒つぶれを画像へ適用する。"""

import cv2
import numpy as np


def encode(linear):
    """線形輝度をガンマ符号化した8bit画像へ戻す。"""
    return np.rint(np.clip(linear, 0.0, 1.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8)


def apply_effect(image, name, severity, seed, *, whiteout_multiplier=2.0):
    """白飛び倍率は評価用2、学習用1。乱数は画像単位で指定する。"""
    if not np.isfinite(whiteout_multiplier) or whiteout_multiplier <= 0:
        raise ValueError("白飛び倍率は有限の正数にしてください。")
    severity = float(severity)
    if not 0 <= severity <= 1:
        raise ValueError("劣化強度は0〜1にしてください。")
    if name not in ("clean", "black_rectangle", "sensor_whiteout", "sensor_black_crush"):
        raise ValueError(f"未対応の劣化です: {name}")
    if name == "clean" or severity == 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    if name == "black_rectangle":
        width = image.shape[1]
        covered = max(1, round(width * severity))
        left = 0 if int(rng.integers(0, 2)) == 0 else width - covered
        result = image.copy()
        result[:, left:left + covered] = 0
        return result
    linear = (image.astype(np.float32) / 255.0) ** 2.2
    if name == "sensor_whiteout":
        strength = severity * whiteout_multiplier
        linear *= 2.0 ** (5.0 * strength)
        saturated = np.clip((linear.mean(axis=2) - 0.82) / 0.18, 0.0, 1.0)
        bloom = cv2.GaussianBlur(saturated, (0, 0), 1.0 + 18.0 * strength)[..., None]
        noise = rng.normal(0.0, 0.002 + 0.006 * strength, linear.shape).astype(np.float32)
        linear += noise * np.sqrt(np.maximum(linear, 0.0))
        return encode(linear + bloom * (0.15 + 0.85 * strength))
    linear *= 2.0 ** (-2.5 * severity ** 1.25)
    linear += rng.normal(0.0, 0.0015 + 0.003 * severity, linear.shape).astype(np.float32)
    black_level = 0.004 + 0.021 * severity ** 1.5
    linear = np.clip((linear - black_level) / (1.0 - black_level), 0.0, 1.0)
    levels = 2 ** max(6, round(8 - 2 * severity)) - 1
    return encode(np.rint(linear * levels) / levels)
