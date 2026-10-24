"""矩形遮蔽とセンサの露光・飽和・黒レベルを近似する撮像効果。"""

import cv2
import numpy as np


def _encode(linear: np.ndarray) -> np.ndarray:
    """線形輝度をクリップし、ガンマ符号化した8bit画像へ戻す。"""
    return np.rint(np.clip(linear, 0.0, 1.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8)


def apply_effect(image: np.ndarray, name: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    """全分割に共通の劣化を適用する。正解形状は参照しない。"""
    severity = float(severity)
    if not 0 <= severity <= 1:
        raise ValueError('劣化強度は0〜1で指定してください')
    if name not in {'clean', 'black_rectangle', 'sensor_whiteout', 'sensor_black_crush'}:
        raise ValueError(f'未対応の撮像効果です: {name}')
    if name == 'clean' or severity == 0:
        return image.copy()
    if name == 'black_rectangle':
        # 強度は画像幅に占める黒矩形の割合。左右の一方から覆う。
        width = image.shape[1]
        patch_width = max(1, round(width * severity))
        left = 0 if int(rng.integers(0, 2)) == 0 else width - patch_width
        result = image.copy()
        result[:, left:left + patch_width] = 0
        return result

    linear = (image.astype(np.float32) / 255.0) ** 2.2
    if name == 'sensor_whiteout':
        # 露光を増やし、飽和した明部の拡散とショットノイズを近似する。
        linear *= 2.0 ** (5.0 * severity)
        saturated = np.clip((linear.mean(axis=2) - 0.82) / 0.18, 0.0, 1.0)
        bloom = cv2.GaussianBlur(saturated, (0, 0), 1.0 + 18.0 * severity)[..., None]
        noise = rng.normal(0.0, 0.002 + 0.006 * severity, linear.shape).astype(np.float32)
        linear += noise * np.sqrt(np.maximum(linear, 0.0))
        return _encode(linear + bloom * (0.15 + 0.85 * severity))

    # 低露光・読み出しノイズ・黒レベル・6〜8bitの量子化を近似する。
    linear *= 2.0 ** (-2.5 * severity ** 1.25)
    linear += rng.normal(0.0, 0.0015 + 0.003 * severity, linear.shape).astype(np.float32)
    black_level = 0.004 + 0.021 * severity ** 1.5
    linear = np.clip((linear - black_level) / (1.0 - black_level), 0.0, 1.0)
    levels = 2 ** max(6, round(8 - 2 * severity)) - 1
    return _encode(np.rint(linear * levels) / levels)
