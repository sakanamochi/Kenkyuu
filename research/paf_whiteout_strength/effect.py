"""露光増加・飽和・局所的なにじみの強度範囲を拡張する。"""

import cv2
import numpy as np

from paf_ring_detection.effects import _encode


def apply_whiteout(image, severity, rng, maximum_strength):
    """表示強度0～1を内部強度0～maximum_strengthへ対応させる。"""
    if not 0 <= severity <= 1 or not 1 <= maximum_strength <= 4:
        raise ValueError('severityは0～1、maximum_strengthは1～4で指定する。')
    if severity == 0:
        return image.copy()
    strength = float(severity) * float(maximum_strength)
    linear = (image.astype(np.float32) / 255.0) ** 2.2
    # 元の白飛び処理と同じ式を用い、最大内部強度だけを変更する。
    linear *= 2.0 ** (5.0 * strength)
    saturated = np.clip((linear.mean(axis=2) - 0.82) / 0.18, 0.0, 1.0)
    bloom = cv2.GaussianBlur(saturated, (0, 0), 1.0 + 18.0 * strength)[..., None]
    noise = rng.normal(0.0, 0.002 + 0.006 * strength, linear.shape).astype(np.float32)
    linear += noise * np.sqrt(np.maximum(linear, 0.0))
    return _encode(linear + bloom * (0.15 + 0.85 * strength))
