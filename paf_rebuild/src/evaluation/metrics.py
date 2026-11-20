"""推定楕円と正解楕円の内部領域の重なりを測る。"""

import cv2
import numpy as np


def ellipse_tuple(value):
    """JSON配列をOpenCVが受け取る楕円形式に戻す。"""
    if value is None:
        return None
    return tuple(value[0]), tuple(value[1]), float(value[2])


def ellipse_iou(estimated, truth, shape=(256, 256)) -> float:
    """未検出を0として、塗りつぶした楕円内部のIoUを計算する。"""
    if estimated is None:
        return 0.0
    masks = [np.zeros(shape, dtype=np.uint8) for _ in range(2)]
    for mask, ellipse in zip(masks, (estimated, truth)):
        cv2.ellipse(mask, ellipse_tuple(ellipse), 255, -1)
    intersection = np.count_nonzero((masks[0] > 0) & (masks[1] > 0))
    union = np.count_nonzero((masks[0] > 0) | (masks[1] > 0))
    return float(intersection / union if union else 0.0)
