"""穴領域CNNの最大領域外周から内周楕円を求める。"""

import cv2
import numpy as np


def region_ellipse(probability: np.ndarray, threshold: float = 0.5):
    """閾値以上の最大外周に楕円を当てはめる。正解情報は使用しない。"""
    mask = (probability >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = [c for c in contours if len(c) >= 5 and cv2.contourArea(c) > 0]
    if not candidates:
        return None
    ellipse = cv2.fitEllipse(max(candidates, key=cv2.contourArea))
    if not np.isfinite(np.asarray([*ellipse[0], *ellipse[1], ellipse[2]])).all():
        return None
    return ellipse if min(ellipse[1]) > 0 else None
