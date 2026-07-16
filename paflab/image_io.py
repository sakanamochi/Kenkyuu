from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: Path, flags: int = cv2.IMREAD_COLOR):
    """Windowsの非ASCIIパスでも動くOpenCV画像読み込み。"""
    data = np.fromfile(Path(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: Path, image, parameters=None) -> bool:
    """Windowsの非ASCIIパスでも動くOpenCV画像書き込み。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image, parameters or [])
    if success:
        encoded.tofile(path)
    return bool(success)
