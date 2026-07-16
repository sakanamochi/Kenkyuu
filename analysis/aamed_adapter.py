"""公式pyAAMEDバインディングを共通評価形式へ変換する任意アダプター。"""

from __future__ import annotations

import cv2


def is_aamed_available() -> bool:
    try:
        import pyAAMED  # noqa: F401
    except ImportError:
        return False
    return True


def detect_aamed_candidates(image, settings: dict) -> list[dict]:
    try:
        from pyAAMED import pyAAMED
    except ImportError as error:
        raise RuntimeError(
            "AAMED比較には公式Li-Zhaoxi/AAMEDのpyAAMEDをビルドしてください。"
            "READMEの『任意: AAMED』を参照してください。"
        ) from error

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = pyAAMED(gray.shape[0] + 1, gray.shape[1] + 1)
    try:
        detector.setParameters(
            float(settings["theta_fsa"]),
            float(settings["length_fsa"]),
            float(settings["validation_threshold"]),
        )
        raw = detector.run_AAMED(gray)
    finally:
        detector.release()
    candidates = []
    for values in raw:
        ellipse = (
            (float(values[0]), float(values[1])),
            (float(values[2]), float(values[3])),
            float(values[4]),
        )
        candidates.append(
            {
                "ellipse": ellipse,
                "selection_score": float(values[5]),
                "aamed_score": float(values[5]),
            }
        )
    candidates.sort(key=lambda candidate: candidate["selection_score"], reverse=True)
    return candidates
