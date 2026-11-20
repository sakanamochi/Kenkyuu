"""統一推論器（多様形状CNN・1194M CNN・Canny・Zhang 2019型）。

単体推論と一括評価の双方が共通の predict_on_256 を呼び出し、推論処理の重複をなくす。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from core.effects import image_to_tensor
from core.geometry import (
    Ellipse,
    draw_ellipse_overlay,
    ellipse_to_dict,
    ellipse_to_list,
    scale_ellipse,
    translate_ellipse,
)
from core.io import read_image, resolve_path, sha256_file, write_image, write_json
from core.model import load_checkpoint, resolve_device
from core.ransac import boundary_ellipse
from inference.canny import detect_canny_contour_ransac
from inference.zhang2019 import ZHANG_2019_REFERENCE, detect_zhang2019

CNN_METHODS = ("cnn_1194m", "cnn_diversified")
GEOMETRY_METHODS = ("canny", "zhang2019")
ALL_METHODS = CNN_METHODS + GEOMETRY_METHODS


def _pad_to_square(image: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """非正方形画像の上下または左右へ黒余白を加えて正方形化する。"""
    height, width = image.shape[:2]
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left
    padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    return padded, side, pad_left, pad_top


def render_probability_heatmap(probability: np.ndarray) -> np.ndarray:
    """境界確率マップ（0〜1）を視認用ヒートマップ画像（BGR uint8）へ変換する。"""
    clipped = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(clipped, cv2.COLORMAP_INFERNO)


def predict_on_256(
    image_256: np.ndarray,
    method: str,
    config: dict[str, Any],
    *,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    geometry_seed: int | None = None,
) -> tuple[Ellipse | None, np.ndarray | None, str]:
    """256pxモノクロ3ch画像に対して1方式の推論を実行し、(楕円, 境界確率, 状態)を返す。"""
    if bool(config["reject_uniform_images"]) and int(np.ptp(image_256)) == 0:
        return None, None, "insufficient_image_evidence"

    cv2.setRNGSeed(int(config["opencv_seed_per_image"]))
    if method in CNN_METHODS:
        if model is None or device is None:
            raise ValueError("CNN方式の推論にはmodelとdeviceが必要です。")
        tensor = image_to_tensor(image_256).to(device)[None]
        with torch.inference_mode():
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        ellipse = boundary_ellipse(
            probability,
            config["ransac_cnn"],
            threshold=float(config["probability_threshold"]),
        )
        status = "detected" if ellipse is not None else "no_supported_prediction"
        return ellipse, probability, status

    if method == "canny":
        seed = int(config["evaluation_seed"] if geometry_seed is None else geometry_seed)
        selected, _ = detect_canny_contour_ransac(
            image_256,
            config["canny_contour_ransac"],
            config["ransac_geometry"],
            seed,
        )
        ellipse = selected["ellipse"] if selected is not None else None
        return ellipse, None, ("detected" if ellipse is not None else "no_supported_prediction")

    if method == "zhang2019":
        selected, _ = detect_zhang2019(image_256, config["zhang2019"])
        ellipse = selected["ellipse"] if selected is not None else None
        return ellipse, None, ("detected" if ellipse is not None else "no_supported_prediction")

    raise ValueError(f"未対応の方式です: {method}")


class PAFDetector:
    """CNN二系統および幾何二方式を同じ入口から扱う統一検出器。"""

    def __init__(
        self,
        config: dict[str, Any],
        base_dir: Path,
        *,
        method: str = "cnn_diversified",
        seed: int = 20260907,
        checkpoint_path: Path | None = None,
        device_name: str = "auto",
    ) -> None:
        if method not in ALL_METHODS:
            raise ValueError(f"未対応のmethodです: {method}")
        self.config = config
        self.base_dir = base_dir
        self.method = method
        self.seed = int(seed)
        self.input_size = int(config["input_size"])

        self.device: torch.device | None = None
        self.model = None
        self.checkpoint_path: Path | None = None
        self.checkpoint_sha256: str | None = None
        self.selected_epoch: int | None = None

        if method in CNN_METHODS:
            self.device = resolve_device(device_name)
            if checkpoint_path is not None:
                resolved = Path(checkpoint_path).resolve()
            else:
                source = "diversified" if method == "cnn_diversified" else "1194m"
                models_dir = resolve_path(base_dir, config["paths"]["models_dir"])
                resolved = models_dir / source / str(self.seed) / "best.pt"
            self.checkpoint_path = resolved
            self.checkpoint_sha256 = sha256_file(resolved)
            self.model, ckpt = load_checkpoint(
                resolved,
                self.device,
                default_base_channels=int(config["training"]["base_channels"]),
            )
            self.selected_epoch = int(ckpt["epoch"]) if "epoch" in ckpt else None

    def predict_array(
        self,
        image: np.ndarray,
        *,
        pad_non_square: bool = False,
        geometry_seed: int | None = None,
    ) -> dict[str, Any]:
        """uint8グレーまたはBGR画像からPAF内周楕円を推定する。"""
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("入力はuint8配列にしてください。")
        if image.ndim not in (2, 3):
            raise ValueError("入力は2次元グレー画像または3次元BGR画像にしてください。")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError("カラー入力は3チャンネルBGR画像にしてください。")
        if min(image.shape[:2]) == 0:
            raise ValueError("空の画像は推論できません。")

        orig_h, orig_w = image.shape[:2]
        if orig_h != orig_w:
            if not pad_non_square:
                raise ValueError(
                    "入力は正方形画像にしてください（非正方形画像を余白付きで扱う場合は --pad-square を指定）。"
                )
            square, side, pad_left, pad_top = _pad_to_square(image)
        else:
            square, side, pad_left, pad_top = image, orig_h, 0, 0

        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if square.ndim == 3 else square
        resized_gray = cv2.resize(gray, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        input_256 = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)

        ellipse_256, probability, status = predict_on_256(
            input_256,
            self.method,
            self.config,
            model=self.model,
            device=self.device,
            geometry_seed=geometry_seed,
        )

        ellipse_orig: Ellipse | None = None
        if ellipse_256 is not None:
            scaled = scale_ellipse(ellipse_256, side / float(self.input_size))
            ellipse_orig = translate_ellipse(scaled, -pad_left, -pad_top)

        return {
            "status": status,
            "ellipse_256": ellipse_256,
            "ellipse": ellipse_orig,
            "probability": probability,
            "input_256": input_256,
            "original_shape": [int(orig_h), int(orig_w)],
            "padded_side": int(side),
            "pad_offset": [int(pad_left), int(pad_top)],
        }

    def predict_file(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        *,
        pad_non_square: bool = False,
    ) -> dict[str, Any]:
        """画像ファイルから推定し、JSON・重ね描き画像・境界確率を保存する。"""
        src_path = Path(image_path).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        image = read_image(src_path)
        prediction = self.predict_array(image, pad_non_square=pad_non_square)

        write_image(out_dir / "overlay.png", draw_ellipse_overlay(image, prediction["ellipse"]))
        write_image(out_dir / "input_256.png", prediction["input_256"])

        prob_files: dict[str, str] = {}
        if prediction["probability"] is not None:
            np.save(out_dir / "probability.npy", prediction["probability"])
            prob_u8 = np.rint(np.clip(prediction["probability"], 0.0, 1.0) * 255.0).astype(np.uint8)
            write_image(out_dir / "probability.png", prob_u8)
            write_image(
                out_dir / "probability_heatmap.png",
                render_probability_heatmap(prediction["probability"]),
            )
            prob_files = {
                "probability_npy": "probability.npy",
                "probability_png": "probability.png",
                "probability_heatmap_png": "probability_heatmap.png",
            }

        payload: dict[str, Any] = {
            "method": self.method,
            "seed": self.seed if self.model is not None else None,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "selected_epoch": self.selected_epoch,
            "input_image": str(src_path),
            "input_image_sha256": sha256_file(src_path),
            "original_shape": prediction["original_shape"],
            "padded_side": prediction["padded_side"],
            "pad_offset": prediction["pad_offset"],
            "status": prediction["status"],
            "ellipse": ellipse_to_dict(prediction["ellipse"]),
            "ellipse_opencv": ellipse_to_list(prediction["ellipse"]),
            "ellipse_256": ellipse_to_dict(prediction["ellipse_256"]),
            "ellipse_256_opencv": ellipse_to_list(prediction["ellipse_256"]),
            "files": {
                "overlay_png": "overlay.png",
                "input_256_png": "input_256.png",
                **prob_files,
            },
        }
        if self.method == "zhang2019":
            payload["reference"] = ZHANG_2019_REFERENCE

        write_json(out_dir / "result.json", payload)
        return payload
