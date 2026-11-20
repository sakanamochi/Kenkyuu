"""画像1枚のPAF内周楕円・境界確率・重ね描きを保存する推論入口。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import cv2
import numpy as np
import torch

from core.images import read_image, write_png
from core.files import read_json, sha
from core.ellipse import EllipseSettings
from inference.predictor import BoundaryPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/predict.json")
    parser.add_argument("--image", type=Path, required=True, help="正方形のグレーまたはBGR画像")
    parser.add_argument("--output", type=Path, required=True, help="新しく作る出力フォルダ")
    parser.add_argument("--checkpoint", type=Path, help="設定のモデルを変更する場合の重み")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("出力先が存在します。別のフォルダ名を指定してください。")
    settings = read_json(args.config)
    checkpoint = args.checkpoint or args.config.resolve().parent / settings["checkpoint"]
    torch.set_num_threads(2)
    predictor = BoundaryPredictor(checkpoint, args.device or settings["device"],
                                  EllipseSettings(**settings.get("ellipse", {})))
    image = read_image(args.image)
    result = predictor.predict(image)
    args.output.mkdir(parents=True, exist_ok=False)
    if result.probability is not None:
        np.save(args.output / "probability.npy", result.probability)
        preview = np.uint8(result.probability * 255)
        write_png(args.output / "probability.png", preview)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if result.ellipse is not None:
        cv2.ellipse(overlay, result.ellipse, (255, 0, 255), 2, cv2.LINE_AA)
    write_png(args.output / "overlay.png", overlay)
    metadata = {
        "status": result.status,
        "image": str(args.image.resolve()),
        "image_sha256": sha(args.image),
        "checkpoint": str(predictor.checkpoint),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "epoch": predictor.epoch,
        "device": str(predictor.device),
        "input_size": 256,
        "ellipse": result.ellipse,
        "ellipse_256": result.ellipse_256,
        "ellipse_format": "[[center_x, center_y], [axis_1_full_length, axis_2_full_length], angle_degrees]",
        "ellipse_settings": asdict(predictor.settings),
        "probability_file": "probability.npy" if result.probability is not None else None,
    }
    (args.output / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{result.status}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
