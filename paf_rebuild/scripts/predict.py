"""画像1枚の境界確率を保存する、利用者向けの推論入口。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import cv2
import numpy as np
import torch

from paf.core.images import read_image
from paf.inference.predictor import BoundaryPredictor


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
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or args.config.resolve().parent / settings["checkpoint"]
    torch.set_num_threads(2)
    predictor = BoundaryPredictor(checkpoint, args.device or settings["device"])
    result = predictor.predict(read_image(args.image))
    args.output.mkdir(parents=True, exist_ok=False)
    if result.probability is not None:
        np.save(args.output / "probability.npy", result.probability)
        preview = np.rint(result.probability * 255).astype(np.uint8)
        encoded, data = cv2.imencode(".png", preview)
        if not encoded:
            raise RuntimeError("境界確率のPNG変換に失敗しました。")
        (args.output / "probability.png").write_bytes(data.tobytes())
    metadata = {
        "status": result.status,
        "image": str(args.image.resolve()),
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "checkpoint": str(predictor.checkpoint),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "epoch": predictor.epoch,
        "device": str(predictor.device),
        "input_size": 256,
        "output_stage": "boundary_probability_only",
        "probability_file": "probability.npy" if result.probability is not None else None,
    }
    (args.output / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{result.status}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
