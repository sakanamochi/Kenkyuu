"""推論用スクリプト：画像1枚からPAF内周楕円を検出する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defaults import DEFAULT_CONFIG_PATH
from core.io import load_config
from inference.predictor import ALL_METHODS, PAFDetector


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)

    parser = argparse.ArgumentParser(description="画像1枚からPAF内周楕円を検出します。")
    parser.add_argument("--image", type=Path, required=True, help="入力画像ファイルのパス")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predict",
        help="結果（result.json / overlay.png / probability.*）の出力先ディレクトリ",
    )
    parser.add_argument(
        "--method",
        choices=list(ALL_METHODS),
        default="cnn_diversified",
        help="検出方式（既定: cnn_diversified）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260907,
        help="CNN方式で読み込む学習seed（既定: 20260907）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="任意の学習済み重み(.pt)を直接指定する場合のパス",
    )
    parser.add_argument(
        "--pad-square",
        action="store_true",
        help="非正方形画像の上下または左右へ黒余白を加えて正方形化してから推論する",
    )
    parser.add_argument("--device", default="auto", help="実行デバイス（auto / cuda / cpu）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="設定JSONのパス")
    args = parser.parse_args()

    config, base_dir = load_config(args.config)
    detector = PAFDetector(
        config,
        base_dir,
        method=args.method,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
    )
    result = detector.predict_file(args.image, args.output, pad_non_square=args.pad_square)
    print(f"status: {result['status']}")
    print(f"ellipse: {result['ellipse']}")
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
