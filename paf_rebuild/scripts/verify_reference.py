"""固定した6モデルを再推論し、保存済みの旧結果と照合する入口。"""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch
from core.ellipse import EllipseSettings
from core.files import read_json
from evaluation.reference import verify_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=PROJECT, help="data・models・referencesのあるフォルダ")
    parser.add_argument("--output", type=Path, required=True, help="新しい照合結果フォルダ")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/reference.json")
    parser.add_argument("--predict-config", type=Path, default=PROJECT / "configs/predict.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", help="確認する学習seedを指定。省略時は全seed")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("出力先が存在します。別のフォルダ名を指定してください。")
    config = read_json(args.config)
    if args.seeds is not None:
        if not set(args.seeds).issubset(config["seeds"]) or len(set(args.seeds)) != len(args.seeds):
            parser.error("--seedsには設定内のseedを重複なく指定してください。")
        config = {**config, "seeds": args.seeds}
    prediction = read_json(args.predict_config)
    torch.set_num_threads(2)
    files = [args.config, args.predict_config, Path(__file__), *sorted((PROJECT / "src").rglob("*.py"))]
    verify_reference(args.reference_root.resolve(), args.output.resolve(), config,
                     EllipseSettings(**prediction["ellipse"]), args.device, files)


if __name__ == "__main__":
    main()
