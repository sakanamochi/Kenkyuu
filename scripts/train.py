"""学習用スクリプト：可視内周境界CNN（多様形状 / 1194M）を学習する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defaults import DEFAULT_CONFIG_PATH
from core.io import load_config, resolve_path
from training.trainer import train_model


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)

    parser = argparse.ArgumentParser(description="Tiny U-Net16を可視内周境界教師で学習します。")
    parser.add_argument(
        "--source",
        choices=["diversified", "1194m"],
        default="diversified",
        help="学習データ源（diversified: 多様形状4ビュー / 1194m: 1194M元材質2ビュー）",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="学習seed（省略時は全3seed）")
    parser.add_argument("--epochs", type=int, default=None, help="学習epoch数（省略時は60epoch）")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="学習結果の出力ベースディレクトリ（既定: outputs/trained_models）",
    )
    parser.add_argument("--device", default="auto", help="実行デバイス（auto / cuda / cpu）")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に先頭N件へ制限する")
    parser.add_argument("--overwrite", action="store_true", help="既存の完了済みフォルダを上書きする")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="設定JSONのパス")
    args = parser.parse_args()

    config, base_dir = load_config(args.config)
    seeds = [int(s) for s in (config["seeds"] if args.seeds is None else args.seeds)]
    base_output = (
        resolve_path(base_dir, config["paths"]["outputs_dir"]) / "trained_models"
        if args.output is None
        else args.output.resolve()
    )

    for seed in seeds:
        run_dir = base_output / args.source / str(seed)
        summary = train_model(
            config,
            base_dir,
            source_name=args.source,
            seed=seed,
            output_dir=run_dir,
            epochs=args.epochs,
            device_name=args.device,
            limit_samples=args.limit,
            overwrite=args.overwrite,
        )
        print(
            f"[DONE] source={args.source} seed={seed} "
            f"best_epoch={summary['best_epoch']} "
            f"best_val_loss={summary['best_validation_loss']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
