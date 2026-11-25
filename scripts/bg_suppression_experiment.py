"""オンライン背景ランダム化＋外周シルエット負例重みの1seed検証スクリプト。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.io import load_config, read_json, resolve_path, write_json
from defaults import DEFAULT_CONFIG_PATH
from training.bg_suppression import evaluate_black_and_earth_bg, train_bg_suppression_model


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description="背景ランダム化＋外周負例重みCNNの学習・比較評価")
    parser.add_argument("--stage", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bg-prob", type=float, default=0.25)
    parser.add_argument("--outer-weight", type=float, default=2.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config, base_dir = load_config(args.config)
    models_root = resolve_path(base_dir, config["paths"]["models_dir"])
    outputs_root = resolve_path(base_dir, config["paths"]["outputs_dir"])
    run_dir = models_root / "bg_outer_suppression" / "diversified" / str(args.seed)
    out_dir = outputs_root / "bg_outer_suppression"
    datasets = list(config["evaluation"]["test_datasets"])

    if args.stage in ("all", "train"):
        summary = train_bg_suppression_model(
            config,
            base_dir,
            source_name="diversified",
            seed=args.seed,
            output_dir=run_dir,
            epochs=args.epochs,
            bg_random_prob=args.bg_prob,
            outer_neg_weight=args.outer_weight,
            device_name=args.device,
            overwrite=args.overwrite,
        )
        print(
            f"[DONE-BG-TRAIN] best_epoch={summary['best_epoch']} "
            f"best_val_loss={summary['best_validation_loss']:.5f}",
            flush=True,
        )

    if args.stage in ("all", "evaluate"):
        res_new = evaluate_black_and_earth_bg(
            config,
            base_dir,
            checkpoint_path=run_dir / "best.pt",
            tag="bg_outer_suppression",
            output_dir=out_dir,
            datasets=datasets,
            device_name=args.device,
        )
        base_summary_path = out_dir / "summary_baseline_diversified.json"
        if base_summary_path.exists() and not args.overwrite:
            res_base = read_json(base_summary_path)
            print(f"[REUSE] {base_summary_path}", flush=True)
        else:
            res_base = evaluate_black_and_earth_bg(
                config,
                base_dir,
                checkpoint_path=models_root / "diversified" / str(args.seed) / "best.pt",
                tag="baseline_diversified",
                output_dir=out_dir,
                datasets=datasets,
                device_name=args.device,
            )
        write_json(out_dir / "comparison.json", {"baseline": res_base, "bg_outer_suppression": res_new})
        print(f"SAVED COMPARISON: {out_dir / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
