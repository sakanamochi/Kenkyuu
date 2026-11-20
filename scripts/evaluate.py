"""評価用スクリプト：4方式×2評価集合×13条件の評価および確定参照結果との照合。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.io import load_config, resolve_path
from defaults import DEFAULT_CONFIG_PATH
from evaluation.evaluator import run_evaluation, verify_against_references
from inference.predictor import ALL_METHODS


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(
        description="固定テスト集合（全13条件）で4方式を評価し、必要に応じて確定参照結果と照合します。"
    )
    parser.add_argument("--methods", nargs="+", choices=list(ALL_METHODS), default=None, help="評価する方式")
    parser.add_argument("--datasets", nargs="+", choices=["1194m", "diversified"], default=None, help="評価対象データセット")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="CNNの評価seed（省略時は全3seed）")
    parser.add_argument("--output", type=Path, default=None, help="出力先ディレクトリ（既定: outputs/evaluation）")
    parser.add_argument("--models-dir", type=Path, default=None, help="CNN重みの格納ディレクトリ")
    parser.add_argument("--device", default="auto", help="実行デバイス（auto / cuda / cpu）")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に先頭N件だけ評価する")
    parser.add_argument("--verify-reference", action="store_true", help="references/の確定結果と全件照合する")
    parser.add_argument("--overwrite", action="store_true", help="既存の予測JSONを上書きする")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="設定JSONのパス")
    args = parser.parse_args()

    config, base_dir = load_config(args.config)
    output_dir = (
        resolve_path(base_dir, config["paths"]["outputs_dir"]) / "evaluation"
        if args.output is None
        else args.output.resolve()
    )

    run_evaluation(
        config,
        base_dir,
        output_dir=output_dir,
        methods=args.methods,
        datasets=args.datasets,
        seeds=args.seeds,
        models_dir=args.models_dir,
        device_name=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    if args.verify_reference:
        result = verify_against_references(config, base_dir, output_dir)
        print(
            f"[VERIFY] checked={result['checked_predictions']} "
            f"mismatches={result['ellipse_mismatches']}/{result['iou_mismatches']}/{result['success_mismatches']} "
            f"hash_mismatches={result['image_hash_mismatches']} "
            f"verified_exact={result['verified_exact']}",
            flush=True,
        )
        if not result["verified_exact"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
