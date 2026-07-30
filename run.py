"""PAF内周リング研究を順番に実行する唯一の入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from paf_ring_detection.data import read_json
from paf_ring_detection.evaluate import evaluate
from paf_ring_detection.prepare import (
    prepare_diagnostic_data,
    prepare_training_data,
)
from paf_ring_detection.report import build_report
from paf_ring_detection.train import train


ROOT = Path(__file__).resolve().parent
ALL_STAGES = ["render", "prepare", "train", "evaluate", "report"]


def _render(config: dict) -> None:
    """Blenderをバックグラウンド起動して基礎CGとOOD CGを作る。"""
    for render_config in config["render_configs"]:
        command = [
            config["blender"],
            "--background",
            "--python",
            str(ROOT / "blender/generate_dataset.py"),
            "--",
            "--config",
            str(ROOT / render_config),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    # PowerShellでも日本語の進捗表示をUTF-8で統一する。
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="PAF内周リング研究を実行する")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
        help="実行する段階。例: --stages evaluate report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="評価画像数を制限する動作確認用オプション",
    )
    parser.add_argument("--epochs", type=int, help="学習epoch数を一時変更する")
    args = parser.parse_args()

    config = read_json(ROOT / "config/experiment.json")
    if "render" in args.stages:
        _render(config)
    if "prepare" in args.stages:
        prepare_training_data(config)
        prepare_diagnostic_data(config)
    if "train" in args.stages:
        train(config, args.epochs)
    if "evaluate" in args.stages:
        evaluate(config, args.limit)
    if "report" in args.stages:
        build_report(config)


if __name__ == "__main__":
    main()
