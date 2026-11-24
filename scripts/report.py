"""集計用スクリプト：評価結果からREPORT.md・summary.json・CSV・比較図を生成する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defaults import DEFAULT_CONFIG_PATH
from core.io import load_config, resolve_path
from evaluation.reporter import generate_report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="評価結果から集計表・REPORT.md・比較図を生成します。")
    parser.add_argument("--output", type=Path, default=None, help="評価結果ディレクトリ（既定: outputs/evaluation）")
    parser.add_argument("--datasets", nargs="+", default=None, help="集計対象データセット（省略時はevaluation_index.jsonまたは既定）")
    parser.add_argument("--no-figures", action="store_true", help="グラフ画像の生成を省略する")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="設定JSONのパス")
    args = parser.parse_args()

    config, base_dir = load_config(args.config)
    output_dir = (
        resolve_path(base_dir, config["paths"]["outputs_dir"]) / "evaluation"
        if args.output is None
        else args.output.resolve()
    )
    summary = generate_report(config, output_dir, datasets=args.datasets, with_figures=not args.no_figures)
    print(f"REPORT: {output_dir / 'REPORT.md'}")
    print(f"SUMMARY: {output_dir / 'summary.json'}")
    for fig in summary["figures"]:
        print(f"FIGURE: {fig}")


if __name__ == "__main__":
    main()
