"""CNNモデル依存性の専用実験を段階別に実行する。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from paf_ring_detection.data import read_json  # noqa: E402

from evaluate import evaluate, evaluate_zhang2019  # noqa: E402
from generate_parametric_paf import generate  # noqa: E402
from report import build_report  # noqa: E402


ALL_STAGES = ["model", "render", "evaluate", "report"]


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _render(experiment: dict, limit: int | None) -> None:
    reference_config = read_json(_project_path(experiment["reference_config"]))
    command = [
        reference_config["blender"],
        "--background",
        "--python",
        str(PROJECT_ROOT / "blender/generate_dataset.py"),
        "--",
        "--config",
        str(_project_path(experiment["render_config"])),
    ]
    if limit is not None:
        command.extend(["--limit-samples", str(limit)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    # PowerShellでも日本語の進捗をUTF-8で表示する。
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="CNNモデル依存性を評価する")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
    )
    parser.add_argument("--limit-render", type=int)
    parser.add_argument("--limit-evaluate", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cnn", "zhang2019"],
        default=["cnn", "zhang2019"],
        help="evaluate段階で実行する方式",
    )
    parser.add_argument(
        "--experiment-config",
        default="model_generalization_study/config/experiment.json",
        help="プロジェクトルート基準の専用実験設定JSON",
    )
    args = parser.parse_args()

    experiment = read_json(_project_path(args.experiment_config))
    if "model" in args.stages:
        metadata = generate(
            _project_path(experiment["model_config"]),
            _project_path(experiment["generated_model"]),
        )
        print(
            f"モデル生成: {metadata['model_id']} "
            f"({metadata['vertex_count']}頂点)"
        )
    if "render" in args.stages:
        _render(experiment, args.limit_render)
    if "evaluate" in args.stages:
        if "cnn" in args.methods:
            print(evaluate(experiment, args.limit_evaluate))
        if "zhang2019" in args.methods:
            print(evaluate_zhang2019(experiment, args.limit_evaluate))
    if "report" in args.stages:
        summary = build_report(experiment)
        print(summary["paired_comparison"])


if __name__ == "__main__":
    main()
