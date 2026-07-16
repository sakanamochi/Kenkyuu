from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAFロバスト性研究を再現可能な順序で一括実行する")
    parser.add_argument("--config", default="config/research_experiment.json")
    parser.add_argument("--blender", default=str(DEFAULT_BLENDER))
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-classic", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    blender = Path(args.blender)
    if not args.skip_render:
        if not blender.exists():
            raise FileNotFoundError(f"Blenderが見つかりません: {blender}")
        run(
            [
                str(blender),
                "--background",
                "--python",
                str(PROJECT_ROOT / "blender" / "generate_dataset.py"),
                "--",
                "--config",
                str(project_path(config["base_render_config"])),
            ]
        )
    if not args.skip_prepare:
        run([sys.executable, "-m", "paflab.prepare_stress_dataset", "--config", str(config_path)])
    if not args.skip_train:
        run([sys.executable, "-m", "paflab.train_cnn", "--config", str(config_path)])
    dataset_dir = project_path(config["stress_dataset_dir"])
    if not args.skip_classic:
        classic_methods = [
            method
            for method in config["evaluation"]["methods"]
            if method != "cnn_ransac"
        ]
        for method in classic_methods:
            run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "analysis" / "evaluate_dataset.py"),
                    "--dataset",
                    str(dataset_dir),
                    "--config",
                    str(PROJECT_ROOT / "config" / "baseline.json"),
                    "--method",
                    method,
                    "--split",
                    str(config["evaluation"]["split"]),
                ]
            )
    if not args.skip_cnn:
        run([sys.executable, "-m", "paflab.evaluate_cnn", "--config", str(config_path)])
    if not args.skip_summary:
        run([sys.executable, "-m", "paflab.reporting.summarize_robustness", "--config", str(config_path)])
        run([sys.executable, "-m", "paflab.reporting.validate_results", "--config", str(config_path)])
        run(
            [
                sys.executable,
                str(PROJECT_ROOT / "analysis" / "compare_methods.py"),
                "--dataset",
                str(dataset_dir),
                "--methods",
                *config["evaluation"]["methods"],
            ]
        )
        run([sys.executable, "-m", "paflab.reporting.build_report_artifact", "--config", str(config_path)])


if __name__ == "__main__":
    main()
