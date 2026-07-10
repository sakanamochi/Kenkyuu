import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PAFのCG生成から古典法評価までを一括実行する"
    )
    parser.add_argument(
        "--experiment-config",
        default="config/factorial_experiment.json",
    )
    parser.add_argument(
        "--baseline-config",
        default="config/baseline.json",
    )
    parser.add_argument("--blender", default=str(DEFAULT_BLENDER))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command))
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    if args.render_only and args.evaluate_only:
        raise ValueError("--render-onlyと--evaluate-onlyは同時に指定できません")

    experiment_config_path = resolve_project_path(args.experiment_config)
    experiment_config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(experiment_config["output_dir"])

    if not args.evaluate_only:
        blender_path = Path(args.blender)
        if not blender_path.exists():
            raise FileNotFoundError(f"Blenderが見つかりません: {blender_path}")
        run(
            [
                str(blender_path),
                "--background",
                "--python",
                str(PROJECT_ROOT / "blender" / "generate_dataset.py"),
                "--",
                "--config",
                str(experiment_config_path),
            ]
        )

    if not args.render_only:
        for method in ("contour_fit", "canny_ransac"):
            run(
                [
                    sys.executable,
                    "-B",
                    str(PROJECT_ROOT / "analysis" / "evaluate_dataset.py"),
                    "--dataset",
                    str(dataset_dir),
                    "--config",
                    str(resolve_project_path(args.baseline_config)),
                    "--method",
                    method,
                ]
            )
        run(
            [
                sys.executable,
                "-B",
                str(PROJECT_ROOT / "analysis" / "compare_methods.py"),
                "--dataset",
                str(dataset_dir),
            ]
        )


if __name__ == "__main__":
    main()
