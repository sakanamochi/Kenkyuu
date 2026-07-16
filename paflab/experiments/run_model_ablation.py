from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN幅とseedのアブレーションを実行する")
    parser.add_argument("--config", default="config/research_second_stage.json")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    suite = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    base_config_path = project_path(suite["base_config"])
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    root = project_path(suite["artifacts_dir"]) / "model_ablation"
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for channels in suite["model_ablation"]["base_channels"]:
        for seed in suite["model_ablation"]["seeds"]:
            run_id = f"c{int(channels):02d}_s{int(seed)}"
            model_dir = root / run_id
            checkpoint = model_dir / "cnn_best.pt"
            if not args.skip_training and not checkpoint.exists():
                run(
                    [
                        sys.executable,
                        "-m",
                        "paflab.train_cnn",
                        "--config",
                        str(base_config_path),
                        "--epochs",
                        str(suite["model_ablation"]["epochs"]),
                        "--seed",
                        str(seed),
                        "--base-channels",
                        str(channels),
                        "--artifacts-dir",
                        str(model_dir),
                        "--cache-data",
                    ]
                )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

            evaluations = {
                "validation": (project_path(base["stress_dataset_dir"]), "validation"),
                "ood_test": (project_path(suite["ood_dataset_dir"]), "ood_test"),
            }
            for evaluation_name, (dataset, split) in evaluations.items():
                result_dir = model_dir / evaluation_name
                summary_path = result_dir / "summary.json"
                if not args.skip_evaluation and not summary_path.exists():
                    run(
                        [
                            sys.executable,
                            "-m",
                            "paflab.evaluate_cnn",
                            "--config",
                            str(base_config_path),
                            "--dataset",
                            str(dataset),
                            "--split",
                            split,
                            "--checkpoint",
                            str(checkpoint),
                            "--results-name",
                            run_id,
                            "--results-dir",
                            str(result_dir),
                            "--no-images",
                        ]
                    )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows.append({"run_id": run_id, "evaluation": evaluation_name, **summary})

    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(root / "summary.csv")


if __name__ == "__main__":
    main()
