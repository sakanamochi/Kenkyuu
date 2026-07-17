from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def project_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def evaluate_classic(
    dataset: Path,
    split: str,
    method: str,
    results_name: str,
    baseline_config: Path,
    *extra: str,
) -> None:
    run(
        [
            sys.executable,
            str(ROOT / "analysis/evaluate_dataset.py"),
            "--dataset",
            str(dataset),
            "--config",
            str(baseline_config),
            "--method",
            method,
            "--split",
            split,
            "--results-name",
            results_name,
            "--no-images",
            *extra,
        ]
    )


def configured_methods(config: dict, include_ablations: bool) -> list[str]:
    methods = list(config["primary_methods"])
    if include_ablations:
        methods = [*config.get("ablation_methods", []), *methods]
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(
        description="文献ベースの古典法とCNNを同一条件で一括比較する"
    )
    parser.add_argument(
        "--config", default="config/literature_baseline_comparison.json"
    )
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="主比較に加えてCanny + 輪郭別RANSAC等も再評価・表示する",
    )
    parser.add_argument("--with-aamed", action="store_true")
    parser.add_argument("--rerun-cnn", action="store_true")
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    baseline_config = project_path(config["baseline_config"])
    cnn_config = project_path(config["cnn_config"])
    methods_to_run = configured_methods(config, args.include_ablations)

    if not args.figures_only:
        for dataset_config in config["datasets"]:
            dataset = project_path(dataset_config["dataset_dir"])
            split = dataset_config["split"]
            results = dataset_config["results"]
            if "zhang2019_arc_reproduction" in methods_to_run:
                evaluate_classic(
                    dataset,
                    split,
                    "zhang2019_arc_reproduction",
                    results["zhang2019_arc_reproduction"],
                    baseline_config,
                    "--resume",
                )
            if "kojima2021_fornaciari_reproduction" in methods_to_run:
                evaluate_classic(
                    dataset,
                    split,
                    "kojima2021_fornaciari_reproduction",
                    results["kojima2021_fornaciari_reproduction"],
                    baseline_config,
                    "--resume",
                )
            if "contour_fit" in methods_to_run:
                evaluate_classic(
                    dataset,
                    split,
                    "contour_fit",
                    results["contour_fit"],
                    baseline_config,
                    "--resume",
                )
            if "canny_ransac_inner_pair" in methods_to_run:
                evaluate_classic(
                    dataset,
                    split,
                    "canny_ransac_inner_pair",
                    results["canny_ransac_inner_pair"],
                    baseline_config,
                    "--paired-ransac-seed",
                    "--resume",
                )
            cnn_summary = dataset / "results" / results["cnn_ransac"] / "summary.json"
            if args.rerun_cnn or not cnn_summary.exists():
                run(
                    [
                        sys.executable,
                        "-m",
                        "paflab.evaluate_cnn",
                        "--config",
                        str(cnn_config),
                        "--dataset",
                        str(dataset),
                        "--split",
                        split,
                        "--results-name",
                        results["cnn_ransac"],
                        "--paired-ransac-seed",
                        "--no-images",
                    ]
                )
            methods = [results[method] for method in methods_to_run]
            if args.with_aamed:
                evaluate_classic(
                    dataset,
                    split,
                    "aamed",
                    results["aamed"],
                    baseline_config,
                    "--resume",
                )
                methods.append(results["aamed"])
            run(
                [
                    sys.executable,
                    str(ROOT / "analysis/compare_methods.py"),
                    "--dataset",
                    str(dataset),
                    "--methods",
                    *methods,
                ]
            )

    run(
        [
            sys.executable,
            "-m",
            "paflab.reporting.build_literature_comparison",
            "--config",
            str(config_path),
            *(
                ["--include-ablations"]
                if args.include_ablations
                else []
            ),
        ]
    )


if __name__ == "__main__":
    main()
