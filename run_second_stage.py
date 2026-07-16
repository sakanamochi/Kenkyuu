from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")


def path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def classic(dataset: Path, method: str, split: str, name: str, *extra: str) -> None:
    run(
        [
            sys.executable,
            str(ROOT / "analysis/evaluate_dataset.py"),
            "--dataset",
            str(dataset),
            "--method",
            method,
            "--split",
            split,
            "--results-name",
            name,
            "--no-images",
            *extra,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PAF第2段階検証を依存順に一括実行する")
    parser.add_argument("--config", default="config/research_second_stage.json")
    parser.add_argument("--blender", default=str(DEFAULT_BLENDER))
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-classic", action="store_true")
    parser.add_argument(
        "--include-classic-ablations",
        action="store_true",
        help="主ベースラインのZhang型に加えて旧Contour/Canny方式も再評価する",
    )
    parser.add_argument(
        "--include-ransac-audit",
        action="store_true",
        help="旧Canny方式の内周候補選択アブレーションも再実行する",
    )
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()

    config_path = path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_config = path(config["base_config"])
    ood = path(config["ood_dataset_dir"])
    diagnostics = path(config["diagnostics"]["dataset_dir"])

    if not args.skip_render:
        blender = Path(args.blender)
        if not blender.exists():
            raise FileNotFoundError(blender)
        run(
            [
                str(blender),
                "--background",
                "--python",
                str(ROOT / "blender/generate_dataset.py"),
                "--",
                "--config",
                str(path(config["ood_render_config"])),
            ]
        )
    if not args.skip_prepare:
        run(
            [
                sys.executable,
                "-m",
                "paflab.prepare_diagnostic_dataset",
                "--config",
                str(config_path),
            ]
        )
    if not args.skip_models:
        run(
            [
                sys.executable,
                "-m",
                "paflab.experiments.run_model_ablation",
                "--config",
                str(config_path),
            ]
        )
    if not args.skip_classic:
        classic(
            ood,
            "zhang2019_arc_reproduction",
            "ood_test",
            "zhang2019_arc_ood",
            "--resume",
        )
        classic(
            diagnostics,
            "zhang2019_arc_reproduction",
            "diagnostic_test",
            "zhang2019_arc_diagnostic",
            "--resume",
        )
        if args.include_classic_ablations:
            classic(ood, "contour_fit", "ood_test", "contour_fit_ood")
            classic(
                ood,
                "canny_ransac",
                "ood_test",
                "canny_ransac_ood",
                "--paired-ransac-seed",
                "--resume",
            )
            classic(
                diagnostics,
                "contour_fit",
                "diagnostic_test",
                "contour_fit_diagnostic",
            )
            classic(
                diagnostics,
                "canny_ransac",
                "diagnostic_test",
                "canny_ransac_diagnostic",
                "--paired-ransac-seed",
                "--resume",
            )
    if args.include_ransac_audit:
        source = ROOT / "output/datasets/paf_robustness_v1"
        for split in ("train", "validation", "test"):
            classic(
                source,
                "canny_ransac",
                split,
                "canny_ransac_tuning",
                "--degradation",
                "clean",
                "--resume",
            )
        run(
            [
                sys.executable,
                "-m",
                "paflab.experiments.analyze_ransac_selection",
                "--dataset",
                str(source),
                "--method",
                "canny_ransac_tuning",
                "--output",
                str(path(config["artifacts_dir"]) / "ransac_audit/clean"),
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "paflab.experiments.tune_inner_pair_selector",
            ]
        )
        classic(
            ood,
            "canny_ransac_inner_pair",
            "ood_test",
            "canny_ransac_inner_pair_ood",
            "--reuse-candidates-from",
            "canny_ransac_ood",
        )
        classic(
            diagnostics,
            "canny_ransac_inner_pair",
            "diagnostic_test",
            "canny_ransac_inner_pair_diagnostic",
            "--reuse-candidates-from",
            "canny_ransac_diagnostic",
        )
    if not args.skip_cnn:
        run(
            [
                sys.executable,
                "-m",
                "paflab.evaluate_cnn",
                "--config",
                str(base_config),
                "--dataset",
                str(ood),
                "--split",
                "ood_test",
                "--results-name",
                "cnn_ransac_support_ood",
                "--paired-ransac-seed",
                "--no-images",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "paflab.evaluate_cnn",
                "--config",
                str(base_config),
                "--dataset",
                str(diagnostics),
                "--split",
                "diagnostic_test",
                "--results-name",
                "cnn_ransac_support_diagnostic",
                "--paired-ransac-seed",
                "--no-images",
            ]
        )
        run([sys.executable, "-m", "paflab.experiments.benchmark_models"])
    if not args.skip_summary:
        run([sys.executable, "-m", "paflab.reporting.validate_second_stage"])
        run(
            [
                sys.executable,
                "-m",
                "paflab.reporting.summarize_second_stage",
                "--config",
                str(config_path),
            ]
        )
        run([sys.executable, "-m", "paflab.reporting.build_second_stage_report"])
        run([sys.executable, "-m", "paflab.reporting.build_summary_figures"])


if __name__ == "__main__":
    main()
