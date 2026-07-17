from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_success(row: dict) -> float:
    return float(str(row["top1_matches_ground_truth"]).lower() == "true")


def cluster_interval(rows: list[dict], *, seed: int = 20260716) -> tuple[float, float]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[row["camera_id"]].append(is_success(row))
    cluster_means = np.asarray(
        [np.mean(values) for values in clusters.values()], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    boot = cluster_means[
        rng.integers(0, len(cluster_means), size=(5000, len(cluster_means)))
    ].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def grouped_rows(
    method_paths: dict[str, Path], group_fields: tuple[str, ...]
) -> list[dict]:
    output = []
    for method, path in method_paths.items():
        if not path.exists():
            continue
        groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
        for row in read_csv(path):
            groups[tuple(row.get(field, "") for field in group_fields)].append(row)
        for keys, rows in groups.items():
            low, high = cluster_interval(rows)
            output.append(
                {
                    "method": method,
                    **dict(zip(group_fields, keys)),
                    "sample_count": len(rows),
                    "camera_cluster_count": len({row["camera_id"] for row in rows}),
                    "detected_count": sum(row["status"] == "detected" for row in rows),
                    "success_count": int(sum(is_success(row) for row in rows)),
                    "success_rate": float(np.mean([is_success(row) for row in rows])),
                    "cluster_ci95_low": low,
                    "cluster_ci95_high": high,
                }
            )
    return output


def model_ablation(root: Path) -> tuple[list[dict], list[dict], dict]:
    run_rows = []
    curve_rows = []
    for run_dir in sorted(root.glob("c*_s*")):
        metadata_path = run_dir / "model_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
        best = min(history, key=lambda row: row["validation_loss"])
        terminal = history[-1]
        evaluations = {}
        for name in ("validation", "ood_test"):
            path = run_dir / name / "summary.json"
            if path.exists():
                evaluations[name] = json.loads(path.read_text(encoding="utf-8"))[
                    "top1_match_rate"
                ]
        run_rows.append(
            {
                "run_id": run_dir.name,
                "base_channels": metadata["base_channels"],
                "parameter_count": metadata["parameter_count"],
                "seed": metadata["seed"],
                "best_epoch": best["epoch"],
                "best_validation_loss": best["validation_loss"],
                "train_loss_at_best": best["train_loss"],
                "loss_gap_at_best": best["validation_loss"] - best["train_loss"],
                "terminal_validation_loss": terminal["validation_loss"],
                "post_best_worsening": terminal["validation_loss"]
                - best["validation_loss"],
                "validation_success_rate": evaluations.get("validation"),
                "ood_success_rate": evaluations.get("ood_test"),
                "training_seconds": metadata["total_training_seconds"],
            }
        )
        for row in history:
            curve_rows.append(
                {
                    "run_id": run_dir.name,
                    "base_channels": metadata["base_channels"],
                    "seed": metadata["seed"],
                    **row,
                }
            )

    aggregate = []
    by_width: dict[int, list[dict]] = defaultdict(list)
    for row in run_rows:
        by_width[int(row["base_channels"])].append(row)
    for width, rows in sorted(by_width.items()):
        record = {
            "base_channels": width,
            "parameter_count": rows[0]["parameter_count"],
            "seed_count": len(rows),
        }
        for key in ("validation_success_rate", "ood_success_rate"):
            values = [float(row[key]) for row in rows if row[key] is not None]
            record[f"mean_{key}"] = float(np.mean(values)) if values else None
            record[f"min_{key}"] = float(np.min(values)) if values else None
            record[f"max_{key}"] = float(np.max(values)) if values else None
        record["mean_best_epoch"] = float(np.mean([row["best_epoch"] for row in rows]))
        record["mean_loss_gap_at_best"] = float(
            np.mean([row["loss_gap_at_best"] for row in rows])
        )
        record["mean_post_best_worsening"] = float(
            np.mean([row["post_best_worsening"] for row in rows])
        )
        aggregate.append(record)

    minimum_width = None
    if aggregate:
        reference = next(row for row in aggregate if row["base_channels"] == 16)
        for row in aggregate:
            if all(
                row[f"mean_{metric}"] >= reference[f"mean_{metric}"] - 0.03
                for metric in ("validation_success_rate", "ood_success_rate")
            ):
                minimum_width = row["base_channels"]
                break
    decision = {
        "rule": "smallest width within 3 percentage points of width 16 mean on both validation and OOD",
        "minimum_noninferior_base_channels": minimum_width,
    }
    overfitting = {
        "run_count": len(run_rows),
        "best_at_final_epoch_count": sum(
            row["best_epoch"] == 18 for row in run_rows
        ),
        "median_best_epoch": float(np.median([row["best_epoch"] for row in run_rows])),
        "mean_loss_gap_at_best": float(
            np.mean([row["loss_gap_at_best"] for row in run_rows])
        ),
        "maximum_post_best_worsening": float(
            np.max([row["post_best_worsening"] for row in run_rows])
        ),
    }
    return run_rows, curve_rows, {
        "aggregate": aggregate,
        "decision": decision,
        "overfitting": overfitting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="第2段階検証を統計的に集約する")
    parser.add_argument("--config", default="config/research_second_stage.json")
    args = parser.parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    output = project_path(config["artifacts_dir"])
    output.mkdir(parents=True, exist_ok=True)
    ood_dataset = project_path(config["ood_dataset_dir"])
    diagnostics = project_path(config["diagnostics"]["dataset_dir"])

    ood_paths = {
        "zhang2019_arc_reproduction": ood_dataset
        / "results/zhang2019_arc_ood/summary.csv",
        "cnn_ransac_support": ood_dataset
        / "results/cnn_ransac_support_ood/summary.csv",
        "contour_fit": ood_dataset / "results/contour_fit_ood/summary.csv",
        "canny_ransac_inner_pair": ood_dataset
        / "results/canny_ransac_inner_pair_ood/summary.csv",
        "cnn_ransac": ood_dataset / "results/cnn_ransac_v3_ood/summary.csv",
    }
    diagnostic_paths = {
        "zhang2019_arc_reproduction": diagnostics
        / "results/zhang2019_arc_diagnostic/summary.csv",
        "cnn_ransac_support": diagnostics
        / "results/cnn_ransac_support_diagnostic/summary.csv",
        "contour_fit": diagnostics / "results/contour_fit_diagnostic/summary.csv",
        "canny_ransac_inner_pair": diagnostics
        / "results/canny_ransac_inner_pair_diagnostic/summary.csv",
        "canny_global_ransac": diagnostics
        / "results/canny_global_ransac_diagnostic/summary.csv",
        "cnn_ransac": diagnostics / "results/cnn_ransac_v3_diagnostic/summary.csv",
    }
    ood_background = grouped_rows(ood_paths, ("background",))
    ood_tilt = grouped_rows(ood_paths, ("camera_tilt_deg",))
    ood_light = grouped_rows(
        ood_paths, ("light_tilt_deg", "light_azimuth_deg", "light_energy")
    )
    diagnostic_curves = grouped_rows(
        diagnostic_paths, ("degradation", "severity")
    )
    write_csv(output / "ood_by_background.csv", ood_background)
    write_csv(output / "ood_by_camera_tilt.csv", ood_tilt)
    write_csv(output / "ood_by_light.csv", ood_light)
    write_csv(output / "diagnostic_curves.csv", diagnostic_curves)

    run_rows, training_curves, ablation = model_ablation(output / "model_ablation")
    write_csv(output / "model_ablation_runs.csv", run_rows)
    write_csv(output / "training_curves.csv", training_curves)
    write_csv(output / "model_ablation_aggregate.csv", ablation["aggregate"])
    summary = {
        "experiment_id": config["experiment_id"],
        "primary_comparison": {
            "classical_method": "zhang2019_arc_reproduction",
            "learned_method": "cnn_ransac_support",
            "reference_doi": "10.3390/s19235243",
            "implementation_note": "Zhang et al. (2019)の処理構成を参考にした再現実装",
        },
        "success_definition": "top-1 ellipse IoU >= 0.80",
        "uncertainty": "camera_id cluster bootstrap, 5000 resamples, deterministic seed 20260716; effect/background comparisons use paired RANSAC seeds",
        "model_ablation": ablation,
        "available_methods": {
            "ood": [name for name, path in ood_paths.items() if path.exists()],
            "diagnostic": [
                name for name, path in diagnostic_paths.items() if path.exists()
            ],
        },
        "overall_rates": {
            "ood": {
                name: json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))[
                    "top1_match_rate"
                ]
                for name, path in ood_paths.items()
                if path.exists()
            },
            "diagnostic": {
                name: json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))[
                    "top1_match_rate"
                ]
                for name, path in diagnostic_paths.items()
                if path.exists()
            },
        },
    }
    audit_path = output / "ransac_audit/clean/summary.json"
    pair_path = output / "ransac_audit/inner_pair/selection_result.json"
    if audit_path.exists() and pair_path.exists():
        summary["ransac_selection"] = {
            "candidate_audit": json.loads(audit_path.read_text(encoding="utf-8")),
            "inner_pair": json.loads(pair_path.read_text(encoding="utf-8")),
        }
    (output / "second_stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
