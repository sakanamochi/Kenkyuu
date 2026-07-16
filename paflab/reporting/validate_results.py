from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def as_bool(value) -> bool:
    return str(value).lower() in ("true", "1", "yes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最終実験結果の完全性と集計を検算する")
    parser.add_argument("--config", default="config/research_experiment.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    dataset = project_path(config["stress_dataset_dir"])
    artifacts = project_path(config["artifacts_dir"])
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    test_samples = {sample["sample_id"]: sample for sample in manifest["samples"] if sample["split"] == "test"}
    expected_severities = {0.0, *map(float, config["degradations"]["test_severities"])}
    checks = []
    failures = []

    assignments = manifest["split_policy"]["group_assignments"]
    split_groups = {
        split: {group for group, assigned in assignments.items() if assigned == split}
        for split in ("train", "validation", "test")
    }
    disjoint = not (
        split_groups["train"] & split_groups["validation"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["validation"] & split_groups["test"]
    )
    checks.append({"check": "camera_group_splits_are_disjoint", "passed": disjoint})
    if not disjoint:
        failures.append("camera_group_splits_are_disjoint")

    for sample in test_samples.values():
        if not (dataset / sample["image"]).exists() or not (dataset / sample["label"]).exists():
            failures.append(f"missing_artifact:{sample['sample_id']}")
            break
    checks.append(
        {"check": "all_test_images_and_labels_exist", "passed": not any(x.startswith("missing_artifact") for x in failures)}
    )

    for method in config["evaluation"]["methods"]:
        result_dir = dataset / "results" / method
        with (result_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        result_ids = [row["sample_id"] for row in rows]
        exact_ids = set(result_ids) == set(test_samples) and len(result_ids) == len(set(result_ids))
        checks.append({"check": f"{method}_sample_ids_match_manifest", "passed": exact_ids, "count": len(rows)})
        if not exact_ids:
            failures.append(f"{method}_sample_ids_match_manifest")
        conditions_match = all(
            row["degradation"] == test_samples[row["sample_id"]]["conditions"]["degradation"]
            and abs(float(row["severity"]) - float(test_samples[row["sample_id"]]["conditions"]["severity"])) < 1e-9
            for row in rows
            if row["sample_id"] in test_samples
        )
        checks.append({"check": f"{method}_conditions_match_manifest", "passed": conditions_match})
        if not conditions_match:
            failures.append(f"{method}_conditions_match_manifest")
        valid_metrics = all(
            row["ellipse_iou"] in ("", None) or 0.0 <= float(row["ellipse_iou"]) <= 1.0
            for row in rows
        )
        checks.append({"check": f"{method}_iou_in_unit_interval", "passed": valid_metrics})
        if not valid_metrics:
            failures.append(f"{method}_iou_in_unit_interval")
        summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
        successes = sum(as_bool(row["top1_matches_ground_truth"]) for row in rows)
        aggregate_matches = (
            int(summary["sample_count"]) == len(rows)
            and int(summary["top1_match_count"]) == successes
            and abs(float(summary["top1_match_rate"]) - successes / len(rows)) < 1e-12
        )
        checks.append({"check": f"{method}_aggregate_recomputed", "passed": aggregate_matches})
        if not aggregate_matches:
            failures.append(f"{method}_aggregate_recomputed")

    observed_severities = {
        float(sample["conditions"]["severity"]) for sample in test_samples.values()
    }
    severity_complete = observed_severities == expected_severities
    checks.append(
        {
            "check": "severity_grid_complete",
            "passed": severity_complete,
            "observed": sorted(observed_severities),
        }
    )
    if not severity_complete:
        failures.append("severity_grid_complete")

    report = {
        "status": "ready_to_share" if not failures else "needs_revision",
        "test_sample_count": len(test_samples),
        "camera_group_counts": {key: len(value) for key, value in split_groups.items()},
        "checks": checks,
        "failures": failures,
        "required_caveats": [
            "CG単一モデルであり実写への一般化は未検証",
            "完全楕円教師は不可視部分の形状補完を学習させる",
            "劣化は制御可能な近似であり物理センサ校正値ではない",
            "95%区間はtestの16カメラ条件をクラスタとして推定する",
        ],
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
