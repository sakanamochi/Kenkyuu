from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    errors = []
    ood = ROOT / "output/datasets/research_ood_base_v1"
    diagnostic = ROOT / "output/datasets/paf_diagnostics_v1"
    output = ROOT / "output/experiments/paf_second_stage_v1"
    ood_manifest = json.loads((ood / "manifest.json").read_text(encoding="utf-8"))
    diagnostic_manifest = json.loads(
        (diagnostic / "manifest.json").read_text(encoding="utf-8")
    )

    if len(ood_manifest["samples"]) != 480:
        errors.append("OOD sample count is not 480")
    backgrounds = Counter(
        sample["conditions"]["background"]["type"] for sample in ood_manifest["samples"]
    )
    if backgrounds != {"space": 160, "earth": 160, "moon": 160}:
        errors.append(f"OOD background imbalance: {backgrounds}")
    if len({s["conditions"]["camera_id"] for s in ood_manifest["samples"]}) != 20:
        errors.append("OOD camera count is not 20")
    if len({s["conditions"]["lighting_id"] for s in ood_manifest["samples"]}) != 8:
        errors.append("OOD lighting count is not 8")

    if len(diagnostic_manifest["samples"]) != 1904:
        errors.append("diagnostic sample count is not 1904")
    diagnostic_groups = Counter(
        (
            sample["conditions"]["degradation"],
            float(sample["conditions"]["severity"]),
        )
        for sample in diagnostic_manifest["samples"]
    )
    if len(diagnostic_groups) != 17 or set(diagnostic_groups.values()) != {112}:
        errors.append(f"diagnostic condition grid is invalid: {diagnostic_groups}")

    result_paths = {
        "ood": {
            "zhang2019_arc_reproduction": ood
            / "results/zhang2019_arc_ood/summary.csv",
            "cnn_ransac_support": ood
            / "results/cnn_ransac_support_ood/summary.csv",
            "contour_fit": ood / "results/contour_fit_ood/summary.csv",
            "canny_ransac": ood / "results/canny_ransac_ood/summary.csv",
            "canny_ransac_inner_pair": ood
            / "results/canny_ransac_inner_pair_ood/summary.csv",
            "cnn_ransac": ood / "results/cnn_ransac_v3_ood/summary.csv",
        },
        "diagnostic": {
            "zhang2019_arc_reproduction": diagnostic
            / "results/zhang2019_arc_diagnostic/summary.csv",
            "cnn_ransac_support": diagnostic
            / "results/cnn_ransac_support_diagnostic/summary.csv",
            "contour_fit": diagnostic / "results/contour_fit_diagnostic/summary.csv",
            "canny_ransac": diagnostic / "results/canny_ransac_diagnostic/summary.csv",
            "canny_ransac_inner_pair": diagnostic
            / "results/canny_ransac_inner_pair_diagnostic/summary.csv",
            "cnn_ransac": diagnostic / "results/cnn_ransac_v3_diagnostic/summary.csv",
        },
    }
    expected_ids = {
        "ood": {sample["sample_id"] for sample in ood_manifest["samples"]},
        "diagnostic": {
            sample["sample_id"] for sample in diagnostic_manifest["samples"]
        },
    }
    loaded = {}
    for family, paths in result_paths.items():
        loaded[family] = {}
        for method, path in paths.items():
            if not path.exists():
                if method in ("zhang2019_arc_reproduction", "cnn_ransac_support"):
                    errors.append(f"{family}/{method} required result is missing")
                continue
            rows = load_csv(path)
            loaded[family][method] = rows
            ids = {row["sample_id"] for row in rows}
            if ids != expected_ids[family] or len(rows) != len(ids):
                errors.append(f"{family}/{method} sample ID mismatch")

    paired_groups = Counter(
        sample["conditions"]["base_sample_id"]
        for sample in diagnostic_manifest["samples"]
    )
    if len(paired_groups) != 112 or set(paired_groups.values()) != {17}:
        errors.append("diagnostic paired groups are not 112 x 17")

    ablation = load_csv(output / "model_ablation/summary.csv")
    if len(ablation) != 24:
        errors.append(f"model ablation row count is {len(ablation)}, expected 24")
    if len({row["run_id"] for row in ablation}) != 12:
        errors.append("model ablation does not contain 12 unique runs")
    if Counter(row["evaluation"] for row in ablation) != {
        "validation": 12,
        "ood_test": 12,
    }:
        errors.append("model ablation evaluation balance is invalid")

    report = {
        "status": "ready_to_share" if not errors else "failed",
        "errors": errors,
        "checks": {
            "ood_sample_count": len(ood_manifest["samples"]),
            "ood_background_counts": backgrounds,
            "diagnostic_sample_count": len(diagnostic_manifest["samples"]),
            "diagnostic_condition_count": len(diagnostic_groups),
            "paired_group_count": len(paired_groups),
            "paired_variants_per_group": sorted(set(paired_groups.values())),
            "model_ablation_run_count": len({row["run_id"] for row in ablation}),
        },
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
