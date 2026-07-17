import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="楕円検出方式の結果を比較する")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("contour_fit", "canny_ransac_inner_pair"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_project_path(args.dataset)
    methods = tuple(args.methods)
    rows_by_method = {}
    for method in methods:
        with (dataset_dir / "results" / method / "summary.csv").open(
            encoding="utf-8-sig", newline=""
        ) as file:
            rows_by_method[method] = {
                row["sample_id"]: row for row in csv.DictReader(file)
            }

    sample_ids = sorted(set.intersection(*(set(rows) for rows in rows_by_method.values())))
    comparison_rows = []
    for sample_id in sample_ids:
        reference = rows_by_method[methods[0]][sample_id]
        row = {
            "sample_id": sample_id,
            "camera_id": reference["camera_id"],
            "lighting_id": reference["lighting_id"],
            "degradation": reference.get("degradation", "clean"),
            "severity": reference.get("severity", "0"),
        }
        for method in methods:
            result = rows_by_method[method][sample_id]
            row[f"{method}_status"] = result["status"]
            row[f"{method}_match"] = result["top1_matches_ground_truth"]
            row[f"{method}_center_error_px"] = result["center_error_px"]
            row[f"{method}_ellipse_iou"] = result["ellipse_iou"]
        comparison_rows.append(row)

    results_dir = dataset_dir / "results"
    with (results_dir / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summaries = {
        method: json.loads(
            (results_dir / method / "summary.json").read_text(encoding="utf-8")
        )
        for method in methods
    }
    (results_dir / "comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
