from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.ellipse_baseline import evaluate_ellipses


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ellipse_from_dict(values: dict):
    return (
        (float(values["center_x"]), float(values["center_y"])),
        (float(values["axis_1"]), float(values["axis_2"])),
        float(values["angle_deg"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RANSAC候補順位とoracle候補存在率を監査する")
    parser.add_argument("--dataset", default="output/datasets/paf_robustness_v3")
    parser.add_argument("--method", default="canny_ransac")
    parser.add_argument("--split", default="test")
    parser.add_argument("--degradation", default="clean")
    parser.add_argument("--iou-threshold", type=float, default=0.8)
    parser.add_argument("--output", default="output/experiments/paf_second_stage_v1/ransac_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = project_path(args.dataset)
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    samples = [
        sample
        for sample in manifest["samples"]
        if sample.get("split") == args.split
        and sample["conditions"].get("degradation", "clean") == args.degradation
    ]
    rows = []
    for sample in samples:
        detail_path = dataset / "results" / args.method / "details" / f"{sample['sample_id']}.json"
        if not detail_path.exists():
            continue
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        ground_truth = ellipse_from_dict(detail["ground_truth"])
        candidates = detail.get("ransac_candidates", [])
        candidate_rows = []
        for candidate in candidates:
            evaluation = evaluate_ellipses(
                ellipse_from_dict(candidate), ground_truth, (480, 480)
            )
            candidate_rows.append(
                {
                    "rank": int(candidate["rank"]),
                    "selection_score": float(candidate["selection_score"]),
                    "ellipse_iou": evaluation["ellipse_iou"],
                }
            )
        successful = [
            row for row in candidate_rows if row["ellipse_iou"] >= args.iou_threshold
        ]
        best = max(candidate_rows, key=lambda row: row["ellipse_iou"], default=None)
        target_rank = min((row["rank"] for row in successful), default=None)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "candidate_count": len(candidate_rows),
                "top1_iou": candidate_rows[0]["ellipse_iou"] if candidate_rows else None,
                "top1_success": bool(
                    candidate_rows and candidate_rows[0]["ellipse_iou"] >= args.iou_threshold
                ),
                "oracle_best_iou": best["ellipse_iou"] if best else None,
                "oracle_success": bool(successful),
                "target_rank": target_rank,
                "top1_score_margin": (
                    candidate_rows[0]["selection_score"] - candidate_rows[1]["selection_score"]
                    if len(candidate_rows) >= 2
                    else None
                ),
            }
        )
    sample_count = len(rows)
    summary = {
        "method": args.method,
        "split": args.split,
        "degradation": args.degradation,
        "sample_count": sample_count,
        "details_found": len(rows),
        "top1_success_rate": sum(row["top1_success"] for row in rows) / sample_count,
        "oracle_success_rate": sum(row["oracle_success"] for row in rows) / sample_count,
        "selection_loss_rate": sum(
            row["oracle_success"] and not row["top1_success"] for row in rows
        )
        / sample_count,
        "target_recall_at_k": {
            str(k): sum(
                row["target_rank"] is not None and row["target_rank"] <= k for row in rows
            )
            / sample_count
            for k in (1, 3, 5, 10)
        },
        "no_candidate_rate": sum(row["candidate_count"] == 0 for row in rows) / sample_count,
    }
    with (output / "samples.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
