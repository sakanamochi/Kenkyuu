"""Zhang 2019型（再現実装）のモデル変更時の失敗段階を分解する。"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paf_ring_detection.data import (  # noqa: E402
    label_ellipse,
    load_samples,
    read_image,
    read_json,
)
from paf_ring_detection.geometry import evaluate_ellipses  # noqa: E402
from paf_ring_detection.methods.zhang2019 import (  # noqa: E402
    _ellipse_samples,
    detect_zhang_arc_candidates,
)
from paf_ring_detection.methods.zhang2019_paf import (  # noqa: E402
    score_zhang_candidates_for_paf,
    select_zhang_inner_boundary,
)


OUTPUT_DIR = (
    PROJECT_ROOT / "model_generalization_study/output/results_comparison_c"
)
SUCCESS_IOU = 0.80


def _best_iou(candidates: list[dict], truth, image_shape) -> float:
    return max(
        (
            evaluate_ellipses(candidate["ellipse"], truth, image_shape)[
                "ellipse_iou"
            ]
            for candidate in candidates
        ),
        default=0.0,
    )


def _failure_stage(
    success: bool,
    raw_oracle_success: bool,
    retained_oracle_success: bool,
) -> str:
    """正解候補がどの段階で失われたかを排他的に分類する。"""
    if success:
        return "success"
    if retained_oracle_success:
        return "selection_failure"
    if raw_oracle_success:
        return "postvalidation_rejection"
    return "core_candidate_failure"


def _ground_truth_edge_support(stages: dict, truth, settings: dict) -> dict:
    """正解楕円そのものがCannyエッジ上に残っている割合を測る。"""
    sample_count = int(
        settings["paf_postprocess"]["candidate_validation"][
            "validation_samples"
        ]
    )
    threshold = float(
        settings["paf_postprocess"]["candidate_validation"][
            "edge_distance_threshold_px"
        ]
    )
    bin_count = int(
        settings["paf_postprocess"]["candidate_validation"]["angular_bins"]
    )
    perimeter, _ = _ellipse_samples(truth, sample_count)
    pixels = np.rint(perimeter).astype(int)
    distance_map = cv2.distanceTransform(
        255 - stages["edges"],
        cv2.DIST_L2,
        3,
    )
    height, width = distance_map.shape
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    distances = np.full(sample_count, np.inf, dtype=np.float32)
    distances[valid] = distance_map[pixels[valid, 1], pixels[valid, 0]]
    supported = distances <= threshold
    coverage = np.mean(
        [
            np.any(supported[indexes])
            for indexes in np.array_split(
                np.arange(sample_count),
                bin_count,
            )
        ]
    )
    finite = distances[np.isfinite(distances)]
    return {
        "gt_edge_density": float(np.mean(supported)),
        "gt_angular_coverage": float(coverage),
        "gt_mean_edge_distance_px": (
            float(np.mean(np.minimum(finite, threshold * 3.0)))
            if len(finite)
            else threshold * 3.0
        ),
    }


def _analyze_dataset(
    dataset_name: str,
    dataset_dir: Path,
    split: str,
    settings: dict,
) -> list[dict]:
    rows = []
    samples = load_samples(dataset_dir, split)
    for index, sample in enumerate(samples, start=1):
        image = read_image(dataset_dir / sample["image"])
        truth = label_ellipse(read_json(dataset_dir / sample["label"]))

        raw_candidates, stages = detect_zhang_arc_candidates(
            image,
            settings["preprocess"],
            settings["detector"],
        )
        retained_candidates = score_zhang_candidates_for_paf(
            raw_candidates,
            stages,
            settings["paf_postprocess"]["candidate_validation"],
        )
        selected = select_zhang_inner_boundary(
            retained_candidates,
            settings["paf_postprocess"]["selector"],
        )

        raw_best_iou = _best_iou(raw_candidates, truth, image.shape)
        retained_best_iou = _best_iou(
            retained_candidates,
            truth,
            image.shape,
        )
        selected_iou = (
            evaluate_ellipses(selected["ellipse"], truth, image.shape)[
                "ellipse_iou"
            ]
            if selected is not None
            else 0.0
        )
        raw_oracle_success = raw_best_iou >= SUCCESS_IOU
        retained_oracle_success = retained_best_iou >= SUCCESS_IOU
        success = selected_iou >= SUCCESS_IOU
        gt_support = _ground_truth_edge_support(stages, truth, settings)
        conditions = sample["conditions"]
        rows.append(
            {
                "dataset": dataset_name,
                "sample_id": sample["sample_id"],
                "camera_tilt_deg": float(
                    conditions["camera"]["tilt_deg"]
                ),
                "background": conditions["background"]["type"],
                "lighting_id": conditions["lighting_id"],
                "edge_pixel_count": int(np.count_nonzero(stages["edges"])),
                "arc_count": len(stages["arcs"]),
                "arc_limit_reached": (
                    len(stages["arcs"])
                    >= int(settings["detector"]["max_arcs"])
                ),
                "raw_candidate_count": len(raw_candidates),
                "retained_candidate_count": len(retained_candidates),
                "raw_best_iou": raw_best_iou,
                "retained_best_iou": retained_best_iou,
                "selected_iou": selected_iou,
                "raw_oracle_success": raw_oracle_success,
                "retained_oracle_success": retained_oracle_success,
                "detected": selected is not None,
                "success": success,
                "failure_stage": _failure_stage(
                    success,
                    raw_oracle_success,
                    retained_oracle_success,
                ),
                **gt_support,
            }
        )
        if index % 80 == 0 or index == len(samples):
            print(f"{dataset_name}: {index}/{len(samples)}")
    return rows


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _summarize(rows: list[dict]) -> dict:
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    summary = {}
    for dataset_name, dataset_rows in by_dataset.items():
        total = len(dataset_rows)
        stage_counts = Counter(row["failure_stage"] for row in dataset_rows)
        by_tilt = {}
        for tilt in sorted({row["camera_tilt_deg"] for row in dataset_rows}):
            subset = [
                row for row in dataset_rows
                if row["camera_tilt_deg"] == tilt
            ]
            by_tilt[str(int(tilt))] = {
                "sample_count": len(subset),
                "success_rate": _rate(
                    sum(bool(row["success"]) for row in subset),
                    len(subset),
                ),
                "raw_oracle_success_rate": _rate(
                    sum(bool(row["raw_oracle_success"]) for row in subset),
                    len(subset),
                ),
                "arc_limit_rate": _rate(
                    sum(bool(row["arc_limit_reached"]) for row in subset),
                    len(subset),
                ),
            }
        summary[dataset_name] = {
            "sample_count": total,
            "edge_pixel_count_mean": float(
                np.mean([row["edge_pixel_count"] for row in dataset_rows])
            ),
            "arc_count_mean": float(
                np.mean([row["arc_count"] for row in dataset_rows])
            ),
            "arc_limit_count": sum(
                bool(row["arc_limit_reached"]) for row in dataset_rows
            ),
            "raw_candidate_count_mean": float(
                np.mean(
                    [row["raw_candidate_count"] for row in dataset_rows]
                )
            ),
            "retained_candidate_count_mean": float(
                np.mean(
                    [
                        row["retained_candidate_count"]
                        for row in dataset_rows
                    ]
                )
            ),
            "gt_edge_density_mean": float(
                np.mean([row["gt_edge_density"] for row in dataset_rows])
            ),
            "gt_angular_coverage_mean": float(
                np.mean(
                    [
                        row["gt_angular_coverage"]
                        for row in dataset_rows
                    ]
                )
            ),
            "gt_mean_edge_distance_px_mean": float(
                np.mean(
                    [
                        row["gt_mean_edge_distance_px"]
                        for row in dataset_rows
                    ]
                )
            ),
            "detected_count": sum(
                bool(row["detected"]) for row in dataset_rows
            ),
            "success_count": sum(
                bool(row["success"]) for row in dataset_rows
            ),
            "raw_oracle_success_count": sum(
                bool(row["raw_oracle_success"]) for row in dataset_rows
            ),
            "retained_oracle_success_count": sum(
                bool(row["retained_oracle_success"])
                for row in dataset_rows
            ),
            "failure_stage_counts": dict(stage_counts),
            "by_camera_tilt_deg": by_tilt,
        }
    return summary


def main() -> None:
    reference_config = read_json(PROJECT_ROOT / "config/experiment.json")
    settings = reference_config["zhang2019"]
    rows = []
    rows.extend(
        _analyze_dataset(
            "1194M_3",
            PROJECT_ROOT / "output/datasets/ood_evaluation",
            "ood_test",
            settings,
        )
    )
    rows.extend(
        _analyze_dataset(
            "comparison_c",
            PROJECT_ROOT
            / "model_generalization_study/output/datasets/comparison_c",
            "model_holdout_test",
            settings,
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "zhang_failure_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    json_path = OUTPUT_DIR / "zhang_failure_diagnostics.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
