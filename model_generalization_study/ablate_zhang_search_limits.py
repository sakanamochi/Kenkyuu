"""Zhang 2019型（再現実装）の弧・組合せ探索上限を評価する。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

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
    detect_zhang_arc_candidates,
)
from paf_ring_detection.methods.zhang2019_paf import (  # noqa: E402
    score_zhang_candidates_for_paf,
    select_zhang_inner_boundary,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "model_generalization_study/config/zhang_search_ablation.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "model_generalization_study/output/results_comparison_c"
)
DIAGNOSTIC_CSV = OUTPUT_DIR / "zhang_failure_diagnostics.csv"


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def _pilot_sample_ids(config: dict) -> set[str]:
    """基準で成功し比較Cで候補生成に失敗した条件を傾斜別に抽出する。"""
    rows = []
    with DIAGNOSTIC_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))
    by_sample = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["dataset"]] = row

    per_tilt = defaultdict(list)
    for sample_id, pair in by_sample.items():
        reference = pair.get("1194M_3")
        comparison = pair.get("comparison_c")
        if reference is None or comparison is None:
            continue
        if (
            reference["success"] == "True"
            and comparison["failure_stage"] == "core_candidate_failure"
        ):
            per_tilt[int(float(comparison["camera_tilt_deg"]))].append(
                sample_id
            )

    selected = set()
    count = int(config["pilot_samples_per_tilt"])
    for tilt in sorted(per_tilt):
        selected.update(sorted(per_tilt[tilt])[:count])
    return selected


def _run_variant(
    samples: list[dict],
    dataset_dir: Path,
    reference_config: dict,
    variant: dict,
    success_iou: float,
) -> list[dict]:
    settings = copy.deepcopy(reference_config["zhang2019"])
    settings["detector"]["max_arcs"] = int(variant["max_arcs"])
    settings["detector"]["max_arc_combinations"] = int(
        variant["max_arc_combinations"]
    )
    settings["detector"]["max_candidates"] = int(
        variant["max_candidates"]
    )
    rows = []
    for index, sample in enumerate(samples, start=1):
        image = read_image(dataset_dir / sample["image"])
        truth = label_ellipse(read_json(dataset_dir / sample["label"]))
        started = time.perf_counter()
        raw_candidates, stages = detect_zhang_arc_candidates(
            image,
            settings["preprocess"],
            settings["detector"],
        )
        retained = score_zhang_candidates_for_paf(
            raw_candidates,
            stages,
            settings["paf_postprocess"]["candidate_validation"],
        )
        selected = select_zhang_inner_boundary(
            retained,
            settings["paf_postprocess"]["selector"],
        )
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        raw_best_iou = _best_iou(raw_candidates, truth, image.shape)
        retained_best_iou = _best_iou(retained, truth, image.shape)
        selected_iou = (
            evaluate_ellipses(selected["ellipse"], truth, image.shape)[
                "ellipse_iou"
            ]
            if selected is not None
            else 0.0
        )
        conditions = sample["conditions"]
        rows.append(
            {
                "variant": variant["id"],
                "max_arcs": int(variant["max_arcs"]),
                "max_arc_combinations": int(
                    variant["max_arc_combinations"]
                ),
                "max_candidates": int(variant["max_candidates"]),
                "sample_id": sample["sample_id"],
                "camera_tilt_deg": float(
                    conditions["camera"]["tilt_deg"]
                ),
                "background": conditions["background"]["type"],
                "lighting_id": conditions["lighting_id"],
                "arc_count": len(stages["arcs"]),
                "raw_candidate_count": len(raw_candidates),
                "retained_candidate_count": len(retained),
                "raw_best_iou": raw_best_iou,
                "retained_best_iou": retained_best_iou,
                "selected_iou": selected_iou,
                "raw_oracle_success": raw_best_iou >= success_iou,
                "retained_oracle_success": (
                    retained_best_iou >= success_iou
                ),
                "detected": selected is not None,
                "success": selected_iou >= success_iou,
                "processing_ms": elapsed_ms,
            }
        )
        if index % 10 == 0 or index == len(samples):
            print(
                f"{variant['id']}: {index}/{len(samples)} "
                f"({elapsed_ms:.0f} ms)"
            )
    return rows


def _summarize(rows: list[dict]) -> dict:
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    summary = {}
    for variant, subset in by_variant.items():
        total = len(subset)
        summary[variant] = {
            "sample_count": total,
            "max_arcs": subset[0]["max_arcs"],
            "max_arc_combinations": subset[0]["max_arc_combinations"],
            "max_candidates": subset[0]["max_candidates"],
            "detected_count": sum(bool(row["detected"]) for row in subset),
            "success_count": sum(bool(row["success"]) for row in subset),
            "success_rate": sum(bool(row["success"]) for row in subset)
            / total,
            "raw_oracle_success_count": sum(
                bool(row["raw_oracle_success"]) for row in subset
            ),
            "raw_oracle_success_rate": sum(
                bool(row["raw_oracle_success"]) for row in subset
            )
            / total,
            "retained_oracle_success_count": sum(
                bool(row["retained_oracle_success"]) for row in subset
            ),
            "processing_ms_mean": float(
                np.mean([row["processing_ms"] for row in subset])
            ),
            "processing_ms_median": float(
                np.median([row["processing_ms"] for row in subset])
            ),
            "processing_ms_p95": float(
                np.percentile(
                    [row["processing_ms"] for row in subset],
                    95,
                )
            ),
            "raw_candidate_count_mean": float(
                np.mean([row["raw_candidate_count"] for row in subset])
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        help="未指定なら設定内の全variantを実行する",
    )
    args = parser.parse_args()

    config = read_json(args.config)
    reference_config = read_json(
        _project_path(config["reference_config"])
    )
    dataset_dir = _project_path(config["dataset"])
    samples = load_samples(dataset_dir, config["split"])
    if args.mode == "pilot":
        pilot_ids = _pilot_sample_ids(config)
        samples = [
            sample
            for sample in samples
            if sample["sample_id"] in pilot_ids
        ]
    selected_variants = [
        variant
        for variant in config["variants"]
        if args.variants is None or variant["id"] in args.variants
    ]
    if not selected_variants:
        raise ValueError("実行対象のvariantがありません")

    rows = []
    for variant in selected_variants:
        rows.extend(
            _run_variant(
                samples,
                dataset_dir,
                reference_config,
                variant,
                float(config["success_iou"]),
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"zhang_search_ablation_{args.mode}"
    csv_path = OUTPUT_DIR / f"{prefix}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "sample_selection": (
            "1194M_3成功・比較Cコア候補生成失敗を傾斜別に最大8件"
            if args.mode == "pilot"
            else "比較C全480件"
        ),
        "variants": _summarize(rows),
    }
    json_path = OUTPUT_DIR / f"{prefix}.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
