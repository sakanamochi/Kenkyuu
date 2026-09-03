"""撮像診断データに残るZhang型楕円候補数を集計する一時監査スクリプト。"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from paf_ring_detection.data import label_ellipse, read_image, read_json
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.zhang2019 import detect_zhang_arc_candidates
from paf_ring_detection.methods.zhang2019_paf import (
    score_zhang_candidates_for_paf,
    select_zhang_inner_boundary,
)


OUTPUT_PATH = Path(__file__).with_name("zhang_candidate_count_audit.json")


def summarize(values: list[int]) -> dict:
    """候補数の分布を発表で説明しやすい統計量へまとめる。"""
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": int(np.max(array)),
        "zero_count": int(np.count_nonzero(array == 0)),
        "one_count": int(np.count_nonzero(array == 1)),
        "multiple_count": int(np.count_nonzero(array >= 2)),
    }


def main() -> None:
    config = read_json(PROJECT_ROOT / "config" / "experiment.json")
    dataset_dir = PROJECT_ROOT / config["paths"]["diagnostic_dataset"]
    samples = [
        sample
        for sample in read_json(dataset_dir / "manifest.json")["samples"]
        if sample["split"] == "diagnostic_test"
    ]
    settings = config["zhang2019"]

    core_counts: list[int] = []
    paf_counts: list[int] = []
    selection_modes: Counter[str] = Counter()
    detected_count = 0
    success_count = 0
    grouped_core: defaultdict[str, list[int]] = defaultdict(list)
    grouped_paf: defaultdict[str, list[int]] = defaultdict(list)
    grouped_success: defaultdict[str, list[int]] = defaultdict(list)

    for index, sample in enumerate(samples, start=1):
        image = read_image(dataset_dir / sample["image"])
        core_candidates, stages = detect_zhang_arc_candidates(
            image,
            settings["preprocess"],
            settings["detector"],
        )
        paf_candidates = score_zhang_candidates_for_paf(
            core_candidates,
            stages,
            settings["paf_postprocess"]["candidate_validation"],
        )
        selected = select_zhang_inner_boundary(
            paf_candidates,
            settings["paf_postprocess"]["selector"],
        )
        success = False
        if selected is not None:
            detected_count += 1
            truth = label_ellipse(read_json(dataset_dir / sample["label"]))
            metrics = evaluate_ellipses(
                selected["ellipse"],
                truth,
                image.shape,
            )
            success = metrics["ellipse_iou"] >= config["success_iou"]
            success_count += int(success)

        core_count = len(core_candidates)
        paf_count = len(paf_candidates)
        core_counts.append(core_count)
        paf_counts.append(paf_count)
        mode = selected["selection_mode"] if selected else "no_detection"
        selection_modes[mode] += 1

        conditions = sample["conditions"]
        key = f"{conditions['degradation']}@{float(conditions['severity']):.2f}"
        grouped_core[key].append(core_count)
        grouped_paf[key].append(paf_count)
        grouped_success[key].append(int(success))

        if index % 100 == 0 or index == len(samples):
            print(f"{index}/{len(samples)}", flush=True)

    result = {
        "dataset": str(dataset_dir),
        "sample_count": len(samples),
        "definition": {
            "core_candidates": (
                "三弧の幾何制約、楕円当てはめ、論文型適合度検証、重複除去後"
            ),
            "paf_candidates": "全周支持・角度被覆の閾値を通過した候補",
        },
        "core_candidates": summarize(core_counts),
        "paf_candidates": summarize(paf_counts),
        "selection_modes": dict(selection_modes),
        "selection_result": {
            "detected_count": detected_count,
            "success_count": success_count,
            "success_rate": success_count / len(samples),
        },
        "by_condition": {
            key: {
                "core_candidates": summarize(grouped_core[key]),
                "paf_candidates": summarize(grouped_paf[key]),
                "success_count": int(sum(grouped_success[key])),
                "success_rate": float(np.mean(grouped_success[key])),
            }
            for key in sorted(grouped_core)
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(OUTPUT_PATH, flush=True)


if __name__ == "__main__":
    main()
