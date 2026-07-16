from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from analysis.ellipse_baseline import evaluate_ellipses
from analysis.ellipse_ransac import select_paf_inner_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ellipse(values: dict):
    return (
        (float(values["center_x"]), float(values["center_y"])),
        (float(values["axis_1"]), float(values["axis_2"])),
        float(values["angle_deg"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAF内周の二重輪郭priorをtrainで調整しholdout評価する")
    parser.add_argument("--dataset", default="output/datasets/paf_robustness_v1")
    parser.add_argument("--results-name", default="canny_ransac_tuning")
    parser.add_argument("--output", default="output/experiments/paf_second_stage_v1/ransac_audit/inner_pair")
    return parser.parse_args()


def load_samples(dataset: Path, results_name: str) -> dict[str, list[dict]]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    groups = {"train": [], "validation": [], "test": []}
    for sample in manifest["samples"]:
        if sample.get("split") not in groups or sample["conditions"].get("degradation") != "clean":
            continue
        detail_path = dataset / "results" / results_name / "details" / f"{sample['sample_id']}.json"
        if not detail_path.exists():
            detail_path = dataset / "results" / "canny_ransac" / "details" / f"{sample['sample_id']}.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        ground_truth = ellipse(detail["ground_truth"])
        candidates = []
        for candidate in detail["ransac_candidates"]:
            fitted = ellipse(candidate)
            candidates.append(
                {
                    **candidate,
                    "ellipse": fitted,
                    "matches_ground_truth": evaluate_ellipses(
                        fitted, ground_truth, (480, 480)
                    )["ellipse_iou"]
                    >= 0.8,
                }
            )
        groups[sample["split"]].append(
            {"sample_id": sample["sample_id"], "ground_truth": ground_truth, "candidates": candidates}
        )
    return groups


def success_rate(samples: list[dict], settings: dict) -> float:
    success = 0
    for sample in samples:
        selected = select_paf_inner_candidate(sample["candidates"], settings)
        if selected is None:
            continue
        success += bool(selected["matches_ground_truth"])
    return success / len(samples)


def main() -> None:
    args = parse_args()
    dataset = project_path(args.dataset)
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    groups = load_samples(dataset, args.results_name)
    trials = []
    for values in itertools.product(
        (1.1, 1.2, 1.5, 2.0),
        (2.0, 3.0, 4.0, 5.0),
        (0.08, 0.12, 0.18, 0.24),
        (0.03, 0.05, 0.08, 0.12),
        (0.1, 0.25, 0.5),
    ):
        settings = dict(
            zip(
                (
                    "area_ratio_min",
                    "area_ratio_max",
                    "center_distance_major_ratio",
                    "axis_ratio_difference",
                    "min_quality_ratio",
                ),
                values,
            )
        )
        settings["max_candidates"] = 10
        trials.append({"settings": settings, "train_rate": success_rate(groups["train"], settings)})
    best_train = max(trial["train_rate"] for trial in trials)
    train_ties = [trial for trial in trials if trial["train_rate"] == best_train]
    # train同率ならvalidationで一度だけ選び、testは最終評価専用にする。
    for trial in train_ties:
        trial["validation_rate"] = success_rate(groups["validation"], trial["settings"])
    best = max(train_ties, key=lambda trial: trial["validation_rate"])
    result = {
        "selection_protocol": "grid search on train; validation breaks train ties; test evaluated once",
        "settings": best["settings"],
        "sample_counts": {split: len(samples) for split, samples in groups.items()},
        "rates": {
            "train": best["train_rate"],
            "validation": best["validation_rate"],
            "test": success_rate(groups["test"], best["settings"]),
        },
        "quality_only_rates": {
            split: sum(
                bool(sample["candidates"])
                and sample["candidates"][0]["matches_ground_truth"]
                for sample in samples
            )
            / len(samples)
            for split, samples in groups.items()
        },
        "trial_count": len(trials),
        "train_best_tie_count": len(train_ties),
    }
    (output / "selection_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
