from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.ellipse_baseline import evaluate_ellipses, fit_ground_truth, preprocess_image
from analysis.ellipse_ransac import fit_contour_ransac_candidates, select_paf_inner_candidate
from paflab.dataset import StressDataset
from paflab.evaluate_cnn import probability_ellipse
from paflab.image_io import imread
from paflab.model import TinyUNet


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/experiments/paf_second_stage_v1/ransac_scoring_ablation"
SCORE_MODES = {
    "density": 1.0,
    "sqrt_perimeter": 0.5,
    "support": 0.0,
}
SCOPES = {
    "validation": {
        "dataset": "output/datasets/paf_robustness_v3",
        "split": "validation",
        "degradation": None,
        "paired_seed": False,
    },
    "validation_clean": {
        "dataset": "output/datasets/paf_robustness_v3",
        "split": "validation",
        "degradation": "clean",
        "paired_seed": False,
    },
    "test": {
        "dataset": "output/datasets/paf_robustness_v3",
        "split": "test",
        "degradation": None,
        "paired_seed": False,
    },
    "test_clean": {
        "dataset": "output/datasets/paf_robustness_v3",
        "split": "test",
        "degradation": "clean",
        "paired_seed": False,
    },
    "ood": {
        "dataset": "output/datasets/research_ood_base_v1",
        "split": "ood_test",
        "degradation": None,
        "paired_seed": True,
    },
    "diagnostic": {
        "dataset": "output/datasets/paf_diagnostics_v1",
        "split": "diagnostic_test",
        "degradation": None,
        "paired_seed": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNN用・Canny用RANSACの周長補正を同一乱数条件で比較する"
    )
    parser.add_argument("--task", choices=("cnn", "classic"), required=True)
    parser.add_argument("--scope", choices=tuple(SCOPES), required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(SCORE_MODES),
        default=list(SCORE_MODES),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_for_sample(base_seed: int, sample: dict, paired: bool) -> int:
    conditions = sample["conditions"]
    seed_key = sample["sample_id"]
    if paired:
        seed_key = conditions.get("base_sample_id") or (
            f"{conditions['camera_id']}|{conditions['lighting_id']}"
        )
    return (int(base_seed) + zlib.crc32(seed_key.encode("utf-8"))) % (2**32)


def ground_truth_for(dataset: Path, sample: dict, scale: float = 1.0):
    label = json.loads((dataset / sample["label"]).read_text(encoding="utf-8"))
    points = np.asarray(label["image_points"], dtype=np.float32) * float(scale)
    return fit_ground_truth(points)


def result_columns(result: dict | None, ground_truth, image_shape, threshold: float) -> dict:
    row = {
        "detected": result is not None,
        "success": False,
        "ellipse_iou": None,
        "center_error_px": None,
        "minor_axis_error_px": None,
        "major_axis_error_px": None,
        "angle_error_deg": None,
        "predicted_center_x": None,
        "predicted_center_y": None,
        "predicted_axis_1": None,
        "predicted_axis_2": None,
        "predicted_angle_deg": None,
        "ransac_score": None,
        "ransac_inlier_count": None,
        "ransac_angular_coverage": None,
        "ransac_mean_inlier_distance_px": None,
    }
    if result is None:
        return row
    evaluation = evaluate_ellipses(result["ellipse"], ground_truth, image_shape)
    ellipse = result["ellipse"]
    row.update(
        {
            "success": evaluation["ellipse_iou"] >= threshold,
            **evaluation,
            "predicted_center_x": float(ellipse[0][0]),
            "predicted_center_y": float(ellipse[0][1]),
            "predicted_axis_1": float(ellipse[1][0]),
            "predicted_axis_2": float(ellipse[1][1]),
            "predicted_angle_deg": float(ellipse[2]),
            "ransac_score": float(result["score"]),
            "ransac_inlier_count": int(result["inlier_count"]),
            "ransac_angular_coverage": float(result["angular_coverage"]),
            "ransac_mean_inlier_distance_px": float(result["mean_inlier_distance_px"]),
        }
    )
    return row


def write_outputs(rows: list[dict], output: Path, stem: str, elapsed_seconds: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["score_mode"], row["selection_rule"])].append(row)
    summaries = []
    for (score_mode, selection_rule), group in grouped.items():
        successes = sum(bool(row["success"]) for row in group)
        detected = [row for row in group if row["detected"]]
        summaries.append(
            {
                "score_mode": score_mode,
                "perimeter_power": SCORE_MODES[score_mode],
                "selection_rule": selection_rule,
                "sample_count": len(group),
                "detected_count": len(detected),
                "success_count": successes,
                "success_rate": successes / len(group),
                "mean_iou_detected": (
                    sum(float(row["ellipse_iou"]) for row in detected) / len(detected)
                    if detected
                    else None
                ),
            }
        )
    payload = {
        "source_csv": csv_path.relative_to(ROOT).as_posix(),
        "elapsed_seconds": elapsed_seconds,
        "summaries": summaries,
    }
    (output / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def filtered_samples(dataset: Path, scope: dict, limit: int | None) -> list[dict]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    samples = [
        sample
        for sample in manifest["samples"]
        if sample.get("split") == scope["split"]
        and (
            scope["degradation"] is None
            or sample["conditions"].get("degradation", "clean") == scope["degradation"]
        )
    ]
    return samples[:limit] if limit is not None else samples


def evaluate_cnn(args: argparse.Namespace, scope: dict, dataset_dir: Path, output: Path) -> None:
    config = json.loads((ROOT / "config/research_experiment.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        ROOT / "output/experiments/paf_robustness_v3/cnn_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    dataset = StressDataset(
        dataset_dir,
        split=scope["split"],
        input_size=int(config["input_size"]),
        return_index=True,
    )
    if scope["degradation"] is not None:
        dataset.samples = [
            sample
            for sample in dataset.samples
            if sample["conditions"].get("degradation", "clean") == scope["degradation"]
        ]
    if args.limit is not None:
        dataset.samples = dataset.samples[: args.limit]
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, _, indices) in enumerate(loader, start=1):
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            for probability, dataset_index in zip(probabilities, indices.tolist()):
                sample = dataset.samples[dataset_index]
                label = json.loads((dataset_dir / sample["label"]).read_text(encoding="utf-8"))
                scale = float(config["input_size"]) / float(label["image_width"])
                ground_truth = fit_ground_truth(
                    np.asarray(label["image_points"], dtype=np.float32) * scale
                )
                sample_seed = seed_for_sample(
                    config["cnn_ransac"]["random_seed"], sample, scope["paired_seed"]
                )
                for mode in args.modes:
                    settings = {
                        **config["cnn_ransac"],
                        "perimeter_power": SCORE_MODES[mode],
                    }
                    result = probability_ellipse(
                        probability, settings, random_seed=sample_seed
                    )
                    rows.append(
                        {
                            "sample_id": sample["sample_id"],
                            "scope": args.scope,
                            "degradation": sample["conditions"].get("degradation", "clean"),
                            "severity": float(sample["conditions"].get("severity", 0.0)),
                            "background": sample["conditions"].get("background", {}).get("type", "space"),
                            "camera_tilt_deg": sample["conditions"].get("camera", {}).get("tilt_deg"),
                            "score_mode": mode,
                            "perimeter_power": SCORE_MODES[mode],
                            "selection_rule": "single_global_ransac",
                            "candidate_count": 1 if result is not None else 0,
                            "selection_mode": "global_ransac",
                            **result_columns(
                                result,
                                ground_truth,
                                (int(config["input_size"]),) * 2,
                                float(config["evaluation"]["success_ellipse_iou"]),
                            ),
                        }
                    )
            if batch_index % 10 == 0:
                print(
                    f"cnn {args.scope}: {min(batch_index * loader.batch_size, len(dataset))}/{len(dataset)}",
                    flush=True,
                )
    write_outputs(
        rows,
        output,
        f"cnn_{args.scope}",
        time.perf_counter() - started,
    )


def evaluate_classic(args: argparse.Namespace, scope: dict, dataset_dir: Path, output: Path) -> None:
    config = json.loads((ROOT / "config/baseline.json").read_text(encoding="utf-8"))
    samples = filtered_samples(dataset_dir, scope, args.limit)
    rows = []
    started = time.perf_counter()
    for sample_index, sample in enumerate(samples, start=1):
        image = imread(dataset_dir / sample["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(dataset_dir / sample["image"])
        preprocessing = preprocess_image(image, config["detector"])
        ground_truth = ground_truth_for(dataset_dir, sample)
        sample_seed = seed_for_sample(
            config["ransac"]["random_seed"], sample, scope["paired_seed"]
        )
        for mode in args.modes:
            settings = {
                **config["ransac"],
                "perimeter_power": SCORE_MODES[mode],
            }
            candidates = fit_contour_ransac_candidates(
                preprocessing["contours"],
                image.shape,
                settings,
                random_seed=sample_seed,
            )
            selected = {
                "quality": candidates[0] if candidates else None,
                "inner_pair": select_paf_inner_candidate(
                    candidates, config["inner_pair_selector"]
                ),
            }
            for rule, result in selected.items():
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "scope": args.scope,
                        "degradation": sample["conditions"].get("degradation", "clean"),
                        "severity": float(sample["conditions"].get("severity", 0.0)),
                        "background": sample["conditions"].get("background", {}).get("type", "space"),
                        "camera_tilt_deg": sample["conditions"].get("camera", {}).get("tilt_deg"),
                        "score_mode": mode,
                        "perimeter_power": SCORE_MODES[mode],
                        "selection_rule": rule,
                        "candidate_count": len(candidates),
                        "selection_mode": (
                            result.get("selection_mode", "quality_only")
                            if result is not None
                            else "no_detection"
                        ),
                        **result_columns(
                            result,
                            ground_truth,
                            image.shape,
                            float(config["success_thresholds"]["ellipse_iou"]),
                        ),
                    }
                )
        if sample_index % 10 == 0 or sample_index == len(samples):
            print(
                f"classic {args.scope}: {sample_index}/{len(samples)}",
                flush=True,
            )
    write_outputs(
        rows,
        output,
        f"classic_{args.scope}",
        time.perf_counter() - started,
    )


def main() -> None:
    args = parse_args()
    scope = SCOPES[args.scope]
    dataset_dir = project_path(scope["dataset"])
    output = project_path(args.output)
    if args.task == "cnn":
        evaluate_cnn(args, scope, dataset_dir, output)
    else:
        evaluate_classic(args, scope, dataset_dir, output)


if __name__ == "__main__":
    main()
