"""未学習のパラメトリックPAFを既存CNNで評価する。"""

from __future__ import annotations

import csv
import sys
import time
import zlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paf_ring_detection.data import (  # noqa: E402
    RingDataset,
    label_ellipse,
    load_samples,
    read_image,
    read_json,
    scaled_ellipse,
    write_json,
)
from paf_ring_detection.geometry import evaluate_ellipses  # noqa: E402
from paf_ring_detection.methods.cnn import load_model  # noqa: E402
from paf_ring_detection.methods.cnn_ransac import (  # noqa: E402
    detect_from_probability,
)
from paf_ring_detection.methods.zhang2019 import (  # noqa: E402
    detect_zhang_arc_candidates,
)
from paf_ring_detection.methods.zhang2019_paf import (  # noqa: E402
    score_zhang_candidates_for_paf,
    select_zhang_inner_boundary,
)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate(experiment: dict, limit: int | None = None) -> dict:
    reference_config = read_json(_project_path(experiment["reference_config"]))
    dataset_dir = _project_path(experiment["comparison_dataset"])
    split = experiment["comparison_split"]
    checkpoint = _project_path(reference_config["paths"]["checkpoint"])
    results_dir = _project_path(experiment["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    dataset = RingDataset(dataset_dir, split, reference_config["input_size"])
    if limit is not None:
        dataset.samples = dataset.samples[:limit]
    loader = DataLoader(
        dataset,
        batch_size=reference_config["cnn"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rows = []

    with torch.inference_mode():
        for images, _, indices in loader:
            started = time.perf_counter()
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            batch_seconds = time.perf_counter() - started
            seconds_per_image = batch_seconds / len(images)
            for probability, index in zip(probabilities, indices.tolist()):
                sample = dataset.samples[index]
                label = read_json(dataset_dir / sample["label"])
                scale = reference_config["input_size"] / float(label["image_width"])
                truth = scaled_ellipse(label_ellipse(label), scale)
                random_seed = (
                    reference_config["seed"]
                    + zlib.crc32(sample["sample_id"].encode("utf-8"))
                ) % (2**32)
                result = detect_from_probability(
                    probability,
                    reference_config["ransac"],
                    random_seed,
                )
                conditions = sample["conditions"]
                row = {
                    "sample_id": sample["sample_id"],
                    "model_id": conditions["model_id"],
                    "camera_id": conditions["camera_id"],
                    "lighting_id": conditions["lighting_id"],
                    "background": conditions["background"]["type"],
                    "camera_tilt_deg": conditions["camera"]["tilt_deg"],
                    "detected": False,
                    "success": False,
                    "ellipse_iou": None,
                    "center_error_px": None,
                    "minor_axis_error_px": None,
                    "major_axis_error_px": None,
                    "angle_error_deg": None,
                    "cnn_inference_ms": 1000.0 * seconds_per_image,
                }
                if result is not None:
                    metrics = evaluate_ellipses(
                        result["ellipse"],
                        truth,
                        probability.shape,
                    )
                    row.update(metrics)
                    row["detected"] = True
                    row["success"] = (
                        metrics["ellipse_iou"] >= reference_config["success_iou"]
                    )
                rows.append(row)
            print(f"比較モデルCNN評価: {len(rows)}/{len(dataset)}")

    results_prefix = experiment.get("results_prefix", "comparison_a")
    output_csv = results_dir / f"{results_prefix}_cnn.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sample_count": len(rows),
        "detected_count": sum(bool(row["detected"]) for row in rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "device": str(device),
        "checkpoint": str(checkpoint),
    }
    write_json(results_dir / f"{results_prefix}_cnn.json", summary)
    return summary


def evaluate_zhang2019(experiment: dict, limit: int | None = None) -> dict:
    """比較モデルをZhang 2019型（再現実装）で評価する。"""
    reference_config = read_json(_project_path(experiment["reference_config"]))
    dataset_dir = _project_path(experiment["comparison_dataset"])
    split = experiment["comparison_split"]
    settings = reference_config["zhang2019"]
    samples = load_samples(dataset_dir, split)
    if limit is not None:
        samples = samples[:limit]
    rows = []

    for index, sample in enumerate(samples, start=1):
        image = read_image(dataset_dir / sample["image"])
        truth = label_ellipse(read_json(dataset_dir / sample["label"]))
        started = time.perf_counter()
        candidates, stages = detect_zhang_arc_candidates(
            image,
            settings["preprocess"],
            settings["detector"],
        )
        candidates = score_zhang_candidates_for_paf(
            candidates,
            stages,
            settings["paf_postprocess"]["candidate_validation"],
        )
        selected = select_zhang_inner_boundary(
            candidates,
            settings["paf_postprocess"]["selector"],
        )
        candidate_ious = [
            evaluate_ellipses(
                candidate["ellipse"],
                truth,
                image.shape,
            )["ellipse_iou"]
            for candidate in candidates
        ]
        oracle_best_iou = max(candidate_ious, default=0.0)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        conditions = sample["conditions"]
        row = {
            "sample_id": sample["sample_id"],
            "model_id": conditions["model_id"],
            "camera_id": conditions["camera_id"],
            "lighting_id": conditions["lighting_id"],
            "background": conditions["background"]["type"],
            "camera_tilt_deg": conditions["camera"]["tilt_deg"],
            "candidate_count": len(candidates),
            "oracle_success": (
                oracle_best_iou >= reference_config["success_iou"]
            ),
            "oracle_best_iou": oracle_best_iou,
            "detected": False,
            "success": False,
            "ellipse_iou": None,
            "center_error_px": None,
            "minor_axis_error_px": None,
            "major_axis_error_px": None,
            "angle_error_deg": None,
            "processing_ms": elapsed_ms,
        }
        if selected is not None:
            metrics = evaluate_ellipses(
                selected["ellipse"],
                truth,
                image.shape,
            )
            row.update(metrics)
            row["detected"] = True
            row["success"] = (
                metrics["ellipse_iou"] >= reference_config["success_iou"]
            )
        rows.append(row)
        if index % 50 == 0 or index == len(samples):
            print(f"比較モデルZhang 2019型（再現実装）: {index}/{len(samples)}")

    results_dir = _project_path(experiment["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_prefix = experiment.get("results_prefix", "comparison_a")
    output_csv = results_dir / f"{results_prefix}_zhang2019.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "Zhang 2019型（再現実装）",
        "sample_count": len(rows),
        "detected_count": sum(bool(row["detected"]) for row in rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "oracle_success_count": sum(
            bool(row["oracle_success"]) for row in rows
        ),
        "oracle_success_rate": sum(
            bool(row["oracle_success"]) for row in rows
        )
        / len(rows),
        "processing_ms_mean": sum(float(row["processing_ms"]) for row in rows)
        / len(rows),
    }
    write_json(results_dir / f"{results_prefix}_zhang2019.json", summary)
    return summary


if __name__ == "__main__":
    experiment_path = (
        PROJECT_ROOT / "model_generalization_study/config/experiment.json"
    )
    experiment = read_json(experiment_path)
    print(evaluate(experiment))
    print(evaluate_zhang2019(experiment))
