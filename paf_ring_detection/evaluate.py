"""Canny、Zhang 2019型（再現実装）、CNNを同じIoU基準で評価する。"""

from __future__ import annotations

import csv
import zlib
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from paf_ring_detection.data import (
    RingDataset,
    label_ellipse,
    load_samples,
    read_image,
    read_json,
    scaled_ellipse,
    write_json,
)
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.canny_contour_ransac import (
    detect_canny_contour_ransac,
)
from paf_ring_detection.methods.cnn import load_model
from paf_ring_detection.methods.cnn_ransac import detect_from_probability
from paf_ring_detection.methods.zhang2019 import (
    detect_zhang_arc_candidates,
)
from paf_ring_detection.methods.zhang2019_paf import (
    score_zhang_candidates_for_paf,
    select_zhang_inner_boundary,
)


def _result_row(sample: dict, method: str) -> dict:
    conditions = sample["conditions"]
    return {
        "sample_id": sample["sample_id"],
        "method": method,
        "camera_id": conditions["camera_id"],
        "lighting_id": conditions["lighting_id"],
        "degradation": conditions.get("degradation", "clean"),
        "severity": conditions.get("severity", 0.0),
        "background": conditions.get("background", {}).get("type", "space"),
        "camera_tilt_deg": conditions["camera"].get("tilt_deg"),
        "detected": False,
        "success": False,
        "ellipse_iou": None,
        "center_error_px": None,
        "minor_axis_error_px": None,
        "major_axis_error_px": None,
        "angle_error_deg": None,
    }


def _add_evaluation(row: dict, detected, truth, image_shape, threshold: float) -> None:
    if detected is None:
        return
    metrics = evaluate_ellipses(detected, truth, image_shape)
    row.update(metrics)
    row["detected"] = True
    row["success"] = metrics["ellipse_iou"] >= threshold


def _evaluate_zhang(
    config: dict,
    dataset_dir: Path,
    split: str,
    limit: int | None,
) -> list[dict]:
    rows = []
    settings = config["zhang2019"]
    samples = load_samples(dataset_dir, split)[:limit]
    for index, sample in enumerate(samples, start=1):
        image = read_image(dataset_dir / sample["image"])
        truth = label_ellipse(read_json(dataset_dir / sample["label"]))
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
        row = _result_row(sample, "zhang2019_reproduction")
        detected = selected["ellipse"] if selected else None
        _add_evaluation(
            row,
            detected,
            truth,
            image.shape,
            config["success_iou"],
        )
        rows.append(row)
        if index % 50 == 0:
            print(f"Zhang型: {index}/{len(samples)}")
    return rows


def _evaluate_canny(
    config: dict,
    dataset_dir: Path,
    split: str,
    limit: int | None,
) -> list[dict]:
    """CNNと同じ入力寸法・同じRANSAC実装でCanny方式を評価する。"""
    rows = []
    samples = load_samples(dataset_dir, split)[:limit]
    input_size = int(config["input_size"])
    for index, sample in enumerate(samples, start=1):
        source_image = read_image(dataset_dir / sample["image"])
        image = cv2.resize(
            source_image,
            (input_size, input_size),
            interpolation=cv2.INTER_AREA,
        )
        label = read_json(dataset_dir / sample["label"])
        scale = input_size / float(label["image_width"])
        truth = scaled_ellipse(label_ellipse(label), scale)
        seed_key = sample["conditions"].get(
            "base_sample_id", sample["sample_id"]
        )
        random_seed = (
            config["seed"] + zlib.crc32(seed_key.encode("utf-8"))
        ) % (2**32)
        selected, _ = detect_canny_contour_ransac(
            image,
            config["canny_contour_ransac"],
            config["ransac"],
            random_seed,
        )
        row = _result_row(sample, "canny_contour_shared_ransac")
        detected = selected["ellipse"] if selected else None
        _add_evaluation(
            row,
            detected,
            truth,
            image.shape,
            config["success_iou"],
        )
        rows.append(row)
        if index % 50 == 0:
            print(f"Canny方式: {index}/{len(samples)}")
    return rows


def _evaluate_cnn(
    config: dict,
    dataset_dir: Path,
    split: str,
    limit: int | None,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config["paths"]["checkpoint"], device)
    dataset = RingDataset(dataset_dir, split, config["input_size"])
    if limit:
        dataset.samples = dataset.samples[:limit]
    loader = DataLoader(
        dataset,
        batch_size=config["cnn"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    rows = []

    with torch.inference_mode():
        for images, _, indices in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            for probability, index in zip(probabilities, indices.tolist()):
                sample = dataset.samples[index]
                label = read_json(dataset_dir / sample["label"])
                scale = config["input_size"] / float(label["image_width"])
                truth = scaled_ellipse(label_ellipse(label), scale)
                seed_key = sample["conditions"].get(
                    "base_sample_id", sample["sample_id"]
                )
                random_seed = (
                    config["seed"] + zlib.crc32(seed_key.encode("utf-8"))
                ) % (2**32)
                result = detect_from_probability(
                    probability,
                    config["ransac"],
                    random_seed,
                )
                row = _result_row(sample, "cnn_weighted_ransac")
                detected = result["ellipse"] if result else None
                _add_evaluation(
                    row,
                    detected,
                    truth,
                    probability.shape,
                    config["success_iou"],
                )
                rows.append(row)
            print(f"CNN方式: {len(rows)}/{len(dataset)}")
    return rows


def _save_results(output: Path, rows: list[dict]) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_count": len(rows),
        "detected_count": sum(row["detected"] for row in rows),
        "success_count": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
    }
    write_json(output.with_suffix(".json"), summary)
    return summary


def evaluate(config: dict, limit: int | None = None) -> None:
    """設定に記載したデータセットを3方式で評価する。"""
    result_root = Path(config["paths"]["results"])
    for dataset_name, dataset in config["evaluation_datasets"].items():
        dataset_dir = Path(dataset["path"])
        for method, evaluator in (
            ("canny_contour_ransac", _evaluate_canny),
            ("zhang2019", _evaluate_zhang),
            ("cnn_ransac", _evaluate_cnn),
        ):
            rows = evaluator(config, dataset_dir, dataset["split"], limit)
            summary = _save_results(
                result_root / dataset_name / f"{method}.csv",
                rows,
            )
            print(dataset_name, method, summary)
