from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.ellipse_baseline import draw_evaluation, evaluate_ellipses, fit_ground_truth
from analysis.ellipse_ransac import fit_ellipse_ransac
from paflab.dataset import StressDataset
from paflab.image_io import imread, imwrite
from paflab.model import TinyUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN確率マップと重み付きRANSACを評価する")
    parser.add_argument("--config", default="config/research_experiment.json")
    parser.add_argument("--split", default=None)
    parser.add_argument("--results-name", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--minimum-severity", type=float, default=None)
    parser.add_argument(
        "--paired-ransac-seed",
        action="store_true",
        help="同じ元画像の劣化・背景違いで同一RANSAC乱数列を使う",
    )
    return parser.parse_args()


def probability_ellipse(
    probability: np.ndarray,
    settings: dict,
    *,
    random_seed: int,
):
    threshold = float(settings["probability_threshold"])
    rows, columns = np.nonzero(probability >= threshold)
    if len(rows) < 5:
        return None
    points = np.column_stack((columns, rows)).astype(np.float32)
    weights = probability[rows, columns].astype(np.float64)
    maximum = int(settings["max_points"])
    if len(points) > maximum:
        rng = np.random.default_rng(random_seed)
        distribution = weights / weights.sum()
        indices = rng.choice(len(points), size=maximum, replace=False, p=distribution)
        points = points[indices]
        weights = weights[indices]
    ransac_settings = {
        key: value
        for key, value in settings.items()
        if key not in ("probability_threshold", "max_points")
    }
    return fit_ellipse_ransac(
        points,
        probability.shape,
        ransac_settings,
        weights=weights,
        random_seed=random_seed,
    )


def main() -> None:
    args = parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    split = args.split or config["evaluation"]["split"]
    dataset_dir = project_path(args.dataset or config["stress_dataset_dir"])
    artifacts_dir = project_path(config["artifacts_dir"])
    checkpoint_path = project_path(args.checkpoint) if args.checkpoint else artifacts_dir / "cnn_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    dataset = StressDataset(
        dataset_dir,
        split=split,
        input_size=int(config["input_size"]),
        return_index=True,
    )
    if args.minimum_severity is not None:
        dataset.samples = [
            sample
            for sample in dataset.samples
            if float(sample["conditions"].get("severity", 0.0)) >= args.minimum_severity
        ]
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    method_name = args.results_name or "cnn_ransac"
    results_dir = project_path(args.results_dir) if args.results_dir else dataset_dir / "results" / method_name
    overlays_dir = results_dir / "overlays"
    probability_dir = results_dir / "probability"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []

    with torch.inference_mode():
        for images, _, indices in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()[:, 0]
            for probability, dataset_index in zip(probabilities, indices.tolist()):
                sample = dataset.samples[dataset_index]
                label = json.loads((dataset_dir / sample["label"]).read_text(encoding="utf-8"))
                scale = float(config["input_size"]) / float(label["image_width"])
                scaled_points = np.asarray(label["image_points"], dtype=np.float32) * scale
                ground_truth = fit_ground_truth(scaled_points)
                conditions = sample["conditions"]
                seed_key = sample["sample_id"]
                if args.paired_ransac_seed:
                    seed_key = conditions.get("base_sample_id") or (
                        f"{conditions['camera_id']}|{conditions['lighting_id']}"
                    )
                sample_seed = (
                    int(config["cnn_ransac"]["random_seed"])
                    + zlib.crc32(seed_key.encode("utf-8"))
                ) % (2**32)
                result = probability_ellipse(
                    probability, config["cnn_ransac"], random_seed=sample_seed
                )
                row = {
                    "sample_id": sample["sample_id"],
                    "method": method_name,
                    "split": split,
                    "camera_id": conditions["camera_id"],
                    "lighting_id": conditions["lighting_id"],
                    "degradation": conditions["degradation"],
                    "severity": conditions["severity"],
                    "background": conditions.get("background", {}).get("type", "space"),
                    "camera_tilt_deg": conditions["camera"].get("tilt_deg"),
                    "camera_azimuth_deg": conditions["camera"].get("azimuth_deg"),
                    "light_tilt_deg": conditions["lighting"].get("tilt_deg"),
                    "light_azimuth_deg": conditions["lighting"].get("azimuth_deg"),
                    "light_energy": conditions["lighting"].get("energy"),
                    "status": "no_detection",
                    "top1_matches_ground_truth": False,
                    "center_error_px": None,
                    "minor_axis_error_px": None,
                    "major_axis_error_px": None,
                    "angle_error_deg": None,
                    "ellipse_iou": None,
                    "ransac_score": None,
                    "ransac_inlier_count": None,
                    "ransac_angular_coverage": None,
                    "ransac_mean_inlier_distance_px": None,
                }
                if result is not None:
                    evaluation = evaluate_ellipses(
                        result["ellipse"], ground_truth, (config["input_size"],) * 2
                    )
                    matched = evaluation["ellipse_iou"] >= float(
                        config["evaluation"]["success_ellipse_iou"]
                    )
                    row.update(
                        {
                            "status": "detected",
                            "top1_matches_ground_truth": matched,
                            "ransac_score": result["score"],
                            "ransac_inlier_count": result["inlier_count"],
                            "ransac_angular_coverage": result["angular_coverage"],
                            "ransac_mean_inlier_distance_px": result["mean_inlier_distance_px"],
                            **evaluation,
                        }
                    )
                    if not args.no_images:
                        source = imread(dataset_dir / sample["image"], cv2.IMREAD_COLOR)
                        source = cv2.resize(
                            source,
                            (int(config["input_size"]), int(config["input_size"])),
                            interpolation=cv2.INTER_AREA,
                        )
                        imwrite(
                            overlays_dir / f"{sample['sample_id']}.png",
                            draw_evaluation(source, result["ellipse"], ground_truth),
                        )
                if not args.no_images:
                    imwrite(
                        probability_dir / f"{sample['sample_id']}.png",
                        np.rint(probability * 255.0).astype(np.uint8),
                    )
                rows_out.append(row)

    fieldnames = list(rows_out[0]) if rows_out else []
    with (results_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    detected = [row for row in rows_out if row["status"] == "detected"]
    aggregate = {
        "method": method_name,
        "checkpoint": str(checkpoint_path),
        "base_channels": int(checkpoint["base_channels"]),
        "parameter_count": int(checkpoint.get("parameter_count", sum(p.numel() for p in model.parameters()))),
        "seed": int(checkpoint.get("seed", checkpoint.get("config", {}).get("seed", 0))),
        "split": split,
        "sample_count": len(rows_out),
        "detected_count": len(detected),
        "top1_match_count": sum(bool(row["top1_matches_ground_truth"]) for row in rows_out),
        "top1_match_rate": (
            sum(bool(row["top1_matches_ground_truth"]) for row in rows_out) / len(rows_out)
            if rows_out
            else 0.0
        ),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
