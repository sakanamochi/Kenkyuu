import argparse
import csv
import json
import zlib
from pathlib import Path

import cv2

from ellipse_baseline import (
    candidate_to_dict,
    detect_candidates,
    draw_candidates,
    draw_evaluation,
    ellipse_to_dict,
    evaluate_ellipses,
    fit_ground_truth,
    preprocess_image,
)
from ellipse_ransac import fit_contour_ransac_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CGデータセットを古典法で評価する")
    parser.add_argument("--dataset", required=True, help="データセットのディレクトリ")
    parser.add_argument(
        "--config",
        default="config/baseline.json",
        help="古典法と成功判定の設定JSON",
    )
    parser.add_argument(
        "--method",
        choices=("contour_fit", "canny_ransac"),
        default="contour_fit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_project_path(args.dataset)
    settings = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))

    results_dir = dataset_dir / "results" / args.method
    overlays_dir = results_dir / "overlays"
    candidates_dir = results_dir / "candidates"
    details_dir = results_dir / "details"
    for directory in (overlays_dir, candidates_dir, details_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        image_path = dataset_dir / sample["image"]
        label_path = dataset_dir / sample["label"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")

        label = json.loads(label_path.read_text(encoding="utf-8"))
        ground_truth = fit_ground_truth(label["image_points"])
        candidates = []
        ransac_result = None
        ransac_candidates = []
        detected = None
        candidate_visualization = None
        method_detail = {}

        if args.method == "contour_fit":
            candidates = detect_candidates(image, settings["detector"])
            if candidates:
                detected = candidates[0]["ellipse"]
            candidate_visualization = draw_candidates(image, candidates)
            method_detail["candidates"] = [
                candidate_to_dict(candidate, rank)
                for rank, candidate in enumerate(candidates, start=1)
            ]
        else:
            preprocessing = preprocess_image(image, settings["detector"])
            sample_seed = (
                int(settings["ransac"]["random_seed"])
                + zlib.crc32(sample_id.encode("utf-8"))
            ) % (2**32)
            ransac_candidates = fit_contour_ransac_candidates(
                preprocessing["contours"],
                image.shape,
                settings["ransac"],
                random_seed=sample_seed,
            )
            if ransac_candidates:
                ransac_result = ransac_candidates[0]
                detected = ransac_result["ellipse"]
            method_detail["ransac_candidates"] = [
                {
                    "rank": rank,
                    **ellipse_to_dict(candidate["ellipse"]),
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key not in ("ellipse", "inlier_mask")
                    },
                }
                for rank, candidate in enumerate(ransac_candidates, start=1)
            ]
            candidate_visualization = image.copy()
            for rank, candidate in enumerate(ransac_candidates, start=1):
                color = (0, 0, 255) if rank == 1 else (255, 160, 0)
                cv2.ellipse(
                    candidate_visualization,
                    candidate["ellipse"],
                    color,
                    1,
                    cv2.LINE_AA,
                )
            if ransac_result is not None:
                contour = preprocessing["contours"][ransac_result["contour_index"]]
                contour_points = contour[:, 0, :]
                inlier_points = contour_points[ransac_result["inlier_mask"]].astype(int)
                valid = (
                    (inlier_points[:, 0] >= 0)
                    & (inlier_points[:, 0] < image.shape[1])
                    & (inlier_points[:, 1] >= 0)
                    & (inlier_points[:, 1] < image.shape[0])
                )
                inlier_points = inlier_points[valid]
                candidate_visualization[
                    inlier_points[:, 1], inlier_points[:, 0]
                ] = (0, 255, 0)

        row = {
            "sample_id": sample_id,
            "method": args.method,
            "camera_id": sample["conditions"]["camera_id"],
            "lighting_id": sample["conditions"]["lighting_id"],
            "candidate_count": len(candidates) if args.method == "contour_fit" else len(ransac_candidates),
            "status": "no_detection",
            "top1_matches_ground_truth": False,
            "center_error_px": None,
            "minor_axis_error_px": None,
            "major_axis_error_px": None,
            "angle_error_deg": None,
            "ellipse_iou": None,
        }
        detail = {
            "sample_id": sample_id,
            "method": args.method,
            "image": sample["image"],
            "label": sample["label"],
            "conditions": sample["conditions"],
            "ground_truth": ellipse_to_dict(ground_truth),
            "detected": None,
            "evaluation": None,
            **method_detail,
        }

        if detected is not None:
            evaluation = evaluate_ellipses(detected, ground_truth, image.shape)
            matched = evaluation["ellipse_iou"] >= float(
                settings["success_thresholds"]["ellipse_iou"]
            )
            row.update(
                {
                    "status": "detected",
                    "top1_matches_ground_truth": matched,
                    **evaluation,
                }
            )
            detail["detected"] = ellipse_to_dict(detected)
            detail["evaluation"] = evaluation
            cv2.imwrite(
                str(overlays_dir / f"{sample_id}.png"),
                draw_evaluation(image, detected, ground_truth),
            )

        cv2.imwrite(
            str(candidates_dir / f"{sample_id}.png"),
            candidate_visualization,
        )
        (details_dir / f"{sample_id}.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(row)

    fieldnames = list(rows[0]) if rows else []
    with (results_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    detected_rows = [row for row in rows if row["status"] == "detected"]
    matched_count = sum(bool(row["top1_matches_ground_truth"]) for row in rows)
    metric_names = (
        "center_error_px",
        "minor_axis_error_px",
        "major_axis_error_px",
        "angle_error_deg",
        "ellipse_iou",
    )
    aggregate = {
        "method": args.method,
        "sample_count": len(rows),
        "detected_count": len(detected_rows),
        "top1_match_count": matched_count,
        "top1_match_rate": matched_count / len(rows) if rows else 0.0,
        "mean_metrics_for_detected": {
            name: (
                sum(float(row[name]) for row in detected_rows) / len(detected_rows)
                if detected_rows
                else None
            )
            for name in metric_names
        },
    }
    (results_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
