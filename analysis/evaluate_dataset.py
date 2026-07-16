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
from ellipse_ransac import fit_contour_ransac_candidates, select_paf_inner_candidate
from image_io import imread, imwrite
from zhang_arc_detector import (
    ZHANG_2019_REFERENCE,
    detect_zhang_arc_candidates,
    draw_zhang_arcs,
    select_zhang_inner_boundary,
)
from aamed_adapter import detect_aamed_candidates


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
        choices=(
            "contour_fit",
            "canny_ransac",
            "canny_ransac_inner_pair",
            "zhang2019_arc_reproduction",
            "aamed",
        ),
        default="contour_fit",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="manifestにsplitがある場合に評価対象を限定する",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既存details JSONを再利用して中断位置から続行する",
    )
    parser.add_argument(
        "--degradation",
        default=None,
        help="特定の劣化条件だけを評価する",
    )
    parser.add_argument(
        "--results-name",
        default=None,
        help="results以下の保存名。既定はmethod名",
    )
    parser.add_argument(
        "--minimum-severity",
        type=float,
        default=None,
        help="指定強度以上の劣化サンプルだけを評価する",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="オーバーレイと候補可視化を書き出さない",
    )
    parser.add_argument(
        "--reuse-candidates-from",
        default=None,
        help="指定結果名のdetailsからRANSAC候補を再利用し、選択規則だけ評価する",
    )
    parser.add_argument(
        "--paired-ransac-seed",
        action="store_true",
        help="同じ元画像の劣化・背景違いで同一RANSAC乱数列を使う",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="指定sample_idだけを評価する。複数回指定可能",
    )
    return parser.parse_args()


def condition_columns(conditions: dict) -> dict:
    """実験後にmanifestを再結合しなくても層別集計できる列を残す。"""
    camera = conditions.get("camera", {})
    lighting = conditions.get("lighting", {})
    background = conditions.get("background", "space")
    if isinstance(background, dict):
        background = background.get("type", "space")
    return {
        "camera_id": conditions["camera_id"],
        "lighting_id": conditions["lighting_id"],
        "background": background,
        "camera_tilt_deg": conditions.get("camera_tilt_deg", camera.get("tilt_deg")),
        "camera_azimuth_deg": conditions.get("camera_azimuth_deg", camera.get("azimuth_deg")),
        "light_tilt_deg": conditions.get("light_tilt_deg", lighting.get("tilt_deg")),
        "light_azimuth_deg": conditions.get("light_azimuth_deg", lighting.get("azimuth_deg")),
        "light_energy": conditions.get("light_energy", lighting.get("energy")),
        "degradation": conditions.get("degradation", "clean"),
        "severity": conditions.get("severity", 0.0),
    }


def ellipse_from_dict(values: dict):
    return (
        (float(values["center_x"]), float(values["center_y"])),
        (float(values["axis_1"]), float(values["axis_2"])),
        float(values["angle_deg"]),
    )


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_project_path(args.dataset)
    settings = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))

    results_dir = dataset_dir / "results" / (args.results_name or args.method)
    overlays_dir = results_dir / "overlays"
    candidates_dir = results_dir / "candidates"
    details_dir = results_dir / "details"
    for directory in (overlays_dir, candidates_dir, details_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    samples = [
        sample
        for sample in manifest["samples"]
        if (args.split is None or sample.get("split") == args.split)
        and (args.sample_id is None or sample["sample_id"] in set(args.sample_id))
        and (
            args.degradation is None
            or sample["conditions"].get("degradation", "clean") == args.degradation
        )
        and (
            args.minimum_severity is None
            or float(sample["conditions"].get("severity", 0.0)) >= args.minimum_severity
        )
    ]
    for sample in samples:
        sample_id = sample["sample_id"]
        image_path = dataset_dir / sample["image"]
        label_path = dataset_dir / sample["label"]
        detail_path = details_dir / f"{sample_id}.json"
        if args.resume and detail_path.exists():
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
            if (
                args.method == "zhang2019_arc_reproduction"
                and detail.get("method_reference") != ZHANG_2019_REFERENCE
            ):
                detail["method_reference"] = ZHANG_2019_REFERENCE
                detail_path.write_text(
                    json.dumps(detail, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            evaluation = detail.get("evaluation")
            conditions = sample["conditions"]
            candidate_key = {
                "contour_fit": "candidates",
                "zhang2019_arc_reproduction": "arc_candidates",
                "aamed": "aamed_candidates",
            }.get(args.method, "ransac_candidates")
            row = {
                "sample_id": sample_id,
                "method": args.method,
                "split": sample.get("split", "all"),
                **condition_columns(conditions),
                "candidate_count": len(detail.get(candidate_key, [])),
                "status": "detected" if evaluation is not None else "no_detection",
                "top1_matches_ground_truth": bool(
                    evaluation
                    and evaluation["ellipse_iou"]
                    >= float(settings["success_thresholds"]["ellipse_iou"])
                ),
                "center_error_px": None,
                "minor_axis_error_px": None,
                "major_axis_error_px": None,
                "angle_error_deg": None,
                "ellipse_iou": None,
            }
            if evaluation is not None:
                row.update(evaluation)
            rows.append(row)
            continue
        image = imread(image_path)
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
        elif args.method == "zhang2019_arc_reproduction":
            arc_candidates, arc_stages = detect_zhang_arc_candidates(
                image,
                settings["detector"],
                settings["zhang2019_arc"],
            )
            if arc_candidates:
                arc_result = select_zhang_inner_boundary(
                    arc_candidates, settings["zhang2019_inner_selector"]
                )
                detected = arc_result["ellipse"]
                method_detail["selection_mode"] = arc_result["selection_mode"]
                method_detail["nested_candidate_ranks"] = arc_result.get(
                    "nested_candidate_ranks"
                )
            method_detail["arc_count"] = len(arc_stages["arcs"])
            method_detail["arc_candidates"] = [
                {
                    "rank": rank,
                    **ellipse_to_dict(candidate["ellipse"]),
                    **{key: value for key, value in candidate.items() if key != "ellipse"},
                }
                for rank, candidate in enumerate(arc_candidates, start=1)
            ]
            candidates = arc_candidates
            if not args.no_images:
                candidate_visualization = draw_zhang_arcs(
                    image, arc_stages, arc_candidates
                )
        elif args.method == "aamed":
            aamed_candidates = detect_aamed_candidates(image, settings["aamed"])
            if aamed_candidates:
                detected = aamed_candidates[0]["ellipse"]
            method_detail["aamed_candidates"] = [
                {
                    "rank": rank,
                    **ellipse_to_dict(candidate["ellipse"]),
                    **{key: value for key, value in candidate.items() if key != "ellipse"},
                }
                for rank, candidate in enumerate(aamed_candidates, start=1)
            ]
            candidates = aamed_candidates
            if not args.no_images:
                candidate_visualization = image.copy()
                for rank, candidate in enumerate(aamed_candidates[:5], start=1):
                    cv2.ellipse(
                        candidate_visualization,
                        candidate["ellipse"],
                        (0, 0, 255) if rank == 1 else (0, 200, 255),
                        2 if rank == 1 else 1,
                        cv2.LINE_AA,
                    )
        else:
            preprocessing = None
            if args.reuse_candidates_from:
                source_detail_path = (
                    dataset_dir
                    / "results"
                    / args.reuse_candidates_from
                    / "details"
                    / f"{sample_id}.json"
                )
                source_detail = json.loads(source_detail_path.read_text(encoding="utf-8"))
                ransac_candidates = [
                    {**candidate, "ellipse": ellipse_from_dict(candidate)}
                    for candidate in source_detail["ransac_candidates"]
                ]
            else:
                preprocessing = preprocess_image(image, settings["detector"])
                conditions = sample["conditions"]
                seed_key = sample_id
                if args.paired_ransac_seed:
                    seed_key = conditions.get("base_sample_id") or (
                        f"{conditions['camera_id']}|{conditions['lighting_id']}"
                    )
                sample_seed = (
                    int(settings["ransac"]["random_seed"])
                    + zlib.crc32(seed_key.encode("utf-8"))
                ) % (2**32)
                ransac_candidates = fit_contour_ransac_candidates(
                    preprocessing["contours"],
                    image.shape,
                    settings["ransac"],
                    random_seed=sample_seed,
                )
            if ransac_candidates:
                ransac_result = (
                    select_paf_inner_candidate(
                        ransac_candidates, settings["inner_pair_selector"]
                    )
                    if args.method == "canny_ransac_inner_pair"
                    else ransac_candidates[0]
                )
                detected = ransac_result["ellipse"]
                method_detail["selection_mode"] = ransac_result.get(
                    "selection_mode", "quality_only"
                )
                method_detail["inner_pair"] = ransac_result.get("inner_pair")
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
            if not args.no_images:
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
            if ransac_result is not None and not args.no_images and preprocessing is not None:
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
            "split": sample.get("split", "all"),
            **condition_columns(sample["conditions"]),
            "candidate_count": (
                len(ransac_candidates)
                if args.method in ("canny_ransac", "canny_ransac_inner_pair")
                else len(candidates)
            ),
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
        if args.method == "zhang2019_arc_reproduction":
            detail["method_reference"] = ZHANG_2019_REFERENCE

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
            if not args.no_images:
                imwrite(
                    overlays_dir / f"{sample_id}.png",
                    draw_evaluation(image, detected, ground_truth),
                )

        if not args.no_images:
            imwrite(
                candidates_dir / f"{sample_id}.png",
                candidate_visualization,
            )
        detail_path.write_text(
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
    if args.method == "zhang2019_arc_reproduction":
        aggregate["method_reference"] = ZHANG_2019_REFERENCE
    (results_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
