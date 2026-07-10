import argparse
import json
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
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1枚の画像を古典法で評価する")
    parser.add_argument("--image", default="output/blender/paf_sample.png")
    parser.add_argument(
        "--label",
        default="output/ground_truth/paf_sample_inner_ring_points.json",
    )
    parser.add_argument("--config", default="config/baseline.json")
    parser.add_argument("--output-dir", default="output/baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = resolve_project_path(args.image)
    label_path = resolve_project_path(args.label)
    output_dir = resolve_project_path(args.output_dir)
    settings = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")
    label = json.loads(label_path.read_text(encoding="utf-8"))
    ground_truth = fit_ground_truth(label["image_points"])
    candidates = detect_candidates(image, settings["detector"])
    if not candidates:
        raise RuntimeError("楕円候補を検出できませんでした")

    detected = candidates[0]["ellipse"]
    evaluation = evaluate_ellipses(detected, ground_truth, image.shape)
    result = {
        "input": str(image_path),
        "detected": ellipse_to_dict(detected),
        "ground_truth": {
            "vertex_group": label["vertex_group"],
            **ellipse_to_dict(ground_truth),
        },
        "evaluation": evaluation,
        "candidate_count": len(candidates),
        "candidates": [
            candidate_to_dict(candidate, rank)
            for rank, candidate in enumerate(candidates, start=1)
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_dir / "paf_sample_evaluation.png"),
        draw_evaluation(image, detected, ground_truth),
    )
    cv2.imwrite(
        str(output_dir / "paf_sample_candidates.png"),
        draw_candidates(image, candidates),
    )
    (output_dir / "paf_sample_detected.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["evaluation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
