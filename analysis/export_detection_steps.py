import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ellipse_baseline import (
    detect_candidates,
    draw_candidates,
    draw_evaluation,
    fit_ground_truth,
    preprocess_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="楕円検出の各画像処理段階を出力する")
    parser.add_argument(
        "--image",
        default="output/datasets/pilot_v1/images/view_high__light_left.png",
    )
    parser.add_argument(
        "--label",
        default="output/datasets/pilot_v1/labels/view_high__light_left.json",
    )
    parser.add_argument("--config", default="config/baseline.json")
    parser.add_argument(
        "--output-dir",
        default="output/detection_steps/view_high__light_left",
    )
    return parser.parse_args()


def to_bgr(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def make_contact_sheet(stages: list[tuple[str, np.ndarray]]) -> np.ndarray:
    tile_width = 320
    tile_height = 320
    columns = 4
    rows = (len(stages) + columns - 1) // columns
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, (label, image) in enumerate(stages):
        resized = cv2.resize(to_bgr(image), (tile_width, tile_height))
        tile = add_label(resized, label)
        row, column = divmod(index, columns)
        sheet[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    return sheet


def main() -> None:
    args = parse_args()
    image_path = resolve_project_path(args.image)
    output_dir = resolve_project_path(args.output_dir)
    settings = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    label = json.loads(resolve_project_path(args.label).read_text(encoding="utf-8"))

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")

    preprocessing = preprocess_image(image, settings["detector"])
    candidates = detect_candidates(image, settings["detector"])
    if not candidates:
        raise RuntimeError("楕円候補を検出できませんでした")
    ground_truth = fit_ground_truth(label["image_points"])
    detected = candidates[0]["ellipse"]

    contours_only = np.zeros_like(image)
    cv2.drawContours(
        contours_only,
        preprocessing["contours"],
        -1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    top1 = image.copy()
    cv2.ellipse(top1, detected, (0, 0, 255), 1, cv2.LINE_AA)

    stages = [
        ("1 Original", image),
        ("2 Grayscale", preprocessing["gray"]),
        ("3 Gaussian blur", preprocessing["blurred"]),
        ("4 Canny edges", preprocessing["edges"]),
        ("5 Contours", contours_only),
        ("6 Ellipse candidates", draw_candidates(image, candidates)),
        ("7 Top-1 ellipse", top1),
        ("8 GT comparison", draw_evaluation(image, detected, ground_truth)),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (_, stage_image) in enumerate(stages, start=1):
        cv2.imwrite(str(output_dir / f"{index:02d}.png"), stage_image)
    cv2.imwrite(str(output_dir / "contact_sheet.png"), make_contact_sheet(stages))
    print(f"Saved stages: {output_dir}")


if __name__ == "__main__":
    main()
