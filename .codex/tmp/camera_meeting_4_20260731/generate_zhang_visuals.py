"""第4回進捗発表用にZhang 2019型（再現実装）の中間処理図を生成する。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paf_ring_detection.data import label_ellipse, read_image, read_json
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.zhang2019 import detect_zhang_arc_candidates
from paf_ring_detection.methods.zhang2019_paf import (
    score_zhang_candidates_for_paf,
    select_zhang_inner_boundary,
)


BUILD_DIR = Path(__file__).resolve().parent
ASSET_DIR = BUILD_DIR / "generated-assets"
SAMPLE_ID = "camera_t030_a135_d039.0_o00__light_t020_a060_e01.5__bg_space"
DATASET_DIR = ROOT / "output" / "datasets" / "ood_evaluation"

# 象限I～IVを一目で追えるよう、明度差の大きい色に固定する。
QUADRANT_COLORS = (
    (41, 180, 246),
    (232, 98, 154),
    (46, 204, 113),
    (255, 193, 7),
)


def ellipse_for_cv(ellipse):
    """OpenCVの描画関数へ渡せる数値型へそろえる。"""
    center, axes, angle = ellipse
    return (
        (int(round(center[0])), int(round(center[1]))),
        (int(round(axes[0])), int(round(axes[1]))),
        float(angle),
    )


def ellipse_samples(ellipse, count: int) -> np.ndarray:
    """楕円周上を等角度でサンプリングする。"""
    (center_x, center_y), (axis_1, axis_2), angle = ellipse
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local = np.column_stack(
        (axis_1 * 0.5 * np.cos(theta), axis_2 * 0.5 * np.sin(theta))
    )
    radians = np.deg2rad(angle)
    rotation = np.asarray(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    return local @ rotation.T + np.asarray([center_x, center_y])


def draw_arcs(base: np.ndarray, arcs: list[dict], indices=None) -> np.ndarray:
    """抽出弧を象限ごとの色で元画像へ重ねる。"""
    canvas = base.copy()
    selected = set(range(len(arcs))) if indices is None else set(indices)
    for index, arc in enumerate(arcs):
        if index not in selected:
            continue
        points = np.rint(arc["points"]).astype(np.int32).reshape(-1, 1, 2)
        color = QUADRANT_COLORS[int(arc["quadrant"])]
        cv2.polylines(canvas, [points], False, color, 3, cv2.LINE_AA)
    return canvas


def save_image(name: str, image: np.ndarray) -> Path:
    """BGR画像をPNGとして保存する。"""
    path = ASSET_DIR / name
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"画像をPNGへ変換できませんでした: {path}")
    path.write_bytes(encoded.tobytes())
    return path


def fit_square(path: Path, size: int) -> Image.Image:
    """処理段階の比較用に、余白を保った正方形へ収める。"""
    image = Image.open(path).convert("RGB")
    return ImageOps.pad(image, (size, size), color=(8, 12, 18), method=Image.Resampling.LANCZOS)


def make_process_strip(paths: list[Path]) -> Path:
    """入力から最終推定までを横一列の処理図へまとめる。"""
    labels = ["入力", "Canny", "弧の分類", "三弧統合", "最終選択"]
    panel_size = 260
    label_height = 54
    gap = 10
    width = len(paths) * panel_size + (len(paths) - 1) * gap
    canvas = Image.new("RGB", (width, panel_size + label_height), "white")
    font_path = Path("C:/Windows/Fonts/YuGothB.ttc")
    font = ImageFont.truetype(str(font_path), 24)
    draw = ImageDraw.Draw(canvas)

    for index, (path, label) in enumerate(zip(paths, labels)):
        left = index * (panel_size + gap)
        canvas.paste(fit_square(path, panel_size), (left, label_height))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (left + (panel_size - text_width) / 2, 10),
            label,
            font=font,
            fill=(20, 35, 50),
        )

    output = ASSET_DIR / "zhang_process_strip.png"
    canvas.save(output)
    return output


def make_evaluation_montage() -> Path:
    """OOD背景と撮像劣化の代表例を2段の図へまとめる。"""
    ood_root = ROOT / "output" / "datasets" / "ood_evaluation" / "images"
    diagnostic_root = ROOT / "output" / "datasets" / "diagnostic_evaluation" / "images"
    ood_paths = [
        ood_root / f"camera_t030_a135_d039.0_o00__light_t020_a060_e01.5__bg_{name}.png"
        for name in ("space", "earth", "moon")
    ]
    diagnostic_names = (
        "camera_t020_a000_d044.0_o01__light_t000_a000_e03.0"
        "__diagnostic__black_rectangle_s0750_v03.png",
        "camera_t020_a000_d044.0_o01__light_t000_a000_e03.0"
        "__diagnostic__sensor_whiteout_s0750_v07.png",
        "camera_t020_a000_d044.0_o01__light_t000_a000_e03.0"
        "__diagnostic__sensor_black_crush_s0750_v11.png",
    )
    diagnostic_paths = [diagnostic_root / name for name in diagnostic_names]
    labels = ["宇宙", "地球", "月", "黒矩形 75%", "白飛び 75%", "黒つぶれ 75%"]

    tile = 230
    label_height = 42
    row_gap = 14
    canvas = Image.new(
        "RGB",
        (tile * 3, (tile + label_height) * 2 + row_gap),
        "white",
    )
    font = ImageFont.truetype("C:/Windows/Fonts/YuGothB.ttc", 21)
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(zip(ood_paths + diagnostic_paths, labels)):
        if not path.exists():
            raise FileNotFoundError(path)
        row, column = divmod(index, 3)
        x = column * tile
        y = row * (tile + label_height + row_gap)
        canvas.paste(fit_square(path, tile), (x, y))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x + (tile - (bbox[2] - bbox[0])) / 2, y + tile + 7),
            label,
            font=font,
            fill=(20, 35, 50),
        )
    output = ASSET_DIR / "evaluation_conditions.png"
    canvas.save(output)
    return output


def make_validation_wide(
    validation_path: Path,
    edge_density: float,
    angular_coverage: float,
) -> Path:
    """横長画像枠向けに、支持点図と数値の読み方を1枚へまとめる。"""
    width, height = 1040, 270
    canvas = Image.new("RGB", (width, height), "white")
    validation = fit_square(validation_path, height)
    canvas.paste(validation, (0, 0))
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.truetype("C:/Windows/Fonts/YuGothB.ttc", 28)
    body = ImageFont.truetype("C:/Windows/Fonts/YuGothR.ttc", 24)
    draw.text((310, 30), "PAF固有の候補再評価", font=heading, fill=(20, 35, 50))
    draw.text((310, 90), "● 緑：近傍にCannyエッジあり", font=body, fill=(40, 150, 80))
    draw.text((310, 130), "● 赤：近傍エッジなし", font=body, fill=(205, 55, 55))
    draw.text(
        (310, 185),
        f"全周支持率 {edge_density * 100:.1f}%　角度被覆 {angular_coverage * 100:.1f}%",
        font=body,
        fill=(20, 35, 50),
    )
    output = ASSET_DIR / "zhang_paf_validation_wide.png"
    canvas.save(output)
    return output


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8"))
    settings = config["zhang2019"]
    image_path = DATASET_DIR / "images" / f"{SAMPLE_ID}.png"
    label_path = DATASET_DIR / "labels" / f"{SAMPLE_ID}.json"
    image = read_image(image_path)
    truth = label_ellipse(read_json(label_path))

    paper_candidates, stages = detect_zhang_arc_candidates(
        image,
        settings["preprocess"],
        settings["detector"],
    )
    scored_candidates = score_zhang_candidates_for_paf(
        paper_candidates,
        stages,
        settings["paf_postprocess"]["candidate_validation"],
    )
    selected = select_zhang_inner_boundary(
        scored_candidates,
        settings["paf_postprocess"]["selector"],
    )
    if selected is None:
        raise RuntimeError("代表画像でZhang型の候補を選択できませんでした。")

    # 1. 入力画像
    input_path = save_image("zhang_input.png", image)

    # 2. Cannyエッジ
    edges_rgb = cv2.cvtColor(stages["edges"], cv2.COLOR_GRAY2BGR)
    edges_path = save_image("zhang_canny.png", edges_rgb)

    # 3. 象限・凸性で残った弧
    dark = (image.astype(np.float32) * 0.34).astype(np.uint8)
    arc_image = draw_arcs(dark, stages["arcs"])
    arcs_path = save_image("zhang_arc_classes.png", arc_image)

    # 4. 選択された三弧と、その弧から当てはめた楕円
    group_image = cv2.cvtColor(stages["edges"] // 3, cv2.COLOR_GRAY2BGR)
    group_image = draw_arcs(group_image, stages["arcs"], selected["arc_indices"])
    cv2.ellipse(group_image, ellipse_for_cv(selected["ellipse"]), (0, 170, 255), 3, cv2.LINE_AA)
    group_path = save_image("zhang_three_arc_fit.png", group_image)

    # 論文コアが複数の楕円候補を返すことを示す。
    candidate_image = (image.astype(np.float32) * 0.58).astype(np.uint8)
    for rank, candidate in enumerate(scored_candidates[:8]):
        color = (80 + rank * 12, 130, 240 - rank * 12)
        cv2.ellipse(
            candidate_image,
            ellipse_for_cv(candidate["ellipse"]),
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.ellipse(candidate_image, ellipse_for_cv(selected["ellipse"]), (46, 204, 113), 4, cv2.LINE_AA)
    candidate_path = save_image("zhang_candidate_ellipses.png", candidate_image)

    # PAF固有処理で使う、楕円全周のエッジ支持を点ごとに可視化する。
    validation_image = (image.astype(np.float32) * 0.42).astype(np.uint8)
    validation_points = ellipse_samples(selected["ellipse"], 180)
    validation_pixels = np.rint(validation_points).astype(int)
    distance_map = cv2.distanceTransform(255 - stages["edges"], cv2.DIST_L2, 3)
    height, width = distance_map.shape
    threshold = float(
        settings["paf_postprocess"]["candidate_validation"][
            "edge_distance_threshold_px"
        ]
    )
    for x, y in validation_pixels:
        if 0 <= x < width and 0 <= y < height:
            supported = distance_map[y, x] <= threshold
            color = (46, 204, 113) if supported else (70, 70, 230)
            cv2.circle(validation_image, (x, y), 3, color, -1, cv2.LINE_AA)
    validation_path = save_image("zhang_paf_validation.png", validation_image)
    validation_wide_path = make_validation_wide(
        validation_path,
        float(selected["edge_density"]),
        float(selected["angular_coverage"]),
    )

    # PAF固有の同心候補対から内側を選ぶ様子を示す。
    nested_image = (image.astype(np.float32) * 0.50).astype(np.uint8)
    nested_ranks = selected.get("nested_candidate_ranks", [1, 2])
    for position, rank in enumerate(nested_ranks[:2]):
        if 1 <= rank <= len(scored_candidates):
            color = (41, 180, 246) if position == 0 else (232, 98, 154)
            cv2.ellipse(
                nested_image,
                ellipse_for_cv(scored_candidates[rank - 1]["ellipse"]),
                color,
                4,
                cv2.LINE_AA,
            )
    cv2.ellipse(nested_image, ellipse_for_cv(selected["ellipse"]), (46, 204, 113), 4, cv2.LINE_AA)
    nested_path = save_image("zhang_nested_selection.png", nested_image)

    # 最終推定（緑）とCG正解（シアン）を重ねる。
    final_image = image.copy()
    cv2.ellipse(final_image, ellipse_for_cv(truth), (255, 210, 50), 4, cv2.LINE_AA)
    cv2.ellipse(final_image, ellipse_for_cv(selected["ellipse"]), (46, 204, 113), 4, cv2.LINE_AA)
    final_path = save_image("zhang_final_overlay.png", final_image)

    process_strip = make_process_strip(
        [input_path, edges_path, arcs_path, group_path, final_path]
    )
    evaluation_montage = make_evaluation_montage()
    metrics = evaluate_ellipses(selected["ellipse"], truth, image.shape)
    metrics.update(
        {
            "sample_id": SAMPLE_ID,
            "extracted_arc_count": len(stages["arcs"]),
            "paper_candidate_count": len(paper_candidates),
            "paf_validated_candidate_count": len(scored_candidates),
            "selection_mode": selected.get("selection_mode"),
            "selected_arc_quadrants": selected.get("arc_quadrants"),
            "selected_group_fit_ratio": selected.get("group_fit_ratio"),
            "selected_edge_density": selected.get("edge_density"),
            "selected_angular_coverage": selected.get("angular_coverage"),
            "generated_assets": {
                "process_strip": str(process_strip),
                "candidate_ellipses": str(candidate_path),
                "paf_validation": str(validation_path),
                "paf_validation_wide": str(validation_wide_path),
                "nested_selection": str(nested_path),
                "evaluation_conditions": str(evaluation_montage),
            },
        }
    )
    (ASSET_DIR / "zhang_visual_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
