from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch

from analysis.ellipse_baseline import evaluate_ellipses, preprocess_image
from analysis.ellipse_ransac import _point_distances
from paflab.reporting.build_summary_figures import Predictor, setup_style
from paflab.image_io import imread, imwrite
from paflab.labels import fit_label_ellipse, rasterize_ring_mask, scale_ellipse


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/datasets/research_base_v1"
STRESS = ROOT / "output/datasets/paf_robustness_v3"
DIAGNOSTIC = ROOT / "output/datasets/paf_diagnostics_v1"
OUTPUT = ROOT / "output/presentation_assets/progress_meeting_1_v1"
FIGURES = OUTPUT / "figures"
PANELS = OUTPUT / "panels"

TEXT = "#182230"
MUTED = "#667085"
GRID = "#d0d5dd"
CLASSIC = "#f59e0b"
CNN = "#2563eb"
SUCCESS = "#21a366"
FAIL = "#e5484d"
GT = "#00a6d6"
LIGHT_BLUE = "#eaf2ff"
LIGHT_ORANGE = "#fff4e5"

BASE_SAMPLE_ID = "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0"
# 検出系の構成説明では、方式差ではなく処理内容へ注目できるよう、
# 旧Canny版とCNN版が同一の遮蔽なし入力でともに成功する代表例を固定する。
DETECTOR_SUCCESS_SAMPLE_ID = (
    "camera_t020_a090_d034.0_o02__light_t000_a000_e03.0"
    "__diagnostic__clean_s0000_v00"
)
DIAGNOSTIC_IDS = {
    "clean": f"{BASE_SAMPLE_ID}__diagnostic__clean_s0000_v00",
    "black_rectangle_025": (
        f"{BASE_SAMPLE_ID}__diagnostic__black_rectangle_s0250_v01"
    ),
    "black_rectangle_050": (
        f"{BASE_SAMPLE_ID}__diagnostic__black_rectangle_s0500_v02"
    ),
    "black_rectangle_075": (
        f"{BASE_SAMPLE_ID}__diagnostic__black_rectangle_s0750_v03"
    ),
    "black_rectangle_100": (
        f"{BASE_SAMPLE_ID}__diagnostic__black_rectangle_s1000_v04"
    ),
    "sensor_whiteout_025": (
        f"{BASE_SAMPLE_ID}__diagnostic__sensor_whiteout_s0250_v05"
    ),
    "sensor_whiteout_050": (
        f"{BASE_SAMPLE_ID}__diagnostic__sensor_whiteout_s0500_v06"
    ),
    "sensor_whiteout_075": (
        f"{BASE_SAMPLE_ID}__diagnostic__sensor_whiteout_s0750_v07"
    ),
    "sensor_whiteout_100": (
        f"{BASE_SAMPLE_ID}__diagnostic__sensor_whiteout_s1000_v08"
    ),
    "sensor_black_crush_025": (
        f"{BASE_SAMPLE_ID}__diagnostic__sensor_black_crush_s0250_v09"
    ),
    "sensor_black_crush_050": (
        f"{BASE_SAMPLE_ID}__diagnostic__sensor_black_crush_s0500_v10"
    ),
    "sensor_black_crush_075": (
        f"{BASE_SAMPLE_ID}__diagnostic__sensor_black_crush_s0750_v11"
    ),
    "sensor_black_crush_100": (
        f"{BASE_SAMPLE_ID}__diagnostic__sensor_black_crush_s1000_v12"
    ),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_manifest(dataset: Path) -> tuple[dict, dict[str, dict]]:
    manifest = read_json(dataset / "manifest.json")
    return manifest, {sample["sample_id"]: sample for sample in manifest["samples"]}


def load_rgb(dataset: Path, sample: dict, size: int | None = None) -> np.ndarray:
    image = imread(dataset / sample["image"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(dataset / sample["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def ellipse_from_dict(values: dict | None):
    if not values:
        return None
    return (
        (float(values["center_x"]), float(values["center_y"])),
        (float(values["axis_1"]), float(values["axis_2"])),
        float(values["angle_deg"]),
    )


def scale_cv_ellipse(ellipse, scale: float):
    if ellipse is None:
        return None
    return (
        (ellipse[0][0] * scale, ellipse[0][1] * scale),
        (ellipse[1][0] * scale, ellipse[1][1] * scale),
        ellipse[2],
    )


def label_ellipse(dataset: Path, sample: dict, size: int = 256):
    label = read_json(dataset / sample["label"])
    ellipse = fit_label_ellipse(label)
    scale = size / float(label["image_width"])
    return scale_ellipse(ellipse, scale, scale)


def probability_rgb(probability: np.ndarray) -> np.ndarray:
    return np.rint(plt.get_cmap("magma")(probability)[..., :3] * 255).astype(
        np.uint8
    )


def threshold_rgb(probability: np.ndarray, threshold: float) -> np.ndarray:
    mask = probability >= threshold
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = (235, 240, 255)
    return image


def tiny_unet_card(size: int = 256) -> np.ndarray:
    image = np.full((size, size, 3), (238, 244, 255), dtype=np.uint8)
    encoder = [(28, 42, 70, 82), (46, 94, 88, 134), (64, 146, 106, 186)]
    decoder = [(150, 146, 192, 186), (168, 94, 210, 134), (186, 42, 228, 82)]
    colors = [(37, 99, 235), (59, 130, 246), (96, 165, 250)]
    for index, box in enumerate(encoder):
        cv2.rectangle(image, box[:2], box[2:], colors[index], -1, cv2.LINE_AA)
        if index:
            cv2.arrowedLine(
                image,
                (encoder[index - 1][2], encoder[index - 1][3]),
                (box[0], box[1]),
                (71, 84, 103),
                2,
                cv2.LINE_AA,
                tipLength=0.2,
            )
    cv2.rectangle(image, (107, 184), (149, 224), (29, 78, 216), -1, cv2.LINE_AA)
    cv2.arrowedLine(
        image,
        (106, 186),
        (108, 204),
        (71, 84, 103),
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )
    for index, box in enumerate(decoder):
        cv2.rectangle(image, box[:2], box[2:], colors[2 - index], -1, cv2.LINE_AA)
        source = (149, 204) if index == 0 else (
            decoder[index - 1][2],
            decoder[index - 1][1],
        )
        target = (box[0], box[3]) if index == 0 else (box[0], box[3])
        cv2.arrowedLine(
            image,
            source,
            target,
            (71, 84, 103),
            2,
            cv2.LINE_AA,
            tipLength=0.2,
        )
    for left, right in zip(encoder, reversed(decoder)):
        cv2.line(
            image,
            (left[2], (left[1] + left[3]) // 2),
            (right[0], (right[1] + right[3]) // 2),
            (148, 163, 184),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "RGB",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (24, 34, 48),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "1ch",
        (202, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (24, 34, 48),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "skip connections",
        (66, 244),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (71, 84, 103),
        1,
        cv2.LINE_AA,
    )
    return image


def draw_dashed_ellipse(
    image: np.ndarray,
    ellipse,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    center = tuple(int(round(value)) for value in ellipse[0])
    axes = tuple(max(1, int(round(value / 2))) for value in ellipse[1])
    points = cv2.ellipse2Poly(
        center,
        axes,
        int(round(ellipse[2])),
        0,
        360,
        4,
    )
    for index in range(0, len(points) - 1, 2):
        cv2.line(
            image,
            tuple(points[index]),
            tuple(points[index + 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )


def overlay_rgb(
    image: np.ndarray,
    ground_truth,
    predicted,
    evaluation: dict | None,
) -> np.ndarray:
    output = image.copy()
    draw_dashed_ellipse(output, ground_truth, (0, 166, 214), 3)
    if predicted is not None:
        success = bool(evaluation and evaluation["ellipse_iou"] >= 0.8)
        color = (33, 163, 102) if success else (229, 72, 77)
        cv2.ellipse(output, predicted, color, 4, cv2.LINE_AA)
    return output


def show_image(ax, image: np.ndarray, title: str | None = None) -> None:
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, pad=8)


def add_flow_arrow(ax) -> None:
    ax.annotate(
        "→",
        xy=(1.12, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=25,
        color=MUTED,
        annotation_clip=False,
    )


def result_label(evaluation: dict | None) -> str:
    if evaluation is None:
        return "検出なし"
    success = evaluation["ellipse_iou"] >= 0.8
    return f"IoU {evaluation['ellipse_iou']:.2f}  {'成功' if success else '失敗'}"


def add_result_badge(ax, evaluation: dict | None) -> None:
    success = bool(evaluation and evaluation["ellipse_iou"] >= 0.8)
    ax.text(
        0.03,
        0.96,
        result_label(evaluation),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": SUCCESS if success else "#111827",
            "alpha": 0.88,
            "edgecolor": "none",
        },
    )


def load_classic_detail(sample_id: str, result_name: str) -> dict:
    return read_json(
        DIAGNOSTIC
        / "results"
        / result_name
        / "details"
        / f"{sample_id}.json"
    )


def classic_prediction(
    sample_id: str,
    result_name: str,
    size: int = 256,
):
    detail = load_classic_detail(sample_id, result_name)
    scale = size / 480.0
    predicted = scale_cv_ellipse(ellipse_from_dict(detail.get("detected")), scale)
    ground_truth = scale_cv_ellipse(
        ellipse_from_dict(detail["ground_truth"]), scale
    )
    evaluation = (
        evaluate_ellipses(predicted, ground_truth, (size, size))
        if predicted is not None
        else None
    )
    return predicted, ground_truth, evaluation


def canny_prediction(
    sample_id: str,
    diagnostic_samples: dict[str, dict],
    size: int = 256,
) -> dict:
    sample = diagnostic_samples[sample_id]
    source = load_rgb(DIAGNOSTIC, sample)
    baseline = read_json(ROOT / "config/baseline.json")
    stages = preprocess_image(
        cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
        baseline["detector"],
    )
    detail = load_classic_detail(sample_id, "canny_ransac_diagnostic")
    candidates = [
        {**candidate, "ellipse": ellipse_from_dict(candidate)}
        for candidate in detail["ransac_candidates"]
    ]

    contour_visualization = source.copy()
    palette = (
        (37, 99, 235),
        (245, 158, 11),
        (33, 163, 102),
        (168, 85, 247),
        (236, 72, 153),
    )
    for index, contour in enumerate(stages["contours"]):
        cv2.drawContours(
            contour_visualization,
            [contour],
            -1,
            palette[index % len(palette)],
            2,
            cv2.LINE_AA,
        )

    candidate_visualization = source.copy()
    for index, candidate in enumerate(candidates[:12]):
        color = FAIL if index == 0 else CLASSIC
        thickness = 4 if index == 0 else 2
        cv2.ellipse(
            candidate_visualization,
            candidate["ellipse"],
            tuple(int(color[offset : offset + 2], 16) for offset in (1, 3, 5)),
            thickness,
            cv2.LINE_AA,
        )
    predicted, ground_truth, evaluation = classic_prediction(
        sample_id,
        "canny_ransac_diagnostic",
        size,
    )
    return {
        "image": cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA),
        "stages": stages,
        "contour_visualization": cv2.resize(
            contour_visualization,
            (size, size),
            interpolation=cv2.INTER_AREA,
        ),
        "candidate_visualization": cv2.resize(
            candidate_visualization,
            (size, size),
            interpolation=cv2.INTER_AREA,
        ),
        "predicted": predicted,
        "ground_truth": ground_truth,
        "evaluation": evaluation,
    }


def cnn_threshold_inlier_rgb(
    probability: np.ndarray,
    ellipse,
    threshold: float,
    distance_threshold: float,
) -> np.ndarray:
    rows, columns = np.nonzero(probability >= threshold)
    points = np.column_stack((columns, rows)).astype(np.float32)
    output = np.zeros((*probability.shape, 3), dtype=np.uint8)
    output[rows, columns] = (125, 135, 150)
    if ellipse is not None and len(points):
        inliers = _point_distances(points, ellipse) <= distance_threshold
        inlier_points = points[inliers].astype(int)
        output[inlier_points[:, 1], inlier_points[:, 0]] = (33, 220, 140)
        cv2.ellipse(output, ellipse, (70, 140, 255), 2, cv2.LINE_AA)
    return output


def save_panel(name: str, image: np.ndarray) -> Path:
    path = PANELS / name
    imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path


def save_figure(fig, name: str, *, dpi: int = 180) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_dataset_overview(base_samples: dict[str, dict]) -> Path:
    camera_ids = [
        f"camera_t{tilt:03d}_a000_d034.0_o00__light_t045_a000_e03.0"
        for tilt in (0, 20, 40, 60, 75)
    ]
    camera_labels = [f"傾斜 {tilt}°" for tilt in (0, 20, 40, 60, 75)]
    lighting_ids = [
        "camera_t040_a090_d034.0_o00__light_t000_a000_e03.0",
        "camera_t040_a090_d034.0_o00__light_t045_a000_e03.0",
        "camera_t040_a090_d034.0_o00__light_t045_a120_e03.0",
        "camera_t040_a090_d034.0_o00__light_t045_a240_e03.0",
        "camera_t040_a090_d034.0_o00__light_t075_a000_e03.0",
        "camera_t040_a090_d034.0_o00__light_t075_a120_e03.0",
        "camera_t040_a090_d034.0_o00__light_t075_a240_e03.0",
    ]
    lighting_labels = [
        "軸方向",
        "45° / 0°",
        "45° / 120°",
        "45° / 240°",
        "75° / 0°",
        "75° / 120°",
        "75° / 240°",
    ]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("CGデータセット：視点と照明を制御して生成", fontsize=23, y=0.975)
    fig.text(
        0.5,
        0.925,
        "102カメラ条件 × 7照明条件 ＝ 714画像",
        ha="center",
        fontsize=16,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f2f4f7",
            "edgecolor": "none",
        },
    )
    outer = fig.add_gridspec(
        2,
        1,
        left=0.045,
        right=0.985,
        top=0.84,
        bottom=0.065,
        hspace=0.32,
    )
    top = outer[0].subgridspec(1, 5, wspace=0.08)
    bottom = outer[1].subgridspec(1, 7, wspace=0.07)
    for index, (sample_id, label) in enumerate(zip(camera_ids, camera_labels)):
        ax = fig.add_subplot(top[0, index])
        show_image(ax, load_rgb(BASE, base_samples[sample_id]), label)
    for index, (sample_id, label) in enumerate(zip(lighting_ids, lighting_labels)):
        ax = fig.add_subplot(bottom[0, index])
        show_image(ax, load_rgb(BASE, base_samples[sample_id]), label)
    fig.text(0.045, 0.855, "カメラ条件の代表例（照明固定）", fontsize=14, color=MUTED)
    fig.text(0.045, 0.435, "照明7条件（カメラ固定）", fontsize=14, color=MUTED)
    fig.text(
        0.985,
        0.02,
        "正解ラベル：PAF内周輪郭",
        ha="right",
        fontsize=12,
        color=MUTED,
    )
    return save_figure(fig, "01-dataset-overview.png")


def build_dataset_construction_flow(
    base_manifest: dict,
    stress_manifest: dict,
    base_samples: dict[str, dict],
    diagnostic_samples: dict[str, dict],
) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6))
    fig.suptitle("学習・評価データの構成", fontsize=22, y=0.985)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.86, bottom=0.12, wspace=0.24)

    base_image = load_rgb(BASE, base_samples[BASE_SAMPLE_ID])
    effect_ids = [
        DIAGNOSTIC_IDS["black_rectangle_025"],
        DIAGNOSTIC_IDS["sensor_whiteout_100"],
        DIAGNOSTIC_IDS["sensor_black_crush_025"],
    ]
    effect_images = [
        load_rgb(DIAGNOSTIC, diagnostic_samples[sample_id]) for sample_id in effect_ids
    ]

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.set_facecolor("#fafafa")

    axes[0].imshow(base_image)
    axes[0].set_title("1. BlenderでCG生成", fontsize=15, pad=10)
    axes[0].text(
        0.5,
        -0.08,
        "PAF姿勢・カメラ・照明を指定",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=11,
        color=MUTED,
    )

    axes[1].imshow(base_image)
    axes[1].set_title("2. ベース画像 714枚", fontsize=15, pad=10)
    axes[1].text(
        0.5,
        -0.08,
        (
            f"{base_manifest['camera_count']}カメラ条件 × "
            f"{base_manifest['lighting_count']}照明条件"
        ),
        transform=axes[1].transAxes,
        ha="center",
        fontsize=11,
        color=MUTED,
    )

    axes[2].axis("off")
    axes[2].set_title("3. カメラ条件単位で分割", fontsize=15, pad=10)
    split_counts = stress_manifest["split_counts"]
    base_split_counts = {
        split: sum(
            sample["split"] == split
            and sample["conditions"]["degradation"] == "clean"
            for sample in stress_manifest["samples"]
        )
        for split in ("train", "validation", "test")
    }
    rows = [
        ("学習", base_split_counts["train"], "#dbeafe"),
        ("検証", base_split_counts["validation"], "#fef3c7"),
        ("テスト", base_split_counts["test"], "#dcfce7"),
    ]
    for index, (label, count, color) in enumerate(rows):
        top = 0.76 - index * 0.25
        axes[2].add_patch(
            FancyBboxPatch(
                (0.12, top),
                0.76,
                0.16,
                boxstyle="round,pad=0.02",
                transform=axes[2].transAxes,
                facecolor=color,
                edgecolor="none",
            )
        )
        axes[2].text(
            0.22,
            top + 0.08,
            label,
            transform=axes[2].transAxes,
            va="center",
            fontsize=13,
        )
        axes[2].text(
            0.78,
            top + 0.08,
            f"{count}元画像",
            transform=axes[2].transAxes,
            va="center",
            ha="right",
            fontsize=13,
            fontweight=500,
        )
    axes[2].text(
        0.5,
        0.05,
        "学習・テストで異なるカメラ条件を使用",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    axes[3].axis("off")
    axes[3].set_title("4. 劣化画像を付与", fontsize=15, pad=10)
    for index, (image, label) in enumerate(
        zip(effect_images, ("遮蔽", "白飛び", "黒つぶれ"))
    ):
        inset = axes[3].inset_axes([0.02 + index * 0.33, 0.38, 0.30, 0.42])
        show_image(inset, image, label)
    axes[3].text(
        0.5,
        0.24,
        f"合計 {stress_manifest['sample_count']:,}枚",
        transform=axes[3].transAxes,
        ha="center",
        fontsize=17,
        fontweight=500,
    )
    axes[3].text(
        0.5,
        0.10,
        (
            f"学習 {split_counts['train']:,} / "
            f"検証 {split_counts['validation']:,} / "
            f"テスト {split_counts['test']:,}"
        ),
        transform=axes[3].transAxes,
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )
    for ax in axes[:-1]:
        add_flow_arrow(ax)
    return save_figure(fig, "02-dataset-construction-flow.png")


def build_ground_truth_and_training_labels(
    diagnostic_samples: dict[str, dict],
) -> Path:
    keys = [
        "clean",
        "black_rectangle_025",
        "sensor_whiteout_100",
        "sensor_black_crush_025",
    ]
    titles = ["Clean", "黒矩形 25%", "白飛び 100%", "黒つぶれ 25%"]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.3))
    fig.suptitle(
        "CNNの学習ペア：入力が劣化しても教師ラベルは完全な内周リング",
        fontsize=21,
        y=0.975,
    )
    for column, (key, title) in enumerate(zip(keys, titles)):
        sample = diagnostic_samples[DIAGNOSTIC_IDS[key]]
        image = load_rgb(DIAGNOSTIC, sample, 256)
        ellipse = label_ellipse(DIAGNOSTIC, sample, 256)
        mask = rasterize_ring_mask(
            256, 256, ellipse, thickness=4, blur_sigma=0.8
        )
        show_image(axes[0, column], image, title)
        show_image(axes[1, column], mask, "教師ラベル")
        if column == 0:
            axes[0, column].set_ylabel("入力", fontsize=13, labelpad=12)
            axes[1, column].set_ylabel("正解", fontsize=13, labelpad=12)
    fig.text(
        0.5,
        0.025,
        "正解ラベル：PAF内周輪郭　│　遮蔽部も幾何学的なリングを描画",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.035, 0.065, 0.99, 0.93), h_pad=1.2, w_pad=0.7)
    return save_figure(fig, "03-cnn-training-input-label-pairs.png")


def build_canny_pipeline(
    diagnostic_samples: dict[str, dict],
) -> tuple[Path, list[Path]]:
    sample_id = DETECTOR_SUCCESS_SAMPLE_ID
    prediction = canny_prediction(
        sample_id,
        diagnostic_samples,
    )
    image = prediction["image"]
    stages = prediction["stages"]
    output = overlay_rgb(
        image,
        prediction["ground_truth"],
        prediction["predicted"],
        prediction["evaluation"],
    )

    panels = [
        ("遮蔽なし入力", image),
        ("Gaussian平滑化", stages["blurred"]),
        ("Cannyエッジ", stages["edges"]),
        ("輪郭ごとに分割", prediction["contour_visualization"]),
        ("輪郭ごとにRANSAC", prediction["candidate_visualization"]),
        ("推定内周楕円", output),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.8))
    fig.suptitle("旧Canny＋輪郭別RANSACの処理ブロック", fontsize=22, y=0.99)
    for index, ((title, panel), ax) in enumerate(zip(panels, axes)):
        show_image(ax, panel, title)
        if index < len(axes) - 1:
            add_flow_arrow(ax)
    add_result_badge(axes[-1], prediction["evaluation"])
    fig.text(
        0.5,
        0.02,
        "Cannyで得た輪郭を分離し、それぞれへ独立にRANSAC楕円推定を適用",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.90), w_pad=1.1)

    saved_panels = [
        save_panel("canny-01-input.png", image),
        save_panel(
            "canny-02-canny-edges.png",
            cv2.cvtColor(stages["edges"], cv2.COLOR_GRAY2RGB),
        ),
        save_panel("canny-03-separated-contours.png", prediction["contour_visualization"]),
        save_panel("canny-04-ransac-candidates.png", prediction["candidate_visualization"]),
        save_panel("canny-05-output.png", output),
    ]
    return save_figure(fig, "04-canny-processing-blocks.png"), saved_panels


def build_cnn_pipeline(
    predictor: Predictor,
    diagnostic_samples: dict[str, dict],
) -> tuple[Path, list[Path]]:
    sample_id = DIAGNOSTIC_IDS["black_rectangle_025"]
    sample = diagnostic_samples[sample_id]
    prediction = predictor.predict(DIAGNOSTIC, sample)
    probability = prediction["probability"]
    threshold = float(predictor.config["cnn_ransac"]["probability_threshold"])
    distance = float(predictor.config["cnn_ransac"]["distance_threshold_px"])
    threshold_image = threshold_rgb(probability, threshold)
    inliers = cnn_threshold_inlier_rgb(
        probability, prediction["predicted"], threshold, distance
    )
    output = overlay_rgb(
        prediction["image"],
        prediction["ground_truth"],
        prediction["predicted"],
        prediction["evaluation"],
    )
    panels = [
        ("遮蔽入力", prediction["image"]),
        ("Tiny U-Net", tiny_unet_card()),
        ("リング尤度", probability_rgb(probability)),
        (f"閾値 ≥ {threshold:.2f}", threshold_image),
        ("重み付きRANSAC", inliers),
        ("推定楕円", output),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(16, 3.8))
    fig.suptitle("CNN版の処理ブロック", fontsize=22, y=0.99)
    for index, ((title, panel), ax) in enumerate(zip(panels, axes)):
        show_image(ax, panel, title)
        if index < len(axes) - 1:
            add_flow_arrow(ax)
    add_result_badge(axes[-1], prediction["evaluation"])
    fig.text(
        0.5,
        0.02,
        "遮蔽で尤度が弧になっても、残った高尤度点からRANSACで楕円を推定",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.90), w_pad=1.1)

    saved_panels = [
        save_panel("cnn-01-occluded-input.png", prediction["image"]),
        save_panel("cnn-02-probability.png", probability_rgb(probability)),
        save_panel("cnn-03-threshold-points.png", threshold_image),
        save_panel("cnn-04-ransac-inliers.png", inliers),
        save_panel("cnn-05-output.png", output),
    ]
    return save_figure(fig, "05-cnn-processing-blocks.png"), saved_panels


def distributed_five_points(points: np.ndarray, ellipse) -> np.ndarray:
    (cx, cy), (axis_1, axis_2), angle = ellipse
    centered = points - np.array([cx, cy], dtype=np.float32)
    radians = math.radians(angle)
    local_x = centered[:, 0] * math.cos(radians) + centered[:, 1] * math.sin(
        radians
    )
    local_y = -centered[:, 0] * math.sin(radians) + centered[:, 1] * math.cos(
        radians
    )
    point_angles = np.mod(
        np.arctan2(local_y / (axis_2 / 2), local_x / (axis_1 / 2)),
        2 * np.pi,
    )
    selected = []
    for target in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        difference = np.abs(np.angle(np.exp(1j * (point_angles - target))))
        for index in np.argsort(difference):
            if int(index) not in selected:
                selected.append(int(index))
                break
    return points[selected].astype(np.float32)


def point_cloud_rgb(
    shape: tuple[int, int],
    points: np.ndarray,
    *,
    highlighted: np.ndarray | None = None,
    ellipse=None,
    inlier_mask: np.ndarray | None = None,
) -> np.ndarray:
    output = np.zeros((*shape, 3), dtype=np.uint8)
    integer_points = np.rint(points).astype(int)
    valid = (
        (integer_points[:, 0] >= 0)
        & (integer_points[:, 0] < shape[1])
        & (integer_points[:, 1] >= 0)
        & (integer_points[:, 1] < shape[0])
    )
    integer_points = integer_points[valid]
    output[integer_points[:, 1], integer_points[:, 0]] = (160, 170, 185)
    if inlier_mask is not None:
        inlier_points = np.rint(points[inlier_mask]).astype(int)
        valid_inliers = (
            (inlier_points[:, 0] >= 0)
            & (inlier_points[:, 0] < shape[1])
            & (inlier_points[:, 1] >= 0)
            & (inlier_points[:, 1] < shape[0])
        )
        inlier_points = inlier_points[valid_inliers]
        output[inlier_points[:, 1], inlier_points[:, 0]] = (33, 220, 140)
    if ellipse is not None:
        cv2.ellipse(output, ellipse, (70, 140, 255), 2, cv2.LINE_AA)
    if highlighted is not None:
        for point in np.rint(highlighted).astype(int):
            cv2.circle(output, tuple(point), 5, (255, 76, 92), -1, cv2.LINE_AA)
            cv2.circle(output, tuple(point), 7, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def build_ransac_concept(
    predictor: Predictor,
    diagnostic_samples: dict[str, dict],
) -> Path:
    sample = diagnostic_samples[DIAGNOSTIC_IDS["clean"]]
    prediction = predictor.predict(DIAGNOSTIC, sample)
    probability = prediction["probability"]
    threshold = float(predictor.config["cnn_ransac"]["probability_threshold"])
    rows, columns = np.nonzero(probability >= threshold)
    points = np.column_stack((columns, rows)).astype(np.float32)
    five = distributed_five_points(points, prediction["predicted"])
    hypothesis = cv2.fitEllipseDirect(five.reshape(-1, 1, 2))
    distances = _point_distances(points, hypothesis)
    inlier_mask = distances <= float(
        predictor.config["cnn_ransac"]["distance_threshold_px"]
    )
    final_inliers = _point_distances(points, prediction["predicted"]) <= float(
        predictor.config["cnn_ransac"]["distance_threshold_px"]
    )

    panels = [
        ("1. 候補点を抽出", point_cloud_rgb(probability.shape, points)),
        (
            "2. 5点をランダム抽出",
            point_cloud_rgb(probability.shape, points, highlighted=five),
        ),
        (
            "3. 楕円仮説を生成",
            point_cloud_rgb(
                probability.shape, points, highlighted=five, ellipse=hypothesis
            ),
        ),
        (
            "4. 支持点を評価",
            point_cloud_rgb(
                probability.shape,
                points,
                ellipse=hypothesis,
                inlier_mask=inlier_mask,
            ),
        ),
        (
            "5. 最良仮説を再フィット",
            point_cloud_rgb(
                probability.shape,
                points,
                ellipse=prediction["predicted"],
                inlier_mask=final_inliers,
            ),
        ),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2))
    fig.suptitle("RANSACによる楕円推定", fontsize=22, y=0.99)
    for index, ((title, image), ax) in enumerate(zip(panels, axes)):
        show_image(ax, image, title)
        if index < len(axes) - 1:
            add_flow_arrow(ax)
    fig.text(
        0.5,
        0.02,
        (
            "1反復の概念図　│　実際は "
            f"{predictor.config['cnn_ransac']['iterations']} 回反復し、"
            "支持量・角度被覆・距離で最良仮説を選択"
        ),
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.09, 0.99, 0.90), w_pad=1.15)
    return save_figure(fig, "06-ransac-concept.png")


def build_degradation_grid(diagnostic_samples: dict[str, dict]) -> Path:
    row_definitions = [
        (
            "単純黒矩形",
            [
                "clean",
                "black_rectangle_025",
                "black_rectangle_050",
                "black_rectangle_075",
                "black_rectangle_100",
            ],
        ),
        (
            "センサ白飛びproxy",
            [
                "clean",
                "sensor_whiteout_025",
                "sensor_whiteout_050",
                "sensor_whiteout_075",
                "sensor_whiteout_100",
            ],
        ),
        (
            "センサ黒つぶれproxy",
            [
                "clean",
                "sensor_black_crush_025",
                "sensor_black_crush_050",
                "sensor_black_crush_075",
                "sensor_black_crush_100",
            ],
        ),
    ]
    columns = ["Clean", "25%", "50%", "75%", "100%"]
    fig, axes = plt.subplots(3, 5, figsize=(14.5, 8.7))
    fig.suptitle("初期実験で付与した劣化の強度", fontsize=22, y=0.985)
    for row, (row_label, keys) in enumerate(row_definitions):
        for column, key in enumerate(keys):
            sample = diagnostic_samples[DIAGNOSTIC_IDS[key]]
            show_image(
                axes[row, column],
                load_rgb(DIAGNOSTIC, sample),
                columns[column] if row == 0 else None,
            )
            if column == 0:
                axes[row, column].set_ylabel(row_label, fontsize=12, labelpad=12)
    fig.text(
        0.5,
        0.018,
        "全セルで同一の元画像・PAF姿勢・照明を使用",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.99, 0.95), h_pad=0.6, w_pad=0.45)
    return save_figure(fig, "07-degradation-severity-grid.png")


def build_method_comparison_examples(
    predictor: Predictor,
    diagnostic_samples: dict[str, dict],
) -> Path:
    keys = [
        "clean",
        "black_rectangle_025",
        "sensor_whiteout_100",
        "sensor_black_crush_025",
        "black_rectangle_050",
    ]
    titles = [
        "Clean",
        "黒矩形 25%",
        "白飛び 100%",
        "黒つぶれ 25%",
        "黒矩形 50%",
    ]
    fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.2))
    fig.suptitle("代表的な成功例・失敗例", fontsize=22, y=0.99)
    for column, (key, title) in enumerate(zip(keys, titles)):
        sample_id = DIAGNOSTIC_IDS[key]
        sample = diagnostic_samples[sample_id]
        cnn_prediction = predictor.predict(DIAGNOSTIC, sample)
        classic_prediction_result = canny_prediction(
            sample_id,
            diagnostic_samples,
        )
        classic_image = overlay_rgb(
            cnn_prediction["image"],
            classic_prediction_result["ground_truth"],
            classic_prediction_result["predicted"],
            classic_prediction_result["evaluation"],
        )
        cnn_image = overlay_rgb(
            cnn_prediction["image"],
            cnn_prediction["ground_truth"],
            cnn_prediction["predicted"],
            cnn_prediction["evaluation"],
        )
        show_image(axes[0, column], cnn_prediction["image"], title)
        show_image(axes[1, column], classic_image)
        show_image(axes[2, column], cnn_image)
        add_result_badge(axes[1, column], classic_prediction_result["evaluation"])
        add_result_badge(axes[2, column], cnn_prediction["evaluation"])
        if column == 0:
            axes[0, column].set_ylabel("入力", fontsize=13, labelpad=12)
            axes[1, column].set_ylabel(
                "旧Canny\n＋輪郭別RANSAC", fontsize=12, labelpad=12
            )
            axes[2, column].set_ylabel(
                "CNN + RANSAC", fontsize=12, labelpad=12
            )
    fig.text(
        0.5,
        0.018,
        "水色破線：正解楕円　緑：IoU ≥ 0.80　赤：IoU < 0.80",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.995, 0.955), h_pad=0.55, w_pad=0.55)
    return save_figure(fig, "08-success-failure-examples.png")


def effect_rows(
    rows: list[dict],
    method: str,
    effect: str,
) -> list[dict]:
    clean = next(
        row
        for row in rows
        if row["method"] == method and row["degradation"] == "clean"
    )
    effect_values = sorted(
        [
            row
            for row in rows
            if row["method"] == method and row["degradation"] == effect
        ],
        key=lambda row: float(row["severity"]),
    )
    return [{**clean, "severity": "0.0"}, *effect_values]


def plot_effect_chart(ax, rows: list[dict], effect: str, title: str) -> None:
    for method, label, color in (
        ("canny_ransac", "旧Canny + 輪郭別RANSAC", CLASSIC),
        ("cnn_ransac_support", "CNN + RANSAC", CNN),
    ):
        values = effect_rows(rows, method, effect)
        x = np.asarray([float(row["severity"]) * 100 for row in values])
        y = np.asarray([float(row["success_rate"]) * 100 for row in values])
        low = np.asarray([float(row["cluster_ci95_low"]) * 100 for row in values])
        high = np.asarray([float(row["cluster_ci95_high"]) * 100 for row in values])
        ax.plot(x, y, marker="o", linewidth=2.6, markersize=6, label=label, color=color)
        ax.fill_between(x, low, high, color=color, alpha=0.10)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("劣化強度（%）")
    ax.set_ylabel("成功率（%）")
    ax.grid(axis="y", color=GRID, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def build_result_charts() -> list[Path]:
    rows = read_csv(
        ROOT / "output/experiments/paf_second_stage_v1/diagnostic_curves.csv"
    )
    definitions = [
        ("black_rectangle", "単純黒矩形"),
        ("sensor_whiteout", "センサ白飛びproxy"),
        ("sensor_black_crush", "センサ黒つぶれproxy"),
    ]
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("初期実験：劣化強度と楕円検出成功率", fontsize=22, y=0.99)
    for ax, (effect, title) in zip(axes, definitions):
        plot_effect_chart(ax, rows, effect, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.text(
        0.985,
        0.02,
        "成功条件：楕円IoU ≥ 0.80　各条件 n=112",
        ha="right",
        fontsize=10.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.10, 0.99, 0.93), w_pad=1.4)
    paths.append(save_figure(fig, "09-initial-results-three-effects.png"))

    for index, (effect, title) in enumerate(definitions, start=1):
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        plot_effect_chart(ax, rows, effect, title)
        ax.legend(frameon=False, fontsize=10, loc="best")
        fig.text(
            0.98,
            0.02,
            "成功条件：楕円IoU ≥ 0.80　各条件 n=112",
            ha="right",
            fontsize=10,
            color=MUTED,
        )
        fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.99))
        paths.append(save_figure(fig, f"09-{index}-{effect}-curve.png"))
    return paths


def build_system_overview(
    predictor: Predictor,
    diagnostic_samples: dict[str, dict],
) -> Path:
    sample_id = DETECTOR_SUCCESS_SAMPLE_ID
    sample = diagnostic_samples[sample_id]
    prediction = predictor.predict(DIAGNOSTIC, sample)
    probability = prediction["probability"]
    classic_prediction_result = canny_prediction(
        sample_id,
        diagnostic_samples,
    )
    canny = classic_prediction_result["stages"]["edges"]
    classic_output = overlay_rgb(
        prediction["image"],
        classic_prediction_result["ground_truth"],
        classic_prediction_result["predicted"],
        classic_prediction_result["evaluation"],
    )
    cnn_output = overlay_rgb(
        prediction["image"],
        prediction["ground_truth"],
        prediction["predicted"],
        prediction["evaluation"],
    )
    cnn_points = threshold_rgb(
        probability, float(predictor.config["cnn_ransac"]["probability_threshold"])
    )

    top_panels = [
        prediction["image"],
        canny,
        classic_prediction_result["candidate_visualization"],
        classic_output,
    ]
    bottom_panels = [
        prediction["image"],
        probability_rgb(probability),
        cnn_points,
        cnn_output,
    ]
    titles = [
        ["入力", "Cannyエッジ", "輪郭別RANSAC候補", "推定楕円"],
        ["入力", "CNNリング尤度", "閾値点群", "推定楕円"],
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.4))
    fig.suptitle("設計した2つの検出系", fontsize=22, y=0.99)
    for row, panels in enumerate((top_panels, bottom_panels)):
        for column, (panel, title) in enumerate(zip(panels, titles[row])):
            show_image(axes[row, column], panel, title if row == 0 else None)
            if row == 1:
                axes[row, column].set_xlabel(title, fontsize=12, labelpad=8)
            if column < 3:
                add_flow_arrow(axes[row, column])
    axes[0, 0].set_ylabel(
        "旧Canny版", fontsize=15, labelpad=14, color=CLASSIC, fontweight=500
    )
    axes[1, 0].set_ylabel(
        "CNN版", fontsize=15, labelpad=14, color=CNN, fontweight=500
    )
    add_result_badge(axes[0, -1], classic_prediction_result["evaluation"])
    add_result_badge(axes[1, -1], prediction["evaluation"])
    fig.text(
        0.5,
        0.02,
        "比較条件：同一入力・同一正解・同一成功基準（楕円IoU ≥ 0.80）",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.035, 0.06, 0.99, 0.94), h_pad=0.8, w_pad=1.2)
    return save_figure(fig, "10-system-overview-with-images.png")


def write_chart_values() -> Path:
    rows = read_csv(
        ROOT / "output/experiments/paf_second_stage_v1/diagnostic_curves.csv"
    )
    target_methods = {"canny_ransac", "cnn_ransac_support"}
    target_effects = {
        "clean",
        "black_rectangle",
        "sensor_whiteout",
        "sensor_black_crush",
    }
    selected = [
        row
        for row in rows
        if row["method"] in target_methods and row["degradation"] in target_effects
    ]
    path = OUTPUT / "chart-source-values.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return path


def main() -> None:
    setup_style()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlecolor": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)

    base_manifest, base_samples = load_manifest(BASE)
    stress_manifest, _ = load_manifest(STRESS)
    _, diagnostic_samples = load_manifest(DIAGNOSTIC)
    predictor = Predictor()

    paths: list[Path] = []
    paths.append(build_dataset_overview(base_samples))
    paths.append(
        build_dataset_construction_flow(
            base_manifest,
            stress_manifest,
            base_samples,
            diagnostic_samples,
        )
    )
    paths.append(build_ground_truth_and_training_labels(diagnostic_samples))
    canny_figure, canny_panels = build_canny_pipeline(diagnostic_samples)
    paths.append(canny_figure)
    paths.extend(canny_panels)
    cnn_figure, cnn_panels = build_cnn_pipeline(predictor, diagnostic_samples)
    paths.append(cnn_figure)
    paths.extend(cnn_panels)
    paths.append(build_ransac_concept(predictor, diagnostic_samples))
    paths.append(build_degradation_grid(diagnostic_samples))
    paths.append(
        build_method_comparison_examples(predictor, diagnostic_samples)
    )
    paths.extend(build_result_charts())
    paths.append(build_system_overview(predictor, diagnostic_samples))
    paths.append(write_chart_values())

    index = {
        "title": "第1回進捗報告向け図版",
        "notes": [
            "714枚 = 102カメラ条件 × 7照明条件",
            "正解ラベルはPAF内周輪郭",
            "成功条件は楕円IoU 0.80以上",
            "第1回の比較対象は旧Canny＋輪郭別RANSAC",
            "学習・テストはカメラ条件単位で分割し、異なる視点条件を使用",
            (
                "輪郭分断の問題を受け、次回までにZhang et al. "
                "(Sensors 2019, DOI: 10.3390/s19235243)型を導入予定"
            ),
        ],
        "files": [
            str(path.relative_to(ROOT)).replace("\\", "/") for path in paths
        ],
    }
    (OUTPUT / "figure-index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
