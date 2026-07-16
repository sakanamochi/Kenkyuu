from __future__ import annotations

import csv
import json
import zlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyBboxPatch

from analysis.ellipse_baseline import evaluate_ellipses, fit_ground_truth
from paflab.evaluate_cnn import probability_ellipse
from paflab.image_io import imread
from paflab.model import TinyUNet


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/experiments/paf_second_stage_v1/figures"
OOD = ROOT / "output/datasets/research_ood_base_v1"
DIAGNOSTIC = ROOT / "output/datasets/paf_diagnostics_v1"
SUCCESS_COLOR = "#21a366"
FAIL_COLOR = "#e5484d"
GT_COLOR = "#00a6d6"
CLASSIC_COLOR = "#f59e0b"
CNN_COLOR = "#2563eb"
TEXT = "#182230"
MUTED = "#5f6b7a"


def setup_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\meiryo.ttc")
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
        }
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_manifest(dataset: Path) -> tuple[dict, dict[str, dict]]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    return manifest, {sample["sample_id"]: sample for sample in manifest["samples"]}


def ellipse_from_dict(values: dict | None):
    if not values:
        return None
    return (
        (float(values["center_x"]), float(values["center_y"])),
        (float(values["axis_1"]), float(values["axis_2"])),
        float(values["angle_deg"]),
    )


def scale_ellipse(ellipse, scale: float):
    if ellipse is None:
        return None
    return (
        (ellipse[0][0] * scale, ellipse[0][1] * scale),
        (ellipse[1][0] * scale, ellipse[1][1] * scale),
        ellipse[2],
    )


class Predictor:
    def __init__(self) -> None:
        self.config = json.loads(
            (ROOT / "config/research_experiment.json").read_text(encoding="utf-8")
        )
        checkpoint = torch.load(
            ROOT / "output/experiments/paf_robustness_v3/cnn_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyUNet(int(checkpoint["base_channels"]))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.cache: dict[tuple[str, str], dict] = {}

    def predict(self, dataset: Path, sample: dict) -> dict:
        key = (str(dataset), sample["sample_id"])
        if key in self.cache:
            return self.cache[key]
        source = imread(dataset / sample["image"], cv2.IMREAD_COLOR)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        size = int(self.config["input_size"])
        image = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        tensor = (tensor / 127.5 - 1.0)[None].to(self.device)
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()
        label = json.loads((dataset / sample["label"]).read_text(encoding="utf-8"))
        scale = size / float(label["image_width"])
        ground_truth = fit_ground_truth(
            np.asarray(label["image_points"], dtype=np.float32) * scale
        )
        conditions = sample["conditions"]
        seed_key = conditions.get("base_sample_id") or (
            f"{conditions['camera_id']}|{conditions['lighting_id']}"
        )
        seed = (
            int(self.config["cnn_ransac"]["random_seed"])
            + zlib.crc32(seed_key.encode("utf-8"))
        ) % (2**32)
        result = probability_ellipse(
            probability, self.config["cnn_ransac"], random_seed=seed
        )
        predicted = result["ellipse"] if result else None
        evaluation = (
            evaluate_ellipses(predicted, ground_truth, (size, size))
            if predicted is not None
            else None
        )
        output = {
            "image": image,
            "probability": probability,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "evaluation": evaluation,
        }
        self.cache[key] = output
        return output


def classic_prediction(dataset: Path, result_name: str, sample_id: str):
    detail = json.loads(
        (
            dataset
            / "results"
            / result_name
            / "details"
            / f"{sample_id}.json"
        ).read_text(encoding="utf-8")
    )
    predicted = scale_ellipse(ellipse_from_dict(detail.get("detected")), 256 / 480)
    ground_truth = scale_ellipse(ellipse_from_dict(detail["ground_truth"]), 256 / 480)
    evaluation = (
        evaluate_ellipses(predicted, ground_truth, (256, 256))
        if predicted is not None
        else None
    )
    return predicted, evaluation


def ellipse_patch(ellipse, *, color: str, dashed: bool, width: float) -> Ellipse:
    return Ellipse(
        ellipse[0],
        width=ellipse[1][0],
        height=ellipse[1][1],
        angle=ellipse[2],
        fill=False,
        edgecolor=color,
        linewidth=width,
        linestyle="--" if dashed else "-",
    )


def show_overlay(ax, image, ground_truth, predicted, evaluation) -> None:
    ax.imshow(image)
    ax.add_patch(ellipse_patch(ground_truth, color=GT_COLOR, dashed=True, width=2.2))
    success = bool(evaluation and evaluation["ellipse_iou"] >= 0.8)
    if predicted is not None:
        ax.add_patch(
            ellipse_patch(
                predicted,
                color=SUCCESS_COLOR if success else FAIL_COLOR,
                dashed=False,
                width=2.6,
            )
        )
    label = (
        f"IoU {evaluation['ellipse_iou']:.2f}  {'成功' if success else '失敗'}"
        if evaluation
        else "検出なし"
    )
    ax.text(
        0.03,
        0.96,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#111827", "alpha": 0.78, "edgecolor": "none"},
    )
    ax.set_xticks([])
    ax.set_yticks([])


def show_probability(ax, probability) -> None:
    ax.imshow(probability, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def finish_grid(fig, path: Path, subtitle: str) -> None:
    fig.text(
        0.5,
        0.018,
        "水色破線: 正解楕円　緑: IoU ≥ 0.80　赤: IoU < 0.80　" + subtitle,
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_angle_grid(predictor: Predictor, samples: dict[str, dict]) -> Path:
    ids = [
        f"camera_t{tilt:03d}_a045_d039.0_o00__light_t020_a060_e01.5__bg_space"
        for tilt in (10, 30, 50, 67, 82)
    ]
    fig, axes = plt.subplots(4, 5, figsize=(15.8, 12.1))
    fig.suptitle(
        "未知カメラ角度：入力 → CNN尤度 → 推定内周楕円",
        fontsize=21,
        fontweight=500,
        y=0.985,
    )
    row_labels = ["入力画像", "CNNリング尤度", "CNN + RANSAC", "Zhang 2019型"]
    for column, (tilt, sample_id) in enumerate(zip((10, 30, 50, 67, 82), ids)):
        sample = samples[sample_id]
        predicted = predictor.predict(OOD, sample)
        classic, classic_eval = classic_prediction(
            OOD, "zhang2019_arc_ood", sample_id
        )
        axes[0, column].imshow(predicted["image"])
        axes[0, column].set_title(f"傾斜 {tilt}°", fontsize=14, pad=8)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        show_probability(axes[1, column], predicted["probability"])
        show_overlay(
            axes[2, column],
            predicted["image"],
            predicted["ground_truth"],
            predicted["predicted"],
            predicted["evaluation"],
        )
        show_overlay(
            axes[3, column],
            predicted["image"],
            predicted["ground_truth"],
            classic,
            classic_eval,
        )
        for row in range(4):
            if column == 0:
                axes[row, column].set_ylabel(row_labels[row], fontsize=12, labelpad=12)
    fig.tight_layout(rect=(0.035, 0.045, 1, 0.96), h_pad=1.0, w_pad=0.7)
    path = OUTPUT / "angle-input-output-grid.png"
    finish_grid(fig, path, "固定条件: space背景、太陽 tilt 20° / azimuth 60°")
    return path


def build_background_grid(predictor: Predictor, samples: dict[str, dict]) -> Path:
    backgrounds = ("space", "earth", "moon")
    ids = [
        f"camera_t067_a045_d039.0_o00__light_t020_a060_e01.5__bg_{background}"
        for background in backgrounds
    ]
    labels = {"space": "宇宙", "earth": "地球", "moon": "月"}
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 12.3))
    fig.suptitle(
        "背景一般化：同じ姿勢・照明で背景だけを変更",
        fontsize=21,
        fontweight=500,
        y=0.985,
    )
    row_labels = ["入力画像", "CNNリング尤度", "CNN + RANSAC", "Zhang 2019型"]
    for column, (background, sample_id) in enumerate(zip(backgrounds, ids)):
        sample = samples[sample_id]
        predicted = predictor.predict(OOD, sample)
        classic, classic_eval = classic_prediction(
            OOD, "zhang2019_arc_ood", sample_id
        )
        axes[0, column].imshow(predicted["image"])
        axes[0, column].set_title(labels[background], fontsize=15, pad=8)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        show_probability(axes[1, column], predicted["probability"])
        show_overlay(
            axes[2, column],
            predicted["image"],
            predicted["ground_truth"],
            predicted["predicted"],
            predicted["evaluation"],
        )
        show_overlay(
            axes[3, column],
            predicted["image"],
            predicted["ground_truth"],
            classic,
            classic_eval,
        )
        for row in range(4):
            if column == 0:
                axes[row, column].set_ylabel(row_labels[row], fontsize=12, labelpad=12)
    fig.tight_layout(rect=(0.045, 0.045, 1, 0.96), h_pad=1.0, w_pad=0.9)
    path = OUTPUT / "background-input-output-grid.png"
    finish_grid(fig, path, "固定条件: カメラ傾斜67°、同一太陽条件")
    return path


def build_diagnostic_grid(predictor: Predictor, samples: dict[str, dict]) -> Path:
    ids = [
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__clean_s0000_v00",
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__black_rectangle_s0250_v01",
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__sensor_whiteout_s1000_v08",
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__sensor_black_crush_s0250_v09",
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__lens_flare_s1000_v16",
    ]
    titles = ["Clean", "黒矩形 25%", "白飛び 100%", "黒つぶれ 25%", "Flare 100%"]
    fig, axes = plt.subplots(4, 5, figsize=(15.8, 12.1))
    fig.suptitle(
        "撮像・遮蔽診断：同じ元画像に効果だけを付与",
        fontsize=21,
        fontweight=500,
        y=0.985,
    )
    row_labels = ["入力画像", "CNNリング尤度", "CNN + RANSAC", "Zhang 2019型"]
    for column, (title, sample_id) in enumerate(zip(titles, ids)):
        sample = samples[sample_id]
        predicted = predictor.predict(DIAGNOSTIC, sample)
        classic, classic_eval = classic_prediction(
            DIAGNOSTIC, "zhang2019_arc_diagnostic", sample_id
        )
        axes[0, column].imshow(predicted["image"])
        axes[0, column].set_title(title, fontsize=14, pad=8)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        show_probability(axes[1, column], predicted["probability"])
        show_overlay(
            axes[2, column],
            predicted["image"],
            predicted["ground_truth"],
            predicted["predicted"],
            predicted["evaluation"],
        )
        show_overlay(
            axes[3, column],
            predicted["image"],
            predicted["ground_truth"],
            classic,
            classic_eval,
        )
        for row in range(4):
            if column == 0:
                axes[row, column].set_ylabel(row_labels[row], fontsize=12, labelpad=12)
    fig.tight_layout(rect=(0.035, 0.045, 1, 0.96), h_pad=1.0, w_pad=0.7)
    path = OUTPUT / "diagnostic-input-output-grid.png"
    finish_grid(fig, path, "全列で同一のPAF姿勢・照明・RANSAC seed key")
    return path


def build_quantitative_summary() -> Path:
    summary = json.loads(
        (ROOT / "output/experiments/paf_second_stage_v1/second_stage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    model = summary["model_ablation"]["aggregate"]
    primary_rates = summary["overall_rates"]
    tilt_rows = [
        row
        for row in read_csv(
            ROOT / "output/experiments/paf_second_stage_v1/ood_by_camera_tilt.csv"
        )
        if row["method"] in ("cnn_ransac_support", "zhang2019_arc_reproduction")
    ]
    diagnostic_rows = read_csv(
        ROOT / "output/experiments/paf_second_stage_v1/diagnostic_curves.csv"
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    fig.suptitle("第2段階検証：定量結果の一枚まとめ", fontsize=22, fontweight=500, y=0.98)

    ax = axes[0, 0]
    params = np.array([row["parameter_count"] for row in model])
    for metric, label, color in (
        ("validation_success_rate", "Validation（clean + 劣化）", CNN_COLOR),
        ("ood_success_rate", "OOD clean", SUCCESS_COLOR),
    ):
        means = np.array([row[f"mean_{metric}"] for row in model]) * 100
        lows = np.array([row[f"min_{metric}"] for row in model]) * 100
        highs = np.array([row[f"max_{metric}"] for row in model]) * 100
        ax.errorbar(
            params,
            means,
            yerr=[means - lows, highs - means],
            marker="o",
            linewidth=2.2,
            capsize=4,
            label=label,
            color=color,
        )
    for row in model:
        ax.annotate(
            f"幅{row['base_channels']}",
            (row["parameter_count"], row["mean_validation_success_rate"] * 100),
            textcoords="offset points",
            xytext=(4, 7),
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_ylim(35, 85)
    ax.set_xlabel("パラメータ数（log）")
    ax.set_ylabel("成功率（%）")
    ax.set_title("A. CNN規模 × 3 seed", loc="left", fontsize=14, fontweight=500)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    values = [
        primary_rates["ood"]["zhang2019_arc_reproduction"] * 100,
        primary_rates["ood"]["cnn_ransac_support"] * 100,
        primary_rates["diagnostic"]["zhang2019_arc_reproduction"] * 100,
        primary_rates["diagnostic"]["cnn_ransac_support"] * 100,
    ]
    bars = ax.bar(
        ["Zhang\nOOD", "CNN\nOOD", "Zhang\n診断", "CNN\n診断"],
        values,
        color=[CLASSIC_COLOR, CNN_COLOR, CLASSIC_COLOR, CNN_COLOR],
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value:.1f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("成功率（%）")
    ax.set_title("B. 主比較の全体成功率", loc="left", fontsize=14, fontweight=500)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    for method, label, color in (
        ("cnn_ransac_support", "CNN + RANSAC", CNN_COLOR),
        ("zhang2019_arc_reproduction", "Zhang 2019型", CLASSIC_COLOR),
    ):
        rows = sorted(
            [row for row in tilt_rows if row["method"] == method],
            key=lambda row: float(row["camera_tilt_deg"]),
        )
        ax.plot(
            [float(row["camera_tilt_deg"]) for row in rows],
            [float(row["success_rate"]) * 100 for row in rows],
            marker="o",
            linewidth=2.2,
            label=label,
            color=color,
        )
    ax.axvline(67, color="#94a3b8", linestyle="--", linewidth=1)
    ax.text(67.8, 5, "外挿域", fontsize=9, color=MUTED)
    ax.set_ylim(-3, 105)
    ax.set_xlabel("カメラ傾斜角（°）")
    ax.set_ylabel("成功率（%）")
    ax.set_title("C. 未知角度の性能境界", loc="left", fontsize=14, fontweight=500)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    keys = [
        ("black_rectangle", 0.25, "黒矩形\n25%"),
        ("sensor_whiteout", 1.0, "白飛び\n100%"),
        ("sensor_black_crush", 0.25, "黒つぶれ\n25%"),
        ("lens_flare", 1.0, "Flare\n100%"),
    ]
    x = np.arange(len(keys))
    width = 0.36
    for offset, method, label, color in (
        (-width / 2, "cnn_ransac_support", "CNN + RANSAC", CNN_COLOR),
        (width / 2, "zhang2019_arc_reproduction", "Zhang 2019型", CLASSIC_COLOR),
    ):
        values = []
        for effect, severity, _ in keys:
            row = next(
                row
                for row in diagnostic_rows
                if row["method"] == method
                and row["degradation"] == effect
                and float(row["severity"]) == severity
            )
            values.append(float(row["success_rate"]) * 100)
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_xticks(x, [key[2] for key in keys])
    ax.set_ylim(0, 105)
    ax.set_ylabel("成功率（%）")
    ax.set_title("D. 遮蔽・撮像効果proxy", loc="left", fontsize=14, fontweight=500)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.03, 0.035, 0.99, 0.95), h_pad=2.3, w_pad=1.7)
    path = OUTPUT / "quantitative-experiment-summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_thumbnail_strip(ax, images: list[np.ndarray], labels: list[str]) -> None:
    ax.axis("off")
    count = len(images)
    for index, (image, label) in enumerate(zip(images, labels)):
        left = index / count + 0.01
        width = 0.94 / count
        inset = ax.inset_axes([left, 0.18, width, 0.76])
        inset.imshow(image)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(label, fontsize=9, pad=3)


def build_overview_poster(predictor: Predictor, ood_samples: dict[str, dict], diagnostic_samples: dict[str, dict]) -> Path:
    summary = json.loads(
        (
            ROOT
            / "output/experiments/paf_second_stage_v1/second_stage_summary.json"
        ).read_text(encoding="utf-8")
    )
    primary_rates = summary["overall_rates"]
    diagnostic_rows = read_csv(
        ROOT / "output/experiments/paf_second_stage_v1/diagnostic_curves.csv"
    )

    def diagnostic_rate(method: str, degradation: str, severity: float) -> float:
        row = next(
            row
            for row in diagnostic_rows
            if row["method"] == method
            and row["degradation"] == degradation
            and float(row["severity"]) == severity
        )
        return float(row["success_rate"])

    fig = plt.figure(figsize=(16, 9), facecolor="#f7f9fc")
    fig.text(0.05, 0.94, "PAF内周リング検出：実験全体の一枚まとめ", fontsize=25, fontweight=500, color=TEXT)
    fig.text(0.05, 0.905, "Blender 5.2 CG → Zhang弧統合 / semantic CNN → 楕円推定 → IoU ≥ 0.80", fontsize=12, color=MUTED)
    grid = fig.add_gridspec(2, 2, left=0.045, right=0.97, top=0.87, bottom=0.14, hspace=0.24, wspace=0.13)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_facecolor("white")
    angle_ids = [
        "camera_t010_a045_d039.0_o00__light_t020_a060_e01.5__bg_space",
        "camera_t067_a045_d039.0_o00__light_t020_a060_e01.5__bg_earth",
        "camera_t082_a045_d039.0_o00__light_t020_a060_e01.5__bg_space",
    ]
    angle_images = [predictor.predict(OOD, ood_samples[sample_id])["image"] for sample_id in angle_ids]
    add_thumbnail_strip(ax, angle_images, ["10° / space", "67° / Earth", "82° / space"])
    ax.set_title("1. 未知角度・照明・背景 OOD（480枚）", loc="left", fontsize=14, fontweight=500, pad=10)
    ax.text(
        0.02,
        0.02,
        (
            f"CNN {primary_rates['ood']['cnn_ransac_support']:.1%}　"
            f"Zhang型 {primary_rates['ood']['zhang2019_arc_reproduction']:.1%}"
        ),
        transform=ax.transAxes,
        fontsize=10.5,
        color=TEXT,
    )

    ax = fig.add_subplot(grid[0, 1])
    ax.set_facecolor("white")
    diag_ids = [
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__clean_s0000_v00",
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__black_rectangle_s0250_v01",
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__sensor_whiteout_s1000_v08",
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__sensor_black_crush_s0250_v09",
        "camera_t020_a090_d034.0_o02__light_t045_a000_e03.0__diagnostic__lens_flare_s1000_v16",
    ]
    diag_images = [predictor.predict(DIAGNOSTIC, diagnostic_samples[sample_id])["image"] for sample_id in diag_ids]
    add_thumbnail_strip(ax, diag_images, ["Clean", "黒矩形25%", "白飛び", "黒つぶれ", "Flare"])
    ax.set_title("2. 遮蔽・撮像効果proxy（1,904枚）", loc="left", fontsize=14, fontweight=500, pad=10)
    ax.text(
        0.02,
        0.02,
        (
            "黒矩形25%: "
            f"CNN {diagnostic_rate('cnn_ransac_support', 'black_rectangle', 0.25):.1%} / "
            f"Zhang型 {diagnostic_rate('zhang2019_arc_reproduction', 'black_rectangle', 0.25):.1%}"
        ),
        transform=ax.transAxes,
        fontsize=10.5,
        color=TEXT,
    )

    model = summary["model_ablation"]["aggregate"]
    ax = fig.add_subplot(grid[1, 0])
    params = [row["parameter_count"] for row in model]
    ax.plot(params, [row["mean_validation_success_rate"] * 100 for row in model], marker="o", linewidth=2.4, color=CNN_COLOR, label="Validation")
    ax.plot(params, [row["mean_ood_success_rate"] * 100 for row in model], marker="o", linewidth=2.4, color=SUCCESS_COLOR, label="OOD clean")
    ax.set_xscale("log")
    ax.set_ylim(35, 85)
    ax.set_xlabel("パラメータ数（log）")
    ax.set_ylabel("成功率（%）")
    ax.set_title("3. CNN最小化（4幅 × 3 seed）", loc="left", fontsize=14, fontweight=500, pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.02, 0.03, "採用: 幅16 / 482,737 params　近似解: 幅8 / 121,177 params", transform=ax.transAxes, fontsize=10.5)

    ax = fig.add_subplot(grid[1, 1])
    values = [
        primary_rates["ood"]["zhang2019_arc_reproduction"] * 100,
        primary_rates["ood"]["cnn_ransac_support"] * 100,
        primary_rates["diagnostic"]["zhang2019_arc_reproduction"] * 100,
        primary_rates["diagnostic"]["cnn_ransac_support"] * 100,
    ]
    bars = ax.bar(
        ["Zhang\nOOD", "CNN\nOOD", "Zhang\n診断", "CNN\n診断"],
        values,
        color=[CLASSIC_COLOR, CNN_COLOR, CLASSIC_COLOR, CNN_COLOR],
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("成功率（%）")
    ax.set_title("4. 主比較：Zhang型とCNN", loc="left", fontsize=14, fontweight=500, pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.03, "同一画像・同一正解楕円・同一IoU基準", transform=ax.transAxes, fontsize=10.5)

    fig.text(0.05, 0.045, "次の優先実験: 既知太陽ベクトル条件付け │ 複数PAF CAD family holdout │ 実機カメラ校正", fontsize=12, color=TEXT)
    path = OUTPUT / "experiment-overview-poster.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#f7f9fc")
    plt.close(fig)
    return path


def main() -> None:
    setup_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, ood_samples = load_manifest(OOD)
    _, diagnostic_samples = load_manifest(DIAGNOSTIC)
    predictor = Predictor()
    paths = [
        build_overview_poster(predictor, ood_samples, diagnostic_samples),
        build_angle_grid(predictor, ood_samples),
        build_background_grid(predictor, ood_samples),
        build_diagnostic_grid(predictor, diagnostic_samples),
        build_quantitative_summary(),
    ]
    (OUTPUT / "figure_index.json").write_text(
        json.dumps([str(path.relative_to(ROOT)).replace("\\", "/") for path in paths], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
