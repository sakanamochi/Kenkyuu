from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from paflab.camera_effects import (
    black_rectangle,
    sensor_black_crush,
    sensor_whiteout,
)
from paflab.image_io import imread
from paflab.labels import fit_label_ellipse


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "output/datasets/research_base_v1"
OUTPUT = ROOT / "output/presentation_assets/progress_meeting_1_v1/figures"

TEXT = "#182230"
MUTED = "#667085"
PANEL = "#f2f4f7"


def setup_style() -> None:
    """Windowsの日本語フォントを明示して図中の文字化けを防ぐ。"""
    font_path = Path(r"C:\Windows\Fonts\meiryo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "text.color": TEXT,
            "axes.titlecolor": TEXT,
        }
    )


def load_samples() -> dict[str, dict]:
    manifest = json.loads(
        (DATASET / "manifest.json").read_text(encoding="utf-8")
    )
    return {sample["sample_id"]: sample for sample in manifest["samples"]}


def load_rgb(samples: dict[str, dict], sample_id: str) -> np.ndarray:
    sample = samples[sample_id]
    image = imread(DATASET / sample["image"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(DATASET / sample["image"])
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_image(axis, image: np.ndarray, label: str) -> None:
    axis.imshow(image)
    axis.set_title(label, fontsize=12, pad=7)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_section_label(fig, axis, title: str, detail: str) -> None:
    box = axis.get_position()
    fig.text(box.x0, box.y1 + 0.035, title, fontsize=14, fontweight=500)
    fig.text(
        box.x0 + 0.18,
        box.y1 + 0.036,
        detail,
        fontsize=10,
        color=MUTED,
    )


def build_parameter_grid(samples: dict[str, dict]) -> Path:
    """各生成パラメータだけを変えた代表画像を一枚へ整理する。"""
    tilt_ids = [
        f"camera_t{tilt:03d}_a000_d034.0_o00__light_t045_a000_e03.0"
        for tilt in (0, 20, 40, 60, 75)
    ]
    distance_ids = [
        f"camera_t040_a000_d{distance:03d}.0_o00__light_t045_a000_e03.0"
        for distance in (34, 44)
    ]
    target_ids = [
        f"camera_t040_a000_d034.0_o{index:02d}__light_t045_a000_e03.0"
        for index in range(3)
    ]
    lighting_ids = [
        "camera_t040_a000_d034.0_o00__light_t000_a000_e03.0",
        "camera_t040_a000_d034.0_o00__light_t045_a000_e03.0",
        "camera_t040_a000_d034.0_o00__light_t045_a120_e03.0",
        "camera_t040_a000_d034.0_o00__light_t045_a240_e03.0",
        "camera_t040_a000_d034.0_o00__light_t075_a000_e03.0",
        "camera_t040_a000_d034.0_o00__light_t075_a120_e03.0",
        "camera_t040_a000_d034.0_o00__light_t075_a240_e03.0",
    ]

    fig = plt.figure(figsize=(16, 11.8))
    fig.suptitle("基礎CG画像の生成パラメータ", fontsize=24, y=0.985)
    fig.text(
        0.5,
        0.94,
        "カメラ角度 17 × 距離 2 × 注視点 3 × 照明 7 ＝ 714画像",
        ha="center",
        fontsize=17,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": PANEL,
            "edgecolor": "none",
        },
    )
    outer = fig.add_gridspec(
        4,
        1,
        left=0.045,
        right=0.985,
        top=0.865,
        bottom=0.045,
        hspace=0.48,
    )

    # 全行を7列基準にそろえ、各パターンを左端から連続して配置する。
    tilt_grid = outer[0].subgridspec(1, 7, wspace=0.07)
    tilt_axes = []
    for index, (sample_id, tilt) in enumerate(zip(tilt_ids, (0, 20, 40, 60, 75))):
        axis = fig.add_subplot(tilt_grid[0, index])
        show_image(axis, load_rgb(samples, sample_id), f"傾斜 {tilt}°")
        tilt_axes.append(axis)
    add_section_label(
        fig,
        tilt_axes[0],
        "カメラ角度",
        "ここでは傾斜5種類を表示。実際は方位を含め17パターン",
    )

    distance_grid = outer[1].subgridspec(1, 7, wspace=0.07)
    distance_axes = []
    for column, (sample_id, distance) in enumerate(
        zip(distance_ids, (34, 44)),
    ):
        axis = fig.add_subplot(distance_grid[0, column])
        show_image(
            axis,
            load_rgb(samples, sample_id),
            f"パターン{column + 1}",
        )
        distance_axes.append(axis)
    add_section_label(fig, distance_axes[0], "距離", "2パターン")

    target_grid = outer[2].subgridspec(1, 7, wspace=0.07)
    target_labels = ("中央（X=0）", "X=-1.5", "X=+1.5")
    target_axes = []
    for column, (sample_id, label) in enumerate(
        zip(target_ids, target_labels),
    ):
        axis = fig.add_subplot(target_grid[0, column])
        show_image(axis, load_rgb(samples, sample_id), label)
        target_axes.append(axis)
    add_section_label(fig, target_axes[0], "注視点", "3パターン")

    lighting_grid = outer[3].subgridspec(1, 7, wspace=0.07)
    lighting_labels = (
        "軸方向",
        "45° / 0°",
        "45° / 120°",
        "45° / 240°",
        "75° / 0°",
        "75° / 120°",
        "75° / 240°",
    )
    lighting_axes = []
    for index, (sample_id, label) in enumerate(
        zip(lighting_ids, lighting_labels)
    ):
        axis = fig.add_subplot(lighting_grid[0, index])
        show_image(axis, load_rgb(samples, sample_id), label)
        lighting_axes.append(axis)
    add_section_label(
        fig,
        lighting_axes[0],
        "照明",
        "傾斜/方位。合計7パターン",
    )

    path = OUTPUT / "11-generation-parameter-grid.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_degradation_grid(samples: dict[str, dict]) -> Path:
    """同一CGへ撮像診断用の3種類の効果を同一強度で付与する。"""
    sample_id = (
        "camera_t040_a090_d034.0_o00__light_t045_a000_e03.0"
    )
    sample = samples[sample_id]
    source_bgr = imread(DATASET / sample["image"], cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise FileNotFoundError(DATASET / sample["image"])
    label = json.loads(
        (DATASET / sample["label"]).read_text(encoding="utf-8")
    )
    ellipse = fit_label_ellipse(label)
    severity = 0.60

    rectangle, _ = black_rectangle(
        source_bgr,
        ellipse,
        severity,
        rng=np.random.default_rng(20260716),
    )
    whiteout, _ = sensor_whiteout(
        source_bgr,
        severity,
        rng=np.random.default_rng(20260717),
    )
    black_crush, _ = sensor_black_crush(
        source_bgr,
        severity,
        rng=np.random.default_rng(20260718),
    )
    images = [
        (cv2.cvtColor(rectangle, cv2.COLOR_BGR2RGB), "単純黒矩形 60%"),
        (cv2.cvtColor(whiteout, cv2.COLOR_BGR2RGB), "センサ白飛びproxy 60%"),
        (cv2.cvtColor(black_crush, cv2.COLOR_BGR2RGB), "センサ黒つぶれproxy 60%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    fig.suptitle("撮像診断で付与する3種類の効果", fontsize=23, y=0.985)
    fig.text(
        0.5,
        0.90,
        "同一の元画像・カメラ・照明に、同じ強度60%を適用",
        ha="center",
        fontsize=13,
        color=MUTED,
    )
    for axis, (image, label_text) in zip(axes, images):
        show_image(axis, image, label_text)
    fig.text(
        0.5,
        0.025,
        "単純黒矩形 ／ センサ白飛びproxy ／ センサ黒つぶれproxy",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.87), w_pad=0.6)

    path = OUTPUT / "12-three-degradation-patterns.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    setup_style()
    samples = load_samples()
    for path in (
        build_parameter_grid(samples),
        build_degradation_grid(samples),
    ):
        print(path)


if __name__ == "__main__":
    main()
