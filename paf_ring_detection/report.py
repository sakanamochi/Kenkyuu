"""評価結果をJSON、全体比較図、劣化強度別比較図にまとめる。"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

from paf_ring_detection.data import read_json, write_json


METHODS = (
    (
        "canny_contour_ransac.csv",
        "Canny + 輪郭別RANSAC",
        "#f59e0b",
        "s",
        "-",
    ),
    (
        "zhang2019.csv",
        "Zhang 2019型（再現実装）",
        "#64748b",
        "^",
        "--",
    ),
    (
        "cnn_ransac.csv",
        "CNN + RANSAC",
        "#2563eb",
        "o",
        "-",
    ),
)

DIAGNOSTIC_PANELS = (
    ("black_rectangle", "黒矩形"),
    ("sensor_whiteout", "白飛び"),
    ("sensor_black_crush", "黒つぶれ"),
)


def _configure_japanese_font() -> None:
    """日本語を含む図で利用可能なフォントを優先順に設定する。"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Yu Gothic", "Meiryo", "Noto Sans CJK JP"):
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            return


def _style_boxed_axes(axis) -> None:
    """四辺を表示し、上下左右の目盛りを内向きにそろえる。"""
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )


def _success_by_severity(
    csv_path: Path,
    degradation: str,
) -> list[dict]:
    """cleanを0%として、指定した劣化の強度別成功率を集計する。"""
    grouped: dict[float, list[bool]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            row_degradation = row["degradation"]
            if row_degradation == "clean":
                severity = 0.0
            elif row_degradation == degradation:
                severity = float(row["severity"])
            else:
                continue
            grouped[severity].append(row["success"].strip().lower() == "true")
    return [
        {
            "severity": severity,
            "sample_count": len(values),
            "success_rate": sum(values) / len(values),
        }
        for severity, values in sorted(grouped.items())
    ]


def _build_diagnostic_severity_figure(
    result_root: Path,
    success_iou: float,
) -> None:
    """3方式の撮像劣化強度別成功率を横並びで保存する。"""
    diagnostic_root = result_root / "diagnostic"
    _configure_japanese_font()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.3),
        sharex=True,
        sharey=True,
    )
    all_counts = set()
    for axis, (degradation, title) in zip(axes, DIAGNOSTIC_PANELS):
        for filename, label, color, marker, line_style in METHODS:
            values = _success_by_severity(
                diagnostic_root / filename,
                degradation,
            )
            if not values:
                raise ValueError(
                    f"{filename}: {degradation}の評価結果がありません"
                )
            all_counts.update(value["sample_count"] for value in values)
            axis.plot(
                [value["severity"] * 100 for value in values],
                [value["success_rate"] * 100 for value in values],
                label=label,
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=2.2,
                markersize=6,
            )
        axis.set_title(title)
        axis.set_xlabel("劣化強度（%）")
        axis.set_xlim(-2, 102)
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.set_ylim(-2, 102)
        axis.set_yticks((0, 20, 40, 60, 80, 100))
        axis.grid(axis="y", color="#d1d5db", linewidth=0.8)
        _style_boxed_axes(axis)
    axes[0].set_ylabel("成功率（%）")
    figure.suptitle("撮像劣化強度別の検出成功率", fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    condition = f"成功条件：楕円IoU ≥ {success_iou:.2f}"
    if len(all_counts) == 1:
        condition += f"　各条件 n={next(iter(all_counts))}"
    figure.text(0.99, 0.025, condition, ha="right", color="#64748b")
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.92))
    figure.savefig(
        result_root / "diagnostic_by_severity.png",
        dpi=160,
    )
    plt.close(figure)


def build_report(config: dict) -> None:
    result_root = Path(config["paths"]["results"])
    summary = {}

    for dataset_name in config["evaluation_datasets"]:
        canny = read_json(
            result_root / dataset_name / "canny_contour_ransac.json"
        )
        zhang = read_json(result_root / dataset_name / "zhang2019.json")
        cnn = read_json(result_root / dataset_name / "cnn_ransac.json")
        sample_counts = {
            canny["sample_count"],
            zhang["sample_count"],
            cnn["sample_count"],
        }
        if len(sample_counts) != 1:
            raise ValueError(f"{dataset_name}: 3方式の評価枚数が一致しません")
        summary[dataset_name] = {
            "sample_count": zhang["sample_count"],
            "canny_contour_shared_ransac": canny["success_rate"],
            "zhang2019_reproduction": zhang["success_rate"],
            "cnn_weighted_ransac": cnn["success_rate"],
        }

    write_json(result_root / "summary.json", summary)

    names = list(summary)
    x = range(len(names))
    canny_rates = [
        summary[name]["canny_contour_shared_ransac"] * 100 for name in names
    ]
    zhang_rates = [summary[name]["zhang2019_reproduction"] * 100 for name in names]
    cnn_rates = [summary[name]["cnn_weighted_ransac"] * 100 for name in names]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        [value - 0.26 for value in x],
        canny_rates,
        0.26,
        label="Canny contour + shared RANSAC",
    )
    axis.bar(
        list(x),
        zhang_rates,
        0.26,
        label="Zhang 2019 reproduction",
    )
    axis.bar(
        [value + 0.26 for value in x],
        cnn_rates,
        0.26,
        label="CNN + weighted RANSAC",
    )
    axis.set_xticks(list(x), names)
    axis.set_ylabel("Success rate [%]")
    axis.set_ylim(0, 100)
    _style_boxed_axes(axis)
    axis.legend()
    figure.tight_layout()
    figure.savefig(result_root / "comparison.png", dpi=160)
    plt.close(figure)
    _build_diagnostic_severity_figure(
        result_root,
        float(config["success_iou"]),
    )
    print(f"集計を作成しました: {result_root}")
