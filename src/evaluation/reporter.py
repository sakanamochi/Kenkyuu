"""評価結果の集計、CSV/JSON/REPORT.md出力、および比較図の生成。"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from core.io import read_json, write_json
from inference.predictor import CNN_METHODS
from inference.zhang2019 import ZHANG_2019_REFERENCE

METHOD_LABELS = {
    "cnn_1194m": "1194M学習CNN",
    "cnn_diversified": "多様形状学習CNN",
    "canny": "Canny＋輪郭別RANSAC",
    "zhang2019": "Zhang 2019型（再現実装）",
}

DATASET_LABELS = {
    "1194m": "1194M評価データ",
    "diversified": "多様形状評価データ",
}

DEGRADATION_LABELS = {
    "clean": "通常",
    "black_rectangle": "矩形遮蔽",
    "sensor_whiteout": "白飛び",
    "sensor_black_crush": "黒つぶれ",
}

DEGRADATION_ORDER = ("clean", "black_rectangle", "sensor_whiteout", "sensor_black_crush")


def _aggregate(config: dict[str, Any], out_root: Path) -> dict[str, Any]:
    methods = list(config["evaluation"]["methods"])
    datasets = list(config["evaluation"]["test_datasets"])
    seeds = [int(s) for s in config["seeds"]]

    overall_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []

    for dataset in datasets:
        for method in methods:
            method_seeds: list[int | None] = seeds if method in CNN_METHODS else [None]
            loaded_per_seed: list[tuple[int | None, list[dict[str, Any]]]] = []
            for seed in method_seeds:
                tag = f"{method}_{seed}" if seed is not None else f"{method}_fixed"
                pred_path = out_root / "predictions" / tag / f"test_{dataset}.json"
                if pred_path.exists():
                    loaded_per_seed.append((seed, read_json(pred_path)["rows"]))
            if not loaded_per_seed:
                continue

            per_seed_rates, per_seed_ious = [], []
            for seed, rows in loaded_per_seed:
                n = len(rows)
                successes = sum(1 for r in rows if r["success"])
                rate = 100.0 * successes / n
                mean_iou = float(np.mean([r["ellipse_iou"] for r in rows]))
                per_seed_rates.append(rate)
                per_seed_ious.append(mean_iou)
                seed_records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "n": n,
                        "successes": successes,
                        "success_percent": rate,
                        "mean_iou": mean_iou,
                    }
                )
            overall_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed_count": len(loaded_per_seed),
                    "n_per_seed": len(loaded_per_seed[0][1]),
                    "success_percent": float(np.mean(per_seed_rates)),
                    "min_success_percent": float(min(per_seed_rates)),
                    "max_success_percent": float(max(per_seed_rates)),
                    "mean_iou": float(np.mean(per_seed_ious)),
                }
            )

            cond_keys = sorted(
                {(r["degradation"], float(r["severity"])) for r in loaded_per_seed[0][1]},
                key=lambda item: (DEGRADATION_ORDER.index(item[0]), item[1]),
            )
            for deg, sev in cond_keys:
                rates, ious = [], []
                n_cond = 0
                for _, rows in loaded_per_seed:
                    subset = [r for r in rows if r["degradation"] == deg and float(r["severity"]) == sev]
                    n_cond = len(subset)
                    rates.append(100.0 * sum(1 for r in subset if r["success"]) / n_cond)
                    ious.append(float(np.mean([r["ellipse_iou"] for r in subset])))
                condition_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "degradation": deg,
                        "severity": sev,
                        "seed_count": len(loaded_per_seed),
                        "n_per_seed": n_cond,
                        "success_percent": float(np.mean(rates)),
                        "min_success_percent": float(min(rates)),
                        "max_success_percent": float(max(rates)),
                        "mean_iou": float(np.mean(ious)),
                    }
                )

    if not overall_rows:
        raise FileNotFoundError(f"集計対象の評価JSONが見つかりません: {out_root / 'predictions'}")
    return {"overall": overall_rows, "by_condition": condition_rows, "by_seed": seed_records}


def _plot_comparison_figures(
    config: dict[str, Any],
    data: dict[str, Any],
    figures_dir: Path,
) -> list[str]:
    """configs/experiment.json の report.figures 設定に従って比較図を出力する。"""
    cache_dir = figures_dir / ".mplcache"
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    fig_cfg = config["report"]["figures"]
    dpi = int(fig_cfg["dpi"])
    ext = str(fig_cfg["format"]).lstrip(".")
    colors = fig_cfg["colors"]
    markers = fig_cfg["markers"]

    plt.rcParams.update(
        {
            "font.family": fig_cfg["font_family"],
            "font.size": float(fig_cfg["font_size"]),
            "axes.unicode_minus": False,
        }
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    overall = data["overall"]
    by_condition = data["by_condition"]
    methods = [m for m in METHOD_LABELS if any(r["method"] == m for r in overall)]
    datasets = [d for d in ("1194m", "diversified") if any(r["dataset"] == d for r in overall)]

    # 図1: 全13条件の総合成功率（棒グラフ）
    fig, ax = plt.subplots(figsize=tuple(fig_cfg["overall_figsize"]))
    x = np.arange(len(datasets))
    width = 0.19
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for idx, method in enumerate(methods):
        heights, yerr_low, yerr_high = [], [], []
        for dataset in datasets:
            row = next(r for r in overall if r["dataset"] == dataset and r["method"] == method)
            heights.append(row["success_percent"])
            yerr_low.append(row["success_percent"] - row["min_success_percent"])
            yerr_high.append(row["max_success_percent"] - row["success_percent"])
        bars = ax.bar(
            x + offsets[idx],
            heights,
            width=width * 0.92,
            label=METHOD_LABELS[method],
            color=colors[method],
            edgecolor="black",
            linewidth=0.6,
        )
        if method in CNN_METHODS:
            ax.errorbar(
                x + offsets[idx],
                heights,
                yerr=[yerr_low, yerr_high],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
            )
        for bar, value in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.5,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("検出成功率 [%]（IoU ≥ 0.8）")
    ax.set_ylim(0, 106)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="both", direction="in", top=False, right=True)
    ax.tick_params(axis="x", which="major", direction="in", top=True)
    ax.set_title("全13条件の総合成功率")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_bar = figures_dir / f"overall.{ext}"
    fig.savefig(out_bar, dpi=dpi)
    plt.close(fig)
    generated.append(str(out_bar))

    # 図2: 劣化条件・強度別曲線
    degs = ("black_rectangle", "sensor_whiteout", "sensor_black_crush")
    fig, axes = plt.subplots(len(datasets), len(degs), figsize=tuple(fig_cfg["degradation_figsize"]), sharex=True, sharey=True)
    if len(datasets) == 1:
        axes = np.asarray([axes])
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]

    for row_idx, dataset in enumerate(datasets):
        for col_idx, deg in enumerate(degs):
            ax = axes[row_idx, col_idx]
            for method in methods:
                clean_row = next(
                    (
                        r
                        for r in by_condition
                        if r["dataset"] == dataset and r["method"] == method and r["degradation"] == "clean"
                    ),
                    None,
                )
                valid_x, points, lows, highs = [], [], [], []
                for s in xs:
                    match = (
                        clean_row
                        if s == 0.0
                        else next(
                            (
                                r
                                for r in by_condition
                                if r["dataset"] == dataset
                                and r["method"] == method
                                and r["degradation"] == deg
                                and abs(r["severity"] - s) < 1e-6
                            ),
                            None,
                        )
                    )
                    if match is not None:
                        valid_x.append(s)
                        points.append(match["success_percent"])
                        lows.append(match["min_success_percent"])
                        highs.append(match["max_success_percent"])
                if not valid_x:
                    continue
                ax.plot(
                    valid_x,
                    points,
                    label=METHOD_LABELS[method],
                    color=colors[method],
                    marker=markers[method],
                    linestyle="-" if method in CNN_METHODS else "--",
                    linewidth=1.8,
                    markersize=5,
                )
                if method in CNN_METHODS and len(config["seeds"]) > 1:
                    ax.fill_between(valid_x, lows, highs, color=colors[method], alpha=0.14)
            if row_idx == 0:
                ax.set_title(DEGRADATION_LABELS[deg])
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("劣化強度")
            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset]}\n検出成功率 [%]")
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-2, 104)
            ax.set_xticks(xs)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="both", direction="in", top=True, right=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("劣化条件・強度別の検出成功率", fontsize=14)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False)
    seeds_cfg = [int(s) for s in config["seeds"]]
    curve_note = (
        f"CNNは固定1seed（{seeds_cfg[0]}）、幾何2方式は固定1回。成功条件：塗りつぶし楕円IoU ≥ 0.8。"
        if len(seeds_cfg) == 1
        else f"帯はCNN {len(seeds_cfg)}seedの最小〜最大（信頼区間ではない）。幾何2方式は固定1回。成功条件：塗りつぶし楕円IoU ≥ 0.8。"
    )
    fig.text(
        0.5,
        0.008,
        curve_note,
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out_curve = figures_dir / f"degradation.{ext}"
    fig.savefig(out_curve, dpi=dpi)
    plt.close(fig)
    generated.append(str(out_curve))

    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    return generated


def generate_report(
    config: dict[str, Any],
    output_dir: Path,
    *,
    with_figures: bool = True,
) -> dict[str, Any]:
    """保存済み評価結果を集計し、summary.json・CSV・REPORT.md・図を出力する。"""
    out_root = Path(output_dir).resolve()
    data = _aggregate(config, out_root)
    verification_path = out_root / "reference_verification.json"
    verification = read_json(verification_path) if verification_path.exists() else None
    whiteout_max = float(config["evaluation"]["whiteout_maximum_strength"])

    summary: dict[str, Any] = {
        "success_iou": float(config["success_iou"]),
        "whiteout_maximum_strength": whiteout_max,
        "seeds": [int(s) for s in config["seeds"]],
        **data,
        "reference_verification": verification,
        "zhang2019_reference": ZHANG_2019_REFERENCE,
        "figures": [],
    }

    if with_figures:
        summary["figures"] = _plot_comparison_figures(config, data, out_root / "figures")

    write_json(out_root / "summary.json", summary)

    for filename, rows in (("scores_overall.csv", data["overall"]), ("scores_by_condition.csv", data["by_condition"])):
        if not rows:
            continue
        with (out_root / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    seeds_cfg = [int(s) for s in config["seeds"]]
    cnn_eval_note = (
        f"- CNN評価：固定1seed（{seeds_cfg[0]}）による単一モデル評価。"
        if len(seeds_cfg) == 1
        else f"- CNN評価：独立に学習した{len(seeds_cfg)}seed（{', '.join(str(s) for s in seeds_cfg)}）の算術平均（モデル統合ではない）。"
    )
    lines = [
        "# PAF内周検出：評価レポート",
        "",
        "## 評価条件",
        "",
        "- 入力仕様：モノクロ256×256（同一輝度3ch複製）、一様画像は未検出。",
        f"- 成功条件：正解楕円と推定楕円の**塗りつぶし領域 IoU ≥ {config['success_iou']:.1f}**（未検出も全件を分母に含む）。",
        f"- 白飛び評価：最大内部強度 {whiteout_max:g}（表示強度 0.25 / 0.5 / 0.75 / 1.0）。",
        cnn_eval_note,
        "- 幾何比較器：学習seedを持たない固定設定1回の評価。",
        f"- 比較器出典：{ZHANG_2019_REFERENCE['label']} は "
        f"{', '.join(ZHANG_2019_REFERENCE['authors'])}, "
        f"\"{ZHANG_2019_REFERENCE['title']},\" {ZHANG_2019_REFERENCE['journal']} "
        f"{ZHANG_2019_REFERENCE['volume']}({ZHANG_2019_REFERENCE['issue']}), "
        f"{ZHANG_2019_REFERENCE['article']}, {ZHANG_2019_REFERENCE['year']}, "
        f"DOI `{ZHANG_2019_REFERENCE['doi']}` の処理構成を参考にした再現実装（著者コードの移植ではない）。",
        "",
        "## 全13条件の総合成功率",
        "",
        "|評価対象|1194M学習CNN|多様形状学習CNN|Canny＋輪郭別RANSAC|Zhang 2019型（再現実装）|",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in ("1194m", "diversified"):
        cells = [DATASET_LABELS[dataset]]
        for method in ("cnn_1194m", "cnn_diversified", "canny", "zhang2019"):
            match = next(
                (r for r in data["overall"] if r["dataset"] == dataset and r["method"] == method),
                None,
            )
            cells.append(f"{match['success_percent']:.2f}%" if match is not None else "-")
        lines.append("|" + "|".join(cells) + "|")

    lines.extend(
        [
            "",
            "## 代表条件別成功率",
            "",
            "|評価対象|方式|通常|遮蔽0.5|白飛び0.5|白飛び1.0|黒つぶれ0.5|黒つぶれ1.0|",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    key_conds = [
        ("clean", 0.0),
        ("black_rectangle", 0.5),
        ("sensor_whiteout", 0.5),
        ("sensor_whiteout", 1.0),
        ("sensor_black_crush", 0.5),
        ("sensor_black_crush", 1.0),
    ]
    for dataset in ("1194m", "diversified"):
        for method in ("cnn_1194m", "cnn_diversified", "canny", "zhang2019"):
            if not any(r["dataset"] == dataset and r["method"] == method for r in data["overall"]):
                continue
            vals = []
            for deg, sev in key_conds:
                m = next(
                    (
                        r
                        for r in data["by_condition"]
                        if r["dataset"] == dataset
                        and r["method"] == method
                        and r["degradation"] == deg
                        and abs(r["severity"] - sev) < 1e-6
                    ),
                    None,
                )
                vals.append(f"{m['success_percent']:.2f}%" if m is not None else "-")
            lines.append(f"|{DATASET_LABELS[dataset]}|{METHOD_LABELS[method]}|" + "|".join(vals) + "|")

    if verification is not None:
        lines.extend(
            [
                "",
                "## 確定参照結果との全件照合",
                "",
                f"- CNN照合総数：{verification['checked_predictions']:,}件",
                f"- CNN不一致：楕円 {verification['ellipse_mismatches']}件 / "
                f"IoU {verification['iou_mismatches']}件 / "
                f"成功判定 {verification['success_mismatches']}件 / "
                f"入力ハッシュ {verification['image_hash_mismatches']}件",
                f"- 判定：{'完全一致（VERIFIED）' if verification['verified_exact'] else '不一致あり'}",
            ]
        )

    lines.extend(
        [
            "",
            "## 制約と未検証事項",
            "",
            "- 評価範囲は黒背景・モノクロ256px・単一開口の合成CG画像であり、既存test集合は検討で参照済みのため新規盲検ではない。",
            "- 実画像一般、任意背景、未知機体一般への汎化性能、強化白飛び条件での再学習は未検証。",
            "",
        ]
    )

    (out_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
