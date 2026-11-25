"""直接楕円パラメータ回帰CNNの学習・統一評価・現行方式との比較レポート生成スクリプト。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.io import load_config, read_json, resolve_path, write_json
from defaults import DEFAULT_CONFIG_PATH
from training.regressor_trainer import evaluate_ellipse_regressor, train_ellipse_regressor

CONDITION_SPECS = [
    ("clean", 0.0, "通常"),
    ("black_rectangle", 0.25, "遮蔽0.25"),
    ("black_rectangle", 0.5, "遮蔽0.50"),
    ("black_rectangle", 0.75, "遮蔽0.75"),
    ("black_rectangle", 1.0, "遮蔽1.00"),
    ("sensor_whiteout", 0.25, "白飛0.25"),
    ("sensor_whiteout", 0.5, "白飛0.50"),
    ("sensor_whiteout", 0.75, "白飛0.75"),
    ("sensor_whiteout", 1.0, "白飛1.00"),
    ("sensor_black_crush", 0.25, "黒潰0.25"),
    ("sensor_black_crush", 0.5, "黒潰0.50"),
    ("sensor_black_crush", 0.75, "黒潰0.75"),
    ("sensor_black_crush", 1.0, "黒潰1.00"),
]


def _rate_by_condition(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for deg, sev, _ in CONDITION_SPECS:
        sub = [r for r in rows if r["degradation"] == deg and abs(float(r["severity"]) - sev) < 1e-6]
        out[f"{deg}:{sev:g}"] = 100.0 * sum(1 for r in sub if r["success"]) / len(sub) if sub else 0.0
    return out


def generate_comparison_report(
    config: dict[str, Any],
    base_dir: Path,
    *,
    direct_out: Path,
    eval_out: Path,
    seed: int = 20260907,
) -> Path:
    """直接楕円回帰CNNと現行4方式の比較表・REPORT.md・比較図を生成する。"""
    datasets = list(config["evaluation"]["test_datasets"])
    method_sources = [
        ("cnn_diversified", "多様形状CNN（境界＋RANSAC）", eval_out / "predictions" / f"cnn_diversified_{seed}"),
        ("cnn_direct", "多様形状CNN（直接楕円回帰）", direct_out / "predictions" / f"cnn_direct_{seed}"),
        ("cnn_1194m", "1194M学習CNN（境界＋RANSAC）", eval_out / "predictions" / f"cnn_1194m_{seed}"),
        ("canny", "Canny＋輪郭別RANSAC", eval_out / "predictions" / "canny_fixed"),
        ("zhang2019", "Zhang 2019型（再現実装）", eval_out / "predictions" / "zhang2019_fixed"),
    ]

    summary: dict[str, Any] = {"seed": seed, "datasets": {}}
    lines = [
        "# 直接楕円パラメータ回帰CNNとの比較結果",
        "",
        f"- 評価条件：固定1seed（`{seed}`）、全13条件（白飛び最大内部強度 2.0）、成功判定 塗りつぶし楕円 IoU ≥ 0.8",
        "- 学習条件：両多様形状CNNとも同一データ（`data/diversified`、無地＋柄3種混合・開口幅オンライン遮蔽）・60epoch・batch 16・AdamW",
        "",
        "## 総合成功率（全13条件）",
        "",
        "| 評価データセット | 境界＋RANSAC（多様） | **直接楕円回帰（多様）** | 境界＋RANSAC（1194M） | Canny＋輪郭別RANSAC | Zhang 2019型（再現実装） |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for ds in datasets:
        ds_label = config["datasets"][ds]["label"]
        ds_summary: dict[str, Any] = {}
        row_cells = [ds_label]
        for key, label, pred_dir in method_sources:
            payload = read_json(pred_dir / f"test_{ds}.json")
            ds_summary[key] = {
                "label": label,
                "n": payload["n"],
                "successes": payload["successes"],
                "success_percent": payload["success_percent"],
                "mean_iou": payload["mean_iou"],
                "by_condition": _rate_by_condition(payload["rows"]),
            }
            row_cells.append(f"{payload['success_percent']:.2f}% (IoU {payload['mean_iou']:.3f})")
        summary["datasets"][ds] = ds_summary
        lines.append("| " + " | ".join(row_cells) + " |")

    for ds in datasets:
        ds_label = config["datasets"][ds]["label"]
        lines.extend(["", f"## 条件別成功率：{ds_label}", ""])
        lines.append("| 条件 | 境界＋RANSAC（多様） | **直接楕円回帰（多様）** | 差分 (直接−境界) | 1194M CNN | Canny | Zhang 2019型（再現実装） |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        d = summary["datasets"][ds]
        for deg, sev, cname in CONDITION_SPECS:
            ckey = f"{deg}:{sev:g}"
            v_div = d["cnn_diversified"]["by_condition"][ckey]
            v_dir = d["cnn_direct"]["by_condition"][ckey]
            v_1194 = d["cnn_1194m"]["by_condition"][ckey]
            v_can = d["canny"]["by_condition"][ckey]
            v_zha = d["zhang2019"]["by_condition"][ckey]
            lines.append(
                f"| {cname} | {v_div:.2f}% | **{v_dir:.2f}%** | {v_dir - v_div:+.2f} pp | "
                f"{v_1194:.2f}% | {v_can:.2f}% | {v_zha:.2f}% |"
            )

    # 比較グラフの生成
    plt.rcParams["font.family"] = config["report"]["figures"]["font_family"]
    fig_dir = direct_out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) 総合比較棒グラフ
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    x = np.arange(len(datasets))
    width = 0.16
    colors = {
        "cnn_diversified": "#d62728",
        "cnn_direct": "#ff7f0e",
        "cnn_1194m": "#1f77b4",
        "canny": "#2ca02c",
        "zhang2019": "#9467bd",
    }
    offsets = np.linspace(-2 * width, 2 * width, len(method_sources))
    for offset, (key, label, _) in zip(offsets, method_sources):
        vals = [summary["datasets"][ds][key]["success_percent"] for ds in datasets]
        bars = ax.bar(x + offset, vals, width=width * 0.92, label=label, color=colors[key], edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{config['datasets'][ds]['label']}評価データ" for ds in datasets])
    ax.set_ylabel("検出成功率 [%] (IoU ≥ 0.8)")
    ax.set_ylim(0, 108)
    ax.set_title("境界マップ＋RANSAC と 直接楕円パラメータ回帰CNN の総合比較")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    overall_fig = fig_dir / "overall_comparison.png"
    fig.savefig(overall_fig, bbox_inches="tight")
    plt.close(fig)

    # 2) 劣化曲線比較（3データセット×3劣化）
    fig, axes = plt.subplots(len(datasets), 3, figsize=(13.5, 3.2 * len(datasets)), dpi=180, sharex=True, sharey=True)
    deg_panels = [("black_rectangle", "矩形遮蔽"), ("sensor_whiteout", "白飛び"), ("sensor_black_crush", "黒つぶれ")]
    sevs = [0.0, 0.25, 0.5, 0.75, 1.0]
    styles = {
        "cnn_diversified": dict(color="#d62728", marker="s", linestyle="-", linewidth=1.8),
        "cnn_direct": dict(color="#ff7f0e", marker="o", linestyle="-", linewidth=1.8),
        "cnn_1194m": dict(color="#1f77b4", marker="o", linestyle="--", linewidth=1.4),
        "canny": dict(color="#2ca02c", marker="^", linestyle=":", linewidth=1.2),
        "zhang2019": dict(color="#9467bd", marker="D", linestyle=":", linewidth=1.2),
    }
    for r_idx, ds in enumerate(datasets):
        for c_idx, (deg_name, deg_title) in enumerate(deg_panels):
            ax = axes[r_idx, c_idx]
            for key, label, _ in method_sources:
                by_c = summary["datasets"][ds][key]["by_condition"]
                ys = [by_c["clean:0"] if s == 0.0 else by_c[f"{deg_name}:{s:g}"] for s in sevs]
                ax.plot(sevs, ys, label=label, **styles[key])
            if r_idx == 0:
                ax.set_title(deg_title)
            if c_idx == 0:
                ax.set_ylabel(f"{config['datasets'][ds]['label']}\n検出成功率 [%]")
            if r_idx == len(datasets) - 1:
                ax.set_xlabel("劣化強度")
            ax.set_ylim(-4, 105)
            ax.set_xticks(sevs)
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("劣化条件・強度別の検出成功率（境界＋RANSAC vs 直接楕円回帰）", y=0.995)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.97))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    deg_fig = fig_dir / "degradation_comparison.png"
    fig.savefig(deg_fig, bbox_inches="tight")
    plt.close(fig)

    summary["figures"] = [str(overall_fig), str(deg_fig)]
    write_json(direct_out / "summary.json", summary)
    report_path = direct_out / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description="直接楕円パラメータ回帰CNNの学習・評価・比較レポート生成")
    parser.add_argument("--stage", choices=["all", "train", "evaluate", "report"], default="all")
    parser.add_argument("--source", choices=["diversified", "1194m"], default="diversified")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config, base_dir = load_config(args.config)
    models_root = resolve_path(base_dir, config["paths"]["models_dir"])
    outputs_root = resolve_path(base_dir, config["paths"]["outputs_dir"])
    run_dir = models_root / "direct_regression" / args.source / str(args.seed)
    direct_out = outputs_root / "direct_regression"
    eval_out = outputs_root / "evaluation"

    if args.stage in ("all", "train"):
        summary = train_ellipse_regressor(
            config,
            base_dir,
            source_name=args.source,
            seed=args.seed,
            output_dir=run_dir,
            epochs=args.epochs,
            device_name=args.device,
            overwrite=args.overwrite,
        )
        print(
            f"[DONE-TRAIN] params={summary['parameter_count']} best_epoch={summary['best_epoch']} "
            f"best_val_loss={summary['best_validation_loss']:.5f}",
            flush=True,
        )

    if args.stage in ("all", "evaluate"):
        evaluate_ellipse_regressor(
            config,
            base_dir,
            checkpoint_path=run_dir / "best.pt",
            datasets=list(config["evaluation"]["test_datasets"]),
            output_dir=direct_out,
            seed=args.seed,
            device_name=args.device,
            overwrite=args.overwrite,
        )

    if args.stage in ("all", "report"):
        report_path = generate_comparison_report(
            config,
            base_dir,
            direct_out=direct_out,
            eval_out=eval_out,
            seed=args.seed,
        )
        print(f"REPORT: {report_path}", flush=True)


if __name__ == "__main__":
    main()
