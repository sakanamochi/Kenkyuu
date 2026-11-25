"""EllSeg型マルチタスクU-Net（可視境界＋Amodal完全穴マスク）の学習・評価・比較レポートスクリプト。"""

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
from training.ellseg_trainer import evaluate_ellseg_model, train_ellseg_model

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


def generate_ellseg_report(
    config: dict[str, Any],
    *,
    ellseg_out: Path,
    eval_out: Path,
    direct_out: Path,
    seed: int = 20260907,
) -> Path:
    """EllSeg型（融合・穴のみ・境界のみ）と現行境界CNN・直接回帰・1194M CNNの比較レポートと図を作る。"""
    datasets = list(config["evaluation"]["test_datasets"])
    method_sources = [
        ("ellseg_fused", "EllSeg型（境界×穴融合）", ellseg_out / "predictions" / f"ellseg_fused_{seed}"),
        ("ellseg_hole_only", "EllSeg型（完全穴マスクのみ）", ellseg_out / "predictions" / f"ellseg_hole_only_{seed}"),
        ("ellseg_boundary_only", "EllSeg型（境界Ch0のみ）", ellseg_out / "predictions" / f"ellseg_boundary_only_{seed}"),
        ("cnn_diversified", "現行多様CNN（境界1ch＋RANSAC）", eval_out / "predictions" / f"cnn_diversified_{seed}"),
        ("cnn_direct", "直接楕円回帰CNN", direct_out / "predictions" / f"cnn_direct_{seed}"),
        ("cnn_1194m", "1194M学習CNN（境界1ch）", eval_out / "predictions" / f"cnn_1194m_{seed}"),
    ]

    summary: dict[str, Any] = {"seed": seed, "datasets": {}}
    lines = [
        "# EllSeg型（可視内周境界＋Amodal完全穴マスク）1seed検証結果",
        "",
        f"- 評価条件：固定1seed（`{seed}`）、全13条件（白飛び最大内部強度 2.0）、成功判定 塗りつぶし楕円 IoU ≥ 0.8",
        "- モデル規模：`TinyEllSegUNet(16)`（482,754パラメータ、現行 `TinyUNet(16)` の 482,737 から +17増）",
        "",
        "## 総合成功率（全13条件）",
        "",
        "| 評価データセット | **EllSeg融合** | EllSeg穴のみ | EllSeg境界Ch0のみ | 現行境界1ch | 直接回帰 | 1194M CNN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for ds in datasets:
        ds_label = config["datasets"][ds]["label"]
        ds_summary: dict[str, Any] = {}
        cells = [ds_label]
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
            cells.append(f"{payload['success_percent']:.2f}% ({payload['mean_iou']:.3f})")
        summary["datasets"][ds] = ds_summary
        lines.append("| " + " | ".join(cells) + " |")

    for ds in datasets:
        ds_label = config["datasets"][ds]["label"]
        lines.extend(["", f"## 条件別成功率：{ds_label}", ""])
        lines.append("| 条件 | **EllSeg融合** | 現行境界1ch | 差分 (融合−現行) | EllSeg穴のみ | EllSeg境界Ch0 | 直接回帰 | 1194M CNN |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        d = summary["datasets"][ds]
        for deg, sev, cname in CONDITION_SPECS:
            ckey = f"{deg}:{sev:g}"
            v_f = d["ellseg_fused"]["by_condition"][ckey]
            v_b = d["cnn_diversified"]["by_condition"][ckey]
            v_h = d["ellseg_hole_only"]["by_condition"][ckey]
            v_c0 = d["ellseg_boundary_only"]["by_condition"][ckey]
            v_dir = d["cnn_direct"]["by_condition"][ckey]
            v_1194 = d["cnn_1194m"]["by_condition"][ckey]
            lines.append(
                f"| {cname} | **{v_f:.2f}%** | {v_b:.2f}% | {v_f - v_b:+.2f} pp | "
                f"{v_h:.2f}% | {v_c0:.2f}% | {v_dir:.2f}% | {v_1194:.2f}% |"
            )

    plt.rcParams["font.family"] = config["report"]["figures"]["font_family"]
    fig_dir = ellseg_out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 総合比較図（主要4CNN比較：EllSeg融合 / 現行境界1ch / 直接回帰 / 1194M）
    plot_keys = [
        ("ellseg_fused", "EllSeg型（境界×穴融合）", "#e31a1c"),
        ("ellseg_hole_only", "EllSeg型（完全穴マスクのみ）", "#fb9a99"),
        ("cnn_diversified", "現行多様CNN（境界1ch＋RANSAC）", "#33a02c"),
        ("cnn_direct", "直接楕円回帰CNN", "#ff7f00"),
        ("cnn_1194m", "1194M学習CNN（境界1ch）", "#1f78b4"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    x = np.arange(len(datasets))
    width = 0.16
    offsets = np.linspace(-2 * width, 2 * width, len(plot_keys))
    for offset, (key, label, color) in zip(offsets, plot_keys):
        vals = [summary["datasets"][ds][key]["success_percent"] for ds in datasets]
        bars = ax.bar(x + offset, vals, width=width * 0.92, label=label, color=color, edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{config['datasets'][ds]['label']}評価データ" for ds in datasets])
    ax.set_ylabel("検出成功率 [%] (IoU ≥ 0.8)")
    ax.set_ylim(0, 108)
    ax.set_title("EllSeg型（2chマルチタスク）と既存CNN方式の総合比較")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    overall_fig = fig_dir / "overall_comparison.png"
    fig.savefig(overall_fig, bbox_inches="tight")
    plt.close(fig)

    summary["figures"] = [str(overall_fig)]
    write_json(ellseg_out / "summary.json", summary)
    report_path = ellseg_out / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description="EllSeg型2chマルチタスクU-Netの学習・評価・比較レポート生成")
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
    run_dir = models_root / "ellseg" / args.source / str(args.seed)
    ellseg_out = outputs_root / "ellseg"
    eval_out = outputs_root / "evaluation"
    direct_out = outputs_root / "direct_regression"

    if args.stage in ("all", "train"):
        summary = train_ellseg_model(
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
            f"[DONE-ELLSEG-TRAIN] params={summary['parameter_count']} "
            f"best_epoch={summary['best_epoch']} best_val_loss={summary['best_validation_loss']:.5f}",
            flush=True,
        )

    if args.stage in ("all", "evaluate"):
        evaluate_ellseg_model(
            config,
            base_dir,
            checkpoint_path=run_dir / "best.pt",
            datasets=list(config["evaluation"]["test_datasets"]),
            output_dir=ellseg_out,
            seed=args.seed,
            device_name=args.device,
            overwrite=args.overwrite,
        )

    if args.stage in ("all", "report"):
        report_path = generate_ellseg_report(
            config,
            ellseg_out=ellseg_out,
            eval_out=eval_out,
            direct_out=direct_out,
            seed=args.seed,
        )
        print(f"REPORT: {report_path}", flush=True)


if __name__ == "__main__":
    main()
