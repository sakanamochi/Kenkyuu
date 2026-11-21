"""現実PAF寸法範囲での体系的要因分解（形状範囲・外観/テクスチャ・投影・遮蔽）の実行・集計スクリプト。"""

from __future__ import annotations

import argparse
import csv
import subprocess
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

from cg.geometry_model import build_split_manifests, generate_diversified_plan
from core.io import load_config, read_json, resolve_path, write_json
from defaults import DEFAULT_CONFIG_PATH, DEFAULT_RENDER_DIVERSIFIED_CONFIG
from evaluation.evaluator import evaluate_checkpoint
from training.trainer import train_model

REALISTIC_VIEWS = [
    "base",
    "base_albedo",
    "base_stripe",
    "base_noise",
    "base_checker",
    "projection_plain",
    "projection_albedo",
    "projection_stripe",
    "projection_noise",
    "projection_checker",
]
SINGLE_CONE_VIEWS = [
    "base",
    "base_stripe",
    "base_noise",
    "base_checker",
    "projection_plain",
    "projection_stripe",
    "projection_noise",
    "projection_checker",
]
# 現実PAF範囲の中央値に近い代表円錐（形状多様化なしの対照用）
SINGLE_CONE_RATIOS = {
    "bottom_outer_diameter_ratio": 1.75,
    "height_ratio": 0.55,
    "hole_diameter_ratio": 0.88,
}
DIAGNOSIS_SEED = 20260907


def _mixed_slots() -> list[str]:
    return [
        "base",
        "base",
        "base",
        "base_stripe",
        "base_noise",
        "base_checker",
        "projection_plain",
        "projection_plain",
        "projection_plain",
        "projection_stripe",
        "projection_noise",
        "projection_checker",
    ]


def _mixed_base_only_slots() -> list[str]:
    return ["base", "base", "base", "base_stripe", "base_noise", "base_checker"]


def _model_run_dir(models_root: Path, name: str, seed: int) -> Path:
    """主構成(B6_mixed)はmodels/diversified_limited_size/、他アブレーションはmodels/ablation/へ統一配置する。"""
    if name == "B6_mixed":
        return models_root / "diversified_limited_size" / str(seed)
    return models_root / "ablation" / name / str(seed)


def _experiment_specs(data_root: Path) -> dict[str, dict[str, Any]]:
    """全アブレーション実験の定義（形状範囲・外観分解・投影/遮蔽分解）。"""
    real_dir = data_root / "diversified_limited_size"
    cone_dir = data_root / "single_cone"

    real_views = {v: real_dir / v for v in REALISTIC_VIEWS}
    cone_views = {v: cone_dir / v for v in SINGLE_CONE_VIEWS}

    return {
        # Group B: 外観・テクスチャの根拠分解（現実PAF範囲＋投影拡張＋開口幅遮蔽を固定）
        "B1_plain": {
            "group": "B_appearance",
            "label": "現実PAF・無地（模様なし）",
            "view_dirs": real_views,
            "view_slots": ["base", "base", "projection_plain", "projection_plain"],
            "use_aperture_occlusion": True,
        },
        "B2_albedo": {
            "group": "B_appearance",
            "label": "現実PAF・明暗のみ（模様なし）",
            "view_dirs": real_views,
            "view_slots": ["base", "base_albedo", "projection_plain", "projection_albedo"],
            "use_aperture_occlusion": True,
        },
        "B3_stripe": {
            "group": "B_appearance",
            "label": "現実PAF・放射ストライプのみ",
            "view_dirs": real_views,
            "view_slots": ["base", "base_stripe", "projection_plain", "projection_stripe"],
            "use_aperture_occlusion": True,
        },
        "B4_noise": {
            "group": "B_appearance",
            "label": "現実PAF・ノイズのみ",
            "view_dirs": real_views,
            "view_slots": ["base", "base_noise", "projection_plain", "projection_noise"],
            "use_aperture_occlusion": True,
        },
        "B5_checker": {
            "group": "B_appearance",
            "label": "現実PAF・チェッカーのみ",
            "view_dirs": real_views,
            "view_slots": ["base", "base_checker", "projection_plain", "projection_checker"],
            "use_aperture_occlusion": True,
        },
        "B6_mixed": {
            "group": "B_appearance",
            "label": "現実PAF・3種混合（縞/ノイズ/格子）",
            "view_dirs": real_views,
            "view_slots": _mixed_slots(),
            "use_aperture_occlusion": True,
        },
        # Group C: 投影・外観・開口幅遮蔽の要因分解（現実PAF範囲、外観は3種混合およびストライプ基準）
        "C0_base_fixed_occ": {
            "group": "C_factors",
            "label": "現実PAF・元構図のみ（固定遮蔽）",
            "view_dirs": real_views,
            "view_slots": ["base"],
            "use_aperture_occlusion": False,
        },
        "C1_proj_only_fixed_occ": {
            "group": "C_factors",
            "label": "現実PAF・＋投影のみ（固定遮蔽）",
            "view_dirs": real_views,
            "view_slots": ["base", "projection_plain"],
            "use_aperture_occlusion": False,
        },
        "C2_tex_only_fixed_occ": {
            "group": "C_factors",
            "label": "現実PAF・＋外観のみ（固定遮蔽）",
            "view_dirs": real_views,
            "view_slots": _mixed_base_only_slots(),
            "use_aperture_occlusion": False,
        },
        "C3_proj_tex_fixed_occ": {
            "group": "C_factors",
            "label": "現実PAF・＋投影＋外観（固定遮蔽）",
            "view_dirs": real_views,
            "view_slots": _mixed_slots(),
            "use_aperture_occlusion": False,
        },
        # Group A: 形状多様化そのものの単独効果（単一代表円錐 vs 現実PAF範囲 vs 現行広範囲）
        "A1_single_cone_mixed": {
            "group": "A_shape",
            "label": "単一代表円錐・3種混合（多様化なし）",
            "view_dirs": cone_views,
            "view_slots": _mixed_slots(),
            "use_aperture_occlusion": True,
        },
    }


def _run_blender_render(
    blender_path: Path,
    render_config: Path,
    output_dir: Path,
    views: list[str],
    *,
    fixed_ratios: dict[str, float] | None = None,
    limit: int | None = None,
) -> None:
    import json

    script = PROJECT_ROOT / "src" / "cg" / "render_diversified.py"
    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(script),
        "--",
        "--config",
        str(render_config.resolve()),
        "--base-dir",
        str(PROJECT_ROOT),
        "--output",
        str(output_dir.resolve()),
        "--views",
        *views,
    ]
    if fixed_ratios is not None:
        cmd.extend(["--fixed-ratios", json.dumps(fixed_ratios)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    subprocess.run(cmd, check=True)


def prepare_datasets(
    config: dict[str, Any],
    base_dir: Path,
    render_config: Path,
    *,
    targets: list[str],
    limit: int | None = None,
) -> None:
    """現実PAF(data/diversified_limited_size)および単一代表円錐(data/single_cone)のCGとmanifestを生成する。"""
    blender_path = Path(config["blender_executable"])
    if not blender_path.exists():
        raise FileNotFoundError(f"Blenderが見つかりません: {blender_path}")
    data_root = resolve_path(base_dir, config["paths"]["data_dir"])

    if "diversified_limited_size" in targets:
        real_dir = data_root / "diversified_limited_size"
        generate_diversified_plan(render_config, real_dir)
        _run_blender_render(blender_path, render_config, real_dir, REALISTIC_VIEWS, limit=limit)
        build_split_manifests(real_dir, REALISTIC_VIEWS, seed=2026112102)
        print(f"[READY] diversified_limited_size dataset: {real_dir}", flush=True)

    if "single_cone" in targets:
        cone_dir = data_root / "single_cone"
        generate_diversified_plan(render_config, cone_dir, fixed_ratios=SINGLE_CONE_RATIOS)
        _run_blender_render(
            blender_path,
            render_config,
            cone_dir,
            SINGLE_CONE_VIEWS,
            fixed_ratios=SINGLE_CONE_RATIOS,
            limit=limit,
        )
        build_split_manifests(cone_dir, SINGLE_CONE_VIEWS, seed=2026112102)
        print(f"[READY] single_cone dataset: {cone_dir}", flush=True)


def _summarize_prediction(pred: dict[str, Any]) -> dict[str, float]:
    rows = pred["rows"]
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "clean" if row["degradation"] == "clean" else f"{row['degradation']}_{row['severity']:g}"
        by_cond.setdefault(key, []).append(row)

    def rate(items: list[dict[str, Any]]) -> float:
        return 100.0 * sum(1 for r in items if r["success"]) / len(items) if items else 0.0

    return {
        "overall": float(pred["success_percent"]),
        "mean_iou": float(pred["mean_iou"]),
        "clean": rate(by_cond.get("clean", [])),
        "occ_0.5": rate(by_cond.get("black_rectangle_0.5", [])),
        "whiteout_0.5": rate(by_cond.get("sensor_whiteout_0.5", [])),
        "whiteout_1.0": rate(by_cond.get("sensor_whiteout_1", [])),
        "black_crush_0.5": rate(by_cond.get("sensor_black_crush_0.5", [])),
        "black_crush_1.0": rate(by_cond.get("sensor_black_crush_1", [])),
    }


def evaluate_baselines_on_realistic(
    config: dict[str, Any],
    base_dir: Path,
    ablation_root: Path,
    *,
    device_name: str = "auto",
    overwrite: bool = False,
) -> None:
    """既存比較基準（1194M学習CNN・現行広範囲CNN・Canny・Zhang 2019型）を現実PAF testで評価する。"""
    data_root = resolve_path(base_dir, config["paths"]["data_dir"])
    real_base = data_root / "diversified_limited_size" / "base"
    models_dir = resolve_path(base_dir, config["paths"]["models_dir"])
    pred_root = ablation_root / "predictions" / "baselines"

    for source, method in (("1194m", "cnn_1194m"), ("diversified", "cnn_diversified")):
        for seed in config["seeds"]:
            ckpt = models_dir / source / str(seed) / "best.pt"
            evaluate_checkpoint(
                config,
                dataset_dir=real_base,
                dataset_name="realistic",
                method=method,
                checkpoint_path=ckpt,
                seed=int(seed),
                destination=pred_root / f"{method}_{seed}" / "test_realistic.json",
                device_name=device_name,
                overwrite=overwrite,
            )

    for method in ("canny", "zhang2019"):
        evaluate_checkpoint(
            config,
            dataset_dir=real_base,
            dataset_name="realistic",
            method=method,
            destination=pred_root / f"{method}_fixed" / "test_realistic.json",
            device_name=device_name,
            overwrite=overwrite,
        )


def run_experiments(
    config: dict[str, Any],
    base_dir: Path,
    ablation_root: Path,
    *,
    names: list[str],
    seeds: list[int],
    epochs: int | None = None,
    device_name: str = "auto",
    overwrite: bool = False,
) -> None:
    """指定実験を学習し、1194M testおよび現実PAF testで統一評価する。"""
    data_root = resolve_path(base_dir, config["paths"]["data_dir"])
    models_root = resolve_path(base_dir, config["paths"]["models_dir"])
    specs = _experiment_specs(data_root)
    eval_sets = {
        "1194m": data_root / "1194m" / "base",
        "realistic": data_root / "diversified_limited_size" / "base",
    }

    for name in names:
        if name not in specs:
            raise ValueError(f"未定義の実験名です: {name}（候補: {', '.join(specs)}）")
        spec = specs[name]
        for seed in seeds:
            run_dir = _model_run_dir(models_root, name, seed)
            if not (run_dir / "complete.json").exists() or overwrite:
                train_model(
                    config,
                    base_dir,
                    source_name=name,
                    seed=seed,
                    output_dir=run_dir,
                    epochs=epochs,
                    device_name=device_name,
                    overwrite=overwrite,
                    view_dirs_override=spec["view_dirs"],
                    view_slots_override=spec["view_slots"],
                    use_aperture_occlusion=spec["use_aperture_occlusion"],
                )
            else:
                print(f"[SKIP TRAIN] {name} seed={seed}", flush=True)

            ckpt = run_dir / "best.pt"
            for ds_name, ds_dir in eval_sets.items():
                evaluate_checkpoint(
                    config,
                    dataset_dir=ds_dir,
                    dataset_name=ds_name,
                    method="cnn_diversified",
                    checkpoint_path=ckpt,
                    seed=seed,
                    destination=ablation_root / "predictions" / name / str(seed) / f"test_{ds_name}.json",
                    device_name=device_name,
                    overwrite=overwrite,
                )


def build_report(config: dict[str, Any], base_dir: Path, ablation_root: Path) -> None:
    """全アブレーション結果と既存基準を集計し、REPORT.md・summary.json・CSV・図を出力する。"""
    data_root = resolve_path(base_dir, config["paths"]["data_dir"])
    specs = _experiment_specs(data_root)
    eval_root = resolve_path(base_dir, config["paths"]["outputs_dir"]) / "evaluation" / "predictions"
    base_pred_root = ablation_root / "predictions" / "baselines"

    rows_1seed: list[dict[str, Any]] = []
    # 既存基準（seed20260907および幾何）
    baseline_entries = [
        ("REF_1194m_cnn", "baseline", "1194M学習CNN（参考・1seed）", eval_root / "cnn_1194m_20260907" / "test_1194m.json", base_pred_root / "cnn_1194m_20260907" / "test_realistic.json"),
        ("A3_legacy_wide", "A_shape", "現行広範囲多様形状CNN（1seed）", eval_root / "cnn_diversified_20260907" / "test_1194m.json", base_pred_root / "cnn_diversified_20260907" / "test_realistic.json"),
        ("REF_canny", "baseline", "Canny＋輪郭別RANSAC", eval_root / "canny_fixed" / "test_1194m.json", base_pred_root / "canny_fixed" / "test_realistic.json"),
        ("REF_zhang2019", "baseline", "Zhang 2019型（再現実装）", eval_root / "zhang2019_fixed" / "test_1194m.json", base_pred_root / "zhang2019_fixed" / "test_realistic.json"),
    ]
    for key, group, label, p1194, preal in baseline_entries:
        if p1194.exists() and preal.exists():
            rows_1seed.append(
                {
                    "id": key,
                    "group": group,
                    "label": label,
                    "1194m": _summarize_prediction(read_json(p1194)),
                    "realistic": _summarize_prediction(read_json(preal)),
                }
            )

    for name, spec in specs.items():
        p1194 = ablation_root / "predictions" / name / str(DIAGNOSIS_SEED) / "test_1194m.json"
        preal = ablation_root / "predictions" / name / str(DIAGNOSIS_SEED) / "test_realistic.json"
        if p1194.exists() and preal.exists():
            rows_1seed.append(
                {
                    "id": name,
                    "group": spec["group"],
                    "label": spec["label"],
                    "1194m": _summarize_prediction(read_json(p1194)),
                    "realistic": _summarize_prediction(read_json(preal)),
                }
            )

    # 3seed集計（既存3seedが揃っているモデルのみ参考集計）
    rows_3seed: list[dict[str, Any]] = []
    seeds = [20260907, 20260908, 20260909]

    def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
        keys = items[0].keys()
        return {k: float(np.mean([it[k] for it in items])) for k in keys}

    for key, label, path_fn in [
        (
            "cnn_1194m_3seed",
            "1194M学習CNN（3seed平均）",
            lambda s, ds: (eval_root / f"cnn_1194m_{s}" / "test_1194m.json")
            if ds == "1194m"
            else (base_pred_root / f"cnn_1194m_{s}" / "test_realistic.json"),
        ),
        (
            "cnn_legacy_wide_3seed",
            "現行広範囲多様形状CNN（3seed平均）",
            lambda s, ds: (eval_root / f"cnn_diversified_{s}" / "test_1194m.json")
            if ds == "1194m"
            else (base_pred_root / f"cnn_diversified_{s}" / "test_realistic.json"),
        ),
    ]:
        p1 = [path_fn(s, "1194m") for s in seeds]
        pr = [path_fn(s, "realistic") for s in seeds]
        if all(p.exists() for p in p1 + pr):
            rows_3seed.append(
                {
                    "id": key,
                    "label": label,
                    "1194m": _mean_metrics([_summarize_prediction(read_json(p)) for p in p1]),
                    "realistic": _mean_metrics([_summarize_prediction(read_json(p)) for p in pr]),
                }
            )

    for name, spec in specs.items():
        p1 = [ablation_root / "predictions" / name / str(s) / "test_1194m.json" for s in seeds]
        pr = [ablation_root / "predictions" / name / str(s) / "test_realistic.json" for s in seeds]
        if all(p.exists() for p in p1 + pr):
            rows_3seed.append(
                {
                    "id": f"{name}_3seed",
                    "label": f"{spec['label']}（3seed平均）",
                    "1194m": _mean_metrics([_summarize_prediction(read_json(p)) for p in p1]),
                    "realistic": _mean_metrics([_summarize_prediction(read_json(p)) for p in pr]),
                }
            )

    for key, label, tag in [
        ("canny", "Canny＋輪郭別RANSAC", "canny_fixed"),
        ("zhang2019", "Zhang 2019型（再現実装）", "zhang2019_fixed"),
    ]:
        p1 = eval_root / tag / "test_1194m.json"
        pr = base_pred_root / tag / "test_realistic.json"
        if p1.exists() and pr.exists():
            rows_3seed.append(
                {
                    "id": key,
                    "label": label,
                    "1194m": _summarize_prediction(read_json(p1)),
                    "realistic": _summarize_prediction(read_json(pr)),
                }
            )

    summary = {"diagnosis_seed": DIAGNOSIS_SEED, "single_seed": rows_1seed, "three_seed": rows_3seed}
    write_json(ablation_root / "summary.json", summary)

    # CSV出力
    csv_path = ablation_root / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "id",
                "group",
                "label",
                "1194m_overall",
                "1194m_clean",
                "1194m_occ05",
                "1194m_wo10",
                "1194m_bc10",
                "realistic_overall",
                "realistic_clean",
                "realistic_occ05",
                "realistic_wo10",
                "realistic_bc10",
            ]
        )
        for row in rows_1seed:
            m1, mr = row["1194m"], row["realistic"]
            writer.writerow(
                [
                    row["id"],
                    row["group"],
                    row["label"],
                    f"{m1['overall']:.2f}",
                    f"{m1['clean']:.2f}",
                    f"{m1['occ_0.5']:.2f}",
                    f"{m1['whiteout_1.0']:.2f}",
                    f"{m1['black_crush_1.0']:.2f}",
                    f"{mr['overall']:.2f}",
                    f"{mr['clean']:.2f}",
                    f"{mr['occ_0.5']:.2f}",
                    f"{mr['whiteout_1.0']:.2f}",
                    f"{mr['black_crush_1.0']:.2f}",
                ]
            )

    # Markdownレポート出力
    lines = [
        "# 現実PAF寸法範囲に基づく要因分解・アブレーション報告",
        "",
        "- 評価基準：モノクロ256×256、塗りつぶし楕円 IoU ≥ 0.8、全13条件（白飛び最大内部強度 2.0）。",
        "- 現実PAF寸法範囲：`B/D ∈ [1.0, 2.5]`、`H/D ∈ [0.20, 1.20]`、`hole/D ∈ [0.75, 0.95]`、`t/D = 0.02`。",
        "- Zhang 2019型（再現実装）出典：Zhang, Pan, Ma, *Sensors*, 19(23), 5243, 2019, DOI `10.3390/s19235243`。",
        "",
        "## 1. 要因分解一覧（固定seed 20260907）",
        "",
        "|ID|区分|構成|1194M総合|1194M通常|1194M遮蔽0.5|1194M白飛1.0|1194M黒潰1.0|現実PAF総合|現実PAF通常|現実PAF遮蔽0.5|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_1seed:
        m1, mr = row["1194m"], row["realistic"]
        lines.append(
            f"|{row['id']}|{row['group']}|{row['label']}|"
            f"{m1['overall']:.2f}%|{m1['clean']:.2f}%|{m1['occ_0.5']:.2f}%|{m1['whiteout_1.0']:.2f}%|{m1['black_crush_1.0']:.2f}%|"
            f"{mr['overall']:.2f}%|{mr['clean']:.2f}%|{mr['occ_0.5']:.2f}%|"
        )

    if rows_3seed:
        lines.extend(
            [
                "",
                "## 2. 最終3seed平均比較（1194M評価データ / 現実PAF評価データ）",
                "",
                "|ID|方式|1194M総合|1194M通常|1194M遮蔽0.5|1194M白飛1.0|1194M黒潰1.0|現実PAF総合|現実PAF通常|現実PAF遮蔽0.5|",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows_3seed:
            m1, mr = row["1194m"], row["realistic"]
            lines.append(
                f"|{row['id']}|{row['label']}|"
                f"{m1['overall']:.2f}%|{m1['clean']:.2f}%|{m1['occ_0.5']:.2f}%|{m1['whiteout_1.0']:.2f}%|{m1['black_crush_1.0']:.2f}%|"
                f"{mr['overall']:.2f}%|{mr['clean']:.2f}%|{mr['occ_0.5']:.2f}%|"
            )

    (ablation_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 比較図出力（外観分解・要因分解）
    fig_dir = ablation_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = config["report"]["figures"]["font_family"]
    plt.rcParams["axes.unicode_minus"] = False

    by_id = {r["id"]: r for r in rows_1seed}
    appearance_ids = [k for k in ("B1_plain", "B2_albedo", "B3_stripe", "B4_noise", "B5_checker", "B6_mixed") if k in by_id]
    if appearance_ids:
        fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=180)
        x = np.arange(len(appearance_ids))
        w = 0.36
        v1 = [by_id[k]["1194m"]["overall"] for k in appearance_ids]
        vr = [by_id[k]["realistic"]["overall"] for k in appearance_ids]
        ax.bar(x - w / 2, v1, w, label="1194M 評価データ（未知実機CAD）", color="#1f77b4")
        ax.bar(x + w / 2, vr, w, label="現実PAF 評価データ", color="#d62728")
        for i, (a, b) in enumerate(zip(v1, vr)):
            ax.text(i - w / 2, a + 1.0, f"{a:.1f}%", ha="center", va="bottom", fontsize=8.5)
            ax.text(i + w / 2, b + 1.0, f"{b:.1f}%", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["無地", "明暗のみ", "放射縞", "ノイズ", "チェッカー", "3種混合"][: len(appearance_ids)])
        ax.set_ylim(0, 100)
        ax.set_ylabel("検出成功率 [%]（全13条件・IoU≥0.8）")
        ax.set_title("外観・テクスチャ分解比較（現実PAF範囲・投影拡張・開口幅遮蔽固定、seed=20260907）")
        ax.tick_params(direction="in")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(fig_dir / "appearance_ablation.png")
        plt.close(fig)

    factor_ids = [
        k
        for k in ("C0_base_fixed_occ", "C1_proj_only_fixed_occ", "C2_tex_only_fixed_occ", "C3_proj_tex_fixed_occ", "B6_mixed")
        if k in by_id
    ]
    shape_ids = [k for k in ("REF_1194m_cnn", "A3_legacy_wide", "A1_single_cone_mixed", "B6_mixed") if k in by_id]
    if factor_ids and shape_ids:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=180)
        x1 = np.arange(len(factor_ids))
        w = 0.36
        f_v1 = [by_id[k]["1194m"]["overall"] for k in factor_ids]
        f_vr = [by_id[k]["realistic"]["overall"] for k in factor_ids]
        ax1.bar(x1 - w / 2, f_v1, w, label="1194M 評価データ", color="#1f77b4")
        ax1.bar(x1 + w / 2, f_vr, w, label="現実PAF 評価データ", color="#d62728")
        for i, (a, b) in enumerate(zip(f_v1, f_vr)):
            ax1.text(i - w / 2, a + 1.0, f"{a:.1f}%", ha="center", va="bottom", fontsize=8.0)
            ax1.text(i + w / 2, b + 1.0, f"{b:.1f}%", ha="center", va="bottom", fontsize=8.0)
        ax1.set_xticks(x1)
        ax1.set_xticklabels(["基準\n(無地)", "＋投影", "＋3種模様", "＋投影\n＋3種模様", "＋開口幅\n遮蔽"][: len(factor_ids)])
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("検出成功率 [%]（全13条件・IoU≥0.8）")
        ax1.set_title("主要設計変更の寄与分解（現実PAF範囲、seed=20260907）")
        ax1.tick_params(direction="in")
        ax1.legend(frameon=True, fontsize=8.5)

        x2 = np.arange(len(shape_ids))
        s_v1 = [by_id[k]["1194m"]["overall"] for k in shape_ids]
        s_vr = [by_id[k]["realistic"]["overall"] for k in shape_ids]
        ax2.bar(x2 - w / 2, s_v1, w, label="1194M 評価データ", color="#1f77b4")
        ax2.bar(x2 + w / 2, s_vr, w, label="現実PAF 評価データ", color="#d62728")
        for i, (a, b) in enumerate(zip(s_v1, s_vr)):
            ax2.text(i - w / 2, a + 1.0, f"{a:.1f}%", ha="center", va="bottom", fontsize=8.0)
            ax2.text(i + w / 2, b + 1.0, f"{b:.1f}%", ha="center", va="bottom", fontsize=8.0)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(["1194M\n専用学習", "現行広範囲\n多様形状", "単一代表\n円錐", "現実PAF\n範囲"][: len(shape_ids)])
        ax2.set_ylim(0, 100)
        ax2.set_title("学習形状分布の比較（seed=20260907）")
        ax2.tick_params(direction="in")
        ax2.legend(frameon=True, fontsize=8.5)

        fig.tight_layout()
        fig.savefig(fig_dir / "factor_and_shape_ablation.png")
        plt.close(fig)

    print(f"[REPORT] {ablation_root / 'REPORT.md'}", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description="現実PAF寸法範囲での要因分解実験を実行・集計します。")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="現実PAFおよび単一代表円錐のマルチビューCGとmanifestを生成する")
    p_prep.add_argument(
        "--targets",
        nargs="+",
        choices=["diversified_limited_size", "single_cone"],
        default=["diversified_limited_size", "single_cone"],
    )
    p_prep.add_argument("--limit", type=int, default=None)

    p_base = sub.add_parser("baselines", help="既存基準4方式を現実PAF testで評価する")
    p_base.add_argument("--device", default="auto")
    p_base.add_argument("--overwrite", action="store_true")

    p_run = sub.add_parser("run", help="指定アブレーション実験を学習・評価する")
    p_run.add_argument("--names", nargs="+", required=True)
    p_run.add_argument("--seeds", nargs="+", type=int, default=[DIAGNOSIS_SEED])
    p_run.add_argument("--epochs", type=int, default=None)
    p_run.add_argument("--device", default="auto")
    p_run.add_argument("--overwrite", action="store_true")

    sub.add_parser("report", help="集計表・REPORT.md・比較図を生成する")

    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--render-config", type=Path, default=DEFAULT_RENDER_DIVERSIFIED_CONFIG)
    args = parser.parse_args()

    config, base_dir = load_config(args.config)
    ablation_root = (
        resolve_path(base_dir, config["paths"]["outputs_dir"]) / "ablation"
        if args.output is None
        else args.output.resolve()
    )

    if args.command == "prepare":
        prepare_datasets(config, base_dir, args.render_config, targets=args.targets, limit=args.limit)
    elif args.command == "baselines":
        evaluate_baselines_on_realistic(config, base_dir, ablation_root, device_name=args.device, overwrite=args.overwrite)
    elif args.command == "run":
        run_experiments(
            config,
            base_dir,
            ablation_root,
            names=args.names,
            seeds=args.seeds,
            epochs=args.epochs,
            device_name=args.device,
            overwrite=args.overwrite,
        )
    elif args.command == "report":
        build_report(config, base_dir, ablation_root)


if __name__ == "__main__":
    main()
