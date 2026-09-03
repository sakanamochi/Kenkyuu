"""基準モデルと比較モデルの対応あり集計・図を作る。"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paf_ring_detection.data import (  # noqa: E402
    label_ellipse,
    read_json,
    write_json,
)
from paf_ring_detection.geometry import evaluate_ellipses  # noqa: E402


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["success"] = row["success"].strip().lower() == "true"
    return rows


def _mcnemar_exact(reference_only: int, comparison_only: int) -> float:
    """不一致ペアだけを使う両側の正確二項検定。"""
    discordant = reference_only + comparison_only
    if discordant == 0:
        return 1.0
    smaller = min(reference_only, comparison_only)
    lower_tail = sum(
        math.comb(discordant, index) for index in range(smaller + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * lower_tail)


def _paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    batch_size = 500
    for start in range(0, samples, batch_size):
        end = min(start + batch_size, samples)
        indices = rng.integers(
            0,
            len(differences),
            size=(end - start, len(differences)),
        )
        estimates[start:end] = differences[indices].mean(axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        return 0.0, 0.0
    z_value = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z_value**2 / count
    center = (proportion + z_value**2 / (2.0 * count)) / denominator
    margin = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z_value**2 / (4.0 * count**2)
        )
        / denominator
    )
    return center - margin, center + margin


def _group_rates(
    reference: dict[str, dict],
    comparison: dict[str, dict],
    field: str,
) -> list[dict]:
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for sample_id, comparison_row in comparison.items():
        groups[comparison_row[field]].append(
            (reference[sample_id]["success"], comparison_row["success"])
        )
    rows = []
    for group, pairs in sorted(groups.items()):
        rows.append(
            {
                "group": group,
                "sample_count": len(pairs),
                "reference_success_rate": sum(pair[0] for pair in pairs) / len(pairs),
                "comparison_success_rate": sum(pair[1] for pair in pairs) / len(pairs),
            }
        )
    return rows


def _geometry_fairness_audit(
    experiment: dict,
    paired_sample_ids: set[str],
) -> dict:
    """正解リングの画面上位置・寸法が両モデルで揃っているか確認する。"""
    reference_root = _project_path(experiment["reference_dataset"])
    comparison_root = _project_path(experiment["comparison_dataset"])
    reference_manifest = read_json(reference_root / "manifest.json")
    comparison_manifest = read_json(comparison_root / "manifest.json")
    reference_samples = {
        sample["sample_id"]: sample for sample in reference_manifest["samples"]
    }
    comparison_samples = {
        sample["sample_id"]: sample for sample in comparison_manifest["samples"]
    }
    center_errors = []
    minor_axis_differences = []
    major_axis_differences = []
    ellipse_ious = []

    for sample_id in sorted(paired_sample_ids):
        reference_label = read_json(
            reference_root / reference_samples[sample_id]["label"]
        )
        comparison_label = read_json(
            comparison_root / comparison_samples[sample_id]["label"]
        )
        reference_ellipse = label_ellipse(reference_label)
        comparison_ellipse = label_ellipse(comparison_label)
        metrics = evaluate_ellipses(
            comparison_ellipse,
            reference_ellipse,
            (
                int(reference_label["image_height"]),
                int(reference_label["image_width"]),
            ),
        )
        reference_axes = sorted(reference_ellipse[1])
        comparison_axes = sorted(comparison_ellipse[1])
        center_errors.append(metrics["center_error_px"])
        ellipse_ious.append(metrics["ellipse_iou"])
        minor_axis_differences.append(
            100.0
            * abs(comparison_axes[0] - reference_axes[0])
            / reference_axes[0]
        )
        major_axis_differences.append(
            100.0
            * abs(comparison_axes[1] - reference_axes[1])
            / reference_axes[1]
        )

    return {
        "paired_sample_count": len(center_errors),
        "ground_truth_ellipse_iou_mean": float(np.mean(ellipse_ious)),
        "ground_truth_ellipse_iou_min": float(np.min(ellipse_ious)),
        "center_difference_px_mean": float(np.mean(center_errors)),
        "center_difference_px_max": float(np.max(center_errors)),
        "minor_axis_difference_percent_mean": float(
            np.mean(minor_axis_differences)
        ),
        "minor_axis_difference_percent_max": float(
            np.max(minor_axis_differences)
        ),
        "major_axis_difference_percent_mean": float(
            np.mean(major_axis_differences)
        ),
        "major_axis_difference_percent_max": float(
            np.max(major_axis_differences)
        ),
    }


def _save_figure(summary: dict, output: Path, title: str) -> None:
    comparison_display_name = summary["comparison"].get(
        "display_name",
        "比較A",
    )
    labels = ["1194M_3\n既知モデル", f"{comparison_display_name}\n未学習モデル"]
    rates = [
        summary["reference"]["success_rate"],
        summary["comparison"]["success_rate"],
    ]
    intervals = [
        summary["reference"]["wilson_95_ci"],
        summary["comparison"]["wilson_95_ci"],
    ]
    lower = [rate - interval[0] for rate, interval in zip(rates, intervals)]
    upper = [interval[1] - rate for rate, interval in zip(rates, intervals)]

    plt.rcParams["font.family"] = [
        "Yu Gothic",
        "Meiryo",
        "DejaVu Sans",
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar(
        labels,
        rates,
        color=["#4C78A8", "#E45756"],
        width=0.58,
        yerr=np.asarray([lower, upper]),
        capsize=6,
    )
    axis.axhline(0.8, color="#555555", linestyle="--", linewidth=1.2)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("成功率（楕円IoU ≥ 0.80）")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.22)
    for bar, rate in zip(bars, rates):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            rate + 0.035,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    difference = summary["paired_comparison"]["success_rate_difference"]
    ci_low, ci_high = summary["paired_comparison"][
        "paired_bootstrap_95_ci"
    ]
    figure.text(
        0.5,
        0.015,
        f"差（{comparison_display_name} − 1194M_3）: {difference:+.1%} "
        f"[対応あり95% CI {ci_low:+.1%}, {ci_high:+.1%}]",
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _build_zhang2019_summary(experiment: dict) -> dict | None:
    results_prefix = experiment.get("results_prefix", "comparison_a")
    comparison_path = (
        _project_path(experiment["results_dir"])
        / f"{results_prefix}_zhang2019.csv"
    )
    if not comparison_path.exists():
        return None
    reference_rows = _read_rows(
        _project_path(experiment["reference_zhang2019_results"])
    )
    comparison_rows = _read_rows(comparison_path)
    reference_all = {row["sample_id"]: row for row in reference_rows}
    comparison = {row["sample_id"]: row for row in comparison_rows}
    missing = sorted(set(comparison) - set(reference_all))
    if missing:
        raise RuntimeError(
            f"Zhang型の基準結果に存在しない撮像条件があります: {missing[:3]}"
        )
    reference = {
        sample_id: reference_all[sample_id] for sample_id in comparison
    }
    pairs = np.asarray(
        [
            (
                int(reference[sample_id]["success"]),
                int(comparison[sample_id]["success"]),
            )
            for sample_id in comparison
        ],
        dtype=np.int8,
    )
    sample_count = len(pairs)
    reference_count = int(pairs[:, 0].sum())
    comparison_count = int(pairs[:, 1].sum())
    reference_only = int(
        np.count_nonzero((pairs[:, 0] == 1) & (pairs[:, 1] == 0))
    )
    comparison_only = int(
        np.count_nonzero((pairs[:, 0] == 0) & (pairs[:, 1] == 1))
    )
    differences = pairs[:, 1].astype(float) - pairs[:, 0].astype(float)
    return {
        "method": "Zhang 2019型（再現実装）",
        "paired_sample_count": sample_count,
        "reference": {
            "model_id": "1194M_3",
            "success_count": reference_count,
            "success_rate": reference_count / sample_count,
            "wilson_95_ci": list(
                _wilson_interval(reference_count, sample_count)
            ),
        },
        "comparison": {
            "model_id": experiment.get(
                "comparison_model_id",
                "comparison_a_sparse_rib",
            ),
            "display_name": experiment.get(
                "comparison_display_name",
                "比較A",
            ),
            "success_count": comparison_count,
            "success_rate": comparison_count / sample_count,
            "wilson_95_ci": list(
                _wilson_interval(comparison_count, sample_count)
            ),
        },
        "paired_comparison": {
            "success_rate_difference": float(differences.mean()),
            "paired_bootstrap_95_ci": _paired_bootstrap_ci(
                differences,
                samples=int(experiment["bootstrap_samples"]),
                seed=int(experiment["seed"]) + 1,
            ),
            "both_success": int(
                np.count_nonzero((pairs[:, 0] == 1) & (pairs[:, 1] == 1))
            ),
            "reference_only_success": reference_only,
            "comparison_only_success": comparison_only,
            "both_failure": int(
                np.count_nonzero((pairs[:, 0] == 0) & (pairs[:, 1] == 0))
            ),
            "mcnemar_exact_two_sided_p": _mcnemar_exact(
                reference_only,
                comparison_only,
            ),
        },
        "by_background": _group_rates(reference, comparison, "background"),
        "by_camera_tilt_deg": _group_rates(
            reference,
            comparison,
            "camera_tilt_deg",
        ),
    }


def build_report(experiment: dict) -> dict:
    reference_rows = _read_rows(_project_path(experiment["reference_results"]))
    results_prefix = experiment.get("results_prefix", "comparison_a")
    comparison_path = (
        _project_path(experiment["results_dir"]) / f"{results_prefix}_cnn.csv"
    )
    comparison_rows = _read_rows(comparison_path)
    reference_all = {row["sample_id"]: row for row in reference_rows}
    comparison = {row["sample_id"]: row for row in comparison_rows}
    missing = sorted(set(comparison) - set(reference_all))
    if missing:
        raise RuntimeError(f"基準結果に存在しない撮像条件があります: {missing[:3]}")
    reference = {sample_id: reference_all[sample_id] for sample_id in comparison}

    pairs = np.asarray(
        [
            (
                int(reference[sample_id]["success"]),
                int(comparison[sample_id]["success"]),
            )
            for sample_id in comparison
        ],
        dtype=np.int8,
    )
    reference_count = int(pairs[:, 0].sum())
    comparison_count = int(pairs[:, 1].sum())
    reference_only = int(np.count_nonzero((pairs[:, 0] == 1) & (pairs[:, 1] == 0)))
    comparison_only = int(np.count_nonzero((pairs[:, 0] == 0) & (pairs[:, 1] == 1)))
    both_success = int(np.count_nonzero((pairs[:, 0] == 1) & (pairs[:, 1] == 1)))
    both_failure = int(np.count_nonzero((pairs[:, 0] == 0) & (pairs[:, 1] == 0)))
    sample_count = len(pairs)
    differences = pairs[:, 1].astype(float) - pairs[:, 0].astype(float)

    summary = {
        "experiment_id": experiment["experiment_id"],
        "success_definition": "推定楕円と正解内周楕円のIoUが0.80以上",
        "paired_sample_count": sample_count,
        "reference": {
            "model_id": "1194M_3",
            "training_domain": "seen_model",
            "success_count": reference_count,
            "success_rate": reference_count / sample_count,
            "wilson_95_ci": list(
                _wilson_interval(reference_count, sample_count)
            ),
        },
        "comparison": {
            "model_id": experiment.get(
                "comparison_model_id",
                "comparison_a_sparse_rib",
            ),
            "display_name": experiment.get(
                "comparison_display_name",
                "比較A",
            ),
            "training_domain": experiment.get(
                "comparison_training_domain",
                "unseen_parametric_model",
            ),
            "success_count": comparison_count,
            "success_rate": comparison_count / sample_count,
            "wilson_95_ci": list(
                _wilson_interval(comparison_count, sample_count)
            ),
        },
        "paired_comparison": {
            "success_rate_difference": float(differences.mean()),
            "paired_bootstrap_95_ci": _paired_bootstrap_ci(
                differences,
                samples=int(experiment["bootstrap_samples"]),
                seed=int(experiment["seed"]),
            ),
            "both_success": both_success,
            "reference_only_success": reference_only,
            "comparison_only_success": comparison_only,
            "both_failure": both_failure,
            "mcnemar_exact_two_sided_p": _mcnemar_exact(
                reference_only,
                comparison_only,
            ),
        },
        "ground_truth_geometry_audit": _geometry_fairness_audit(
            experiment,
            set(comparison),
        ),
        "by_background": _group_rates(reference, comparison, "background"),
        "by_camera_tilt_deg": _group_rates(
            reference,
            comparison,
            "camera_tilt_deg",
        ),
        "limitations": experiment.get(
            "limitations",
            [
                "比較Aは実在機CADではなく、PAFらしい構成を持つ合成モデルである",
                "形状パラメータを同時に変更したため、性能差の原因となる部位は特定できない",
                "同じ内周寸法と撮像条件を使うため、未知スケールへの一般化は評価していない",
            ],
        ),
    }
    results_dir = _project_path(experiment["results_dir"])
    zhang2019_summary = _build_zhang2019_summary(experiment)
    if zhang2019_summary is not None:
        summary["zhang2019_reproduction"] = zhang2019_summary
        _save_figure(
            zhang2019_summary,
            results_dir / "zhang2019_model_dependency_comparison.png",
            "Zhang 2019型（再現実装）のモデル変更耐性",
        )
    write_json(results_dir / "summary.json", summary)
    _save_figure(
        summary,
        results_dir / "model_dependency_comparison.png",
        "CNN + weighted RANSAC のモデル変更耐性",
    )
    return summary


if __name__ == "__main__":
    experiment = read_json(
        PROJECT_ROOT / "model_generalization_study/config/experiment.json"
    )
    print(build_report(experiment))
