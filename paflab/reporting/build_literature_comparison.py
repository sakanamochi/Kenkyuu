from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from analysis.ellipse_baseline import preprocess_image
from analysis.zhang_arc_detector import draw_zhang_arcs, extract_zhang_arcs
from paflab.reporting.build_summary_figures import (
    MUTED,
    Predictor,
    classic_prediction,
    setup_style,
    show_overlay,
    show_probability,
)
from paflab.image_io import imread


ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_methods(config: dict, include_ablations: bool) -> list[str]:
    methods = list(config["primary_methods"])
    if include_ablations:
        methods = [*config.get("ablation_methods", []), *methods]
    return methods


def available_methods(
    dataset: Path,
    dataset_config: dict,
    methods: list[str],
) -> list[str]:
    return [
        method
        for method in methods
        if method in dataset_config["results"]
        for result_name in [dataset_config["results"][method]]
        if (dataset / "results" / result_name / "summary.json").exists()
    ]


def build_aggregate(
    config: dict,
    output: Path,
    methods: list[str],
) -> list[dict]:
    rows = []
    for dataset_config in config["datasets"]:
        dataset = project_path(dataset_config["dataset_dir"])
        for method in available_methods(dataset, dataset_config, methods):
            result_name = dataset_config["results"][method]
            summary = load_config(dataset / "results" / result_name / "summary.json")
            rows.append(
                {
                    "dataset_id": dataset_config["id"],
                    "dataset": dataset_config["label"],
                    "method": method,
                    "method_label": config["methods"][method],
                    "result_name": result_name,
                    "sample_count": int(summary["sample_count"]),
                    "detected_count": int(summary["detected_count"]),
                    "top1_match_count": int(summary["top1_match_count"]),
                    "top1_match_rate": float(summary["top1_match_rate"]),
                }
            )
    (output / "comparison-summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "comparison-summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_summary_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def build_stratified_and_audit(
    config: dict,
    output: Path,
    methods: list[str],
) -> dict:
    stratified = []
    audit = {"datasets": {}}
    factors = ("camera_tilt_deg", "background", "degradation", "severity")
    for dataset_config in config["datasets"]:
        dataset = project_path(dataset_config["dataset_dir"])
        rows_by_method = {}
        for method in available_methods(dataset, dataset_config, methods):
            result_name = dataset_config["results"][method]
            rows_by_method[method] = _read_summary_rows(
                dataset / "results" / result_name / "summary.csv"
            )
        id_sets = {
            method: {row["sample_id"] for row in rows}
            for method, rows in rows_by_method.items()
        }
        reference_method = next(iter(id_sets))
        reference_ids = id_sets[reference_method]
        mismatches = {
            method: {
                "missing_from_method": len(reference_ids - ids),
                "extra_in_method": len(ids - reference_ids),
            }
            for method, ids in id_sets.items()
            if ids != reference_ids
        }
        if mismatches:
            raise ValueError(
                f"{dataset_config['id']}で方式間のsample_idが一致しません: {mismatches}"
            )
        audit["datasets"][dataset_config["id"]] = {
            "sample_count": len(reference_ids),
            "methods": list(rows_by_method),
            "sample_ids_identical": True,
        }

        for method, method_rows in rows_by_method.items():
            for factor in factors:
                groups = {}
                for row in method_rows:
                    level = row.get(factor, "")
                    if level == "":
                        continue
                    groups.setdefault(level, []).append(row)
                for level, group in sorted(groups.items()):
                    success = sum(
                        row["top1_matches_ground_truth"].strip().lower()
                        in ("true", "1")
                        for row in group
                    )
                    stratified.append(
                        {
                            "dataset_id": dataset_config["id"],
                            "method": method,
                            "method_label": config["methods"][method],
                            "factor": factor,
                            "level": level,
                            "sample_count": len(group),
                            "success_count": success,
                            "success_rate": success / len(group),
                        }
                    )
    with (output / "comparison-stratified.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(stratified[0]))
        writer.writeheader()
        writer.writerows(stratified)
    (output / "comparison-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def build_bar_chart(
    config: dict,
    rows: list[dict],
    output: Path,
    methods: list[str],
) -> Path:
    method_order = [*methods, "aamed"]
    present = {row["method"] for row in rows}
    method_order = [method for method in method_order if method in present]
    datasets = [item["id"] for item in config["datasets"]]
    labels = {item["id"]: item["label"] for item in config["datasets"]}
    lookup = {(row["dataset_id"], row["method"]): row for row in rows}
    x = np.arange(len(datasets), dtype=float)
    width = 0.78 / len(method_order)
    colors = (
        "#6b7280",
        "#f59e0b",
        "#8b5cf6",
        "#2563eb",
        "#10b981",
        "#ef4444",
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    for index, method in enumerate(method_order):
        values = [lookup[(dataset, method)]["top1_match_rate"] * 100 for dataset in datasets]
        positions = x - 0.39 + width * (index + 0.5)
        bars = ax.bar(
            positions,
            values,
            width=width * 0.92,
            label=config["methods"][method],
            color=colors[index],
        )
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_title("文献ベースライン比較：同一データ・同一IoU基準", fontsize=18, pad=14)
    ax.set_ylabel("成功率（IoU ≥ 0.80）[%]")
    ax.set_xticks(x, [labels[dataset] for dataset in datasets])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    fig.tight_layout()
    path = output / "method-success-rate-comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _show_image(ax, image) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])


def build_grid(
    config: dict,
    dataset_config: dict,
    output: Path,
    predictor: Predictor,
    include_ablations: bool,
) -> Path:
    dataset = project_path(dataset_config["dataset_dir"])
    manifest = load_config(dataset / "manifest.json")
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
    baseline = load_config(project_path(config["baseline_config"]))
    sample_ids = dataset_config["representative_samples"]
    titles = dataset_config["representative_titles"]
    rows = [
        "入力画像",
        "Cannyエッジ",
        "Zhang型の弧抽出",
        "Zhang 2019型",
        "CNNリング尤度",
        "CNN + RANSAC",
    ]
    if include_ablations:
        rows[3:3] = ["Contour fit", "Canny + 輪郭別RANSAC"]
    fig, axes = plt.subplots(len(rows), len(sample_ids), figsize=(16.3, 22.5))
    fig.suptitle(
        f"{dataset_config['label']}：入力・中間表現・推定楕円の対応比較",
        fontsize=22,
        y=0.995,
    )
    for column, (sample_id, title) in enumerate(zip(sample_ids, titles)):
        sample = samples[sample_id]
        source_bgr = imread(dataset / sample["image"], cv2.IMREAD_COLOR)
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        source_256 = cv2.resize(source_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        predicted = predictor.predict(dataset, sample)
        axes[0, column].set_title(title, fontsize=13, pad=7)
        _show_image(axes[0, column], source_256)

        stages = preprocess_image(source_bgr, baseline["detector"])
        _show_image(axes[1, column], stages["edges"])
        arc_stages = extract_zhang_arcs(
            source_bgr, baseline["detector"], baseline["zhang2019_arc"]
        )
        arcs_rgb = cv2.cvtColor(
            draw_zhang_arcs(source_bgr, arc_stages, []), cv2.COLOR_BGR2RGB
        )
        _show_image(
            axes[2, column],
            cv2.resize(arcs_rgb, (256, 256), interpolation=cv2.INTER_AREA),
        )

        classic_methods = ["zhang2019_arc_reproduction"]
        if include_ablations:
            classic_methods = [
                "contour_fit",
                "canny_ransac_inner_pair",
                *classic_methods,
            ]
        for row_index, method in enumerate(classic_methods, start=3):
            result_name = dataset_config["results"][method]
            ellipse, evaluation = classic_prediction(dataset, result_name, sample_id)
            show_overlay(
                axes[row_index, column],
                predicted["image"],
                predicted["ground_truth"],
                ellipse,
                evaluation,
            )
        cnn_probability_row = len(rows) - 2
        cnn_output_row = len(rows) - 1
        show_probability(axes[cnn_probability_row, column], predicted["probability"])
        show_overlay(
            axes[cnn_output_row, column],
            predicted["image"],
            predicted["ground_truth"],
            predicted["predicted"],
            predicted["evaluation"],
        )
        for row_index, row_label in enumerate(rows):
            if column == 0:
                axes[row_index, column].set_ylabel(row_label, fontsize=11, labelpad=11)
    fig.text(
        0.5,
        0.008,
        "水色破線: 正解楕円　緑: IoU ≥ 0.80　赤: IoU < 0.80　全方式で同一画像・同一正解楕円",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.035, 0.022, 1, 0.985), h_pad=0.65, w_pad=0.65)
    path = output / f"{dataset_config['id']}-input-intermediate-output-grid.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="文献ベースライン比較の集計と図を生成する")
    parser.add_argument(
        "--config", default="config/literature_baseline_comparison.json"
    )
    parser.add_argument("--include-ablations", action="store_true")
    args = parser.parse_args()
    config = load_config(project_path(args.config))
    output = project_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    methods = selected_methods(config, args.include_ablations)
    rows = build_aggregate(config, output, methods)
    build_stratified_and_audit(config, output, methods)
    build_bar_chart(config, rows, output, methods)
    predictor = Predictor()
    for dataset_config in config["datasets"]:
        build_grid(
            config,
            dataset_config,
            output,
            predictor,
            args.include_ablations,
        )
    print(output)


if __name__ == "__main__":
    main()
