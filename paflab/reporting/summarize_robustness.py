from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import zlib
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="方式別の劣化強度曲線と頑健性AUCを集計する")
    parser.add_argument("--config", default="config/research_experiment.json")
    return parser.parse_args()


def as_bool(value) -> bool:
    return str(value).lower() in ("true", "1", "yes")


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    probability = successes / count
    denominator = 1.0 + z**2 / count
    center = (probability + z**2 / (2 * count)) / denominator
    margin = z / denominator * math.sqrt(
        probability * (1 - probability) / count + z**2 / (4 * count**2)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def cluster_bootstrap_interval(rows: list[dict], iterations: int = 4000):
    """カメラ条件をクラスタとして成功率の95%区間を推定する。"""
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["camera_id"]].append(float(as_bool(row["top1_matches_ground_truth"])))
    values = list(clusters.values())
    seed_text = "|".join(sorted(row["sample_id"] for row in rows))
    rng = random.Random(zlib.crc32(seed_text.encode("utf-8")))
    estimates = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        flattened = [value for cluster in sample for value in cluster]
        estimates.append(statistics.fmean(flattened))
    estimates.sort()
    low = estimates[round(0.025 * (iterations - 1))]
    high = estimates[round(0.975 * (iterations - 1))]
    return low, high, len(values)


def normalized_auc(points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    if len(points) < 2 or points[-1][0] <= points[0][0]:
        return points[0][1] if points else 0.0
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def critical_severity(points: list[tuple[float, float]], threshold: float = 0.5):
    points = sorted(points)
    for index, (severity, value) in enumerate(points):
        if value >= threshold:
            continue
        if index == 0:
            return severity
        left_s, left_v = points[index - 1]
        if left_v == value:
            return severity
        ratio = (threshold - left_v) / (value - left_v)
        return left_s + ratio * (severity - left_s)
    return None


def main() -> None:
    args = parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    dataset_dir = project_path(config["stress_dataset_dir"])
    artifacts_dir = project_path(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for method in config["evaluation"]["methods"]:
        path = dataset_dir / "results" / method / "summary.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            all_rows[method] = list(csv.DictReader(file))

    grouped = defaultdict(list)
    curve_rows = []
    for method, rows in all_rows.items():
        clean_rows = [row for row in rows if row["degradation"] == "clean"]
        for degradation in config["degradations"]["types"]:
            grouped[(method, degradation, 0.0)].extend(clean_rows)
        for row in rows:
            if row["degradation"] != "clean":
                grouped[(method, row["degradation"], float(row["severity"]))].append(row)

    for (method, degradation, severity), rows in sorted(grouped.items()):
        successes = sum(as_bool(row["top1_matches_ground_truth"]) for row in rows)
        detected = sum(row["status"] == "detected" for row in rows)
        ious = [float(row["ellipse_iou"]) for row in rows if row["ellipse_iou"] not in ("", None)]
        low, high = wilson_interval(successes, len(rows))
        cluster_low, cluster_high, cluster_count = cluster_bootstrap_interval(rows)
        curve_rows.append(
            {
                "method": method,
                "degradation": degradation,
                "severity": severity,
                "sample_count": len(rows),
                "detection_rate": detected / len(rows),
                "success_rate": successes / len(rows),
                "success_ci95_low": low,
                "success_ci95_high": high,
                "camera_cluster_count": cluster_count,
                "cluster_bootstrap_ci95_low": cluster_low,
                "cluster_bootstrap_ci95_high": cluster_high,
                "mean_ellipse_iou": statistics.fmean(ious) if ious else None,
                "median_ellipse_iou": statistics.median(ious) if ious else None,
            }
        )

    with (artifacts_dir / "robustness_curves.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    aggregate = []
    for method in config["evaluation"]["methods"]:
        for degradation in config["degradations"]["types"]:
            points = [
                (float(row["severity"]), float(row["success_rate"]))
                for row in curve_rows
                if row["method"] == method and row["degradation"] == degradation
            ]
            aggregate.append(
                {
                    "method": method,
                    "degradation": degradation,
                    "robustness_auc": normalized_auc(points),
                    "critical_severity_50": critical_severity(points),
                }
            )
    summary = {
        "uncertainty_method": "camera_id cluster bootstrap, 4000 resamples",
        "curves": curve_rows,
        "aggregate": aggregate,
    }
    (artifacts_dir / "robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
