"""評価結果を1つのJSONと比較図にまとめる。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from paf_ring_detection.data import read_json, write_json


def build_report(config: dict) -> None:
    result_root = Path(config["paths"]["results"])
    summary = {}

    for dataset_name in config["evaluation_datasets"]:
        zhang = read_json(result_root / dataset_name / "zhang2019.json")
        cnn = read_json(result_root / dataset_name / "cnn_ransac.json")
        if zhang["sample_count"] != cnn["sample_count"]:
            raise ValueError(f"{dataset_name}: 2方式の評価枚数が一致しません")
        summary[dataset_name] = {
            "sample_count": zhang["sample_count"],
            "zhang2019_reproduction": zhang["success_rate"],
            "cnn_weighted_ransac": cnn["success_rate"],
        }

    write_json(result_root / "summary.json", summary)

    names = list(summary)
    x = range(len(names))
    zhang_rates = [summary[name]["zhang2019_reproduction"] * 100 for name in names]
    cnn_rates = [summary[name]["cnn_weighted_ransac"] * 100 for name in names]

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([value - 0.18 for value in x], zhang_rates, 0.36, label="Zhang 2019 reproduction")
    axis.bar([value + 0.18 for value in x], cnn_rates, 0.36, label="CNN + weighted RANSAC")
    axis.set_xticks(list(x), names)
    axis.set_ylabel("Success rate [%]")
    axis.set_ylim(0, 100)
    axis.legend()
    figure.tight_layout()
    figure.savefig(result_root / "comparison.png", dpi=160)
    plt.close(figure)
    print(f"集計を作成しました: {result_root}")
