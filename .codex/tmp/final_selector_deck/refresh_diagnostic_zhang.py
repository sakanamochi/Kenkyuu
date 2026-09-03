"""最終選択変更後の撮像診断結果と強度別図だけを更新する。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.evaluate import _evaluate_zhang, _save_results
from paf_ring_detection.report import _build_diagnostic_severity_figure


def main() -> None:
    config = read_json(PROJECT_ROOT / "config" / "experiment.json")
    dataset = config["evaluation_datasets"]["diagnostic"]
    rows = _evaluate_zhang(
        config,
        PROJECT_ROOT / dataset["path"],
        dataset["split"],
        None,
    )
    result_root = PROJECT_ROOT / config["paths"]["results"]
    summary = _save_results(
        result_root / "diagnostic" / "zhang2019.csv",
        rows,
    )
    overall = read_json(result_root / "summary.json")
    overall["diagnostic"]["zhang2019_reproduction"] = summary["success_rate"]
    write_json(result_root / "summary.json", overall)
    _build_diagnostic_severity_figure(result_root, config["success_iou"])
    print(summary)


if __name__ == "__main__":
    main()
