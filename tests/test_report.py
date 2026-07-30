import csv
from pathlib import Path

from paf_ring_detection.report import _success_by_severity


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("degradation", "severity", "success"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_success_by_severity_uses_clean_as_zero_percent(tmp_path):
    path = tmp_path / "results.csv"
    _write_rows(
        path,
        [
            {"degradation": "clean", "severity": 0.0, "success": True},
            {"degradation": "clean", "severity": 0.0, "success": False},
            {
                "degradation": "black_rectangle",
                "severity": 0.25,
                "success": True,
            },
            {
                "degradation": "black_rectangle",
                "severity": 0.25,
                "success": True,
            },
            {
                "degradation": "sensor_whiteout",
                "severity": 0.25,
                "success": False,
            },
        ],
    )

    values = _success_by_severity(path, "black_rectangle")

    assert values == [
        {"severity": 0.0, "sample_count": 2, "success_rate": 0.5},
        {"severity": 0.25, "sample_count": 2, "success_rate": 1.0},
    ]
