"""4方式×2評価集合×13条件の統一評価と確定参照照合。"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.effects import load_sample_and_truth
from core.geometry import ellipse_to_list, evaluate_ellipses
from core.io import (
    load_manifest_samples,
    read_json,
    resolve_path,
    sha256_bytes,
    sha256_file,
    write_json,
)
from core.model import load_checkpoint, resolve_device
from inference.predictor import CNN_METHODS, predict_on_256


def sample_geometry_seed(evaluation_seed: int, sample_id: str) -> int:
    """サンプルIDごとに固定した幾何RANSAC用乱数seedを計算する。"""
    return (int(evaluation_seed) + zlib.crc32(sample_id.encode("utf-8"))) % (2**32)


def _method_tag(method: str, seed: int | None) -> str:
    return f"{method}_{seed}" if seed is not None else f"{method}_fixed"


def _evaluate_dataset(
    config: dict[str, Any],
    dataset_dir: Path,
    dataset_name: str,
    samples: list[dict[str, Any]],
    method: str,
    *,
    seed: int | None,
    model: torch.nn.Module | None,
    checkpoint_path: Path | None,
    device: torch.device | None,
    whiteout_maximum_strength: float,
    destination: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if destination.exists() and not overwrite:
        saved = read_json(destination)
        if len(saved.get("rows", [])) == len(samples):
            print(f"[SKIP] 既存結果を使用: {destination.parent.name}/{destination.name}", flush=True)
            return saved

    input_size = int(config["input_size"])
    success_iou = float(config["success_iou"])
    eval_seed = int(config["evaluation_seed"])
    rows: list[dict[str, Any]] = []

    print(
        f"[EVAL] dataset={dataset_name} method={method} seed={seed} "
        f"n={len(samples)} whiteout_max={whiteout_maximum_strength:g}",
        flush=True,
    )
    for index, sample in enumerate(samples, start=1):
        image, truth = load_sample_and_truth(
            dataset_dir,
            sample,
            input_size=input_size,
            whiteout_maximum_strength=whiteout_maximum_strength,
        )
        geo_seed = sample_geometry_seed(eval_seed, sample["sample_id"]) if method == "canny" else None
        ellipse, _, _ = predict_on_256(
            image,
            method,
            config,
            model=model,
            device=device,
            geometry_seed=geo_seed,
        )
        iou = (
            evaluate_ellipses(ellipse, truth, (input_size, input_size))["ellipse_iou"]
            if ellipse is not None
            else 0.0
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "base_sample_id": sample["conditions"]["base_sample_id"],
                "group_id": sample["group_id"],
                "degradation": sample["conditions"]["degradation"],
                "severity": float(sample["conditions"]["severity"]),
                "ellipse": ellipse_to_list(ellipse),
                "truth_ellipse": ellipse_to_list(truth),
                "ellipse_iou": float(iou),
                "success": bool(ellipse is not None and iou >= success_iou),
                "image_sha256": sha256_bytes(image.tobytes()),
            }
        )
        if index % 400 == 0:
            print(f"  progress {index}/{len(samples)}", flush=True)

    successes = sum(1 for row in rows if row["success"])
    payload = {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "whiteout_maximum_strength": float(whiteout_maximum_strength),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "n": len(rows),
        "successes": successes,
        "success_percent": 100.0 * successes / len(rows) if rows else 0.0,
        "mean_iou": float(np.mean([row["ellipse_iou"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }
    write_json(destination, payload)
    print(
        f"  -> {successes}/{len(rows)} ({payload['success_percent']:.2f}%, "
        f"mean_iou={payload['mean_iou']:.4f})",
        flush=True,
    )
    return payload


def run_evaluation(
    config: dict[str, Any],
    base_dir: Path,
    *,
    output_dir: Path,
    methods: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    models_dir: Path | None = None,
    device_name: str = "auto",
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """指定方式・評価集合で全13条件の評価を実行し、結果JSONを保存する。"""
    active_methods = list(config["evaluation"]["methods"] if methods is None else methods)
    active_datasets = list(config["evaluation"]["test_datasets"] if datasets is None else datasets)
    active_seeds = [int(s) for s in (config["seeds"] if seeds is None else seeds)]
    whiteout_max = float(config["evaluation"]["whiteout_maximum_strength"])

    data_root = resolve_path(base_dir, config["paths"]["data_dir"])
    resolved_models = (
        resolve_path(base_dir, config["paths"]["models_dir"])
        if models_dir is None
        else Path(models_dir).resolve()
    )
    out_root = Path(output_dir).resolve()
    device = resolve_device(device_name) if any(m in CNN_METHODS for m in active_methods) else None

    written: list[dict[str, Any]] = []
    for method in active_methods:
        run_seeds: list[int | None] = active_seeds if method in CNN_METHODS else [None]
        for seed in run_seeds:
            model = None
            checkpoint_path = None
            if method in CNN_METHODS:
                source = "1194m" if method == "cnn_1194m" else "diversified"
                checkpoint_path = resolved_models / source / str(seed) / "best.pt"
                assert device is not None
                model, _ = load_checkpoint(
                    checkpoint_path,
                    device,
                    default_base_channels=int(config["training"]["base_channels"]),
                )

            tag = _method_tag(method, seed)
            for dataset_name in active_datasets:
                dataset_dir = (data_root / config["datasets"][dataset_name]["base"]).resolve()
                all_test = load_manifest_samples(dataset_dir, "test")
                samples = all_test[:limit] if limit is not None else all_test
                dest = out_root / "predictions" / tag / f"test_{dataset_name}.json"
                res = _evaluate_dataset(
                    config,
                    dataset_dir,
                    dataset_name,
                    samples,
                    method,
                    seed=seed,
                    model=model,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    whiteout_maximum_strength=whiteout_max,
                    destination=dest,
                    overwrite=overwrite,
                )
                written.append(
                    {
                        "method": method,
                        "seed": seed,
                        "dataset": dataset_name,
                        "n": res["n"],
                        "successes": res["successes"],
                        "success_percent": res["success_percent"],
                        "path": str(dest),
                    }
                )

    manifest = {
        "methods": active_methods,
        "datasets": active_datasets,
        "seeds": active_seeds,
        "whiteout_maximum_strength": whiteout_max,
        "limited": limit,
        "runs": written,
    }
    write_json(out_root / "evaluation_index.json", manifest)
    return manifest


def _ellipses_close(a: list[Any] | None, b: list[Any] | None, atol: float = 1e-4) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    flat_a = np.array([a[0][0], a[0][1], a[1][0], a[1][1], a[2]], dtype=np.float64)
    flat_b = np.array([b[0][0], b[0][1], b[1][0], b[1][1], b[2]], dtype=np.float64)
    return bool(np.allclose(flat_a, flat_b, rtol=1e-5, atol=atol))


def verify_against_references(
    config: dict[str, Any],
    base_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """全13条件（白飛び最大内部強度2.0）のCNN評価出力を確定参照JSONと全件照合する。"""
    references_dir = resolve_path(base_dir, config["paths"]["references_dir"])
    out_root = Path(output_dir).resolve()
    datasets = list(config["evaluation"]["test_datasets"])
    seeds = [int(s) for s in config["seeds"]]

    checked = 0
    ellipse_mismatches = 0
    iou_mismatches = 0
    success_mismatches = 0
    image_hash_mismatches = 0
    details: list[dict[str, Any]] = []

    for source, method in (("1194m", "cnn_1194m"), ("diversified", "cnn_diversified")):
        for seed in seeds:
            for dataset in datasets:
                new_path = out_root / "predictions" / f"{method}_{seed}" / f"test_{dataset}.json"
                base_ref_path = references_dir / "results" / source / str(seed) / f"test_{dataset}.json"
                if not new_path.exists() or not base_ref_path.exists():
                    continue

                # 非白飛び9条件はresults参照、白飛び4条件はwhiteout参照（最大強度2.0）から期待値を構築する。
                expected_rows: dict[str, dict[str, Any]] = {}
                for row in read_json(base_ref_path)["rows"]:
                    if row["degradation"] != "sensor_whiteout":
                        expected_rows[row["sample_id"]] = row
                for severity in (0.25, 0.5, 0.75, 1.0):
                    w_path = references_dir / "whiteout" / str(seed) / f"{dataset}_s{severity:g}.json"
                    if not w_path.exists():
                        continue
                    for row in read_json(w_path)["rows"]:
                        if row["method"] == source:
                            expected_rows[row["sample_id"]] = row

                new_rows = {row["sample_id"]: row for row in read_json(new_path)["rows"]}
                assert set(expected_rows) == set(new_rows), f"sample_id集合不一致: {new_path}"

                file_mismatch = 0
                for sample_id, ref_row in expected_rows.items():
                    new_row = new_rows[sample_id]
                    checked += 1
                    e_ok = _ellipses_close(ref_row["ellipse"], new_row["ellipse"])
                    iou_ok = abs(float(ref_row["ellipse_iou"]) - float(new_row["ellipse_iou"])) < 1e-9
                    suc_ok = bool(ref_row["success"]) == bool(new_row["success"])
                    hash_ok = (
                        True
                        if "image_sha256" not in ref_row
                        else ref_row["image_sha256"] == new_row["image_sha256"]
                    )
                    if not e_ok:
                        ellipse_mismatches += 1
                    if not iou_ok:
                        iou_mismatches += 1
                    if not suc_ok:
                        success_mismatches += 1
                    if not hash_ok:
                        image_hash_mismatches += 1
                    if not (e_ok and iou_ok and suc_ok and hash_ok):
                        file_mismatch += 1

                details.append(
                    {
                        "method": method,
                        "seed": seed,
                        "dataset": dataset,
                        "n": len(expected_rows),
                        "mismatches": file_mismatch,
                    }
                )

    verified = (
        checked > 0
        and ellipse_mismatches == 0
        and iou_mismatches == 0
        and success_mismatches == 0
        and image_hash_mismatches == 0
    )
    summary = {
        "checked_predictions": checked,
        "ellipse_mismatches": ellipse_mismatches,
        "iou_mismatches": iou_mismatches,
        "success_mismatches": success_mismatches,
        "image_hash_mismatches": image_hash_mismatches,
        "verified_exact": verified,
        "details": details,
    }
    write_json(out_root / "reference_verification.json", summary)
    return summary
