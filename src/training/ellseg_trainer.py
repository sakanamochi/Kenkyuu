"""EllSeg型マルチタスクU-Net（可視内周境界＋Amodal完全穴マスク）の学習・評価ループ。

既存の MultiViewBoundaryDataset（同一データ・同一スロット・同一開口幅オンライン遮蔽）を
そのまま使用して公平に比較する。
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from core.effects import image_to_tensor, load_sample_and_truth
from core.ellseg_model import (
    TinyEllSegUNet,
    ellseg_loss,
    load_ellseg_checkpoint,
    predict_ellseg_ellipse,
)
from core.geometry import ellipse_to_list, evaluate_ellipses
from core.io import (
    load_manifest_samples,
    read_json,
    resolve_path,
    sha256_bytes,
    sha256_file,
    write_json,
)
from core.model import count_parameters, resolve_device
from training.dataset import MultiViewBoundaryDataset
from training.trainer import resolve_preset_views


def train_ellseg_model(
    config: dict[str, Any],
    base_dir: Path,
    *,
    source_name: str = "diversified",
    seed: int = 20260907,
    output_dir: Path,
    epochs: int | None = None,
    device_name: str = "auto",
    limit_samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """境界CNNと同一条件でEllSeg型2chモデルを学習し、最小validation loss重みを保存する。"""
    preset, view_dirs = resolve_preset_views(config, base_dir, source_name)
    view_slots = list(preset["view_slots"])
    train_cfg = config["training"]
    total_epochs = int(train_cfg["epochs"] if epochs is None else epochs)
    input_size = int(config["input_size"])

    run_dir = Path(output_dir).resolve()
    checkpoint_path = run_dir / "best.pt"
    complete_path = run_dir / "complete.json"
    history_path = run_dir / "training_history.json"

    if complete_path.exists() and not overwrite:
        raise FileExistsError(f"完了済みのEllSeg学習フォルダが存在します: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)

    train_whiteout_max = float(train_cfg.get("whiteout_maximum_strength", 1.0))
    train_ds = MultiViewBoundaryDataset(
        view_dirs=view_dirs,
        view_slots=view_slots,
        split="train",
        input_size=input_size,
        whiteout_maximum_strength=train_whiteout_max,
        cache=True,
        seed=seed,
        limit=limit_samples,
        use_aperture_occlusion=True,
    )
    val_ds = MultiViewBoundaryDataset(
        view_dirs=view_dirs,
        view_slots=["base"],
        split="validation",
        input_size=input_size,
        whiteout_maximum_strength=train_whiteout_max,
        cache=True,
        seed=seed,
        limit=limit_samples,
        use_aperture_occlusion=False,
    )

    batch_size = int(train_cfg["batch_size"])
    generator = torch.Generator().manual_seed(seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    base_channels = int(train_cfg["base_channels"])
    model = TinyEllSegUNet(base_channels=base_channels).to(device)
    param_count = count_parameters(model)
    boundary_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(train_cfg["positive_weight"])], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(
        f"[TRAIN-ELLSEG] source={source_name} seed={seed} epochs={total_epochs} "
        f"params={param_count} device={device_label}",
        flush=True,
    )

    for epoch in range(1, total_epochs + 1):
        ep_start = time.time()
        model.train()
        train_sum = 0.0
        train_n = 0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, _ = ellseg_loss(logits, masks, boundary_bce)
            loss.backward()
            optimizer.step()
            bs = images.size(0)
            train_sum += float(loss.item()) * bs
            train_n += bs

        model.eval()
        val_sum = 0.0
        val_rim_sum = 0.0
        val_hole_sum = 0.0
        val_n = 0
        with torch.inference_mode():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss, parts = ellseg_loss(logits, masks, boundary_bce)
                bs = images.size(0)
                val_sum += float(loss.item()) * bs
                val_rim_sum += parts["rim"] * bs
                val_hole_sum += parts["hole"] * bs
                val_n += bs

        train_loss = train_sum / max(1, train_n)
        val_loss = val_sum / max(1, val_n)
        val_rim = val_rim_sum / max(1, val_n)
        val_hole = val_hole_sum / max(1, val_n)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rim_loss": val_rim,
                "val_hole_loss": val_hole,
            }
        )
        print(
            f"  epoch {epoch:02d}/{total_epochs} train={train_loss:.5f} val={val_loss:.5f} "
            f"(rim={val_rim:.4f}, hole={val_hole:.4f}) ({time.time() - ep_start:.1f}s)",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "architecture": "TinyEllSegUNet",
                    "base_channels": base_channels,
                    "parameter_count": param_count,
                    "source": source_name,
                    "seed": seed,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_state": model.state_dict(),
                },
                checkpoint_path,
            )

    write_json(history_path, {"history": history})
    summary = {
        "architecture": "TinyEllSegUNet",
        "parameter_count": param_count,
        "source": source_name,
        "seed": seed,
        "epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "elapsed_seconds": time.time() - start_time,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    write_json(complete_path, summary)
    return summary


def evaluate_ellseg_model(
    config: dict[str, Any],
    base_dir: Path,
    *,
    checkpoint_path: Path,
    datasets: list[str],
    output_dir: Path,
    modes: tuple[str, ...] = ("fused", "hole_only", "boundary_only"),
    seed: int = 20260907,
    device_name: str = "auto",
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    """EllSeg型モデルの1回のフォワードから複数推論モードを一括統一評価する。"""
    device = resolve_device(device_name)
    model, ckpt = load_ellseg_checkpoint(
        checkpoint_path,
        device,
        default_base_channels=int(config["training"]["base_channels"]),
    )
    input_size = int(config["input_size"])
    success_iou = float(config["success_iou"])
    threshold = float(config["probability_threshold"])
    whiteout_max = float(config["evaluation"]["whiteout_maximum_strength"])
    reject_uniform = bool(config["reject_uniform_images"])
    opencv_seed = int(config["opencv_seed_per_image"])
    ransac_cfg = config["ransac_cnn"]

    data_root = resolve_path(base_dir, config["paths"]["data_dir"])
    out_root = Path(output_dir).resolve()
    all_results: dict[str, dict[str, Any]] = {m: {} for m in modes}

    for dataset_name in datasets:
        dataset_dir = (data_root / config["datasets"][dataset_name]["base"]).resolve()
        samples = load_manifest_samples(dataset_dir, "test")

        pending_modes = []
        for mode in modes:
            dest = out_root / "predictions" / f"ellseg_{mode}_{seed}" / f"test_{dataset_name}.json"
            if dest.exists() and not overwrite:
                saved = read_json(dest)
                if len(saved.get("rows", [])) == len(samples):
                    print(f"[SKIP] 既存結果を使用: ellseg_{mode}_{seed}/{dest.name}", flush=True)
                    all_results[mode][dataset_name] = saved
                    continue
            pending_modes.append(mode)

        if not pending_modes:
            continue

        print(
            f"[EVAL-ELLSEG] dataset={dataset_name} modes={pending_modes} seed={seed} n={len(samples)}",
            flush=True,
        )
        rows_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in pending_modes}

        for index, sample in enumerate(samples, start=1):
            image, truth = load_sample_and_truth(
                dataset_dir,
                sample,
                input_size=input_size,
                whiteout_maximum_strength=whiteout_max,
            )
            img_hash = sha256_bytes(image.tobytes())
            is_uniform = reject_uniform and int(np.ptp(image)) == 0
            if not is_uniform:
                tensor = image_to_tensor(image).to(device)[None]
                with torch.inference_mode():
                    probs = torch.sigmoid(model(tensor))[0].cpu().numpy()
                prob_boundary = probs[0]
                prob_hole = probs[1]
            else:
                prob_boundary = None
                prob_hole = None

            for mode in pending_modes:
                if is_uniform:
                    ellipse = None
                else:
                    cv2.setRNGSeed(opencv_seed)
                    assert prob_boundary is not None and prob_hole is not None
                    ellipse = predict_ellseg_ellipse(
                        prob_boundary,
                        prob_hole,
                        ransac_cfg,
                        mode=mode,
                        threshold=threshold,
                    )
                iou = (
                    evaluate_ellipses(ellipse, truth, (input_size, input_size))["ellipse_iou"]
                    if ellipse is not None
                    else 0.0
                )
                rows_by_mode[mode].append(
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
                        "image_sha256": img_hash,
                    }
                )
            if index % 400 == 0:
                print(f"  progress {index}/{len(samples)}", flush=True)

        for mode in pending_modes:
            rows = rows_by_mode[mode]
            successes = sum(1 for row in rows if row["success"])
            payload = {
                "dataset": dataset_name,
                "method": f"ellseg_{mode}",
                "architecture": "TinyEllSegUNet",
                "parameter_count": int(ckpt.get("parameter_count", count_parameters(model))),
                "seed": seed,
                "whiteout_maximum_strength": whiteout_max,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "n": len(rows),
                "successes": successes,
                "success_percent": 100.0 * successes / len(rows) if rows else 0.0,
                "mean_iou": float(np.mean([row["ellipse_iou"] for row in rows])) if rows else 0.0,
                "rows": rows,
            }
            dest = out_root / "predictions" / f"ellseg_{mode}_{seed}" / f"test_{dataset_name}.json"
            write_json(dest, payload)
            print(
                f"  -> [{mode:13s}] {successes}/{len(rows)} "
                f"({payload['success_percent']:.2f}%, mean_iou={payload['mean_iou']:.4f})",
                flush=True,
            )
            all_results[mode][dataset_name] = payload

    return all_results
