"""可視内周境界教師とTiny U-Net16の学習ループ。"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core.io import resolve_path, sha256_file, write_json
from core.model import TinyUNet, count_parameters, resolve_device
from training.dataset import MultiViewBoundaryDataset


def boundary_target(masks: torch.Tensor) -> torch.Tensor:
    """穴領域マスクから約2px幅の内周境界を作り、既知の矩形遮蔽部を除く。"""
    region = masks[:, 0:1]
    rim = F.max_pool2d(region, 3, 1, 1) + F.max_pool2d(-region, 3, 1, 1)
    return rim.clamp(0.0, 1.0) * masks[:, 1:2]


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """バッチ平均のSoft Dice損失を計算する。"""
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = (probability + target).sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def boundary_loss(logits: torch.Tensor, masks: torch.Tensor, bce: nn.Module) -> torch.Tensor:
    """正例重み付きBCE＋Diceによる可視境界損失。"""
    target = boundary_target(masks)
    return bce(logits, target) + dice_loss(logits, target)


def resolve_preset_views(
    config: dict[str, Any],
    base_dir: Path,
    source_name: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """学習元設定と使用する各ビューのディレクトリを解決する。"""
    presets = config["training"]["presets"]
    if source_name not in presets:
        raise ValueError(f"未定義の学習元です: {source_name}（候補: {', '.join(presets)}）")
    preset = presets[source_name]
    source = preset["source"]
    data_root = resolve_path(base_dir, config["paths"]["data_dir"])
    view_dirs = {
        name: (data_root / rel_path).resolve()
        for name, rel_path in config["datasets"][source]["views"].items()
    }
    return preset, view_dirs


def train_model(
    config: dict[str, Any],
    base_dir: Path,
    *,
    source_name: str,
    seed: int,
    output_dir: Path,
    epochs: int | None = None,
    device_name: str = "auto",
    limit_samples: int | None = None,
    overwrite: bool = False,
    view_dirs_override: dict[str, Path] | None = None,
    view_slots_override: list[str] | None = None,
    use_aperture_occlusion: bool = True,
) -> dict[str, Any]:
    """指定学習元とseedでTiny U-Net16を学習し、最小validation loss重みを保存する。"""
    if view_dirs_override is not None and view_slots_override is not None:
        source = source_name
        view_dirs = {k: Path(v).resolve() for k, v in view_dirs_override.items()}
        view_slots = list(view_slots_override)
    else:
        preset, view_dirs = resolve_preset_views(config, base_dir, source_name)
        source = preset["source"]
        view_slots = list(preset["view_slots"])
    train_cfg = config["training"]
    total_epochs = int(train_cfg["epochs"] if epochs is None else epochs)
    input_size = int(config["input_size"])

    run_dir = Path(output_dir).resolve()
    checkpoint_path = run_dir / "best.pt"
    complete_path = run_dir / "complete.json"
    history_path = run_dir / "training_history.json"
    provenance_path = run_dir / "provenance.json"

    if complete_path.exists() and not overwrite:
        raise FileExistsError(
            f"完了済みの学習フォルダが既に存在します: {run_dir}（上書きするには --overwrite を指定）"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)

    train_whiteout_max = float(train_cfg.get("whiteout_maximum_strength", 1.0))
    train_dataset = MultiViewBoundaryDataset(
        view_dirs=view_dirs,
        view_slots=view_slots,
        split="train",
        input_size=input_size,
        whiteout_maximum_strength=train_whiteout_max,
        cache=True,
        seed=seed,
        limit=limit_samples,
        use_aperture_occlusion=use_aperture_occlusion,
    )
    val_dataset = MultiViewBoundaryDataset(
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

    loader_kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = TinyUNet(int(train_cfg["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(train_cfg["positive_weight"])], device=device)
    )

    provenance = {
        "training_source": source,
        "target_data_used_for_training": source == "1194m",
        "seed": int(seed),
        "epochs": total_epochs,
        "input_size": input_size,
        "base_channels": int(train_cfg["base_channels"]),
        "parameter_count": count_parameters(model),
        "batch_size": int(train_cfg["batch_size"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "positive_weight": float(train_cfg["positive_weight"]),
        "view_slots": list(view_slots),
        "use_aperture_occlusion": bool(use_aperture_occlusion),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    write_json(provenance_path, provenance)

    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    print(
        f"[TRAIN] source={source} seed={seed} epochs={total_epochs} device={provenance['device']}",
        flush=True,
    )

    for epoch in range(1, total_epochs + 1):
        started = time.monotonic()
        model.train()
        train_total = 0.0
        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = boundary_loss(model(images), masks, bce)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(images)

        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for images, masks, _ in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                val_total += float(boundary_loss(model(images), masks, bce)) * len(images)

        row = {
            "epoch": epoch,
            "train_loss": train_total / len(train_dataset),
            "validation_loss": val_total / len(val_dataset),
            "elapsed_seconds": time.monotonic() - started,
        }
        if not (math.isfinite(row["train_loss"]) and math.isfinite(row["validation_loss"])):
            raise ValueError(f"損失が非有限値になりました: {row}")
        history.append(row)

        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_channels": int(train_cfg["base_channels"]),
                    "input_size": input_size,
                    "epoch": epoch,
                    "validation_loss": best_loss,
                    "seed": int(seed),
                    "training_source": source,
                    "provenance": provenance,
                },
                checkpoint_path,
            )
        write_json(history_path, history)
        print(
            f"  epoch {epoch:02d}/{total_epochs:02d} "
            f"train={row['train_loss']:.5f} val={row['validation_loss']:.5f} "
            f"({row['elapsed_seconds']:.1f}s)",
            flush=True,
        )

    summary = {
        "training_source": source,
        "seed": int(seed),
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    write_json(complete_path, summary)
    return summary
