from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from paflab.dataset import StressDataset
from paflab.model import TinyUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAF全リング確率マップCNNを学習する")
    parser.add_argument("--config", default="config/research_experiment.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--cache-data", action="store_true")
    return parser.parse_args()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = torch.sum(probability * target, dim=(1, 2, 3))
    denominator = torch.sum(probability + target, dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def evaluate_loss(model, loader, bce, dice_weight, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            batch_dice_loss = dice_loss(logits, masks)
            loss = bce(logits, masks) + dice_weight * batch_dice_loss
            batch_size = images.shape[0]
            total_loss += float(loss) * batch_size
            total_dice += float(1.0 - batch_dice_loss) * batch_size
            total_samples += batch_size
    return total_loss / total_samples, total_dice / total_samples


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = project_path(config["stress_dataset_dir"])
    artifacts_dir = project_path(args.artifacts_dir or config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = StressDataset(
        dataset_dir,
        split="train",
        input_size=config["input_size"],
        cache_in_memory=args.cache_data,
    )
    validation_dataset = StressDataset(
        dataset_dir,
        split="validation",
        input_size=config["input_size"],
        cache_in_memory=args.cache_data,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    base_channels = int(
        args.base_channels if args.base_channels is not None else config["model"]["base_channels"]
    )
    model = TinyUNet(base_channels).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    positive_weight = torch.tensor(
        [float(config["training"]["positive_weight"])], device=device
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(args.epochs or config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(config["training"]["learning_rate"]) * 0.05
    )
    dice_weight = float(config["training"]["dice_weight"])
    history = []
    best_validation_loss = float("inf")
    checkpoint_path = artifacts_dir / "cnn_best.pt"

    print(f"device={device} train={len(train_dataset)} validation={len(validation_dataset)}")
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_samples = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_weight * dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * images.shape[0]
            running_samples += images.shape[0]

        train_loss = running_loss / running_samples
        validation_loss, validation_dice = evaluate_loss(
            model, validation_loader, bce, dice_weight, device
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_soft_dice": validation_dice,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_channels": base_channels,
                    "parameter_count": parameter_count,
                    "seed": seed,
                    "input_size": int(config["input_size"]),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "config": config,
                },
                checkpoint_path,
            )
        scheduler.step()

    with (artifacts_dir / "training_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (artifacts_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    metadata = {
        "base_channels": base_channels,
        "parameter_count": parameter_count,
        "seed": seed,
        "epochs": epochs,
        "best_validation_loss": best_validation_loss,
        "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
        "total_training_seconds": sum(row["epoch_seconds"] for row in history),
        "checkpoint": str(checkpoint_path),
    }
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={checkpoint_path} best_validation_loss={best_validation_loss:.6f}")


if __name__ == "__main__":
    main()
