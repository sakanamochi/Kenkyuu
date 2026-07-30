"""リング尤度CNNを学習する。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from paf_ring_detection.data import RingDataset, write_json
from paf_ring_detection.methods.cnn import TinyUNet


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = (probability + target).sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def _validation_loss(model, loader, loss_function, dice_weight, device) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for images, masks, _ in loader:
            logits = model(images.to(device))
            masks = masks.to(device)
            loss = loss_function(logits, masks) + dice_weight * _dice_loss(
                logits, masks
            )
            total += float(loss) * len(images)
    return total / len(loader.dataset)


def train(config: dict, epochs: int | None = None) -> Path:
    """validation lossが最小のCNNだけを保存する。"""
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = config["cnn"]
    dataset_dir = Path(config["paths"]["training_dataset"])
    train_data = RingDataset(dataset_dir, "train", config["input_size"])
    validation_data = RingDataset(dataset_dir, "validation", config["input_size"])
    loader_options = {
        "batch_size": settings["batch_size"],
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    validation_loader = DataLoader(
        validation_data, shuffle=False, **loader_options
    )

    model = TinyUNet(settings["base_channels"]).to(device)
    positive_weight = torch.tensor([settings["positive_weight"]], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    checkpoint_path = Path(config["paths"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    for epoch in range(1, int(epochs or settings["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + settings["dice_weight"] * _dice_loss(
                logits, masks
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(images)

        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_data),
            "validation_loss": _validation_loss(
                model,
                validation_loader,
                bce,
                settings["dice_weight"],
                device,
            ),
        }
        history.append(row)
        print(row)

        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_channels": settings["base_channels"],
                    "input_size": config["input_size"],
                    "epoch": epoch,
                    "validation_loss": best_loss,
                },
                checkpoint_path,
            )

    write_json(checkpoint_path.parent / "training_history.json", history)
    return checkpoint_path
