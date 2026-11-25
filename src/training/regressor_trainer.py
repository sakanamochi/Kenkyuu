"""直接楕円パラメータ回帰CNNのデータセット・学習ループ・統一評価。

境界CNN（MultiViewBoundaryDataset・60epoch・batch 16・AdamW・開口幅オンライン遮蔽）と
完全に同一のデータ分割・スロット抽選・劣化条件で学習し、公平に比較する。
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.effects import image_to_tensor, load_sample_and_truth, render_sample_image
from core.ellipse_regressor import (
    TinyEllipseRegressor,
    ellipse_to_normalized_vector,
    load_regressor_checkpoint,
    normalized_vector_to_ellipse,
    prob_iou_loss,
)
from core.geometry import Ellipse, ellipse_to_list, evaluate_ellipses, label_to_ellipse, scale_ellipse
from core.io import (
    load_manifest_samples,
    read_image,
    read_json,
    resolve_path,
    sha256_bytes,
    sha256_file,
    write_json,
)
from core.model import count_parameters, resolve_device
from training.dataset import aperture_cut_x
from training.trainer import resolve_preset_views


class EllipseViewDataset(Dataset):
    """単一ビューの画像と256px換算の正規化正解楕円ベクトルを返すデータセット。"""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        input_size: int = 256,
        *,
        whiteout_maximum_strength: float = 1.0,
        cache: bool = False,
        limit: int | None = None,
    ) -> None:
        self.root = Path(dataset_dir)
        self.split = split
        self.input_size = int(input_size)
        self.whiteout_maximum_strength = float(whiteout_maximum_strength)
        samples = load_manifest_samples(self.root, split)
        self.samples = samples[:limit] if limit is not None else samples
        self._packed: dict[int, tuple[np.ndarray, np.ndarray]] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.samples)

    def _build_item(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[index]
        shared = render_sample_image(
            self.root,
            sample,
            input_size=self.input_size,
            whiteout_maximum_strength=self.whiteout_maximum_strength,
        )
        label = read_json(self.root / sample["label"])
        ellipse = label_to_ellipse(label)
        scale = self.input_size / float(label["image_width"])
        vec = np.asarray(ellipse_to_normalized_vector(scale_ellipse(ellipse, scale), self.input_size), dtype=np.float32)
        gray = shared[:, :, 0].copy()
        return gray, vec

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self._packed is None:
            gray, vec = self._build_item(index)
        else:
            if index not in self._packed:
                self._packed[index] = self._build_item(index)
            gray, vec = self._packed[index]

        image_tensor = (torch.from_numpy(gray).float() / 127.5 - 1.0).unsqueeze(0).expand(3, -1, -1)
        return image_tensor, torch.from_numpy(vec.copy()), index


class MultiViewEllipseDataset(Dataset):
    """境界CNNと同一のビュー抽選・開口幅基準オンライン遮蔽で直接回帰用ペアを返す。"""

    def __init__(
        self,
        view_dirs: dict[str, Path],
        view_slots: list[str],
        split: str,
        *,
        input_size: int = 256,
        whiteout_maximum_strength: float = 1.0,
        cache: bool = False,
        seed: int | None = None,
        limit: int | None = None,
        use_aperture_occlusion: bool = True,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("splitはtrainまたはvalidationを指定してください。")
        self.split = split
        self.input_size = int(input_size)
        self.use_aperture_occlusion = bool(use_aperture_occlusion)

        active_slots = list(view_slots) if split == "train" else ["base"]
        loaded_views: dict[str, EllipseViewDataset] = {}
        for name in active_slots:
            if name not in loaded_views:
                loaded_views[name] = EllipseViewDataset(
                    view_dirs[name],
                    split,
                    input_size=self.input_size,
                    whiteout_maximum_strength=whiteout_maximum_strength,
                    cache=cache,
                    limit=limit,
                )

        self.slot_names = active_slots
        self.views = [loaded_views[name] for name in active_slots]
        self.samples = self.views[0].samples

        base_seed = int(torch.initial_seed() if seed is None else seed)
        self.view_rng = np.random.default_rng(base_seed + 5200)
        self.mask_rng = np.random.default_rng(base_seed + 6200)
        self._raw_cache: dict[tuple[str, str], tuple[np.ndarray, Ellipse]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        choice = int(self.view_rng.integers(0, len(self.views))) if len(self.views) > 1 else 0
        view = self.views[choice]
        slot_name = self.slot_names[choice]
        image, target_vec, _ = view[index]
        sample = view.samples[index]

        if (
            self.use_aperture_occlusion
            and self.split == "train"
            and sample["conditions"]["degradation"] == "black_rectangle"
        ):
            cache_key = (slot_name, sample["image"])
            if cache_key not in self._raw_cache:
                raw = read_image(view.root / sample["image"])
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                ellipse = label_to_ellipse(read_json(view.root / sample["label"]))
                self._raw_cache[cache_key] = (gray, ellipse)
            gray, ellipse = self._raw_cache[cache_key]
            left = bool(self.mask_rng.integers(0, 2) == 0)
            draw = float(self.mask_rng.random())
            cut = aperture_cut_x(ellipse, gray.shape[1], draw)
            occluded = gray.copy()
            if left:
                occluded[:, :cut] = 0
            else:
                occluded[:, cut:] = 0
            resized = cv2.resize(occluded, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            image = (torch.from_numpy(resized).float() / 127.5 - 1.0).unsqueeze(0).expand(3, -1, -1)

        # 全画素均一（全遮蔽等で視覚手がかりゼロ）のサンプルは回帰損失から除外する。
        has_signal = float((image[0].max() - image[0].min()).item() > 1e-4)
        return image, target_vec, torch.tensor(has_signal, dtype=torch.float32)


def train_ellipse_regressor(
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
    """境界CNNと同一条件で直接楕円パラメータ回帰CNNを学習し、最小validation loss重みを保存する。"""
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
        raise FileExistsError(f"完了済みの直接回帰フォルダが存在します: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)

    train_whiteout_max = float(train_cfg.get("whiteout_maximum_strength", 1.0))
    train_ds = MultiViewEllipseDataset(
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
    val_ds = MultiViewEllipseDataset(
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
    model = TinyEllipseRegressor(base_channels=base_channels).to(device)
    param_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(
        f"[TRAIN-DIRECT] source={source_name} seed={seed} epochs={total_epochs} "
        f"params={param_count} device={device_label}",
        flush=True,
    )

    for epoch in range(1, total_epochs + 1):
        ep_start = time.time()
        model.train()
        train_sum = 0.0
        train_n = 0
        for images, targets, valid in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            valid = valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = prob_iou_loss(preds, targets, valid_mask=valid)
            if float(valid.sum().item()) > 0.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                bs = int(valid.sum().item())
                train_sum += float(loss.item()) * bs
                train_n += bs
        scheduler.step()

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.inference_mode():
            for images, targets, valid in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                valid = valid.to(device)
                preds = model(images)
                loss = prob_iou_loss(preds, targets, valid_mask=valid)
                bs = int(valid.sum().item())
                val_sum += float(loss.item()) * bs
                val_n += bs

        train_loss = train_sum / max(1, train_n)
        val_loss = val_sum / max(1, val_n)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"  epoch {epoch:02d}/{total_epochs} train={train_loss:.5f} val={val_loss:.5f} "
            f"({time.time() - ep_start:.1f}s)",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "architecture": "TinyEllipseRegressor",
                    "base_channels": base_channels,
                    "max_axis_ratio": model.max_axis_ratio,
                    "min_axis_ratio": model.min_axis_ratio,
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
        "architecture": "TinyEllipseRegressor",
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


def evaluate_ellipse_regressor(
    config: dict[str, Any],
    base_dir: Path,
    *,
    checkpoint_path: Path,
    datasets: list[str],
    output_dir: Path,
    seed: int = 20260907,
    device_name: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """学習済み直接楕円回帰モデルを全13条件のテスト集合で統一評価する。"""
    device = resolve_device(device_name)
    model, ckpt = load_regressor_checkpoint(
        checkpoint_path,
        device,
        default_base_channels=int(config["training"]["base_channels"]),
    )
    input_size = int(config["input_size"])
    success_iou = float(config["success_iou"])
    whiteout_max = float(config["evaluation"]["whiteout_maximum_strength"])
    reject_uniform = bool(config["reject_uniform_images"])
    data_root = resolve_path(base_dir, config["paths"]["data_dir"])
    out_root = Path(output_dir).resolve()

    results: dict[str, Any] = {}
    for dataset_name in datasets:
        from core.io import load_evaluation_test_samples

        dataset_dir, samples = load_evaluation_test_samples(config, data_root, dataset_name)
        dest = out_root / "predictions" / f"cnn_direct_{seed}" / f"test_{dataset_name}.json"

        if dest.exists() and not overwrite:
            saved = read_json(dest)
            if len(saved.get("rows", [])) == len(samples):
                print(f"[SKIP] 既存結果を使用: {dest.name}", flush=True)
                results[dataset_name] = saved
                continue

        print(
            f"[EVAL-DIRECT] dataset={dataset_name} seed={seed} n={len(samples)} whiteout_max={whiteout_max:g}",
            flush=True,
        )
        rows: list[dict[str, Any]] = []
        for index, sample in enumerate(samples, start=1):
            image, truth = load_sample_and_truth(
                dataset_dir,
                sample,
                input_size=input_size,
                whiteout_maximum_strength=whiteout_max,
            )
            if reject_uniform and int(np.ptp(image)) == 0:
                ellipse = None
            else:
                tensor = image_to_tensor(image).to(device)[None]
                with torch.inference_mode():
                    vec = model(tensor)[0]
                ellipse = normalized_vector_to_ellipse(vec, input_size)

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
            "method": "cnn_direct",
            "architecture": "TinyEllipseRegressor",
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
        write_json(dest, payload)
        print(
            f"  -> {successes}/{len(rows)} ({payload['success_percent']:.2f}%, mean_iou={payload['mean_iou']:.4f})",
            flush=True,
        )
        results[dataset_name] = payload
    return results
