"""オンライン背景ランダム化（地球背景耐性）＋外周シルエット負例重み学習・評価モジュール。

既存の黒背景CGから画像境界に連結する外側背景のみを抽出し、
  1) 穴の内部空間を汚さずに背景へ均一明背景・雲風ノイズ・地球リム風円弧をオンライン合成する
  2) 合成前の物体シルエット外周エッジへ強い負例重みを付与して外周誤反応を学習段階で抑える
既存モデル（TinyUNet16: 482,737パラメータ）と完全同一構造で比較する。
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from core.effects import apply_effect, image_to_tensor, load_sample_and_truth
from core.geometry import Ellipse, ellipse_to_list, evaluate_ellipses, label_to_ellipse, region_mask, scale_ellipse
from core.io import (
    load_manifest_samples,
    read_image,
    read_json,
    resolve_path,
    sha256_bytes,
    sha256_file,
    write_json,
)
from core.model import TinyUNet, count_parameters, load_checkpoint, resolve_device
from core.ransac import boundary_ellipse
from training.dataset import aperture_cut_x
from training.trainer import dice_loss, resolve_preset_views


def extract_outer_background_mask(gray: np.ndarray, threshold: int = 2) -> np.ndarray:
    """画像外周4辺に連結した黒背景領域のみをboolマスクで返す（穴内部の黒は含めない）。"""
    h, w = gray.shape[:2]
    dark = (gray <= int(threshold)).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(dark, connectivity=4)
    if num_labels <= 1:
        return np.zeros((h, w), dtype=bool)
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    border_labels.discard(0)
    if not border_labels:
        return np.zeros((h, w), dtype=bool)
    return np.isin(labels, list(border_labels))


def extract_outer_silhouette_edge(outer_bg_mask: np.ndarray, band_px: int = 3) -> np.ndarray:
    """外側背景マスクの反転（PAF本体＋穴の閉シルエット）から外周エッジ帯（0/1 float32）を作る。"""
    silhouette = (~outer_bg_mask).astype(np.uint8)
    k = max(3, int(band_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(silhouette, kernel)
    eroded = cv2.erode(silhouette, kernel)
    return ((dilated > 0) & (eroded == 0)).astype(np.float32)


def _cloud_noise_layer(size: int, rng: np.random.Generator) -> np.ndarray:
    """地球の雲・大気模様を模擬したマルチスケール滑らかノイズ（0..1 float32）を生成する。"""
    acc = np.zeros((size, size), dtype=np.float32)
    weight_sum = 0.0
    for grid_sz, w in ((6, 0.50), (14, 0.30), (32, 0.20)):
        small = rng.random((grid_sz, grid_sz), dtype=np.float32)
        up = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        acc += w * up
        weight_sum += w
    acc /= weight_sum
    acc = (acc - acc.min()) / max(float(acc.max() - acc.min()), 1e-6)
    return np.clip(acc, 0.0, 1.0)


def _earth_limb_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """画面外に中心を持つ巨大な地球の縁（円弧＋大気ぼかし、0..1 float32）を生成する。"""
    radius = float(rng.uniform(0.95, 2.20)) * size
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    # 画面内に縁が横切るように中心距離を決める
    offset = radius + float(rng.uniform(-0.35, 0.25)) * size
    cx = 0.5 * size + offset * math.cos(angle)
    cy = 0.5 * size + offset * math.sin(angle)
    ys, xs = np.ogrid[:size, :size]
    dist = np.hypot(xs - cx, ys - cy)
    blur_width = float(rng.uniform(2.5, 10.0))
    limb = np.clip((radius - dist) / blur_width + 0.5, 0.0, 1.0).astype(np.float32)
    return limb


def synthesize_background(size: int, rng: np.random.Generator, mode: str | None = None) -> np.ndarray:
    """宇宙・地球背景を模擬した2Dグレースケール背景画像（uint8）を生成する。"""
    chosen = mode or str(rng.choice(["uniform", "clouds", "earth_limb", "earth_clouds"]))
    if chosen == "uniform":
        val = int(rng.integers(25, 225))
        return np.full((size, size), val, dtype=np.uint8)

    if chosen == "clouds":
        noise = _cloud_noise_layer(size, rng)
        lo = float(rng.uniform(15.0, 110.0))
        hi = float(rng.uniform(max(lo + 40.0, 120.0), 245.0))
        return np.clip(np.rint(lo + (hi - lo) * noise), 0, 255).astype(np.uint8)

    if chosen == "earth_limb":
        limb = _earth_limb_mask(size, rng)
        bright = float(rng.uniform(90.0, 240.0))
        return np.clip(np.rint(bright * limb), 0, 255).astype(np.uint8)

    # earth_clouds: 巨大な地球リムの内部に雲模様とグラデーションを乗せる
    limb = _earth_limb_mask(size, rng)
    noise = _cloud_noise_layer(size, rng)
    lo = float(rng.uniform(45.0, 130.0))
    hi = float(rng.uniform(max(lo + 50.0, 160.0), 250.0))
    earth_tex = lo + (hi - lo) * noise
    return np.clip(np.rint(earth_tex * limb), 0, 255).astype(np.uint8)


def composite_background_on_gray(
    gray: np.ndarray,
    outer_bg_mask: np.ndarray,
    bg_gray: np.ndarray,
) -> np.ndarray:
    """外側背景領域のみを背景画像で置き換え、PAF本体と穴内部をそのまま保持する。"""
    out = gray.copy()
    out[outer_bg_mask] = bg_gray[outer_bg_mask]
    return out


class BgSuppressionDataset(Dataset):
    """オンライン背景ランダム化と外周シルエット負例マスクを返す学習用データセット。"""

    def __init__(
        self,
        view_dirs: dict[str, Path],
        view_slots: list[str],
        split: str,
        *,
        input_size: int = 256,
        whiteout_maximum_strength: float = 1.0,
        bg_random_prob: float = 0.25,
        seed: int = 20260907,
        limit: int | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("splitはtrainまたはvalidationを指定してください。")
        self.split = split
        self.input_size = int(input_size)
        self.whiteout_maximum_strength = float(whiteout_maximum_strength)
        self.bg_random_prob = float(bg_random_prob) if split == "train" else 0.0

        active_slots = list(view_slots) if split == "train" else ["base"]
        self.slot_names = active_slots
        self.view_roots = {name: Path(view_dirs[name]).resolve() for name in set(active_slots)}
        self.samples_by_view = {
            name: (load_manifest_samples(root, split)[:limit] if limit is not None else load_manifest_samples(root, split))
            for name, root in self.view_roots.items()
        }
        self.samples = self.samples_by_view[active_slots[0]]

        self.view_rng = np.random.default_rng(seed + 5200)
        self.mask_rng = np.random.default_rng(seed + 6200)
        self.bg_rng = np.random.default_rng(seed + 7200)
        # 原画像(256px)キャッシュと黒背景劣化済みキャッシュを持ち、毎epochのI/Oと480px演算を削減する
        self._raw_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, Ellipse, np.ndarray]] = {}
        self._degraded_black_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_raw_assets(
        self, slot_name: str, sample: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Ellipse, np.ndarray]:
        """256px原画像グレー・外側背景マスク・外周エッジ・256px楕円・穴マスクをキャッシュ取得。"""
        key = (slot_name, sample["image"])
        if key not in self._raw_cache:
            root = self.view_roots[slot_name]
            raw = read_image(root / sample["image"])
            gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            outer_bg_raw = extract_outer_background_mask(gray_raw)
            gray_256 = cv2.resize(gray_raw, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            outer_bg_256 = (
                cv2.resize(
                    outer_bg_raw.astype(np.uint8), (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
            outer_edge_256 = extract_outer_silhouette_edge(outer_bg_256, band_px=3)
            label = read_json(root / sample["label"])
            scale = self.input_size / float(label["image_width"])
            ellipse_256 = scale_ellipse(label_to_ellipse(label), scale)
            hole_256 = region_mask(self.input_size, ellipse_256)
            self._raw_cache[key] = (gray_256, outer_bg_256, outer_edge_256, ellipse_256, hole_256)
        return self._raw_cache[key]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        choice = int(self.view_rng.integers(0, len(self.slot_names))) if len(self.slot_names) > 1 else 0
        slot_name = self.slot_names[choice]
        sample = self.samples_by_view[slot_name][index]
        gray_256, outer_bg_256, outer_edge_256, ellipse_256, hole_256 = self._get_raw_assets(slot_name, sample)

        use_random_bg = self.bg_random_prob > 0.0 and float(self.bg_rng.random()) < self.bg_random_prob
        deg = sample["conditions"]["degradation"]

        if not use_random_bg and not (self.split == "train" and deg == "black_rectangle"):
            cache_k = (slot_name, index)
            if cache_k not in self._degraded_black_cache:
                work_gray = gray_256.copy()
                vis_256 = np.ones((self.input_size, self.input_size), dtype=np.float32)
                effect = sample.get("augmentation")
                if effect and effect["name"] != "clean":
                    bgr = cv2.cvtColor(work_gray, cv2.COLOR_GRAY2BGR)
                    degraded = apply_effect(
                        bgr,
                        effect["name"],
                        float(effect["severity"]),
                        np.random.default_rng(int(effect["seed"])),
                        whiteout_maximum_strength=self.whiteout_maximum_strength,
                    )
                    work_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
                    if effect["name"] == "black_rectangle":
                        vis_bgr = apply_effect(
                            np.full((self.input_size, self.input_size, 3), 255, dtype=np.uint8),
                            effect["name"],
                            float(effect["severity"]),
                            np.random.default_rng(int(effect["seed"])),
                        )
                        vis_256 = vis_bgr[:, :, 0].astype(np.float32) / 255.0
                self._degraded_black_cache[cache_k] = (work_gray, vis_256)
            resized_gray, resized_vis = self._degraded_black_cache[cache_k]
        else:
            if use_random_bg:
                bg_layer = synthesize_background(self.input_size, self.bg_rng)
                work_gray = composite_background_on_gray(gray_256, outer_bg_256, bg_layer)
            else:
                work_gray = gray_256.copy()

            resized_vis = np.ones((self.input_size, self.input_size), dtype=np.float32)
            if self.split == "train" and deg == "black_rectangle":
                left = bool(self.mask_rng.integers(0, 2) == 0)
                draw = float(self.mask_rng.random())
                cut = aperture_cut_x(ellipse_256, self.input_size, draw)
                if left:
                    work_gray[:, :cut] = 0
                    resized_vis[:, :cut] = 0.0
                else:
                    work_gray[:, cut:] = 0
                    resized_vis[:, cut:] = 0.0
            else:
                effect = sample.get("augmentation")
                if effect and effect["name"] != "clean":
                    bgr = cv2.cvtColor(work_gray, cv2.COLOR_GRAY2BGR)
                    degraded = apply_effect(
                        bgr,
                        effect["name"],
                        float(effect["severity"]),
                        np.random.default_rng(int(effect["seed"])),
                        whiteout_maximum_strength=self.whiteout_maximum_strength,
                    )
                    work_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
            resized_gray = work_gray

        image_tensor = (torch.from_numpy(resized_gray).float() / 127.5 - 1.0).unsqueeze(0).expand(3, -1, -1)
        # masks: [0]=完全穴マスク, [1]=可視マスク, [2]=外周シルエットエッジ帯
        masks = np.stack([hole_256, resized_vis, outer_edge_256], axis=0).astype(np.float32)
        return image_tensor, torch.from_numpy(masks)


def outer_weighted_boundary_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    pos_weight: float = 12.0,
    outer_neg_weight: float = 2.5,
) -> torch.Tensor:
    """正例重み付きBCE（外周シルエットの負例に強い重みを付与）＋Soft Dice損失。"""
    region = masks[:, 0:1]
    visibility = masks[:, 1:2]
    outer_edge = masks[:, 2:3]

    rim = (F.max_pool2d(region, 3, 1, 1) + F.max_pool2d(-region, 3, 1, 1)).clamp(0.0, 1.0)
    target = rim * visibility

    # 正解内周の直近2pxは外周負例ペナルティ対象から除外する（薄肉・直円筒での競合を防ぐ）
    near_target = F.max_pool2d(rim, 5, 1, 2)
    hard_outer = outer_edge * (1.0 - near_target) * visibility

    pixel_weight = torch.ones_like(target)
    pixel_weight = pixel_weight + (float(outer_neg_weight) - 1.0) * hard_outer
    pixel_weight = torch.where(target > 0.5, torch.full_like(target, float(pos_weight)), pixel_weight)

    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (bce_map * pixel_weight).mean()
    return weighted_bce + dice_loss(logits, target)


def train_bg_suppression_model(
    config: dict[str, Any],
    base_dir: Path,
    *,
    source_name: str = "diversified",
    seed: int = 20260907,
    output_dir: Path,
    epochs: int | None = None,
    bg_random_prob: float = 0.25,
    outer_neg_weight: float = 2.5,
    device_name: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """背景ランダム化＋外周負例重みでTinyUNet(16)を学習する。"""
    preset, view_dirs = resolve_preset_views(config, base_dir, source_name)
    view_slots = list(preset["view_slots"])
    train_cfg = config["training"]
    total_epochs = int(train_cfg["epochs"] if epochs is None else epochs)
    input_size = int(config["input_size"])

    run_dir = Path(output_dir).resolve()
    ckpt_path = run_dir / "best.pt"
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not overwrite:
        raise FileExistsError(f"完了済みの学習フォルダが存在します: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)

    train_whiteout_max = float(train_cfg.get("whiteout_maximum_strength", 1.0))
    train_ds = BgSuppressionDataset(
        view_dirs=view_dirs,
        view_slots=view_slots,
        split="train",
        input_size=input_size,
        whiteout_maximum_strength=train_whiteout_max,
        bg_random_prob=bg_random_prob,
        seed=seed,
    )
    val_ds = BgSuppressionDataset(
        view_dirs=view_dirs,
        view_slots=["base"],
        split="validation",
        input_size=input_size,
        whiteout_maximum_strength=train_whiteout_max,
        bg_random_prob=0.0,
        seed=seed,
    )

    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 1000)
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    base_channels = int(train_cfg["base_channels"])
    model = TinyUNet(base_channels).to(device)
    param_count = count_parameters(model)
    pos_weight = float(train_cfg["positive_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    start = time.time()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(
        f"[TRAIN-BG-SUPP] source={source_name} seed={seed} epochs={total_epochs} "
        f"bg_prob={bg_random_prob} outer_w={outer_neg_weight} params={param_count} device={dev_name}",
        flush=True,
    )

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        tr_sum = 0.0
        tr_n = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = outer_weighted_boundary_loss(
                model(imgs), masks, pos_weight=pos_weight, outer_neg_weight=outer_neg_weight
            )
            loss.backward()
            optimizer.step()
            bs = imgs.size(0)
            tr_sum += float(loss.item()) * bs
            tr_n += bs

        model.eval()
        va_sum = 0.0
        va_n = 0
        with torch.inference_mode():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                loss = outer_weighted_boundary_loss(
                    model(imgs), masks, pos_weight=pos_weight, outer_neg_weight=outer_neg_weight
                )
                bs = imgs.size(0)
                va_sum += float(loss.item()) * bs
                va_n += bs

        tr_loss = tr_sum / max(1, tr_n)
        va_loss = va_sum / max(1, va_n)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"  epoch {epoch:02d}/{total_epochs} train={tr_loss:.5f} val={va_loss:.5f} ({time.time()-t0:.1f}s)", flush=True)

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            torch.save(
                {
                    "architecture": "TinyUNet",
                    "base_channels": base_channels,
                    "parameter_count": param_count,
                    "source": source_name,
                    "seed": seed,
                    "epoch": epoch,
                    "val_loss": va_loss,
                    "bg_random_prob": bg_random_prob,
                    "outer_neg_weight": outer_neg_weight,
                    "model_state": model.state_dict(),
                },
                ckpt_path,
            )

    write_json(run_dir / "training_history.json", {"history": history})
    summary = {
        "architecture": "TinyUNet",
        "parameter_count": param_count,
        "source": source_name,
        "seed": seed,
        "epochs": total_epochs,
        "bg_random_prob": bg_random_prob,
        "outer_neg_weight": outer_neg_weight,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "elapsed_seconds": time.time() - start,
        "checkpoint_sha256": sha256_file(ckpt_path),
    }
    write_json(complete_path, summary)
    return summary


def evaluate_black_and_earth_bg(
    config: dict[str, Any],
    base_dir: Path,
    *,
    checkpoint_path: Path,
    tag: str,
    output_dir: Path,
    datasets: list[str],
    device_name: str = "auto",
) -> dict[str, Any]:
    """黒背景（単一RANSAC＆残差RANSAC）と擬似地球背景（通常画像）の両面で評価する。"""
    device = resolve_device(device_name)
    model, _ = load_checkpoint(
        checkpoint_path, device, default_base_channels=int(config["training"]["base_channels"])
    )
    input_size = int(config["input_size"])
    success_iou = float(config["success_iou"])
    threshold = float(config["probability_threshold"])
    whiteout_max = float(config["evaluation"]["whiteout_maximum_strength"])
    opencv_seed = int(config["opencv_seed_per_image"])
    ransac_single = {**config["ransac_cnn"], "prefer_inner_ellipse": False}
    ransac_inner = {**config["ransac_cnn"], "prefer_inner_ellipse": True}

    data_root = resolve_path(base_dir, config["paths"]["data_dir"])
    out_root = Path(output_dir).resolve()
    summary: dict[str, Any] = {"tag": tag, "checkpoint": str(checkpoint_path), "black_bg": {}, "earth_bg_clean": {}}

    for ds in datasets:
        ds_dir = (data_root / config["datasets"][ds]["base"]).resolve()
        samples = load_manifest_samples(ds_dir, "test")
        rows_single: list[dict[str, Any]] = []
        rows_inner: list[dict[str, Any]] = []
        outer_prob_clean: list[float] = []

        print(f"[EVAL-BG] tag={tag} dataset={ds} n={len(samples)}", flush=True)
        for idx, sample in enumerate(samples, start=1):
            img, truth = load_sample_and_truth(
                ds_dir, sample, input_size=input_size, whiteout_maximum_strength=whiteout_max
            )
            if int(np.ptp(img)) == 0:
                e_s = e_i = None
                prob = None
            else:
                with torch.inference_mode():
                    prob = torch.sigmoid(model(image_to_tensor(img).to(device)[None]))[0, 0].cpu().numpy()
                cv2.setRNGSeed(opencv_seed)
                e_s = boundary_ellipse(prob, ransac_single, threshold=threshold)
                cv2.setRNGSeed(opencv_seed)
                e_i = boundary_ellipse(prob, ransac_inner, threshold=threshold)

            if sample["conditions"]["degradation"] == "clean" and prob is not None:
                outer_bg = extract_outer_background_mask(img[:, :, 0])
                outer_edge = extract_outer_silhouette_edge(outer_bg, band_px=3) > 0.5
                hole = region_mask(input_size, truth)
                inner_band = cv2.dilate((hole >= 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)) - cv2.erode(
                    (hole >= 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)
                )
                pure_outer = outer_edge & (inner_band == 0)
                if np.any(pure_outer):
                    outer_prob_clean.append(float(prob[pure_outer].mean()))

            for container, ell in ((rows_single, e_s), (rows_inner, e_i)):
                iou = evaluate_ellipses(ell, truth, (input_size, input_size))["ellipse_iou"] if ell else 0.0
                container.append(
                    {
                        "sample_id": sample["sample_id"],
                        "degradation": sample["conditions"]["degradation"],
                        "severity": float(sample["conditions"]["severity"]),
                        "ellipse": ellipse_to_list(ell),
                        "ellipse_iou": float(iou),
                        "success": bool(ell is not None and iou >= success_iou),
                        "image_sha256": sha256_bytes(img.tobytes()),
                    }
                )
            if idx % 600 == 0:
                print(f"  progress {idx}/{len(samples)}", flush=True)

        def _pack(rows: list[dict[str, Any]]) -> dict[str, Any]:
            clean = [r for r in rows if r["degradation"] == "clean"]
            occ50 = [r for r in rows if r["degradation"] == "black_rectangle" and abs(r["severity"] - 0.5) < 1e-6]
            w100 = [r for r in rows if r["degradation"] == "sensor_whiteout" and abs(r["severity"] - 1.0) < 1e-6]
            b100 = [r for r in rows if r["degradation"] == "sensor_black_crush" and abs(r["severity"] - 1.0) < 1e-6]
            rate = lambda xs: 100.0 * sum(1 for x in xs if x["success"]) / len(xs) if xs else 0.0
            return {
                "overall": rate(rows),
                "mean_iou": float(np.mean([r["ellipse_iou"] for r in rows])),
                "clean": rate(clean),
                "occ_0.5": rate(occ50),
                "whiteout_1.0": rate(w100),
                "blackout_1.0": rate(b100),
            }

        summary["black_bg"][ds] = {
            "single_ransac": _pack(rows_single),
            "inner_ransac": _pack(rows_inner),
            "clean_outer_edge_mean_prob": float(np.mean(outer_prob_clean)) if outer_prob_clean else 0.0,
        }
        write_json(out_root / "predictions" / tag / f"test_{ds}_inner.json", {"rows": rows_inner})

        # 擬似地球背景（earth_clouds）を通常テスト画像へ固定seedで合成して背景耐性を比較評価
        clean_samples = [s for s in samples if s["conditions"]["degradation"] == "clean"]
        earth_rng = np.random.default_rng(20261125)
        earth_single = earth_inner = 0
        earth_iou_inner: list[float] = []
        for s in clean_samples:
            raw = read_image(ds_dir / s["image"])
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            outer_bg = extract_outer_background_mask(gray)
            earth_bg = synthesize_background(gray.shape[0], earth_rng, mode="earth_clouds")
            comp_gray = composite_background_on_gray(gray, outer_bg, earth_bg)
            resized = cv2.resize(comp_gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
            img_256 = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            label = read_json(ds_dir / s["label"])
            truth = scale_ellipse(label_to_ellipse(label), input_size / float(label["image_width"]))
            with torch.inference_mode():
                prob = torch.sigmoid(model(image_to_tensor(img_256).to(device)[None]))[0, 0].cpu().numpy()
            cv2.setRNGSeed(opencv_seed)
            e_s = boundary_ellipse(prob, ransac_single, threshold=threshold)
            cv2.setRNGSeed(opencv_seed)
            e_i = boundary_ellipse(prob, ransac_inner, threshold=threshold)
            iou_s = evaluate_ellipses(e_s, truth, (input_size, input_size))["ellipse_iou"] if e_s else 0.0
            iou_i = evaluate_ellipses(e_i, truth, (input_size, input_size))["ellipse_iou"] if e_i else 0.0
            earth_single += int(iou_s >= success_iou)
            earth_inner += int(iou_i >= success_iou)
            earth_iou_inner.append(float(iou_i))

        n_c = len(clean_samples)
        summary["earth_bg_clean"][ds] = {
            "n": n_c,
            "single_ransac_percent": 100.0 * earth_single / n_c if n_c else 0.0,
            "inner_ransac_percent": 100.0 * earth_inner / n_c if n_c else 0.0,
            "inner_mean_iou": float(np.mean(earth_iou_inner)) if earth_iou_inner else 0.0,
        }
        print(
            f"  -> black(single={summary['black_bg'][ds]['single_ransac']['overall']:.2f}%, "
            f"inner={summary['black_bg'][ds]['inner_ransac']['overall']:.2f}%, "
            f"outer_prob={summary['black_bg'][ds]['clean_outer_edge_mean_prob']:.4f}) | "
            f"earth_clean(single={summary['earth_bg_clean'][ds]['single_ransac_percent']:.2f}%, "
            f"inner={summary['earth_bg_clean'][ds]['inner_ransac_percent']:.2f}%)",
            flush=True,
        )

    write_json(out_root / f"summary_{tag}.json", summary)
    return summary
