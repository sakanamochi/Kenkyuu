"""境界教師用データセット、複数ビュー抽選、開口幅基準オンライン遮蔽。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.effects import apply_effect, image_to_tensor, render_sample_image
from core.geometry import Ellipse, label_to_ellipse, region_mask, scale_ellipse
from core.io import load_manifest_samples, read_image, read_json


def aperture_cut_x(ellipse: Ellipse, width: int, draw: float) -> int:
    """開口の横幅を基準に矩形遮蔽の縦辺x座標を決定する（学習時専用）。"""
    (cx, _), (a, b), angle = ellipse
    radians = np.deg2rad(angle)
    half_width = np.hypot(a / 2.0 * np.cos(radians), b / 2.0 * np.sin(radians))
    # 開口幅の外側25%まで含め、全被覆・無被覆を各約10%残す。
    return int(np.clip(round(cx + (2.5 * draw - 1.25) * half_width), 1, width - 1))


def masked_view_relative(
    gray: np.ndarray,
    ellipse: Ellipse,
    draw: float,
    left: bool,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """原解像度で開口幅基準の黒矩形遮蔽と可視マスクを同時適用し、input_sizeへ縮小する。"""
    cut = aperture_cut_x(ellipse, gray.shape[1], draw)
    image = gray.copy()
    visibility = np.full(gray.shape, 255, dtype=np.uint8)
    selected = np.s_[:, :cut] if left else np.s_[:, cut:]
    image[selected] = 0
    visibility[selected] = 0

    resized_img = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    resized_vis = cv2.resize(visibility, (input_size, input_size), interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(resized_img).float().unsqueeze(0).expand(3, -1, -1) / 127.5 - 1.0
    vis_tensor = torch.from_numpy(resized_vis.astype(np.float32) / 255.0)[None]
    return image_tensor, vis_tensor


class BoundaryViewDataset(Dataset):
    """単一ビューの穴領域マスク＋既知矩形遮蔽の可視マスクを返すデータセット。"""

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

    def _build_item(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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

        image_tensor = image_to_tensor(shared)
        hole_mask = region_mask(self.input_size, scale_ellipse(ellipse, scale))
        hole_tensor = torch.from_numpy(hole_mask[None]).float()

        visibility = np.full(
            (int(label["image_height"]), int(label["image_width"]), 3), 255, dtype=np.uint8
        )
        effect = sample.get("augmentation")
        if effect and effect["name"] == "black_rectangle":
            visibility = apply_effect(
                visibility,
                effect["name"],
                effect["severity"],
                np.random.default_rng(effect["seed"]),
            )
        vis_resized = (
            cv2.resize(visibility[:, :, 0], (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            / 255.0
        )
        vis_tensor = torch.from_numpy(vis_resized.astype(np.float32))[None]
        return image_tensor, torch.cat([hole_tensor, vis_tensor], dim=0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self._packed is None:
            image_tensor, masks_tensor = self._build_item(index)
            return image_tensor, masks_tensor, index

        if index not in self._packed:
            image_tensor, masks_tensor = self._build_item(index)
            gray = torch.round((image_tensor[0] + 1.0) * 127.5).to(torch.uint8).numpy()
            target = torch.round(masks_tensor * 255.0).to(torch.uint8).numpy()
            self._packed[index] = (gray, target)

        gray, target = self._packed[index]
        image_tensor = (torch.from_numpy(gray).float() / 127.5 - 1.0).unsqueeze(0).expand(3, -1, -1)
        masks_tensor = torch.from_numpy(target.astype(np.float32) / 255.0)
        return image_tensor, masks_tensor, index


class MultiViewBoundaryDataset(Dataset):
    """複数ビューのスロット抽選と開口幅基準オンライン遮蔽を行う学習用データセット。"""

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
        loaded_views: dict[str, BoundaryViewDataset] = {}
        for name in active_slots:
            if name not in loaded_views:
                loaded_views[name] = BoundaryViewDataset(
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
        expected_ids = [sample["sample_id"] for sample in self.samples]
        for view in loaded_views.values():
            if [sample["sample_id"] for sample in view.samples] != expected_ids:
                raise ValueError(f"ビュー間でsample_id列が一致しません: {view.root}")

        base_seed = int(torch.initial_seed() if seed is None else seed)
        self.view_rng = np.random.default_rng(base_seed + 5200)
        self.mask_rng = np.random.default_rng(base_seed + 6200)
        self._raw_cache: dict[tuple[str, str], tuple[np.ndarray, Ellipse]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        choice = int(self.view_rng.integers(0, len(self.views))) if len(self.views) > 1 else 0
        view = self.views[choice]
        slot_name = self.slot_names[choice]
        image, masks, _ = view[index]
        sample = view.samples[index]

        if (
            not self.use_aperture_occlusion
            or self.split != "train"
            or sample["conditions"]["degradation"] != "black_rectangle"
        ):
            return image, masks, index

        cache_key = (slot_name, sample["image"])
        if cache_key not in self._raw_cache:
            raw = read_image(view.root / sample["image"])
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            ellipse = label_to_ellipse(read_json(view.root / sample["label"]))
            self._raw_cache[cache_key] = (gray, ellipse)
        gray, ellipse = self._raw_cache[cache_key]

        left = bool(self.mask_rng.integers(0, 2) == 0)
        draw = float(self.mask_rng.random())
        occluded_image, visibility = masked_view_relative(gray, ellipse, draw, left, self.input_size)
        return occluded_image, torch.cat([masks[:1], visibility], dim=0), index
