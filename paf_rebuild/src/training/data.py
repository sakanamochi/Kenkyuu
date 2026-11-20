"""manifestで指定した投影・外観と、開口幅基準の遮蔽を使う学習入力。"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from core.effects import apply_effect
from core.ellipse import label_ellipse, scale_ellipse
from core.files import read_json
from core.images import INPUT_SIZE, prepare_image, read_image


def region_mask(ellipse):
    """480px座標の正解楕円を256pxの穴領域にする。"""
    region = np.zeros((INPUT_SIZE, INPUT_SIZE), np.uint8)
    cv2.ellipse(region, scale_ellipse(ellipse, INPUT_SIZE / 480), 255, -1, cv2.LINE_AA)
    return region


def masked_view(gray, ellipse, draw, left):
    """原画像上の同じ矩形で入力と可視マスクを覆い、256pxへ縮小する。"""
    if not 0 <= draw <= 1:
        raise ValueError("遮蔽の抽選値は0〜1にしてください。")
    (cx, _), (a, b), angle = ellipse
    radians = np.deg2rad(angle)
    half_width = np.hypot(a / 2 * np.cos(radians), b / 2 * np.sin(radians))
    # 開口の外側まで抽選範囲を広げ、全被覆と無被覆も含める。
    cut = int(np.clip(round(cx + (2.5 * draw - 1.25) * half_width), 1, gray.shape[1] - 1))
    image = gray.copy()
    visibility = np.full(gray.shape, 255, np.uint8)
    selected = np.s_[:, :cut] if left else np.s_[:, cut:]
    image[selected] = 0
    visibility[selected] = 0
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    visibility = cv2.resize(visibility, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).float().unsqueeze(0).expand(3, -1, -1) / 127.5 - 1
    return tensor, torch.from_numpy(visibility.astype(np.float32) / 255)[None]


class BoundaryView(Dataset):
    """固定劣化と穴領域・可視マスクを生成し、必要なら8bitで保持する。"""

    def __init__(self, root, split, *, cache=True, whiteout_multiplier=1.0):
        self.root = Path(root).resolve()
        self.samples = [s for s in read_json(self.root / "manifest.json")["samples"] if s["split"] == split]
        ids = [s["sample_id"] for s in self.samples]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"空の分割または重複sample IDです: {self.root} / {split}")
        self.packed = {} if cache else None
        self.whiteout_multiplier = whiteout_multiplier

    def __len__(self):
        return len(self.samples)

    def raw(self, index):
        sample = self.samples[index]
        image = read_image(self.root / sample["image"])
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        label = read_json(self.root / sample["label"])
        if image.shape != (480, 480, 3) or image.dtype != np.uint8 or label["image_width"] != 480:
            raise ValueError("現在の学習データは480×480のuint8画像・対応ラベルに限ります。")
        return image, label_ellipse(label)

    def __getitem__(self, index):
        if self.packed is None or index not in self.packed:
            image, ellipse = self.raw(index)
            visibility = np.full(image.shape, 255, np.uint8)
            effect = self.samples[index].get("augmentation")
            if effect:
                arguments = (effect["name"], effect["severity"], effect["seed"])
                image = apply_effect(image, *arguments, whiteout_multiplier=self.whiteout_multiplier)
                if effect["name"] == "black_rectangle":
                    visibility = apply_effect(visibility, *arguments)
            gray = prepare_image(image)
            region = region_mask(ellipse)
            visible = cv2.resize(visibility[:, :, 0], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            packed = gray, np.stack((region, visible))
            if self.packed is not None:
                self.packed[index] = packed
        else:
            packed = self.packed[index]
        gray, masks = packed
        image = (torch.from_numpy(gray).float() / 127.5 - 1).unsqueeze(0).expand(3, -1, -1)
        return image, torch.from_numpy(masks.astype(np.float32) / 255), index


class BoundaryDataset(Dataset):
    """4スロットを抽選し、trainの矩形遮蔽例だけ位置を再抽選する。"""

    def __init__(self, roots, split, seed, *, cache=True, whiteout_multiplier=1.0):
        if split not in ("train", "validation") or len(roots) != (4 if split == "train" else 1):
            raise ValueError("trainには4スロット、validationには元画像1スロットを指定してください。")
        # 同じ材質・投影を参照するスロットは、画素キャッシュを共有する。
        unique = {}
        self.views = []
        for root in roots:
            path = Path(root).resolve()
            if path not in unique:
                unique[path] = BoundaryView(path, split, cache=cache, whiteout_multiplier=whiteout_multiplier)
            self.views.append(unique[path])
        self.samples = self.views[0].samples
        for view in list(unique.values())[1:]:
            if len(view) != len(self.samples):
                raise ValueError("ビュー間の例数が一致しません。")
            for expected, actual in zip(self.samples, view.samples):
                if any(expected[key] != actual[key] for key in ("sample_id", "group_id", "conditions", "augmentation")):
                    raise ValueError("ビュー間のsample ID・分割群・劣化条件が一致しません。")
        self.rng = np.random.default_rng(seed + 5200)
        self.mask_rng = np.random.default_rng(seed + 6200)
        self.raw_cache = {} if cache else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if get_worker_info() is not None:
            raise RuntimeError("抽選順を固定するためDataLoaderのnum_workersは0にしてください。")
        choice = int(self.rng.integers(0, len(self.views))) if len(self.views) > 1 else 0
        view = self.views[choice]
        sample = view.samples[index]
        if len(self.views) == 1 or sample["conditions"]["degradation"] != "black_rectangle":
            return view[index]
        key = view.root, sample["image"], sample["label"]
        if self.raw_cache is None or key not in self.raw_cache:
            raw, ellipse = view.raw(index)
            raw_value = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), ellipse, region_mask(ellipse)
            if self.raw_cache is not None:
                self.raw_cache[key] = raw_value
        else:
            raw_value = self.raw_cache[key]
        gray, ellipse, region = raw_value
        left = bool(self.mask_rng.integers(0, 2) == 0)
        draw = float(self.mask_rng.random())
        image, visibility = masked_view(gray, ellipse, draw, left)
        # 再抽選する遮蔽例には、同じ原画像の穴領域と抽選した可視マスクを使う。
        region = torch.from_numpy(region.astype(np.float32) / 255)[None]
        return image, torch.cat([region, visibility], dim=0), index


def dataset_from_config(config_path, source, split, seed, *, cache=True):
    """パスは設定ファイル基準で解決し、学習元と乱数を明示する。"""
    path = Path(config_path).resolve()
    config = read_json(path)
    settings = config["sources"][source]
    roots = settings["train_views"] if split == "train" else [settings["base"]]
    return BoundaryDataset([path.parent / root for root in roots], split, seed, cache=cache,
                           whiteout_multiplier=config["whiteout_multiplier"])
