"""同じCNNと教師で、学習時の遮蔽位置の抽選方法を比較する。"""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, read_image, label_ellipse
from research.paf_generalization import experiment, boundary, boundary_surface
from research.paf_generalization_v2.projection import PackedBoundaryDataset

MODE = 'relative'
BASE_TRAIN = experiment.train_variant
BASE_FREEZE = boundary_surface.freeze_boundary_surface


def cut_location(ellipse, width, draw, mode):
    """矩形の縦辺を画像幅または開口の横幅から定める。教師の利用は学習時だけ。"""
    if mode == 'image':
        return max(1, round(width*(.1+.7*draw)))
    (cx, _), (a, b), angle = ellipse
    radians = np.deg2rad(angle)
    half_width = np.hypot(a/2*np.cos(radians), b/2*np.sin(radians))
    # 開口幅の外側も25%広げ、全被覆と無被覆を各約10%残す。
    return int(np.clip(round(cx+(2.5*draw-1.25)*half_width), 1, width-1))


def masked_view(gray, ellipse, draw, left, mode, input_size):
    """原解像度で黒矩形と可視マスクを同じ位置へ適用してから縮小する。"""
    cut = cut_location(ellipse, gray.shape[1], draw, mode)
    # 画像幅基準の右遮蔽ではcutは幅、開口基準では絶対x座標を表す。
    if not left and mode == 'image':
        cut = gray.shape[1]-cut
    image = gray.copy()
    visibility = np.full(gray.shape, 255, np.uint8)
    selected = np.s_[:, :cut] if left else np.s_[:, cut:]
    image[selected] = 0
    visibility[selected] = 0
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    visibility = cv2.resize(visibility, (input_size, input_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(image).float().unsqueeze(0).expand(3, -1, -1)/127.5-1
    return x, torch.from_numpy(visibility.astype(np.float32)/255)[None]


class OcclusionDataset(PackedBoundaryDataset):
    """1epochの例数は固定し、矩形遮蔽例だけ位置を毎回再抽選する。"""

    def __init__(self, root, split, input_size, cache=False):
        super().__init__(root, split, input_size, cache)
        self.active = Path(root).name == 'diversified' and split == 'train'
        self.surface = None
        self.raw_cache = {}
        if self.active:
            self.surface = PackedBoundaryDataset(boundary_surface.DATA, split, input_size, cache)
            assert [s['sample_id'] for s in self.samples] == [s['sample_id'] for s in self.surface.samples]
            self.style_rng = np.random.default_rng(torch.initial_seed()+5200)
            self.mask_rng = np.random.default_rng(torch.initial_seed()+6200)

    def __getitem__(self, index):
        surface = self.active and self.style_rng.random() < .5
        view = self.surface if surface else self
        image, masks, _ = self.surface[index] if surface else super().__getitem__(index)
        if not self.active or self.samples[index]['conditions']['degradation'] != 'black_rectangle':
            return image, masks, index
        sample = view.samples[index]
        key = surface, sample['image']
        if key not in self.raw_cache:
            raw = read_image(view.root/sample['image'])
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            ellipse = label_ellipse(read_json(view.root/sample['label']))
            self.raw_cache[key] = gray, ellipse
        gray, ellipse = self.raw_cache[key]
        left = bool(self.mask_rng.integers(0, 2) == 0)
        draw = float(self.mask_rng.random())
        image, visibility = masked_view(gray, ellipse, draw, left, MODE, self.input_size)
        # 完全穴の教師は元のまま。矩形の可視マスクだけを置き換える。
        return image, torch.cat([masks[:1], visibility], dim=0), index


def freeze(folder, config, variant, seed):
    """遮蔽抽選の実装と元の対応画像を固定する。"""
    config = copy.deepcopy(config)
    config['occlusion_sampling_files'] = {str(p.relative_to(ROOT)): experiment.sha(p)
        for p in (Path(__file__), Path(__file__).with_name('projection.py'))}
    return BASE_FREEZE(folder, config, variant, seed)


def train(variant, seed):
    global MODE
    MODE = read_json(experiment.CONFIG)['variants'][variant]['occlusion_mode']
    return BASE_TRAIN(variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = ROOT/'output/paf_generalization_v2'
    experiment.RingDataset = OcclusionDataset
    experiment.objective = boundary.boundary_objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.train_variant = train
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
