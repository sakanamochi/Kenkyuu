"""投影サイズ・扁平率の拡張と、開口幅を基準にした部分遮蔽を同時に学ぶ。"""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_image, read_json, label_ellipse
from research.paf_generalization import experiment, boundary
from research.paf_generalization_v2 import projection
from research.paf_generalization_v2.occlusion_sampling import masked_view


class ProjectionOcclusionDataset(projection.ProjectionDataset):
    """4ビューの抽選を元実装と揃え、選ばれた開口に矩形を毎回生成する。"""

    def __init__(self, dataset_dir, split, input_size, cache=False):
        super().__init__(dataset_dir, split, input_size, cache)
        self.mask_rng = np.random.default_rng(torch.initial_seed()+6200)
        self.raw_cache = {}

    def __getitem__(self, index):
        choice = int(self.rng.integers(0, len(self.views))) if len(self.views) > 1 else 0
        view = self.views[choice]
        image, masks, _ = view[index]
        sample = view.samples[index]
        if len(self.views) == 1 or sample['conditions']['degradation'] != 'black_rectangle':
            return image, masks, index
        key = choice, sample['image']
        if key not in self.raw_cache:
            gray = cv2.cvtColor(read_image(view.root/sample['image']), cv2.COLOR_BGR2GRAY)
            ellipse = label_ellipse(read_json(view.root/sample['label']))
            self.raw_cache[key] = gray, ellipse
        gray, ellipse = self.raw_cache[key]
        left = bool(self.mask_rng.integers(0, 2) == 0)
        draw = float(self.mask_rng.random())
        image, visibility = masked_view(gray, ellipse, draw, left, 'relative', self.input_size)
        # 完全穴教師と画像は選択した同じビューから取得し、既知の可視マスクだけ更新する。
        return image, torch.cat([masks[:1], visibility], dim=0), index


def freeze(folder, config, variant, seed):
    """投影CG・遮蔽抽選の両方を固定し、推定器は元のRANSACを保つ。"""
    config = copy.deepcopy(config)
    config['occlusion_sampling_files'] = {str(p.relative_to(ROOT)): experiment.sha(p)
        for p in (Path(__file__), Path(__file__).with_name('occlusion_sampling.py'))}
    return projection.freeze(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = projection.DATA_ROOT
    experiment.RingDataset = ProjectionOcclusionDataset
    experiment.objective = boundary.boundary_objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.train_variant = projection.train
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
