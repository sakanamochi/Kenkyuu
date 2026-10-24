"""同じ投影・遮蔽画像から、解析楕円の境界教師を整合する単一BCEで学習する。"""

import copy
import sys
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_image, read_json, label_ellipse, scaled_ellipse
from research.paf_generalization import experiment, boundary
from research.paf_generalization_v2 import projection, projection_occlusion, analytic_teacher
from research.paf_generalization_v2.occlusion_sampling import masked_view


class AnalyticProjectionDataset(projection_occlusion.ProjectionOcclusionDataset):
    """画像・開口領域・可視マスクの抽選を揃え、正確なラベルから境界教師を加える。"""

    def __init__(self, root, split, input_size, cache=False):
        super().__init__(root, split, input_size, cache)
        self.band_cache = {}

    def __getitem__(self, index):
        choice = int(self.rng.integers(0, len(self.views))) if len(self.views) > 1 else 0
        view = self.views[choice]
        image, masks, _ = view[index]
        sample = view.samples[index]
        label_path = view.root/sample['label']
        if label_path not in self.band_cache:
            label = read_json(label_path)
            ellipse = scaled_ellipse(label_ellipse(label), self.input_size/label['image_width'])
            self.band_cache[label_path] = torch.from_numpy(analytic_teacher.analytic_band(ellipse, self.input_size))[None]
        if len(self.views) > 1 and sample['conditions']['degradation'] == 'black_rectangle':
            key = choice, sample['image']
            if key not in self.raw_cache:
                gray = cv2.cvtColor(read_image(view.root/sample['image']), cv2.COLOR_BGR2GRAY)
                self.raw_cache[key] = gray, label_ellipse(read_json(label_path))
            gray, ellipse = self.raw_cache[key]
            left = bool(self.mask_rng.integers(0, 2) == 0)
            draw = float(self.mask_rng.random())
            image, visibility = masked_view(gray, ellipse, draw, left, 'relative', self.input_size)
            masks = torch.cat([masks[:1], visibility], dim=0)
        # 既知の矩形は損失で可視マスクとして掛け、完全領域から境界を再抽出しない。
        return image, torch.cat([masks, self.band_cache[label_path]], dim=0), index


def objective(logits, masks, settings, config, bce):
    """BCE全体を画素重み付けし、画素独立の最適値を教師値に一致させる。"""
    target = masks[:, 2:3]*masks[:, 1:2]
    weight = 1+(config['positive_weight']-1)*target
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*weight).mean()


def freeze(folder, config, variant, seed):
    """教師定義と元の投影・遮蔽処理を固定し、別の実験として保存する。"""
    assert config['analytic_sigma_px'] == 1. and not config['variants'][variant]['spatial']
    assert config['analytic_loss'] == 'whole_bce_pixel_weight_no_dice'
    config = copy.deepcopy(config)
    config['analytic_teacher_files'] = {str(p.relative_to(ROOT)): experiment.sha(p)
                                      for p in (Path(__file__), Path(analytic_teacher.__file__))}
    return projection_occlusion.freeze(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = projection.DATA_ROOT
    experiment.RingDataset = AnalyticProjectionDataset
    experiment.objective = objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.train_variant = projection.train
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
