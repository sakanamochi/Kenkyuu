"""学習を保ち、元・模様・投影拡張の4source検証ビューで採用epochを選ぶ。"""

import copy
import sys
from pathlib import Path

from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_generalization import experiment, boundary, boundary_surface
from research.paf_generalization_v2 import projection, projection_occlusion


class ValidationViews(Dataset):
    """同じ105形状の4ビューを固定順で全件評価し、抽選や追加遮蔽を行わない。"""

    def __init__(self, root, input_size, cache):
        roots = [Path(root), boundary_surface.DATA,
                 projection.DATA_ROOT/'projection_coverage/native_plain',
                 projection.DATA_ROOT/'projection_coverage/native_surface']
        self.views = [projection.PackedBoundaryDataset(p, 'validation', input_size, cache) for p in roots]
        self.n = len(self.views[0])
        assert self.n == 735 and all(len(view) == self.n for view in self.views)
        assert all([s['sample_id'] for s in view.samples] == [s['sample_id'] for s in self.views[0].samples]
                   for view in self.views)

    def __len__(self):
        return len(self.views)*self.n

    def __getitem__(self, index):
        view, position = divmod(index, self.n)
        image, masks, _ = self.views[view][position]
        return image, masks, index


def dataset(root, split, input_size, cache=False):
    """検証だけを置き換え、train/testの入力値と乱数消費は元実装へ委ねる。"""
    if Path(root).name == 'diversified' and split == 'validation':
        return ValidationViews(root, input_size, cache)
    return projection_occlusion.ProjectionOcclusionDataset(root, split, input_size, cache)


def freeze(folder, config, variant, seed):
    """検証対象はsourceのみとし、元の学習・投影・遮蔽の依存hashも保持する。"""
    config = copy.deepcopy(config)
    config['source_validation_files'] = {str(Path(__file__).relative_to(ROOT)): experiment.sha(Path(__file__))}
    return projection_occlusion.freeze(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = projection.DATA_ROOT
    experiment.RingDataset = dataset
    experiment.objective = boundary.boundary_objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.train_variant = projection.train
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
