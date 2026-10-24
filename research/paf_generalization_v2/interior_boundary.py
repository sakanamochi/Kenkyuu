"""同じ開口の内部を変え、可視内周と内部の紛らわしい輪郭を区別して学習する。"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json
from research.paf_generalization import experiment, boundary, boundary_surface
from research.paf_generalization_v2.projection import PackedBoundaryDataset

INTERIOR = ROOT/'output/paf_generalization_v1/interior_counterfactual'
BASE_FREEZE = boundary_surface.freeze_boundary_surface


class InteriorBoundaryDataset(Dataset):
    """元画像・表面模様・内部構造の対応画像を、固定確率で切り替える。"""

    def __init__(self, root, split, input_size, cache=False):
        self.root, self.input_size = Path(root), input_size
        roots = [self.root]
        if self.root.name == 'diversified' and split == 'train':
            roots += [boundary_surface.DATA, INTERIOR]
        self.views = [PackedBoundaryDataset(p, split, input_size, cache) for p in roots]
        self.samples = self.views[0].samples
        assert all([s['sample_id'] for s in v.samples] == [s['sample_id'] for s in self.samples] for v in self.views)
        self.rng = np.random.default_rng(torch.initial_seed()+5200)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        view = int(self.rng.choice(3, p=[.25, .25, .5])) if len(self.views) == 3 else 0
        return self.views[view][index]


def freeze(folder, config, variant, seed):
    """内周投影を維持した内部構造データと、省メモリ読込実装を固定する。"""
    verification = read_json(INTERIOR/'verification.json')
    assert verification['available_splits'] == ['train', 'validation']
    assert verification['manifest_sha256'] == experiment.sha(INTERIOR/'manifest.json')
    seen = set()
    for sample in read_json(INTERIOR/'manifest.json')['samples']:
        for field in ('image', 'label'):
            path = Path(sample[field])
            if path not in seen:
                assert experiment.sha(path) == sample[field+'_sha256']
                seen.add(path)
    config = copy.deepcopy(config)
    paths = [Path(__file__), Path(__file__).with_name('projection.py'), INTERIOR/'plan.json',
             INTERIOR/'manifest.json', INTERIOR/'verification.json',
             ROOT/'research/paf_generalization/render_interior.py', ROOT/'research/paf_generalization/render_structure.py']
    config['interior_boundary_files'] = {str(p.relative_to(ROOT)): experiment.sha(p) for p in paths}
    return BASE_FREEZE(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = ROOT/'output/paf_generalization_v2'
    experiment.RingDataset = InteriorBoundaryDataset
    experiment.objective = boundary.boundary_objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
