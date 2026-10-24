"""元の撮影を保持し、内周の投影分布を広げたCGを混ぜて可視境界を学習する。"""

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

DATA_ROOT = ROOT/'output/paf_generalization_v2'
NEW_VIEW = 'native_framing'
BASE_FREEZE = boundary_surface.freeze_boundary_surface
BASE_TRAIN = experiment.train_variant


class PackedBoundaryDataset(boundary.BoundaryDataset):
    """元の入力値を保ち、同一輝度3chとAA教師を8bitで省メモリ保持する。"""

    def __init__(self, root, split, input_size, cache=False):
        super().__init__(root, split, input_size, cache=False)
        self.packed = {} if cache else None

    def __getitem__(self, index):
        if self.packed is None:
            return super().__getitem__(index)
        if index not in self.packed:
            image, masks, _ = super().__getitem__(index)
            gray = torch.round((image[0]+1)*127.5).to(torch.uint8).numpy()
            target = torch.round(masks*255).to(torch.uint8).numpy()
            self.packed[index] = gray, target
            self.visibility_cache.pop(index, None)
        gray, target = self.packed[index]
        image = (torch.from_numpy(gray).float()/127.5-1).unsqueeze(0).expand(3, -1, -1)
        masks = torch.from_numpy(target.astype(np.float32)/255.)
        return image, masks, index


class ProjectionDataset(Dataset):
    """学習時に元構図・新構図と元材質・模様の4組を等確率で選ぶ。"""

    def __init__(self, dataset_dir, split, input_size, cache=False):
        self.root = Path(dataset_dir)
        self.input_size = input_size
        roots = [self.root]
        if self.root.name == 'diversified' and split == 'train':
            roots += [boundary_surface.DATA, DATA_ROOT/NEW_VIEW/'native_plain', DATA_ROOT/NEW_VIEW/'native_surface']
        self.views = [PackedBoundaryDataset(root, split, input_size, cache) for root in roots]
        self.samples = self.views[0].samples
        assert all([s['sample_id'] for s in view.samples] == [s['sample_id'] for s in self.samples] for view in self.views)
        self.rng = np.random.default_rng(torch.initial_seed()+5200)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        view = int(self.rng.integers(0, len(self.views))) if len(self.views) > 1 else 0
        return self.views[view][index]


def freeze(folder, config, variant, seed):
    """元画像と追加投影画像の依存関係を保存し、設定を上書きしない。"""
    directory = DATA_ROOT/config['variants'][variant]['new_view']
    verification = read_json(directory/'verification.json')
    assert verification['plan_sha256'] == experiment.sha(directory/'plan.json')
    seen = set()
    for style, digest in verification['manifest_hashes'].items():
        path = directory/f'native_{style}/manifest.json'
        assert experiment.sha(path) == digest
        for sample in read_json(path)['samples']:
            assert sample['split'] in ('train', 'validation')
            for field in ('image', 'label'):
                p = Path(sample[field])
                if p not in seen:
                    assert experiment.sha(p) == sample[field+'_sha256']
                    seen.add(p)
    config = copy.deepcopy(config)
    renderer = 'render_native.py' if directory.name == 'native_framing' else 'render_projection.py'
    config['projection_files'] = {str(p.relative_to(ROOT)): experiment.sha(p) for p in
                                  (Path(__file__), Path(__file__).with_name(renderer), directory/'plan.json', directory/'verification.json')}
    return BASE_FREEZE(folder, config, variant, seed)


def train(variant, seed):
    global NEW_VIEW
    NEW_VIEW = read_json(experiment.CONFIG)['variants'][variant]['new_view']
    return BASE_TRAIN(variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = DATA_ROOT
    experiment.RingDataset = ProjectionDataset
    experiment.objective = boundary.boundary_objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.train_variant = train
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
