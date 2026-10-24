"""表面模様への不変性と、CNN境界確率・幾何補完の分担を組み合わせる。"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json
from research.paf_generalization import experiment,boundary

DATA=experiment.OUT/'surface_counterfactual'


class SurfaceBoundaryDataset(boundary.BoundaryDataset):
    """同じ境界教師で元画像と模様画像を等確率で学習する。"""

    def __init__(self,dataset_dir,split,input_size,cache=False):
        super().__init__(dataset_dir,split,input_size,cache)
        self.intervention=None
        if Path(dataset_dir).name=='diversified' and split=='train':
            self.intervention=boundary.BoundaryDataset(DATA,split,input_size,cache)
            assert [s['sample_id'] for s in self.samples]==[s['sample_id'] for s in self.intervention.samples]
            self.rng=np.random.default_rng(torch.initial_seed()+5200)

    def __getitem__(self,index):
        if self.intervention is not None and self.rng.random()<.5:
            return self.intervention[index]
        return super().__getitem__(index)


base_freeze=boundary.freeze_boundary


def freeze_boundary_surface(folder,config,variant,seed):
    """境界学習・模様生成・追加画像を一緒に照合して固定する。"""
    check=read_json(DATA/'verification.json')
    assert check['available_splits']==['train','validation']
    assert check['manifest_sha256']==experiment.sha(DATA/'manifest.json')
    seen=set()
    for sample in read_json(DATA/'manifest.json')['samples']:
        for field in ('image','label'):
            path=Path(sample[field])
            if path not in seen:
                assert experiment.sha(path)==sample[field+'_sha256']
                seen.add(path)
    config=copy.deepcopy(config)
    config['boundary_surface_implementation_sha256']=experiment.sha(Path(__file__))
    config['surface_data_hashes']={str(p.relative_to(ROOT)):experiment.sha(p) for p in
        (DATA/'manifest.json',DATA/'plan.json',DATA/'verification.json',Path(__file__).with_name('render_surface.py'))}
    return base_freeze(folder,config,variant,seed)


if __name__=='__main__':
    experiment.CONFIG=Path(__file__).with_name('boundary_surface.json')
    experiment.RingDataset=SurfaceBoundaryDataset
    experiment.objective=boundary.boundary_objective
    experiment.freeze_run=freeze_boundary_surface
    boundary.freeze_boundary=freeze_boundary_surface
    experiment.evaluate_variant=boundary.evaluate_boundary
    experiment.main()
