"""同じ上端環に異なる内部輪郭を与え、開口と内容物を切り分けて学習する。"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import RingDataset,read_json
from research.paf_generalization import experiment
from research.paf_generalization.context import ContextUNet

DATA=experiment.OUT/'interior_counterfactual'


class InteriorDataset(RingDataset):
    """学習時だけ元画像と内部構造介入画像を等確率で選び、例数を維持する。"""

    def __init__(self,dataset_dir,split,input_size,cache=False):
        super().__init__(dataset_dir,split,input_size,cache)
        self.intervention=None
        if Path(dataset_dir).name=='diversified' and split=='train':
            self.intervention=RingDataset(DATA,split,input_size,cache)
            assert [s['sample_id'] for s in self.samples]==[s['sample_id'] for s in self.intervention.samples]
            self.rng=np.random.default_rng(torch.initial_seed()+5200)

    def __getitem__(self,index):
        if self.intervention is not None and self.rng.random()<.5:
            return self.intervention[index]
        return super().__getitem__(index)


original_freeze=experiment.freeze_run


def freeze_interior(folder,config,variant,seed):
    """元教師との対応と追加画像のhashを固定し、testへの混入を防ぐ。"""
    verification=read_json(DATA/'verification.json')
    assert verification['available_splits']==['train','validation']
    assert verification['manifest_sha256']==experiment.sha(DATA/'manifest.json')
    manifest=read_json(DATA/'manifest.json')
    checked=set()
    for sample in manifest['samples']:
        for field in ('image','label'):
            path=Path(sample[field])
            if path not in checked:
                assert experiment.sha(path)==sample[field+'_sha256']
                checked.add(path)
    config=copy.deepcopy(config)
    config['interior_implementation_sha256']=experiment.sha(Path(__file__))
    config['context_implementation_sha256']=experiment.sha(Path(__file__).with_name('context.py'))
    config['interior_data_hashes']={str(p.relative_to(ROOT)):experiment.sha(p) for p in
        (DATA/'manifest.json',DATA/'plan.json',DATA/'verification.json',
         Path(__file__).with_name('render_interior.py'),Path(__file__).with_name('render_structure.py'))}
    return original_freeze(folder,config,variant,seed)


if __name__=='__main__':
    experiment.CONFIG=Path(__file__).with_name('interior.json')
    experiment.TinyUNet=ContextUNet
    experiment.RingDataset=InteriorDataset
    experiment.freeze_run=freeze_interior
    experiment.main()
