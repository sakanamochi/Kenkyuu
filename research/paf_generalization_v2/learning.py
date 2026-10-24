"""同じ小さなCNNと学習入力で、完全内周境界と完全穴領域の教師を比較する。"""

import copy
import sys
from pathlib import Path

from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.train import _dice_loss
from research.paf_generalization import experiment,boundary,boundary_surface

BASE_FREEZE=boundary_surface.freeze_boundary_surface
REGION_EVALUATE=experiment.evaluate_variant


def target_mask(masks,kind):
    """内周の一部が矩形外に残る場合だけ、完全な境界を教師として与える。"""
    region=masks[:,0:1]
    if kind=='complete_region':
        return region
    rim=(F.max_pool2d(region,3,1,1)+F.max_pool2d(-region,3,1,1)).clamp(0,1)
    observed=(rim*masks[:,1:2]).sum((1,2,3),keepdim=True)>0
    # 境界が矩形で全て隠れた例では、根拠のない完全内周を要求しない。
    return rim*observed


def objective(logits,masks,settings,config,bce):
    target=target_mask(masks,settings['target'])
    return bce(logits,target)+_dice_loss(logits,target)


def freeze(folder,config,variant,seed):
    """既存の入力・推定器のhashと、教師を定める本実装を同時に固定する。"""
    config=copy.deepcopy(config)
    config['target_implementation_sha256']=experiment.sha(Path(__file__))
    return BASE_FREEZE(folder,config,variant,seed)


def evaluate(variant,seed):
    if variant=='boundary_amodal':
        return boundary.evaluate_boundary(variant,seed)
    return REGION_EVALUATE(variant,seed)


if __name__=='__main__':
    experiment.CONFIG=Path(__file__).with_name('learning.json')
    experiment.OUT=ROOT/'output/paf_generalization_v2'
    experiment.RingDataset=boundary_surface.SurfaceBoundaryDataset
    experiment.objective=objective
    experiment.freeze_run=freeze
    boundary.freeze_boundary=freeze
    experiment.evaluate_variant=evaluate
    experiment.main()
