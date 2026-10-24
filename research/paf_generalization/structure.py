"""明暗の極性に不変な局所構造テンソルから、完全な内周を学習する。"""

import copy
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.paf_generalization import experiment
from research.paf_generalization.context import ContextUNet


class StructureUNet(ContextUNet):
    """境界の強さ・向きを保持し、白黒反転で変わらない3成分を入力する。"""

    def __init__(self,base_channels=16):
        super().__init__(base_channels)
        kernel=torch.tensor([1.,4.,6.,4.,1.])/16
        self.register_buffer('smooth_kernel',torch.outer(kernel,kernel)[None,None])
        gx=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/8
        self.register_buffer('gradient_kernel',torch.stack([gx,gx.T])[:,None])

    def encode(self,images):
        gray=images.mean(1,keepdim=True)
        smooth=F.conv2d(F.pad(gray,(2,2,2,2),mode='reflect'),self.smooth_kernel)
        gradients=F.conv2d(F.pad(smooth,(1,1,1,1),mode='reflect'),self.gradient_kernel)
        gx,gy=gradients[:,0:1],gradients[:,1:2]
        # 二次積により勾配の符号を消す。大きさは残し、雑音を無条件に増幅しない。
        magnitude=torch.sqrt(gx.square()+gy.square()+1e-8)
        xx=(4*gx.square()/magnitude).clamp(0,1)
        yy=(4*gy.square()/magnitude).clamp(0,1)
        xy=(4*gx*gy/magnitude).clamp(-1,1)
        return torch.cat([2*xx-1,xy,2*yy-1],dim=1)

    def forward(self,images):
        return super().forward(self.encode(images))


original_freeze=experiment.freeze_run


def freeze_structure(folder,config,variant,seed):
    """共通学習器と文脈モデルに加えて、画像表現の実装を固定する。"""
    config=copy.deepcopy(config)
    config['structure_implementation_sha256']=experiment.sha(Path(__file__))
    config['context_implementation_sha256']=experiment.sha(Path(__file__).with_name('context.py'))
    return original_freeze(folder,config,variant,seed)


if __name__=='__main__':
    experiment.CONFIG=Path(__file__).with_name('structure.json')
    experiment.TinyUNet=StructureUNet
    experiment.freeze_run=freeze_structure
    experiment.main()
