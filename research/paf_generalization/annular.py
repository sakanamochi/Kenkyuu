"""開口と上面外周の包含関係を教師にし、環状構造を同時に学習する。"""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import RingDataset,read_json,region_mask,scaled_ellipse
from paf_ring_detection.train import _dice_loss
from research.paf_generalization import experiment
from research.paf_generalization.context import ContextUNet


def outer_ellipse(label):
    """同一平面の内周点から射影を復元し、既知の外径を上面外周へ投影する。"""
    points=np.asarray(label['image_points'],dtype=np.float64)
    theta=np.arange(len(points))*2*np.pi/len(points)
    circle=np.stack([np.cos(theta),np.sin(theta)],axis=1)
    homography,_=cv2.findHomography(circle,points,method=0)
    reconstructed=cv2.perspectiveTransform(circle[None],homography)[0]
    if np.max(np.abs(reconstructed-points))>.005:
        raise ValueError('内周の平面射影を十分な精度で復元できません')
    ratio=label['parameters']['top_outer_diameter']/label['parameters']['hole_diameter']
    outer=cv2.perspectiveTransform((circle*ratio)[None],homography)[0]
    return cv2.fitEllipse(outer.astype(np.float32).reshape(-1,1,2))


class AnnularDataset(RingDataset):
    """画像は元の共通処理のまま、完全開口と完全外周内部の二つを教師にする。"""

    def __init__(self,dataset_dir,split,input_size,cache=False):
        super().__init__(dataset_dir,split,input_size,cache=False)
        self.annular_cache={} if cache else None
        self.outer_cache={}
        self.with_outer=Path(dataset_dir).name=='diversified' and split in ('train','validation')

    def __getitem__(self,index):
        if self.annular_cache is not None and index in self.annular_cache:
            return self.annular_cache[index]
        image,inner,index=super().__getitem__(index)
        if self.with_outer:
            label_path=self.root/self.samples[index]['label']
            if label_path not in self.outer_cache:
                label=read_json(label_path)
                ellipse=scaled_ellipse(outer_ellipse(label),self.input_size/label['image_width'])
                self.outer_cache[label_path]=torch.from_numpy(region_mask(self.input_size,ellipse)[None])
            # AA描画の端数差でも、外周内部が開口を必ず含む教師にする。
            outer=torch.maximum(self.outer_cache[label_path],inner)
            inner=torch.cat([inner,outer],dim=0)
        result=image,inner,index
        if self.annular_cache is not None:
            self.annular_cache[index]=result
        return result


class AnnularUNet(ContextUNet):
    """外周内部の確率と、その内部で開口になる条件付き確率を出力する。"""

    def __init__(self,base_channels=16):
        super().__init__(base_channels)
        self.output=nn.Conv2d(base_channels,2,1)

    def forward(self,images):
        raw=super().forward(images)
        outer=torch.sigmoid(raw[:,1:2])
        # p(開口)=p(外周内部)×p(開口|外周内部)として点ごとの包含を保証する。
        inner=(outer*torch.sigmoid(raw[:,0:1])).clamp(1e-6,1-1e-6)
        return torch.cat([torch.logit(inner),raw[:,1:2]],dim=1)


def annular_objective(logits,masks,settings,config,bce):
    inner=bce(logits[:,0:1],masks[:,0:1])+_dice_loss(logits[:,0:1],masks[:,0:1])
    outer=bce(logits[:,1:2],masks[:,1:2])+_dice_loss(logits[:,1:2],masks[:,1:2])
    return inner+config['outer_loss_weight']*outer


original_freeze=experiment.freeze_run


def freeze_annular(folder,config,variant,seed):
    """追加教師と包含関係を含む実装を、元の学習仕様とともに固定する。"""
    config=copy.deepcopy(config)
    config['annular_implementation_sha256']=experiment.sha(Path(__file__))
    config['context_implementation_sha256']=experiment.sha(Path(__file__).with_name('context.py'))
    return original_freeze(folder,config,variant,seed)


if __name__=='__main__':
    experiment.CONFIG=Path(__file__).with_name('annular.json')
    experiment.TinyUNet=AnnularUNet
    experiment.RingDataset=AnnularDataset
    experiment.objective=annular_objective
    experiment.freeze_run=freeze_annular
    experiment.main()
