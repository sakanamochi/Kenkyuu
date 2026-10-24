"""投影楕円を保つアフィン変換で、位置・寸法・向き・扁平率の依存を減らす。"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.paf_generalization import experiment
from research.paf_generalization.context import ContextUNet


def affine_augment(images,masks,rng,config):
    """画像と完全穴教師へ同じアフィン写像を適用し、教師全体を画面内へ保つ。"""
    batch=len(images)
    cx,cy,*_=experiment.moments(masks)
    scales=np.exp(rng.uniform(np.log(config['scale_range'][0]),np.log(config['scale_range'][1]),batch))
    ratios=np.exp(rng.uniform(np.log(config['flatten_ratio_range'][0]),np.log(config['flatten_ratio_range'][1]),batch))
    angles=rng.uniform(0,2*np.pi,batch)
    target=rng.uniform(*config['center_range'],(batch,2))*2-1
    active=rng.random(batch)<config['spatial_probability']
    matrices=np.zeros((batch,2,3),np.float32)
    for i in range(batch):
        if not active[i]:
            matrices[i]=np.array([[1,0,0],[0,1,0]])
            continue
        ys,xs=torch.where(masks[i,0]>.5)
        center=np.array([float(cx[i]),float(cy[i])])
        bounds=np.array([[float(xs.min()),float(ys.min())],[float(xs.min()),float(ys.max())],
                         [float(xs.max()),float(ys.min())],[float(xs.max()),float(ys.max())]])
        bounds=(bounds+.5)/128-1-center
        c,s=np.cos(angles[i]),np.sin(angles[i])
        # 面積を保つ伸縮と回転により、円錐曲線の楕円性を保ったまま扁平率を広げる。
        shape=np.array([[c,-s],[s,c]])@np.diag([1/np.sqrt(ratios[i]),np.sqrt(ratios[i])])
        extent=np.max(np.abs(bounds@shape.T),axis=0)
        scale=min(scales[i],float(np.min((.95-np.abs(target[i]))/np.maximum(extent,1e-3))))
        inverse=np.linalg.inv(scale*shape)
        matrices[i,:,:2]=inverse
        matrices[i,:,2]=center-inverse@target[i]
    theta=torch.as_tensor(matrices,device=images.device)
    grid=F.affine_grid(theta,images.shape,align_corners=False)
    return F.grid_sample(images+1,grid,align_corners=False)-1,F.grid_sample(masks,grid,align_corners=False)


original_freeze=experiment.freeze_run


def freeze_affine(folder,config,variant,seed):
    """アフィン写像と参照範囲を定める実装を保存する。"""
    config=copy.deepcopy(config)
    config['affine_implementation_sha256']=experiment.sha(Path(__file__))
    config['context_implementation_sha256']=experiment.sha(Path(__file__).with_name('context.py'))
    return original_freeze(folder,config,variant,seed)


if __name__=='__main__':
    experiment.CONFIG=Path(__file__).with_name('affine.json')
    experiment.TinyUNet=ContextUNet
    experiment.spatial_augment=affine_augment
    experiment.freeze_run=freeze_affine
    experiment.main()
