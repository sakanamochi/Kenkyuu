"""CNNが内周の境界位置を学び、確率で重み付けしたRANSACが完全楕円を推定する。"""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import RingDataset,read_json,write_json,region_mask
from paf_ring_detection.effects import apply_effect
from paf_ring_detection.train import _dice_loss
from paf_ring_detection.methods.ransac import fit_ellipse_ransac
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization import experiment


class BoundaryDataset(RingDataset):
    """完全穴領域と既知の矩形遮蔽の可視マスクを返す。穴は空間変換の基準にも使う。"""

    def __init__(self,dataset_dir,split,input_size,cache=False):
        super().__init__(dataset_dir,split,input_size,cache)
        self.visibility_cache={}

    def __getitem__(self,index):
        image,region,_=super().__getitem__(index)
        if index not in self.visibility_cache:
            visibility=np.ones((480,480,3),np.uint8)*255
            effect=self.samples[index].get('augmentation')
            if effect and effect['name']=='black_rectangle':
                visibility=apply_effect(visibility,effect['name'],effect['severity'],np.random.default_rng(effect['seed']))
            visibility=cv2.resize(visibility[:,:,0],(self.input_size,self.input_size),interpolation=cv2.INTER_AREA)/255
            self.visibility_cache[index]=torch.from_numpy(visibility.astype(np.float32))[None]
        return image,torch.cat([region,self.visibility_cache[index]],dim=0),index


def boundary_target(masks):
    """変換後の領域から幅約2pxの境界を作り、矩形に隠れた部分を除く。"""
    region=masks[:,0:1]
    rim=F.max_pool2d(region,3,1,1)+F.max_pool2d(-region,3,1,1)
    # 露光による可視性は判定せず、矩形の既知の被覆だけを教師から除く。
    return rim.clamp(0,1)*masks[:,1:2]


def boundary_objective(logits,masks,settings,config,bce):
    target=boundary_target(masks)
    return bce(logits,target)+_dice_loss(logits,target)


def skeleton_points(probability,threshold=.5):
    """二値化した帯の形態学的スケルトンを求め、確率を点の重みにする。"""
    remaining=(probability>=threshold).astype(np.uint8)
    skeleton=np.zeros_like(remaining)
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    while np.any(remaining):
        eroded=cv2.erode(remaining,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        opened=cv2.dilate(eroded,kernel)
        skeleton|=remaining & (1-opened)
        remaining=eroded
    ys,xs=np.where(skeleton>0)
    return np.stack([xs,ys],axis=1).astype(np.float32),probability[ys,xs]


def boundary_ellipse(probability,settings=None):
    """境界点のCNN確率をサンプリングと支持量の両方に使う。"""
    if settings is None:
        settings=read_json(Path(__file__).with_name('boundary.json'))['ransac']
    points,weights=skeleton_points(probability)
    if len(points)<settings['min_inliers']:
        return None
    fitted=fit_ellipse_ransac(points,probability.shape,settings,weights=weights)
    return fitted['ellipse'] if fitted else None


original_freeze=experiment.freeze_run


def freeze_boundary(folder,config,variant,seed):
    """境界教師と幾何推定器の双方を、共通学習仕様と一緒に固定する。"""
    config=copy.deepcopy(config)
    config['boundary_implementation_sha256']=experiment.sha(Path(__file__))
    config['ransac_implementation_sha256']=experiment.sha(ROOT/'paf_ring_detection/methods/ransac.py')
    return original_freeze(folder,config,variant,seed)


def evaluate_boundary(variant,seed):
    """全画像を分母に楕円を採点し、境界マスクを穴マスクのIoUと混同しない。"""
    config=read_json(experiment.CONFIG)
    folder=experiment.OUT/'runs'/variant/str(seed)
    provenance=freeze_boundary(folder,config,variant,seed)
    path=folder/'best.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    assert checkpoint['provenance']==provenance
    model=experiment.TinyUNet(16).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])
    for source in ('1194m','diversified'):
        destination=folder/f'test_{source}.json'
        if destination.exists():
            assert read_json(destination)['checkpoint_sha256']==experiment.sha(path)
            continue
        dataset=BoundaryDataset(experiment.BASE/'datasets'/source,'test',256)
        rows=[]
        for index,sample in enumerate(dataset.samples):
            image,truth=experiment.image_and_truth(dataset.root,sample)
            x=torch.from_numpy(image.transpose(2,0,1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                probability=torch.sigmoid(model(x))[0,0].cpu().numpy()
            ellipse=boundary_ellipse(probability,config['ransac'])
            iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id':sample['sample_id'],'group_id':sample['group_id'],
                'base_sample_id':sample['conditions']['base_sample_id'],
                'degradation':sample['conditions']['degradation'],'severity':sample['conditions']['severity'],
                'ellipse':ellipse,'truth_ellipse':truth,'ellipse_iou':iou,'success':iou>=.8,'mask_iou':None})
        write_json(destination,{'dataset':source,'variant':variant,'seed':seed,
            'checkpoint_sha256':experiment.sha(path),'provenance':provenance,'rows':rows,
            'mask_metric':'境界確率のため、完全穴マスクIoUは定義しない。'})
        print('EVAL',variant,seed,source,sum(r['success'] for r in rows),len(rows),flush=True)


if __name__=='__main__':
    experiment.CONFIG=Path(__file__).with_name('boundary.json')
    experiment.RingDataset=BoundaryDataset
    experiment.objective=boundary_objective
    experiment.freeze_run=freeze_boundary
    experiment.evaluate_variant=evaluate_boundary
    experiment.main()
