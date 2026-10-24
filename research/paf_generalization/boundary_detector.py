"""CNNの内周境界確率と固定RANSACによる画像単体の推論を提供する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,scaled_ellipse
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_generalization.detector import predict_file
from research.paf_generalization.diagnose import OUT,sha


class BoundaryPAFDetector:
    """矩形遮蔽の外にある内周境界を証拠として完全楕円を推定する。"""

    def __init__(self,variant='boundary_surface',seed=20260907):
        if variant not in ('boundary_spatial','boundary_surface'):
            raise ValueError('boundary_spatialまたはboundary_surfaceを指定してください')
        folder=OUT/'runs'/variant/str(seed)
        path=folder/'best.pt'
        assert read_json(folder/'complete.json')['checkpoint_sha256']==sha(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        metadata=read_json(folder/'provenance.json')
        assert checkpoint['provenance']==metadata
        for relative,digest in metadata['files'].items():
            assert sha(ROOT/relative)==digest
        config=metadata['config']
        assert config['boundary_implementation_sha256']==sha(Path(__file__).with_name('boundary.py'))
        assert config['ransac_implementation_sha256']==sha(ROOT/'paf_ring_detection/methods/ransac.py')
        self.settings=config['ransac']
        self.model=TinyUNet(16).cuda().eval()
        self.model.load_state_dict(checkpoint['model_state'])
        self.provenance={'variant':variant,'seed':seed,'checkpoint_sha256':sha(path),
            'selected_epoch':checkpoint['epoch'],'settings':self.settings,
            'implementation_sha256':sha(Path(__file__)),
            'boundary_implementation_sha256':config['boundary_implementation_sha256'],
            'reject_uniform_images':True,'input':'square_grayscale_256'}

    def predict(self,image):
        """楕円を元の正方形画像の座標へ戻し、256pxの境界確率も返す。"""
        if image.dtype!=np.uint8 or image.shape[0]!=image.shape[1]:
            raise ValueError('入力はuint8の正方形BGRまたはグレー画像にしてください')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        gray=cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
        if np.ptp(gray)==0:
            return {'ellipse':None,'probability':None,'status':'insufficient_image_evidence'}
        x=torch.from_numpy(np.repeat(gray[None,None],3,axis=1).copy()).float().cuda()/127.5-1
        with torch.inference_mode():
            probability=torch.sigmoid(self.model(x))[0,0].cpu().numpy()
        ellipse=boundary_ellipse(probability,self.settings)
        if ellipse is not None:
            ellipse=scaled_ellipse(ellipse,image.shape[0]/256)
        return {'ellipse':ellipse,'probability':probability,
                'status':'detected' if ellipse is not None else 'no_supported_prediction'}


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['boundary_surface','boundary_spatial'],default='boundary_surface')
    parser.add_argument('--seed',type=int,default=20260907)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    predict_file(BoundaryPAFDetector(args.variant,args.seed),args.image,args.output)
