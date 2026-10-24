"""学習済みモデルの確率を平均し、情報のない一様画像では検出を保留する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,read_image,write_image,load_samples,region_mask,scaled_ellipse
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.methods.cnn_region import region_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.context import ContextUNet
from research.paf_generalization.diagnose import BASE,OUT,sha,image_and_truth


class PAFDetector:
    """画像だけを入力に、完全穴の確率平均と楕円を返す研究用検出器。"""

    def __init__(self,variant='surface_context_spatial',seeds=(20260907,20260908,20260909)):
        self.variant=variant
        self.seeds=list(seeds)
        self.models=[]
        self.provenance={'variant':variant,'seeds':self.seeds,'aggregation':'mean_probability',
                         'reject_uniform_images':True,'input':'square_grayscale_256',
                         'implementation_sha256':sha(Path(__file__)),'checkpoints':[]}
        self.provenance['implementation_files']={str(p.relative_to(ROOT)):sha(p) for p in
            (ROOT/'paf_ring_detection/data.py',ROOT/'paf_ring_detection/methods/cnn.py',
             ROOT/'paf_ring_detection/methods/cnn_region.py',Path(__file__).with_name('context.py'))}
        model_class=ContextUNet if 'context' in variant else TinyUNet
        if variant.startswith(('annular','structure','boundary')):
            raise ValueError('この推論器は通常の完全穴領域を出力するモデル専用です')
        for seed in seeds:
            folder=OUT/'runs'/variant/str(seed)
            complete=read_json(folder/'complete.json')
            path=folder/'best.pt'
            assert complete['checkpoint_sha256']==sha(path)
            model=model_class(16).cuda().eval()
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            assert checkpoint['provenance']==read_json(folder/'provenance.json')
            if 'context_implementation_sha256' in checkpoint['provenance']['config']:
                assert checkpoint['provenance']['config']['context_implementation_sha256']==sha(Path(__file__).with_name('context.py'))
            model.load_state_dict(checkpoint['model_state'])
            self.models.append(model)
            self.provenance['checkpoints'].append({'path':str(path),'sha256':sha(path),
                                                  'selected_epoch':checkpoint['epoch']})

    def predict(self,image):
        """正方形のBGR/グレー画像を受け、元画像座標の楕円と256px確率を返す。"""
        if image.shape[0]!=image.shape[1]:
            raise ValueError('入力は正方形画像にしてください。非等方の自動リサイズは行いません')
        if image.dtype!=np.uint8:
            raise ValueError('入力画像の型はuint8、輝度範囲は0〜255にしてください')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        gray=cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
        if np.ptp(gray)==0:
            # 一様画像は対象なしとは限らないため、「情報不足」として区別する。
            return {'ellipse':None,'probability':None,'status':'insufficient_image_evidence'}
        x=torch.from_numpy(np.repeat(gray[None,None],3,axis=1).copy()).float().cuda()/127.5-1
        with torch.inference_mode():
            probability=torch.stack([torch.sigmoid(model(x))[0,0] for model in self.models]).mean(0).cpu().numpy()
        ellipse=region_ellipse(probability)
        if ellipse is not None:
            ellipse=scaled_ellipse(ellipse,image.shape[0]/256)
        return {'ellipse':ellipse,'probability':probability,
                'status':'detected' if ellipse is not None else 'no_supported_prediction'}


def evaluate(detector):
    """同じ全test集合で確率平均と保留を評価し、単体モデルの記録は保持する。"""
    suffix='_'.join(str(s) for s in detector.seeds)
    folder=OUT/'detectors'/f'{detector.variant}_{suffix}'
    settings=folder/'provenance.json'
    if settings.exists():
        assert read_json(settings)==detector.provenance
    else:
        write_json(settings,detector.provenance)
    summaries=[]
    for source in ('1194m','diversified'):
        destination=folder/f'test_{source}.json'
        rows=[]
        for sample in load_samples(BASE/'datasets'/source,'test'):
            image,truth=image_and_truth(BASE/'datasets'/source,sample)
            result=detector.predict(image)
            ellipse=result['ellipse']
            iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
            truth_mask=region_mask(256,truth)>.5
            binary=result['probability']>=.5 if result['probability'] is not None else np.zeros((256,256),bool)
            rows.append({'sample_id':sample['sample_id'],'group_id':sample['group_id'],
                'degradation':sample['conditions']['degradation'],'severity':sample['conditions']['severity'],
                'ellipse':ellipse,'truth_ellipse':truth,'ellipse_iou':iou,'success':iou>=.8,
                'status':result['status'],'mask_iou':float((binary&truth_mask).sum()/max(1,(binary|truth_mask).sum()))})
        write_json(destination,{'rows':rows,'provenance':detector.provenance,'dataset':source})
        for effect in ('all','clean','black_rectangle','sensor_whiteout','sensor_black_crush'):
            subset=rows if effect=='all' else [r for r in rows if r['degradation']==effect]
            levels=[None] if effect in ('all','clean') else [None,.25,.5,.75,1.]
            for level in levels:
                selected=subset if level is None else [r for r in subset if r['severity']==level]
                summaries.append({'dataset':source,'degradation':effect,'severity':level,'n_images':len(selected),
                    'success_rate':float(np.mean([r['success'] for r in selected])),
                    'mean_iou':float(np.mean([r['ellipse_iou'] for r in selected])),
                    'abstained_uniform':sum(r['status']=='insufficient_image_evidence' for r in selected)})
        print(source,'success',sum(r['success'] for r in rows),len(rows),flush=True)
    write_json(folder/'summary.json',{'rows':summaries,'n_models':len(detector.models),'provenance':detector.provenance,
        'interpretation':'学習seed別成功率の平均とは異なり、画像ごとにモデルの確率を平均してから一つの楕円を推定する。完全一様画像は情報不足として未検出にする。'})


def predict_file(detector,path,destination):
    """入力を保存し直さず、推定・確率・重ね描き画像を指定先へ出力する。"""
    image=read_image(path)
    if image is None:
        raise ValueError(f'画像を読み込めません: {path}')
    result=detector.predict(image)
    overlay=image.copy()
    if result['ellipse']:
        cv2.ellipse(overlay,result['ellipse'],(255,0,255),2,cv2.LINE_AA)
    write_json(destination,{'image':str(path.resolve()),'image_sha256':sha(path),'ellipse':result['ellipse'],
        'status':result['status'],'provenance':detector.provenance})
    write_image(destination.with_name(destination.stem+'_overlay.png'),overlay)
    if result['probability'] is not None:
        write_image(destination.with_name(destination.stem+'_probability.png'),np.uint8(result['probability']*255))


if __name__=='__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['evaluate','predict'])
    parser.add_argument('--variant',default='surface_context_spatial')
    parser.add_argument('--seeds',type=int,nargs='+',default=[20260907,20260908,20260909])
    parser.add_argument('--image',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    detector=PAFDetector(args.variant,args.seeds)
    if args.stage=='evaluate':
        evaluate(detector)
    else:
        if args.image is None or args.output is None:
            parser.error('predictには--imageと--outputが必要です')
        predict_file(detector,args.image,args.output)
