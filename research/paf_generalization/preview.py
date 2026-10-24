"""成否で選ばない固定画像で、学習入力と内周推定の根拠を可視化する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import load_samples, read_json, write_json, RingDataset
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.context import ContextUNet
from research.paf_generalization.structure import StructureUNet
from research.paf_generalization.annular import AnnularUNet
from research.paf_generalization.experiment import spatial_augment
from research.paf_generalization.diagnose import BASE, OUT, sha, image_and_truth
from research.paf_generalization.detector import PAFDetector


def font(size=15):
    return ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc', size)


def overlay(image, truth, prediction):
    image = image.copy()
    cv2.ellipse(image, truth, (0,255,0), 1, cv2.LINE_AA)
    if prediction:
        ellipse = (tuple(prediction[0]),tuple(prediction[1]),prediction[2])
        cv2.ellipse(image, ellipse, (255,0,255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def predictions(variant, seed):
    """固定視点ごとのtest順先頭を使い、成否で例を選ばない。"""
    folder = OUT/'runs'/variant/str(seed)
    detector=None
    if variant=='surface_ensemble':
        detector=PAFDetector()
        folder=OUT/'detectors/surface_context_spatial_20260907_20260908_20260909'
    all_samples = load_samples(BASE/'datasets/1194m','test')
    selected = []
    for tilt in (20,40,60,75):
        sample = next(s for s in all_samples if s['conditions']['degradation']=='clean'
                      and s['conditions']['camera']['tilt_deg']==tilt)
        selected.append(('1194m',sample))
    selected.append(('diversified',next(s for s in load_samples(BASE/'datasets/diversified','test')
                                       if s['conditions']['degradation']=='clean')))
    model_class = (AnnularUNet if variant.startswith('annular') else StructureUNet if variant.startswith('structure')
                   else ContextUNet if 'context' in variant else TinyUNet)
    if detector is None:
        model = model_class(16).cuda().eval()
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model_state'])
    width, height = 4*256, 48+len(selected)*300
    canvas = Image.new('RGB',(width,height),'white')
    draw=ImageDraw.Draw(canvas)
    probability_title='新CNNの境界確率' if variant.startswith('boundary') else '新CNNの穴確率'
    for i,title in enumerate(('入力','既存CNN（緑：正解）','新CNN（紫：推定）',probability_title)):
        draw.text((i*256+4,8),title,fill='black',font=font(14))
    saved_rows=[]
    for index,(source,sample) in enumerate(selected):
        image,truth=image_and_truth(BASE/'datasets'/source,sample)
        old={r['sample_id']:r for r in read_json(BASE/f'results/{source}/cnn_diversified_{seed}.json')['rows']}[sample['sample_id']]
        new={r['sample_id']:r for r in read_json(folder/f'test_{source}.json')['rows']}[sample['sample_id']]
        x=torch.from_numpy(image.transpose(2,0,1).copy()).float().cuda()[None]/127.5-1
        if detector is not None:
            result=detector.predict(image)
            probability=result['probability']
            assert np.allclose(np.array([*result['ellipse'][0],*result['ellipse'][1],result['ellipse'][2]]),
                               np.array([*new['ellipse'][0],*new['ellipse'][1],new['ellipse'][2]]))
        else:
            with torch.inference_mode():
                probability=torch.sigmoid(model(x))[0,0].cpu().numpy()
        panels=[Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),overlay(image,truth,old['ellipse']),
                overlay(image,truth,new['ellipse']),Image.fromarray(np.uint8(np.clip(probability*255,0,255))).convert('RGB')]
        y=48+index*300
        for i,panel in enumerate(panels):
            canvas.paste(panel,(i*256,y))
        draw.text((4,y+257),f"{source}: 既存 IoU={old['ellipse_iou']:.3f} → {variant} IoU={new['ellipse_iou']:.3f}",fill='black',font=font(14))
        draw.text((4,y+277),sample['sample_id'],fill='black',font=font(11))
        saved_rows.append({'source':source,'sample_id':sample['sample_id'],
                           'baseline_iou':old['ellipse_iou'],'new_iou':new['ellipse_iou']})
    path=OUT/f'preview_{variant}_{seed}.png'
    canvas.save(path)
    hashes=[r['sha256'] for r in detector.provenance['checkpoints']] if detector else [sha(folder/'best.pt')]
    write_json(path.with_suffix('.json'),{'variant':variant,'baseline_seed':seed,'checkpoint_sha256':hashes,
                                        'selection':'固定視点20/40/60/75度のtest順先頭と多様形状の通常test順先頭', 'rows':saved_rows})


def training():
    """変換後も画像と完全穴教師が一致することを、固定例で確認する。"""
    dataset=RingDataset(BASE/'datasets/diversified','train',256)
    indices=[i for i,s in enumerate(dataset.samples) if s['conditions']['degradation']=='clean'][:6]
    images=torch.stack([dataset[i][0] for i in indices]).cuda()
    masks=torch.stack([dataset[i][1] for i in indices]).cuda()
    config=read_json(Path(__file__).with_name('experiment.json'))
    config['spatial_probability']=1
    transformed,targets=spatial_augment(images,masks,np.random.default_rng(20261025),config)
    canvas=Image.new('RGB',(4*256,48+6*280),'white')
    draw=ImageDraw.Draw(canvas)
    for i,title in enumerate(('元の学習入力','元の完全穴教師','空間変換後の入力','変換後の完全穴教師')):
        draw.text((i*256+4,8),title,fill='black',font=font(14))
    for row,index in enumerate(indices):
        for col,tensor in enumerate((images[row],masks[row]*2-1,transformed[row],targets[row]*2-1)):
            data=((tensor.detach().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
            data=data[0] if data.shape[0]==1 else data.transpose(1,2,0)
            canvas.paste(Image.fromarray(data).convert('RGB'),(col*256,48+row*280))
        draw.text((4,48+row*280+257),dataset.samples[index]['sample_id'],fill='black',font=font(12))
    canvas.save(OUT/'training_spatial.png')


if __name__=='__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant')
    parser.add_argument('--seed',type=int,default=20260907)
    args=parser.parse_args()
    training()
    if args.variant:
        predictions(args.variant,args.seed)
