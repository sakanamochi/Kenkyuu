"""同じ投影・劣化で、補間拡大とカメラ再描画の部分境界推定を比較する。"""

import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,read_image,load_samples,label_ellipse,scaled_ellipse
from paf_ring_detection.effects import apply_effect
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_generalization.diagnose import BASE,OUT as OLD,sha,image_and_truth,predict
from research.paf_generalization_v2.fitting import load


def main():
    torch.set_num_threads(2)
    model,protocol=load()
    data=OLD/'framing_counterfactual'
    verification=read_json(data/'verification.json')
    assert verification['manifest_sha256']==sha(data/'manifest.json')
    originals={s['sample_id']:s for s in load_samples(BASE/'datasets/diversified','validation')
               if s['conditions']['degradation']=='clean'}
    effects=[('clean',0.),('black_rectangle',.5),('black_rectangle',.75),
             ('sensor_whiteout',.5),('sensor_black_crush',.5)]
    rows=[]
    for index,sample in enumerate(load_samples(data,'validation')):
        sid=sample['sample_id']; original=originals[sid]
        for field in ('image','label'):
            assert sha(data/sample[field])==sample[field+'_sha256']
        image,truth=image_and_truth(BASE/'datasets/diversified',original)
        label=read_json(data/sample['label'])
        native_truth=scaled_ellipse(label_ellipse(label),256/480)
        scale=label['framing_scale']
        expected=((128.,128.),tuple(scale*x for x in truth[1]),truth[2])
        assert evaluate_ellipses(expected,native_truth,(256,256))['ellipse_iou']>.999
        factor=scale*480/256
        transform=np.array([[factor,0,240-factor*truth[0][0]],[0,factor,240-factor*truth[0][1]]],np.float32)
        interpolated=cv2.warpAffine(image,transform,(480,480))
        native=read_image(data/sample['image'])
        for effect,severity in effects:
            seed=zlib.crc32(f'{sid}:{effect}:{severity}:20261025'.encode())
            for kind,input_image in [('interpolated',interpolated),('rendered',native)]:
                changed=apply_effect(input_image,effect,severity,np.random.default_rng(seed))
                gray=cv2.cvtColor(changed,cv2.COLOR_BGR2GRAY)
                x=cv2.resize(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),(256,256),interpolation=cv2.INTER_AREA)
                probability=predict(model,[x])[0]
                ellipse=boundary_ellipse(probability,protocol['ransac'])
                iou=evaluate_ellipses(ellipse,native_truth,(256,256))['ellipse_iou'] if ellipse else 0.
                rows.append({'sample_id':sid,'kind':kind,'degradation':effect,'severity':severity,
                    'ellipse':ellipse,'truth':native_truth,'ellipse_iou':iou,'success':iou>=.8,'effect_seed':seed})
        if (index+1)%21==0:
            print('paired resolution',index+1,105,flush=True)
    summary=[]
    for effect,severity in effects:
        for kind in ('interpolated','rendered'):
            selected=[r for r in rows if (r['degradation'],r['severity'],r['kind'])==(effect,severity,kind)]
            summary.append({'degradation':effect,'severity':severity,'kind':kind,'n_images':len(selected),
                            'success_rate':float(np.mean([r['success'] for r in selected]))})
    destination=ROOT/'output/paf_generalization_v2/partial_resolution.json'
    write_json(destination,{'rows':rows,'summary':summary,'protocol':protocol,'implementation_sha256':sha(Path(__file__)),
        'manifest_sha256':sha(data/'manifest.json'),
        'interpretation':'source validationの同じ投影へ480pxで同じ劣化を適用後256px化。補間側は元の256px画像から拡大。再描画ではレンダラの画面依存処理も変わるため純粋な解像度の単独効果とは断定しない。'})
    for row in summary:
        print(row['degradation'],row['severity'],row['kind'],round(row['success_rate']*100,2))


if __name__=='__main__':
    main()
