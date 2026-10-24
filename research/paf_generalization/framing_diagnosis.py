"""同じ拡大投影を持つ画像について、画像の補間拡大と再描画を比較する。"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,load_samples
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.methods.cnn_region import region_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE,OUT,sha,image_and_truth,predict
from research.paf_generalization.structure_diagnosis import enlarged
from research.paf_generalization.context import ContextUNet


def main():
    torch.set_num_threads(3)
    changed=OUT/'framing_counterfactual'
    check=read_json(changed/'verification.json')
    assert check['manifest_sha256']==sha(changed/'manifest.json')
    assert check['plan_sha256']==sha(changed/'plan.json')
    assert read_json(changed/'plan.json')['renderer_sha256']==sha(Path(__file__).with_name('render_framing.py'))
    originals=[s for s in load_samples(BASE/'datasets/diversified','validation') if s['conditions']['degradation']=='clean']
    rendered=load_samples(changed,'validation')
    assert [s['sample_id'] for s in originals]==[s['sample_id'] for s in rendered]
    pairs=[]
    for original,sample in zip(originals,rendered):
        for field in ('image','label'):
            assert sha(changed/sample[field])==sample[field+'_sha256']
        x,y=image_and_truth(BASE/'datasets/diversified',original)
        enlarged_image,enlarged_truth,scale=enlarged(x,y)
        native_image,native_truth=image_and_truth(changed,sample)
        # 中心・軸長の一致を独立したラベル復元で確認する。角度は180度周期。
        assert np.allclose(enlarged_truth[0],native_truth[0],atol=.01)
        assert np.allclose(enlarged_truth[1],native_truth[1],atol=.01)
        assert evaluate_ellipses(enlarged_truth,native_truth,(256,256))['ellipse_iou']>.999
        pairs.append((sample['sample_id'],enlarged_truth,enlarged_image,native_image,scale))
    models=[('baseline',s,BASE/f'models/diversified/{s}/best.pt',TinyUNet) for s in (20260907,20260908,20260909)]
    for name,cls in [('spatial',TinyUNet),('context_spatial',ContextUNet),('interior_context_spatial',ContextUNet),
                     ('surface_context_spatial',ContextUNet),('affine_context',ContextUNet)]:
        path=OUT/f'runs/{name}/20260907/best.pt'
        if (path.parent/'complete.json').exists():
            models.append((name,20260907,path,cls))
    rows=[]
    for name,seed,path,cls in models:
        model=cls(16).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model_state'])
        for sid,truth,interpolated,native,scale in pairs:
            for condition,image in [('interpolated',interpolated),('rendered',native)]:
                ellipse=region_ellipse(predict(model,[image])[0])
                iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
                rows.append({'variant':name,'seed':seed,'sample_id':sid,'condition':condition,
                    'ellipse':ellipse,'truth':truth,'iou':iou,'success':iou>=.8,'scale':scale})
    summary=[]
    for name in dict.fromkeys(r['variant'] for r in rows):
        for condition in ('interpolated','rendered'):
            subset=[r for r in rows if r['variant']==name and r['condition']==condition]
            summary.append({'variant':name,'condition':condition,'n_predictions':len(subset),
                'success_rate':float(np.mean([r['success'] for r in subset])),
                'mean_iou':float(np.mean([r['iou'] for r in subset]))})
    write_json(OUT/'framing_diagnosis.json',{'rows':rows,'summary':summary,'n_shapes':len(pairs),
        'models':[{'variant':n,'seed':s,'sha256':sha(p)} for n,s,p,c in models],
        'manifest_sha256':sha(changed/'manifest.json'),
        'interpretation':'同じ投影の補間拡大とカメラ再描画。レンダラの画面依存処理も変わるため、画素解像度単独の因果効果とは断定しない。'})
    print(summary)


if __name__=='__main__':
    main()
