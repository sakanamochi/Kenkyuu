"""同じ上端環に対する胴体・底の介入を、対応付きのvalidation画像で診断する。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,load_samples
from paf_ring_detection.methods.cnn import load_model,TinyUNet
from paf_ring_detection.methods.cnn_region import region_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE,OUT,sha,image_and_truth,predict
from research.paf_generalization.context import ContextUNet
from research.paf_generalization.annular import AnnularUNet
from research.paf_generalization.boundary import boundary_ellipse


def enlarged(image,truth):
    """正解を用いた3倍拡大診断。画面内に楕円を残し、全構造へ同一変換を使う。"""
    scale=min(3.,224/max(truth[1]))
    cx,cy=truth[0]
    matrix=np.array([[scale,0,128-scale*cx],[0,scale,128-scale*cy]],dtype=np.float32)
    image=cv2.warpAffine(image,matrix,(256,256),flags=cv2.INTER_LINEAR)
    return image,((128.,128.),tuple(scale*x for x in truth[1]),truth[2]),scale


def main():
    torch.set_num_threads(3)
    root=OUT/'structure_counterfactual'
    verification=read_json(root/'verification.json')
    assert verification['plan_sha256']==sha(root/'plan.json')
    plan=read_json(root/'plan.json')
    assert plan['renderer_sha256']==sha(Path(__file__).with_name('render_structure.py'))
    folders={'original':BASE/'datasets/diversified',**{v:root/v for v in plan['variants']}}
    for factor in ('interior','surface'):
        changed=OUT/f'{factor}_counterfactual'
        if (changed/'verification.json').exists():
            verified=read_json(changed/'verification.json')
            assert verified['manifest_sha256']==sha(changed/'manifest.json')
            assert verified['plan_sha256']==sha(changed/'plan.json')
            assert read_json(changed/'plan.json')['renderer_sha256']==sha(Path(__file__).with_name(f'render_{factor}.py'))
            folders[f'{factor}_clutter']=changed
    datasets={name:[s for s in load_samples(folder,'validation') if s['conditions']['degradation']=='clean']
              for name,folder in folders.items()}
    for name,rows in datasets.items():
        assert [s['sample_id'] for s in rows]==[s['sample_id'] for s in datasets['original']]
        for sample in rows:
            for field in ('image','label'):
                assert sha(folders[name]/sample[field])==sample[field+'_sha256']
    images={name:[image_and_truth(folders[name],s) for s in rows] for name,rows in datasets.items()}
    models=[('baseline',seed,BASE/f'models/diversified/{seed}/best.pt',TinyUNet)
            for seed in (20260907,20260908,20260909)]
    for name,cls in [('spatial',TinyUNet),('context_spatial',ContextUNet),('annular_context_spatial',AnnularUNet),
                     ('interior_context_spatial',ContextUNet),('surface_context_spatial',ContextUNet),
                     ('boundary_spatial',TinyUNet),('boundary_surface',TinyUNet)]:
        seeds=(20260907,20260908,20260909) if name in ('surface_context_spatial','boundary_surface') else (20260907,)
        for seed in seeds:
            path=OUT/f'runs/{name}/{seed}/best.pt'
            if (path.parent/'complete.json').exists():
                models.append((name,seed,path,cls))
    results,weights=[],[]
    for name,seed,path,cls in models:
        model=cls(16).to('cuda')
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model_state'])
        model.eval()
        weights.append({'variant':name,'seed':seed,'path':str(path),'sha256':sha(path)})
        for body,pairs in images.items():
            for index,(image,truth) in enumerate(pairs):
                assert truth==images['original'][index][1]
                for framing in ('native','enlarged'):
                    x,y,scale=enlarged(image,truth) if framing=='enlarged' else (image,truth,1.)
                    probability=predict(model,[x])[0]
                    ellipse=boundary_ellipse(probability) if name.startswith('boundary') else region_ellipse(probability)
                    iou=evaluate_ellipses(ellipse,y,(256,256))['ellipse_iou'] if ellipse else 0.
                    results.append({'variant':name,'seed':seed,'body':body,'framing':framing,'scale':scale,
                        'sample_id':datasets['original'][index]['sample_id'],
                        'ellipse':ellipse,'truth':y,'iou':iou,'success':iou>=.8})
        print('DIAGNOSED',name,seed,flush=True)
    summary=[]
    for name in dict.fromkeys(r['variant'] for r in results):
        for framing in ('native','enlarged'):
            for body in folders:
                rows=[r for r in results if (r['variant'],r['framing'],r['body'])==(name,framing,body)]
                summary.append({'variant':name,'framing':framing,'body':body,'n_predictions':len(rows),
                    'success_rate':float(np.mean([r['success'] for r in rows])),
                    'mean_iou':float(np.mean([r['iou'] for r in rows]))})
    write_json(OUT/'structure_diagnosis.json',{'summary':summary,'rows':results,'models':weights,
        'n_shapes':len(datasets['original']),'split':'validation_clean',
        'dataset_manifests':{str(p/'manifest.json'):sha(p/'manifest.json') for p in folders.values()},
        'verification_sha256':sha(root/'verification.json'),
        'interpretation':'上端内外周・材質・カメラ・照明を固定し底/胴体を変える診断。拡大は正解を使用する感度診断であり、検出器の推論方式ではない。'})
    print(summary,flush=True)


if __name__=='__main__':
    main()
