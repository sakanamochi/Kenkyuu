"""同一形状・同一投影の材質介入で、既存モデルの判断が変わるかを測る。"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,load_samples,region_mask
from paf_ring_detection.methods.cnn import load_model
from paf_ring_detection.methods.cnn_region import region_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE,OUT,sha,image_and_truth,predict
MATERIAL=OUT/'material_counterfactual'


def verify_materials():
    """介入画像・元教師・投影一致を実ファイルから照合する。"""
    verification=read_json(MATERIAL/'verification.json')
    assert verification['manifest_sha256']==sha(MATERIAL/'manifest.json')
    plan=read_json(MATERIAL/'plan.json')
    assert plan['renderer_sha256']==sha(Path(__file__).with_name('render_materials.py'))
    manifest=read_json(MATERIAL/'manifest.json')
    checked=set()
    for sample in manifest['samples']:
        for key in ('image','label'):
            path=Path(sample[key])
            if path not in checked:
                assert sha(path)==sample[key+'_sha256']
                checked.add(path)
    for item in plan['plan']:
        label=read_json(MATERIAL/f"labels/{item['base_sample_id']}.json")
        assert label['projection_max_error_px']<1e-3


def main():
    torch.set_num_threads(4)
    verify_materials()
    originals=[s for s in load_samples(BASE/'datasets/diversified','validation') if s['conditions']['degradation']=='clean']
    interventions={s['sample_id']:s for s in load_samples(MATERIAL,'validation') if s['conditions']['degradation']=='clean'}
    rows,appearance=[],[]
    for seed in (20260907,20260908,20260909):
        model=load_model(BASE/f'models/diversified/{seed}/best.pt',torch.device('cuda'))
        for sample in originals:
            original,truth=image_and_truth(BASE/'datasets/diversified',sample)
            changed,changed_truth=image_and_truth(MATERIAL,interventions[sample['sample_id']])
            assert truth==changed_truth
            mask=region_mask(256,truth)>.5
            if seed==20260907:
                appearance.append({'sample_id':sample['sample_id'],
                    'original_interior_mean':float(original[:,:,0][mask].mean()),
                    'changed_interior_mean':float(changed[:,:,0][mask].mean()),
                    'mean_absolute_pixel_difference':float(np.abs(original.astype(float)-changed.astype(float)).mean())})
            result={'seed':seed,'sample_id':sample['sample_id']}
            for name,image in [('original',original),('intervention',changed)]:
                probability=predict(model,[image])[0]
                ellipse=region_ellipse(probability)
                iou=evaluate_ellipses(ellipse,truth,(256,256))['ellipse_iou'] if ellipse else 0.
                binary=probability>=.5
                result[name]={'ellipse':ellipse,'iou':iou,'success':iou>=.8,
                              'mask_iou':float((binary&mask).sum()/max(1,(binary|mask).sum()))}
            rows.append(result)
    summary={name:{'success_rate':float(np.mean([r[name]['success'] for r in rows])),
                   'mean_iou':float(np.mean([r[name]['iou'] for r in rows])),
                   'mean_mask_iou':float(np.mean([r[name]['mask_iou'] for r in rows]))}
             for name in ('original','intervention')}
    write_json(OUT/'counterfactual_validation.json',{'rows':rows,'appearance':appearance,'summary':summary,
        'n_shapes':len(originals),'n_seeds':3,'split':'diversified_validation_clean',
        'material_manifest_sha256':sha(MATERIAL/'manifest.json'),
        'interpretation':'105同一形状の材質介入。既存モデルの外観感度を測る診断であり、1194Mの失敗原因全体を確定しない。'})
    print(summary)


if __name__=='__main__':
    main()
