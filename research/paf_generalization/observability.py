"""受容野と、矩形遮蔽後に残る正解境界の量を診断する。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import load_samples, read_json, write_json, label_ellipse
from research.paf_generalization.diagnose import BASE, OUT


def dependency_interval(pixel, dilation):
    """最深経路を逆向きに辿り、推論時の1次元依存区間の上限を得る。"""
    lo, hi = pixel-2, pixel+2
    # 各decoderの3x3畳み込み2回と、重複しない2x2転置畳み込み。
    lo, hi = lo//2-2, hi//2+2
    lo, hi = lo//2-2, hi//2+2
    lo, hi = lo//2-2*dilation, hi//2+2*dilation
    # encoderの2x2poolと3x3畳み込み2回。
    for _ in range(3):
        lo, hi = 2*lo-2, 2*hi+3
    return {'output_pixel':pixel,'left':lo,'right':hi,'span':hi-lo+1}


def main():
    write_json(OUT/'receptive_field.json', {
        'assumption':'BatchNormはeval時の固定統計。画像外paddingを含む理論上限。実効受容野ではない。',
        'baseline':[dependency_interval(x,1) for x in range(128,136)],
        'context':[dependency_interval(x,4) for x in range(128,136)]})
    records=[]
    for source in ('diversified','1194m'):
        folder=BASE/'datasets'/source
        for split in ('train','validation','test'):
            for sample in load_samples(folder,split):
                if sample['conditions']['degradation']!='black_rectangle':
                    continue
                label=read_json(folder/sample['label'])
                (cx,cy),(a,b),angle=label_ellipse(label)
                theta=np.arange(720)*2*np.pi/720
                alpha=np.deg2rad(angle)
                x=cx+a/2*np.cos(theta)*np.cos(alpha)-b/2*np.sin(theta)*np.sin(alpha)
                y=cy+a/2*np.cos(theta)*np.sin(alpha)+b/2*np.sin(theta)*np.cos(alpha)
                # 投影後の弧長で重み付けし、円の方位角の割合との混同を避ける。
                lengths=np.hypot(np.roll(x,-1)-x,np.roll(y,-1)-y)
                midx=(x+np.roll(x,-1))/2
                augmentation=sample['augmentation']
                patch_width=max(1,round(label['image_width']*augmentation['severity']))
                left_side=int(np.random.default_rng(augmentation['seed']).integers(0,2))==0
                visible=midx>=patch_width if left_side else midx<label['image_width']-patch_width
                fraction=float(lengths[visible].sum()/lengths.sum())
                records.append({'dataset':source,'split':split,'sample_id':sample['sample_id'],
                    'severity':augmentation['severity'],'remaining_projected_rim_fraction':fraction,
                    'fully_covered_rim':fraction==0})
    summary=[]
    for source in ('diversified','1194m'):
        for split in ('train','validation','test'):
            rows=[r for r in records if r['dataset']==source and r['split']==split]
            summary.append({'dataset':source,'split':split,'n':len(rows),
                'fully_covered_rim_count':sum(r['fully_covered_rim'] for r in rows),
                'mean_remaining_fraction':float(np.mean([r['remaining_projected_rim_fraction'] for r in rows]))})
    write_json(OUT/'observability.json',{'rows':records,'summary':summary,
        'interpretation':'正解を使う事後診断。遮蔽矩形の外に残る投影境界の弧長であり、自己遮蔽・暗部や他の本体情報は評価しない。検出器の入力・選択には使用しない。'})
    print(summary)


if __name__=='__main__':
    main()
