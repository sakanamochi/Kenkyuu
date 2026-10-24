"""一様画像での偶然の正解を、画像に残る証拠に基づく検出と区別する。"""

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json,write_json,load_samples
from research.paf_generalization.diagnose import BASE,OUT,image_and_truth,sha


def main():
    constants={}
    for source in ('1194m','diversified'):
        constants[source]={}
        folder=BASE/'datasets'/source
        for sample in load_samples(folder,'test'):
            image,_=image_and_truth(folder,sample)
            if np.ptp(image)==0:
                constants[source][sample['sample_id']]=int(image[0,0,0])
    summaries=[]
    for source in ('1194m','diversified'):
        files=list((BASE/'results'/source).glob('*.json'))+list((OUT/'runs').glob(f'*/*/test_{source}.json'))
        for path in files:
            data=read_json(path)
            if 'rows' not in data:
                continue
            rows=data['rows']
            no_evidence=[r for r in rows if r['sample_id'] in constants[source]]
            guessed=sum(bool(r['success']) for r in no_evidence)
            total=sum(bool(r['success']) for r in rows)
            summaries.append({'dataset':source,'variant':data.get('variant',data.get('method')),'seed':data['seed'],
                'n_images':len(rows),'n_uniform':len(no_evidence),'uniform_successes':guessed,
                'uniform_detections':sum(r['ellipse'] is not None for r in no_evidence),
                'raw_success_rate':total/len(rows),'reject_uniform_success_rate':(total-guessed)/len(rows),
                'predictions_sha256':sha(path)})
    write_json(OUT/'evidence_audit.json',{'constant_images':constants,'rows':summaries,
        'interpretation':'一様入力では未知の楕円を画像から同定できない。同じ全画像分母を保ち、一様入力を未検出とする場合の成績を併記する。正式な保存済み推定は変更しない。'})
    print('uniform images',{k:len(v) for k,v in constants.items()},'evaluated models',len(summaries))


if __name__=='__main__':
    main()
