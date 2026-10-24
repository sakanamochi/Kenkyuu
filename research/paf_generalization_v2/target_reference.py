"""多様形状学習候補を1194M学習CNNと同じ画像・条件・3seedで比較する。"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE, sha
from research.paf_generalization.report import paired_interval

OUT = ROOT/'output/paf_generalization_v2'
SEEDS = (20260907, 20260908, 20260909)


def compare(variant):
    """情報消失例も除外せず、全条件を明示して参照CNNとの差を保存する。"""
    data = BASE/'datasets/1194m'
    samples = {s['sample_id']: s for s in load_samples(data, 'test')}
    truths = {}
    for sid, sample in samples.items():
        label = read_json(data/sample['label'])
        truths[sid] = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    records = {'candidate': [], 'reference': []}
    files = {str((data/'manifest.json').relative_to(ROOT)): sha(data/'manifest.json')}
    for seed in SEEDS:
        paths = {'candidate': OUT/f'runs/{variant}/{seed}/test_1194m.json',
                 'reference': BASE/f'results/1194m/cnn_1194m_{seed}.json'}
        for name, path in paths.items():
            saved = read_json(path)
            rows = saved['rows']
            assert len(rows) == len(samples) and {r['sample_id'] for r in rows} == set(samples)
            files[str(path.relative_to(ROOT))] = sha(path)
            for row in rows:
                sample = samples[row['sample_id']]
                assert (row['degradation'], row['severity']) == (sample['conditions']['degradation'], sample['conditions']['severity'])
                e = row['ellipse']
                ellipse = (tuple(e[0]), tuple(e[1]), e[2]) if e else None
                iou = evaluate_ellipses(ellipse, truths[row['sample_id']], (256, 256))['ellipse_iou'] if ellipse else 0.
                assert np.isclose(row['ellipse_iou'], iou, atol=1e-10, rtol=0)
                assert row['success'] == (iou >= .8)
                records[name].append({**row, 'seed': seed, 'group_id': sample['group_id']})
    conditions = [('all', None)]+sorted({(r['degradation'], r['severity']) for r in records['candidate']})
    summaries = []
    for effect, severity in conditions:
        selected = {name: [r for r in rows if effect == 'all' or (r['degradation'], r['severity']) == (effect, severity)]
                    for name, rows in records.items()}
        rates = {name: float(np.mean([r['success'] for r in rows])) for name, rows in selected.items()}
        interval = paired_interval(selected['candidate'], selected['reference'])
        # 区間はカメラ群の再標本化であり、探索全体の多重性やseed母集団を含まない。
        interval['uncertainty_scope'] = '固定3seedに対するカメラ群の再標本化。seed母集団・探索の多重性は含まない。'
        summaries.append({'degradation': effect, 'severity': severity,
                          'n_images': len(selected['candidate'])//len(SEEDS), **rates, **interval})
    result = {'variant': variant, 'reference': 'cnn_1194m', 'seeds': list(SEEDS),
              'criterion': '情報が残る条件で1194M学習CNNに近い性能を基準にする。',
              'scope': '既存testを使った探索比較。情報量や参照CNNの成否で画像を選別しない。近さの許容幅は未指定。',
              'success_iou': .8, 'prediction_checks': sum(map(len, records.values())),
              'files': files, 'rows': summaries}
    write_json(OUT/f'target_reference/{variant}.json', result)
    for r in summaries:
        print(r['degradation'], r['severity'], 'reference', round(100*r['reference'], 3),
              'candidate', round(100*r['candidate'], 3), 'delta_pp', round(100*r['delta'], 3), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='boundary_projection_relative')
    compare(parser.parse_args().variant)
