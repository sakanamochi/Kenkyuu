"""遮蔽学習で残る内周の量を、投影弧長により測定する。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse
from research.paf_generalization.diagnose import BASE, sha
from research.paf_generalization_v2.occlusion_sampling import cut_location


def rim_segments(ellipse):
    """720分割した投影弧の中点xと弧長を求める。画像の明暗は判定しない。"""
    (cx, cy), (a, b), angle = ellipse
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    radians = np.deg2rad(angle)
    x = cx+a/2*np.cos(theta)*np.cos(radians)-b/2*np.sin(theta)*np.sin(radians)
    y = cy+a/2*np.cos(theta)*np.sin(radians)+b/2*np.sin(theta)*np.cos(radians)
    return (x+np.roll(x, -1))/2, np.hypot(np.roll(x, -1)-x, np.roll(y, -1)-y)


def visible_fraction(midpoint, lengths, cut, left):
    visible = midpoint >= cut if left else midpoint < cut
    return float(lengths[visible].sum()/lengths.sum())


def main():
    roots = {'original': BASE/'datasets/diversified',
             'native': ROOT/'output/paf_generalization_v2/native_framing/native_plain',
             'projection': ROOT/'output/paf_generalization_v2/projection_coverage/native_plain'}
    rows = []
    rng = np.random.default_rng(2026102519)
    for name, directory in roots.items():
        for sample in load_samples(directory, 'train'):
            if sample['conditions']['degradation'] != 'black_rectangle':
                continue
            label = read_json(directory/sample['label'])
            ellipse = label_ellipse(label)
            midpoint, lengths = rim_segments(ellipse)
            effect = sample['augmentation']
            width = label['image_width']
            left = bool(np.random.default_rng(effect['seed']).integers(0, 2) == 0)
            patch = max(1, round(width*effect['severity']))
            cut = patch if left else width-patch
            cases = [(name+'_fixed', 0, cut, left)]
            if name == 'original':
                # 二つのオンライン方式には同じ方向と一様乱数を割り当てる。
                for repeat in range(20):
                    draw = float(rng.random())
                    side = bool(rng.integers(0, 2) == 0)
                    for mode in ('image', 'relative'):
                        cut = cut_location(ellipse, width, draw, mode)
                        if mode == 'image' and not side:
                            cut = width-cut
                        cases.append((mode+'_online', repeat, cut, side))
            for method, repeat, cut, left in cases:
                rows.append({'method': method, 'sample_id': sample['sample_id'], 'repeat': repeat,
                             'cut_x': cut, 'left_side': left,
                             'remaining_projected_rim_fraction': visible_fraction(midpoint, lengths, cut, left)})
    summary = []
    for method in sorted({r['method'] for r in rows}):
        values = np.array([r['remaining_projected_rim_fraction'] for r in rows if r['method'] == method])
        summary.append({'method': method, 'n': len(values), 'none': int(sum(values == 0)),
                        'full': int(sum(values == 1)), 'partial': int(sum((values > 0)&(values < 1))),
                        'remaining_25_to_75_percent': int(sum((values >= .25)&(values <= .75))),
                        'mean_remaining': float(values.mean())})
    write_json(ROOT/'output/paf_generalization_v2/occlusion_coverage.json', {
        'implementation_sha256': sha(Path(__file__)),
        'sampling_sha256': sha(Path(__file__).with_name('occlusion_sampling.py')),
        'manifests': {name: sha(root/'manifest.json') for name, root in roots.items()},
        'seed': 2026102519, 'rows': rows, 'summary': summary,
        'interpretation': 'source trainの投影内周が矩形の外へ残る割合。自己遮蔽・明暗・本体の残存情報は含まない。オンライン方式は20回抽選の分布診断で、実学習の乱数列とは別。'})
    for row in summary:
        print(row, flush=True)


if __name__ == '__main__':
    main()
