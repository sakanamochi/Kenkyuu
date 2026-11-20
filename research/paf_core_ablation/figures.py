"""固定した一形状の対応学習画像を並べ、投影と外観の変更を図示する。"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, label_ellipse, scaled_ellipse
from research.paf_generalization.diagnose import sha
from research.paf_core_ablation.experiment import FactorialDataset, CONFIG, OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['main', 'boundary'], default='main')
    args = parser.parse_args()
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    assert font.exists()
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 12, 'svg.fonttype': 'path'})
    config = read_json(CONFIG)
    data = FactorialDataset(config['variants']['main_projection_surface'], 'train', 20260907, cache=False)
    index = next(i for i, sample in enumerate(data.samples) if sample['conditions']['degradation'] == 'clean')
    views = [((False, False), '元の構図・元の表面'), ((False, True), '表面模様のみ変更'),
             ((True, False), '投影のみ変更'), ((True, True), '投影＋表面模様')]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    evidence = []
    for ax, (key, title) in zip(axes, views):
        image, target, _ = data.get_view(key, index)
        sample = data.views[key][index]
        root = data.roots[key]
        label = read_json(root/sample['label'])
        center, sizes, angle = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
        ax.imshow((image[0].numpy()+1)/2, cmap='gray', vmin=0, vmax=1)
        ax.add_patch(Ellipse(center, sizes[0], sizes[1], angle=angle, fill=False, color='#ff63da', linewidth=1.1))
        ax.set_title(title, pad=12)
        ax.set_xlabel(f'内周長径 {max(sizes):.1f}px / 軸比 {min(sizes)/max(sizes):.2f}', labelpad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        evidence.append({'view': list(key), 'sample_id': sample['sample_id'],
                         'image_sha256': sha(root/sample['image']), 'label_sha256': sha(root/sample['label']),
                         'ellipse': [center, sizes, angle]})
    figure.suptitle('同じ合成形状で、投影条件と表面模様を別々に変える', fontsize=17, y=.97)
    teacher = '完全リング教師' if args.study == 'main' else '可視境界教師'
    figure.text(.5, .055, f'紫線は正解内周の説明用表示。CNN・{teacher}・RANSACは全条件で共通。', ha='center', fontsize=11)
    figure.subplots_adjust(left=.025, right=.985, bottom=.20, top=.83, wspace=.12)
    study_out = OUT if args.study == 'main' else ROOT/'output/paf_core_ablation_v2'
    folder = study_out/'figures'
    folder.mkdir(exist_ok=True)
    for suffix in ('png', 'svg'):
        figure.savefig(folder/f'factorial_views.{suffix}', dpi=160, facecolor='white')
    plt.close(figure)
    write_json(folder/'factorial_views.json', {'selection': '固定train順の先頭通常画像。成否や外観で選択しない。',
        'plan_sha256': sha(OUT/'plan.json' if args.study == 'main' else study_out/'references.json'),
        'study': args.study, 'implementation_sha256': sha(Path(__file__)), 'views': evidence,
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in (folder/'factorial_views.png', folder/'factorial_views.svg')}})
    print('paired training views figure saved', flush=True)


if __name__ == '__main__':
    main()
