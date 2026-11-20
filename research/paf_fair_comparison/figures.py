"""共通CNN二系統と幾何二方式を、通常画像から全劣化強度まで同じ軸で描く。"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from research.paf_fair_comparison.experiment import OUT, CONFIG, sha

METHODS = {'1194m': ('① 1194M学習CNN', '#4b66b0', '-'),
           'diversified': ('② 多様形状学習CNN', '#147d64', '-'),
           'canny': ('③ Canny＋輪郭別RANSAC', '#d48427', '--'),
           'zhang2019': ('③ Zhang 2019型（再現実装）', '#9b639b', '--')}
EFFECTS = {'black_rectangle': '矩形遮蔽', 'sensor_whiteout': '白飛び', 'sensor_black_crush': '黒つぶれ'}


def main():
    summary = read_json(OUT/'summary.json')
    config = read_json(CONFIG)
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 12, 'svg.fonttype': 'path'})
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    evidence = []
    x = np.array([0., .25, .5, .75, 1.])
    for ri, dataset in enumerate(config['sources']):
        for ci, (effect, title) in enumerate(EFFECTS.items()):
            ax = axes[ri, ci]
            for method, (label, color, style) in METHODS.items():
                conditions = [('clean', 0.)]+[(effect, float(s)) for s in x[1:]]
                rows = [next(r for r in summary['rows'] if r['dataset'] == dataset and r['method'] == method
                             and r['degradation'] == e and r['severity'] == s) for e, s in conditions]
                seeds = [None] if method in ('canny', 'zhang2019') else config['seeds']
                assert all(r['seeds'] == seeds for r in rows)
                values = np.array([r['success_rate']*100 for r in rows])
                ax.plot(x, values, color=color, linestyle=style, marker='o', markersize=4,
                        linewidth=1.8, label=label)
                if len(seeds) == 3:
                    low = [min(r['per_seed'].values())*100 for r in rows]
                    high = [max(r['per_seed'].values())*100 for r in rows]
                    ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)
                evidence.extend(rows)
            ax.set_ylim(-2, 104)
            ax.set_xlim(-.03, 1.03)
            ax.set_xticks(x, ['0', '0.25', '0.5', '0.75', '1.0'])
            ax.set_yticks(np.arange(0, 101, 20))
            ax.grid(alpha=.22)
            ax.spines[['top', 'right']].set_visible(False)
            if ri == 0:
                ax.set_title(title, fontsize=16, pad=12)
            else:
                ax.set_xlabel('劣化強度（0は通常画像）')
            if ci == 0:
                name = '1194M test' if dataset == '1194m' else '多様形状 test'
                ax.set_ylabel(name+'\n成功率（IoU ≥ 0.8）%')
    figure.suptitle('学習元の公平比較：教師・中心線RANSAC・学習条件を共通化', fontsize=18, y=.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .06), ncol=2, frameon=False)
    figure.text(.5, .02, 'CNNは各3seed平均、帯は最小〜最大。通常＋3劣化×4強度の全画像を分母に含む。',
                ha='center', fontsize=11)
    figure.subplots_adjust(left=.075, right=.985, top=.90, bottom=.21, hspace=.13, wspace=.09)
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    artifacts = []
    for suffix in ('png', 'svg'):
        path = folder/f'degradation_comparison.{suffix}'
        figure.savefig(path, dpi=160, facecolor='white')
        artifacts.append(path)
    plt.close(figure)
    assert len(evidence) == 120
    write_json(folder/'degradation_comparison.json', {'summary_sha256': sha(OUT/'summary.json'),
        'implementation_sha256': sha(Path(__file__)), 'rows': evidence,
        'displayed_values': 120, 'unique_conditions': 104, 'interval': '固定3seedの範囲。信頼区間ではない。',
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in artifacts}})
    print('FIGURE', len(evidence), 'values', flush=True)


if __name__ == '__main__':
    main()
