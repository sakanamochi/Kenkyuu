"""固定候補・幾何・main参考の全劣化強度を同一尺度で比較する。"""

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
from research.paf_generalization.diagnose import sha
from research.paf_core_ablation.boundary_report import OUT, CONFIG

VARIANTS = {
    'boundary_projection_mix': ('投影＋外観', '#643f97', '-'),
    'boundary_projection_relative': ('投影＋外観＋開口幅遮蔽', '#19785c', '-'),
    'canny': ('Canny＋輪郭別RANSAC', '#d98427', '--'),
    'zhang2019': ('Zhang 2019型（再現実装）', '#4c8fa5', '--'),
    'main_1194m': ('参考：main構成・1194M学習（1seed）', '#666666', ':'),
}
EFFECTS = {'black_rectangle': '矩形遮蔽', 'sensor_whiteout': '白飛び',
           'sensor_black_crush': '黒つぶれ'}


def main():
    summary = read_json(OUT/'summary.json')
    seeds = read_json(CONFIG)['seeds']
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 12, 'svg.fonttype': 'path'})
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), sharex=True, sharey=True)
    evidence = []
    x = np.array([0., .25, .5, .75, 1.])
    for row_index, source in enumerate(('1194m', 'diversified')):
        for column_index, (effect, title) in enumerate(EFFECTS.items()):
            ax = axes[row_index, column_index]
            for variant, (label, color, style) in VARIANTS.items():
                conditions = [('clean', 0.)]+[(effect, float(s)) for s in x[1:]]
                rows = [next(r for r in summary['rows'] if r['dataset'] == source and
                             r['variant'] == variant and r['degradation'] == e and r['severity'] == s)
                        for e, s in conditions]
                expected = [None] if variant in ('canny', 'zhang2019') else [20260907] if variant == 'main_1194m' else seeds
                assert all(r['seeds'] == expected for r in rows), (source, variant)
                values = np.array([r['success_rate']*100 for r in rows])
                ax.plot(x, values, color=color, linestyle=style, marker='o', markersize=4,
                        linewidth=1.8, label=label, zorder=3)
                if len(expected) == 3:
                    # 帯は固定3seedの範囲であり、信頼区間や統合モデルの性能ではない。
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
            if row_index == 0:
                ax.set_title(title, fontsize=16, pad=12)
            if row_index == 1:
                ax.set_xlabel('劣化強度（0は通常画像）')
            if column_index == 0:
                dataset = '未知1194M' if source == '1194m' else '多様形状の未学習寸法'
                ax.set_ylabel(dataset+'\n成功率（IoU ≥ 0.8）%')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045),
                  ncol=3, frameon=False, fontsize=11)
    figure.suptitle('投影・外観の学習と遮蔽学習：全強度の性能と限界', fontsize=18, y=.98)
    figure.text(.5, .018, '候補は3seed平均、帯は最小〜最大。幾何は1回。main参考は別教師・別学習率の60epoch比較。全画像を分母に含む。',
                ha='center', fontsize=10.5)
    figure.subplots_adjust(left=.075, right=.985, top=.91, bottom=.19, hspace=.13, wspace=.09)
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    artifacts = []
    for suffix in ('png', 'svg'):
        path = folder/f'degradation_comparison.{suffix}'
        figure.savefig(path, dpi=160, facecolor='white')
        artifacts.append(path)
    plt.close(figure)
    write_json(folder/'degradation_comparison.json', {
        'summary_sha256': sha(OUT/'summary.json'), 'implementation_sha256': sha(Path(__file__)),
        'rows': evidence, 'interval': '固定3seedの最小〜最大。信頼区間ではない。',
        'scope': '通常画像を各劣化系列の強度0として再表示する。系列をまたいで独立画像数を増やさない。',
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in artifacts}})
    print('All-severity comparison figure saved', flush=True)


if __name__ == '__main__':
    main()
