"""全3seedが揃った2×2対照を、同じ尺度の成功率図として保存する。"""

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
from research.paf_core_ablation.boundary_report import OUT, CONFIG, DISPLAY

VARIANTS = {'core_base': ('元の学習分布', '#6d7278'),
            'core_surface': ('外観のみ', '#d98427'),
            'core_projection': ('投影のみ', '#4c8fa5'),
            'boundary_projection_mix': ('投影＋外観', '#643f97')}


def main():
    summary = read_json(OUT/'summary.json')
    seeds = read_json(CONFIG)['seeds']
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 12, 'svg.fonttype': 'path'})
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.1), sharey=True)
    evidence = []
    for ax, source in zip(axes, ('1194m', 'diversified')):
        for index, (variant, (label, color)) in enumerate(VARIANTS.items()):
            rows = [next(r for r in summary['rows'] if r['dataset'] == source and r['variant'] == variant
                         and r['degradation'] == effect and r['severity'] == severity) for effect, severity in DISPLAY]
            # 未完了・不均等seedの図を最終比較と誤認させない。
            assert all(r['seeds'] == seeds for r in rows), (source, variant, '全3seedの集計が必要')
            values = np.array([r['success_rate']*100 for r in rows])
            low = np.array([min(r['per_seed'].values())*100 for r in rows])
            high = np.array([max(r['per_seed'].values())*100 for r in rows])
            x = np.arange(len(DISPLAY))+(index-1.5)*.19
            ax.bar(x, values, width=.18, color=color, label=label,
                   yerr=np.stack([np.maximum(0, values-low), np.maximum(0, high-values)]),
                   error_kw={'elinewidth': 1, 'capsize': 2, 'capthick': 1}, zorder=3)
            evidence.extend(rows)
        ax.set_title('未知1194M' if source == '1194m' else '多様形状の未学習寸法', fontsize=16, pad=14)
        ax.set_xticks(np.arange(len(DISPLAY)), ['全13条件', '通常', '遮蔽0.5', '白飛び0.5', '黒つぶれ0.5'])
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis='y', alpha=.25, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('楕円検出の成功率（IoU ≥ 0.8）%')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=4, frameon=False)
    figure.suptitle('推定器を固定し、学習時の投影・外観だけを変える', fontsize=18, y=.97)
    figure.text(.5, .025, '棒は3seed平均、ひげはseed最小〜最大（信頼区間ではない）。1194Mは学習・validationに含めない。',
                ha='center', fontsize=11)
    figure.subplots_adjust(left=.065, right=.985, bottom=.23, top=.83, wspace=.14)
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    artifacts = []
    for suffix in ('png', 'svg'):
        path = folder/f'factorial_success.{suffix}'
        figure.savefig(path, dpi=160, facecolor='white')
        artifacts.append(path)
    plt.close(figure)
    write_json(folder/'factorial_success.json', {'summary_sha256': sha(OUT/'summary.json'),
        'implementation_sha256': sha(Path(__file__)), 'rows': evidence,
        'interval': '固定3seedの最小〜最大。モデル統合・信頼区間ではない。',
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in artifacts}})
    print('three-seed factorial figure saved', flush=True)


if __name__ == '__main__':
    main()
