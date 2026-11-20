"""実際の学習画像から各模様の先頭例を選び、元材質との対応を表示する。"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from research.paf_generalization.diagnose import image_and_truth, sha

BASE = ROOT/'output/hole_region_comparison_v1/datasets/diversified'
SURFACE = ROOT/'output/paf_generalization_v1/surface_counterfactual'
OUT = ROOT/'output/paf_core_ablation_v2/figures'
KINDS = {'noise': 'ノイズ：不規則な濃淡', 'bands': '縞：方向性のある濃淡',
         'checker': 'チェッカー：格子状の濃淡'}


def main():
    plan = [r for r in read_json(SURFACE/'plan.json')['plan'] if r['split'] == 'train']
    counts = Counter(r['texture']['kind'] for r in plan)
    sources = [(BASE, '元の材質'), (SURFACE, '模様を付けた材質')]
    lookups = [{s['conditions']['base_sample_id']: s for s in load_samples(root, 'train')
                if s['conditions']['degradation'] == 'clean'} for root, _ in sources]
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 13, 'svg.fonttype': 'path'})
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.9))
    evidence = []
    for column, (kind, title) in enumerate(KINDS.items()):
        # 見栄えや検出成否では選ばず、固定計画の学習順で各種類の先頭を使う。
        item = next(r for r in plan if r['texture']['kind'] == kind)
        sid = item['base_sample_id']
        record = {'kind': kind, 'base_sample_id': sid, 'texture': item['texture'], 'inputs': []}
        first_truth = None
        for row, ((root, label), lookup) in enumerate(zip(sources, lookups)):
            sample = lookup[sid]
            for field in ('image', 'label'):
                assert sha(root/sample[field]) == sample[field+'_sha256']
            image, truth = image_and_truth(root, sample)
            if first_truth is None:
                first_truth = truth
            else:
                assert truth == first_truth
            ax = axes[row, column]
            # 入力の濃淡を保ち、パネルごとの自動コントラスト補正を行わない。
            ax.imshow(image[:, :, 0], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=15, pad=12)
            else:
                ax.set_xlabel(f'{sid} / 学習形状中 {counts[kind]}例', labelpad=9)
            if column == 0:
                ax.set_ylabel(label, fontsize=14, labelpad=14)
            record['inputs'].append({'sample_id': sample['sample_id'],
                'image_path': str((root/sample['image']).resolve()),
                'image_sha256': sample['image_sha256'], 'label_sha256': sample['label_sha256']})
        evidence.append(record)
    figure.suptitle('採用した表面模様：実際の学習画像の対応例', fontsize=19, y=.98)
    figure.text(.5, .035, '各列の上下は同じ形状・カメラ・光源。列間の形状は異なる。\n反射色の模様と明るさを変え、形状・背景は変えない。入力と同じモノクロ256、露光補正なし。',
                ha='center', fontsize=11)
    figure.subplots_adjust(left=.065, right=.98, bottom=.12, top=.90, hspace=.08, wspace=.07)
    OUT.mkdir(exist_ok=True)
    artifacts = []
    for suffix in ('png', 'svg'):
        path = OUT/f'surface_examples.{suffix}'
        figure.savefig(path, dpi=160, facecolor='white')
        artifacts.append(path)
    plt.close(figure)
    write_json(OUT/'surface_examples.json', {
        'selection': '固定計画のtrainで各模様の先頭。性能・見栄えによる選択なし。',
        'plan_sha256': sha(SURFACE/'plan.json'), 'train_shape_counts': dict(counts),
        'implementation_sha256': sha(Path(__file__)), 'examples': evidence,
        'scope': '種類の見え方を示す対応例。種類別の効果・最適性を測る比較実験ではない。',
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in artifacts}})
    print('Three surface-pattern examples saved with matched geometry and input hashes', flush=True)


if __name__ == '__main__':
    main()
