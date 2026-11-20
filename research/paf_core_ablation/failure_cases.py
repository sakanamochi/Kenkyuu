"""基礎対照の各条件で最初に失敗した画像を固定し、境界確率を対応表示する。"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.diagnose import sha, image_and_truth
from research.paf_core_ablation.boundary_report import OUT, BASE


def draw_ellipse(ax, ellipse, color):
    if ellipse is not None:
        center, sizes, angle = ellipse
        ax.add_patch(Ellipse(center, sizes[0], sizes[1], angle=angle,
                             fill=False, color=color, linewidth=1.15))


def main():
    seed = 20260907
    folders = [OUT/f'runs/core_base/{seed}', OUT/f'runs/core_surface/{seed}',
               OUT/f'runs/core_projection/{seed}',
               ROOT/f'output/paf_generalization_v2/runs/boundary_projection_mix/{seed}']
    results = [read_json(p/'test_1194m.json') for p in folders]
    conditions = [('clean', 0., '通常'), ('black_rectangle', .5, '遮蔽0.5'),
                  ('sensor_whiteout', .5, '白飛び0.5'), ('sensor_black_crush', .5, '黒つぶれ0.5')]
    # 良くなった画像を探さず、基礎対照の固定順・最初の失敗という規則で選ぶ。
    selected = [next(r for r in results[0]['rows'] if r['degradation'] == effect
                     and r['severity'] == severity and not r['success']) for effect, severity, _ in conditions]
    input_samples = {s['sample_id']: s for s in load_samples(BASE/'1194m', 'test')}
    lookups = [{r['sample_id']: r for r in result['rows']} for result in results]
    models = []
    for folder, result in zip(folders, results):
        assert sha(folder/'best.pt') == result['checkpoint_sha256']
        model = TinyUNet(16).cpu().eval()
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        assert checkpoint['provenance'] == result['provenance']
        model.load_state_dict(checkpoint['model_state'])
        models.append(model)
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 11, 'svg.fonttype': 'path'})
    figure, axes = plt.subplots(4, 5, figsize=(16, 12))
    evidence = []
    for index, (reference, (_, _, condition)) in enumerate(zip(selected, conditions)):
        sample = input_samples[reference['sample_id']]
        for field in ('image', 'label'):
            assert sha(BASE/'1194m'/sample[field]) == sample[field+'_sha256']
        image, truth = image_and_truth(BASE/'1194m', sample)
        target = reference['truth_ellipse']
        assert np.allclose([*truth[0], *truth[1], truth[2]], [*target[0], *target[1], target[2]])
        axes[index, 0].imshow(image[..., 0], cmap='gray', vmin=0, vmax=255)
        draw_ellipse(axes[index, 0], truth, '#53df8c')
        axes[index, 0].set_ylabel(condition, fontsize=13)
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None]/127.5-1
        pair = []
        for column, (model, lookup) in enumerate(zip(models, lookups), 1):
            with torch.inference_mode():
                probability = torch.sigmoid(model(x))[0, 0].numpy()
            row = lookup[reference['sample_id']]
            ax = axes[index, column]
            ax.imshow(probability, cmap='gray', vmin=0, vmax=1)
            draw_ellipse(ax, truth, '#53df8c')
            # 楕円と成績は照合済み評価の保存値。図のCPU推論で幾何を再抽選しない。
            draw_ellipse(ax, row['ellipse'], '#ff67ce')
            ax.set_xlabel(f"保存済み推定のIoU {row['ellipse_iou']:.3f}")
            pair.append({'variant': row.get('variant', results[column-1]['variant']),
                         'ellipse': row['ellipse'], 'ellipse_iou': row['ellipse_iou'],
                         'probability_min': float(probability.min()), 'probability_max': float(probability.max())})
        for ax in axes[index]:
            ax.set_xticks([])
            ax.set_yticks([])
        evidence.append({'sample_id': sample['sample_id'], 'truth_ellipse': truth, 'predictions': pair,
                         'image_sha256': sample['image_sha256'], 'label_sha256': sample['label_sha256']})
    for ax, title in zip(axes[0], ('入力と正解内周', '元の学習分布', '外観のみ', '投影のみ', '投影＋外観')):
        ax.set_title(title, fontsize=13, pad=10)
    figure.suptitle('基礎対照の失敗例で境界の選び方を比較する', fontsize=16, y=.985)
    figure.text(.5, .035, '緑：正解内周　桃：保存済み推定楕円　白：高い境界確率（共通0〜1尺度）', ha='center')
    figure.text(.5, .014, '各条件で基礎対照が最初に失敗した画像。改善例の選別ではなく、全体性能を表す図でもない。', ha='center', fontsize=10)
    figure.subplots_adjust(left=.065, right=.98, bottom=.085, top=.93, wspace=.13, hspace=.22)
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    paths = [folder/f'boundary_failure_cases.{suffix}' for suffix in ('png', 'svg')]
    for path in paths:
        figure.savefig(path, dpi=145, facecolor='white')
    plt.close(figure)
    write_json(folder/'boundary_failure_cases.json', {'seed': seed, 'cases': evidence,
        'selection': '各条件の基礎対照・固定test順の最初の失敗。改善の有無では選ばない。',
        'scope': '境界確率の説明用。画像は主評価と同じ、図のCNNはCPU推論、楕円・IoUは保存済み評価値。',
        'implementation_sha256': sha(Path(__file__)),
        'inputs': {str((p/name).relative_to(ROOT)): sha(p/name) for p in folders for name in ('best.pt', 'test_1194m.json')},
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in paths}})
    print('paired failure cases saved', flush=True)


if __name__ == '__main__':
    main()
