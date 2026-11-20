"""共通データ入口の等価性と、両学習元の投影・外観パラメータ対応を確認する。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_image
from research.paf_fair_comparison import experiment as fair
from research.paf_generalization_v2 import projection
from research.paf_generalization_v2.projection_occlusion import ProjectionOcclusionDataset


def main():
    torch.set_num_threads(4)
    projection.NEW_VIEW = 'projection_coverage'
    checks = []
    for seed in fair.read_json(fair.CONFIG)['seeds']:
        torch.manual_seed(seed)
        old = ProjectionOcclusionDataset(fair.BASE/'diversified', 'train', 256, cache=True)
        new = fair.FairDataset(fair.BASE/'diversified', 'train', 256, cache=True)
        indices = np.random.default_rng(seed).choice(len(old), size=128, replace=False)
        for i in indices:
            a, b = old[int(i)], new[int(i)]
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            assert torch.equal(fair.boundary.boundary_target(a[1][None]), fair.boundary.boundary_target(b[1][None]))
        assert old.rng.bit_generator.state == new.rng.bit_generator.state
        assert old.mask_rng.bit_generator.state == new.mask_rng.bit_generator.state
        checks.append({'seed': seed, 'same_training_input_and_target': len(indices),
                       'view_rng_equal': True, 'mask_rng_equal': True})
        del old, new
    validation_counts = {}
    for source in ('1194m', 'diversified'):
        new = fair.FairDataset(fair.BASE/source, 'validation', 256, cache=True)
        old = projection.PackedBoundaryDataset(fair.BASE/source, 'validation', 256, cache=True)
        indices = list(range(0, len(old), 17))
        for i in indices:
            a, b = old[i], new[i]
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert len(new.views) == 1
        validation_counts[source] = len(indices)
        del old, new
    plans = fair.read_json(fair.OUT/'augmentation_1194m/plan.json')['plan']
    source_plans = [p for p in fair.read_json(ROOT/'output/paf_generalization_v2/projection_coverage/plan.json')['plan'] if p['split'] == 'train']
    textures = [p['texture'] for p in fair.read_json(ROOT/'output/paf_generalization_v1/surface_counterfactual/plan.json')['plan'] if p['split'] == 'train']
    errors = []
    for target, source, texture in zip(plans, source_plans, textures):
        assert target['split'] == 'train' and target['texture'] == texture
        for field in ('center_480', 'major_axis_256', 'target_axis_ratio'):
            assert target[field] == source[field]
        label = fair.read_json(fair.OUT/f"augmentation_1194m/labels/{target['base_sample_id']}.json")
        assert label['original_projection_error_px'] < .01
        assert label['projection_transform_max_error_px'] < .01
        assert abs(label['actual_axis_ratio']-target['target_axis_ratio']) < 1e-4
        errors.append(label['projection_transform_max_error_px'])
    assert len(plans) == len(source_plans) == len(textures) == 497
    paths = [Path(__file__), Path(fair.__file__), fair.CONFIG]
    fair.write_json(fair.OUT/'implementation_checks.json', {'training_checks': checks,
        'validation_samples': validation_counts, 'matched_augmentation_parameters': len(plans),
        'max_projection_error_px': max(errors),
        'same_getitem_function': fair.FairDataset.__getitem__ is ProjectionOcclusionDataset.__getitem__,
        'shared_training_function': 'research.paf_generalization.experiment.train_variant',
        'files': {str(p.relative_to(ROOT)): fair.sha(p) for p in paths}})
    # 同じ拡張パラメータを使う先頭train例を表示し、都合の良い例を選び直さない。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(), 'font.size': 12})
    figure, axes = plt.subplots(2, 4, figsize=(12, 6.7))
    evidence = []
    for row, source in enumerate(('1194m', 'diversified')):
        for column, root in enumerate(fair.view_roots(source)):
            sample = next(s for s in fair.read_json(root/'manifest.json')['samples']
                          if s['split'] == 'train' and s['conditions']['degradation'] == 'clean')
            path = root/sample['image']
            gray = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
            axes[row, column].imshow(gray, cmap='gray', vmin=0, vmax=255)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(['元画像', '外観変更', '投影変更', '投影＋外観変更'][column])
            if column == 0:
                axes[row, column].set_ylabel('1194M学習' if source == '1194m' else '多様形状学習')
            evidence.append({'source': source, 'view': column, 'sample_id': sample['sample_id'],
                             'image': str(path), 'sha256': fair.sha(path)})
    figure.suptitle('両学習元に同じ4ビュー抽選と拡張パラメータを適用', fontsize=17)
    figure.text(.5, .03, '学習時は各25%。通常画像の表示例。遮蔽例では選んだビューの開口幅を基準に矩形を再抽選。',
                ha='center', fontsize=11)
    figure.subplots_adjust(left=.05, right=.99, bottom=.09, top=.90, wspace=.04, hspace=.04)
    folder = fair.OUT/'figures'
    folder.mkdir(exist_ok=True)
    path = folder/'common_training_views.png'
    figure.savefig(path, dpi=150, facecolor='white')
    plt.close(figure)
    fair.write_json(folder/'common_training_views.json', {'rows': evidence, 'sha256': fair.sha(path)})
    print('INPUT_CHECKS', len(checks)*128, 'training samples,', len(plans), 'paired augmentation plans', flush=True)


if __name__ == '__main__':
    main()
