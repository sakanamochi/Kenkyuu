"""1194Mだけ元材質で学習し、投影・教師・遮蔽・学習器・推定器を固定する対照。"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_fair_comparison import experiment as common
from paf_ring_detection.data import read_json, write_json

BASELINE = ROOT/'output/paf_fair_comparison_v1'
OUT = ROOT/'output/paf_1194m_no_surface_v1'
CONFIG = Path(__file__).with_suffix('.json')
ORIGINAL_ROOTS = common.view_roots
ORIGINAL_DATASET = common.FairDataset


def roots(source):
    """1194Mは元材質の元投影と拡張投影を半分ずつ使う。"""
    if source == '1194m':
        plain = common.BASE/source
        projected = BASELINE/'augmentation_1194m/projection_plain'
        # 抽選列・投影ビットを維持し、外観ビットだけを無効化する。
        return [plain, plain, projected, projected]
    return ORIGINAL_ROOTS(source)


class OriginalAppearanceDataset(ORIGINAL_DATASET):
    """学習時だけ4スロットを2つの元材質ビューへ対応させる。"""

    def __init__(self, dataset_dir, split, input_size, cache=False):
        super().__init__(dataset_dir, split, input_size, cache)
        if self.root.name == '1194m' and split == 'train':
            self.views = [self.views[0], self.views[0], self.views[2], self.views[2]]


def prepare():
    """元比較を保全し、外観変更の除去だけを変更した設定・入力・再利用結果を固定する。"""
    config = read_json(CONFIG)
    previous = read_json(BASELINE/'plan.json')
    previous_audit = read_json(BASELINE/'audit.json')
    assert previous_audit['plan_sha256'] == common.sha(BASELINE/'plan.json')
    assert previous_audit['incomplete_runs'] == []
    for key in ('seeds', 'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'positive_weight',
                'ransac', 'selection', 'occlusion_sampling', 'opencv_seed_per_image', 'reject_uniform_images'):
        assert config[key] == previous['config'][key], key
    assert config['variants']['diversified'] == previous['config']['variants']['diversified']
    files = dict(previous['files'])
    files.update(previous_audit['files'])
    for relative, digest in files.items():
        assert common.sha(ROOT/relative) == digest, relative
    for path in (Path(__file__), CONFIG, BASELINE/'plan.json', BASELINE/'audit.json',
                 BASELINE/'summary.json', BASELINE/'implementation_checks.json'):
        files[str(path.relative_to(ROOT))] = common.sha(path)
    active = {}
    for source in config['sources']:
        active[source] = [str(p.relative_to(ROOT)) for p in roots(source)]
        for path in set(roots(source)):
            assert (path/'manifest.json').exists()
        assert len(common.load_samples(common.BASE/source, 'train')) == 1988
        assert len(common.load_samples(common.BASE/source, 'validation')) == 735
    plan = {'config': config, 'files': files, 'active_training_views': active,
        'counts': previous['counts'], 'references': previous['references'],
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'intervention': '1194M trainの外観ビットのみ除去。元材質の元投影/拡張投影を各50%。',
        'scope': '両CNNで教師・推論・最適化を共通化し、外観拡張は多様形状側だけに使用する。'}
    path = OUT/'plan.json'
    if path.exists():
        assert read_json(path) == plan
    else:
        write_json(path, plan)
    # 多様形状側は入力・重み・推論条件が同じなので、固定済み全推定を再利用する。
    for seed in config['seeds']:
        for dataset in config['sources']:
            src = BASELINE/f'results/diversified/{seed}/test_{dataset}.json'
            dst = OUT/f'results/diversified/{seed}/test_{dataset}.json'
            result = read_json(src)
            assert result['plan_sha256'] == common.sha(BASELINE/'plan.json')
            result['reused_result'] = {'path': str(src.relative_to(ROOT)), 'sha256': common.sha(src)}
            result['plan_sha256'] = common.sha(path)
            if dst.exists():
                assert read_json(dst) == result
            else:
                write_json(dst, result)
    print('PREPARED: 1194M original appearance only; three paired seeds', flush=True)
    return plan


def check_inputs():
    """抽選・投影・遮蔽の乱数を保ち、外観変更だけが除かれたことを確認する。"""
    checks = []
    for seed in read_json(CONFIG)['seeds']:
        torch.manual_seed(seed)
        dataset = OriginalAppearanceDataset(common.BASE/'1194m', 'train', 256, cache=True)
        choice_rng = np.random.default_rng(seed+5200)
        mask_rng = np.random.default_rng(seed+6200)
        draws = []
        for index in np.random.default_rng(seed).choice(len(dataset), 128, replace=False):
            choice = int(choice_rng.integers(0, 4))
            assert dataset.views[choice].root == roots('1194m')[choice]
            image, masks, _ = dataset[int(index)]
            view = dataset.views[choice]
            sample = view.samples[int(index)]
            expected_image, expected_masks, _ = view[int(index)]
            if sample['conditions']['degradation'] == 'black_rectangle':
                from research.paf_generalization_v2.projection_occlusion import read_image, label_ellipse, masked_view, cv2
                gray = cv2.cvtColor(read_image(view.root/sample['image']), cv2.COLOR_BGR2GRAY)
                ellipse = label_ellipse(read_json(view.root/sample['label']))
                left, draw = bool(mask_rng.integers(0, 2) == 0), float(mask_rng.random())
                expected_image, visibility = masked_view(gray, ellipse, draw, left, 'relative', 256)
                expected_masks = torch.cat([expected_masks[:1], visibility], dim=0)
            assert torch.equal(image, expected_image) and torch.equal(masks, expected_masks)
            draws.append(choice)
        assert dataset.rng.bit_generator.state == choice_rng.bit_generator.state
        assert dataset.mask_rng.bit_generator.state == mask_rng.bit_generator.state
        checks.append({'seed': seed, 'checked_samples': len(draws), 'slots_seen': sorted(set(draws)),
                       'surface_views_used': False, 'projection_and_occlusion_rng_preserved': True})
        validation = OriginalAppearanceDataset(common.BASE/'1194m', 'validation', 256, cache=True)
        original = ORIGINAL_DATASET(common.BASE/'1194m', 'validation', 256, cache=True)
        for i in range(0, len(original), 17):
            a, b = validation[i], original[i]
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # 多様形状の等価性は保全済み共通入口の検証結果を明示的に参照する。
    evidence = read_json(BASELINE/'implementation_checks.json')
    evidence['no_surface_checks'] = checks
    evidence['baseline_checks_reference'] = str((BASELINE/'implementation_checks.json').relative_to(ROOT))
    evidence['files'][str(Path(__file__).relative_to(ROOT))] = common.sha(Path(__file__))
    evidence['files'][str(CONFIG.relative_to(ROOT))] = common.sha(CONFIG)
    write_json(OUT/'implementation_checks.json', evidence)
    print('CHECKED: 384 training samples; original validation; no surface-randomized images', flush=True)


def configure():
    """保存先とデータ入口だけを差し替え、既存の学習・評価関数を共有する。"""
    common.OUT, common.CONFIG = OUT, CONFIG
    common.view_roots, common.FairDataset, common.prepare = roots, OriginalAppearanceDataset, prepare


if __name__ == '__main__':
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        torch.set_num_threads(4)
        prepare()
        check_inputs()
    else:
        common.main()
