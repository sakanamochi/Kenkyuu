"""学習元以外の教師・拡張・学習器・推定器を共通化したPAF比較。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization import experiment as shared, boundary
from research.paf_generalization.diagnose import sha, image_and_truth
from research.paf_generalization_v2.projection import PackedBoundaryDataset
from research.paf_generalization_v2.projection_occlusion import ProjectionOcclusionDataset

OUT = ROOT/'output/paf_fair_comparison_v1'
BASE = ROOT/'output/hole_region_comparison_v1/datasets'
CONFIG = Path(__file__).with_suffix('.json')
REFERENCE = ROOT/'output/paf_generalization_v2/runs/boundary_projection_relative'
SOURCE = None
SHARED_TRAIN = shared.train_variant


def view_roots(source):
    """学習元によらず、元/拡張投影×元/拡張外観の順で4ビューを返す。"""
    if source == '1194m':
        root = OUT/'augmentation_1194m'
        return [BASE/source]+[root/v for v in ('original_surface', 'projection_plain', 'projection_surface')]
    assert source == 'diversified'
    return [BASE/source, ROOT/'output/paf_generalization_v1/surface_counterfactual',
            ROOT/'output/paf_generalization_v2/projection_coverage/native_plain',
            ROOT/'output/paf_generalization_v2/projection_coverage/native_surface']


class FairDataset(ProjectionOcclusionDataset):
    """入力元を明示し、4ビュー抽選と開口幅遮蔽は採用実装をそのまま共有する。"""

    def __init__(self, dataset_dir, split, input_size, cache=False):
        self.root = Path(dataset_dir)
        self.input_size = input_size
        roots = view_roots(self.root.name) if split == 'train' else [self.root]
        self.views = [PackedBoundaryDataset(p, split, input_size, cache) for p in roots]
        self.samples = self.views[0].samples
        assert all([s['sample_id'] for s in v.samples] == [s['sample_id'] for s in self.samples] for v in self.views)
        self.rng = np.random.default_rng(torch.initial_seed()+5200)
        self.mask_rng = np.random.default_rng(torch.initial_seed()+6200)
        self.raw_cache = {}


def training_dataset(dataset_dir, split, input_size, cache=False):
    """既存の共通学習ループへ指定した学習元を渡す。validationも同じ学習元とする。"""
    assert SOURCE in ('1194m', 'diversified') and split in ('train', 'validation')
    return FairDataset(BASE/SOURCE, split, input_size, cache)


def prepare():
    """入力・実装・再利用重みのhash、分割分離、共通条件を学習前に固定する。"""
    config = read_json(CONFIG)
    files = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        key = str(path.relative_to(ROOT))
        if key not in files:
            files[key] = sha(path)
        if expected is not None:
            assert files[key] == expected, key

    for path in [CONFIG, Path(__file__), Path(__file__).with_name('render_1194m.py'),
                 ROOT/'research/paf_generalization/experiment.py', ROOT/'research/paf_generalization/boundary.py',
                 ROOT/'research/paf_generalization_v2/projection.py',
                 ROOT/'research/paf_generalization_v2/projection_occlusion.py',
                 ROOT/'research/paf_generalization_v2/occlusion_sampling.py',
                 ROOT/'paf_ring_detection/data.py', ROOT/'paf_ring_detection/effects.py',
                 ROOT/'paf_ring_detection/geometry.py', ROOT/'paf_ring_detection/train.py',
                 ROOT/'paf_ring_detection/methods/cnn.py', ROOT/'paf_ring_detection/methods/ransac.py']:
        check(path)
    counts = {}
    for source in config['sources']:
        samples = read_json(BASE/source/'manifest.json')['samples']
        groups = {split: {s['group_id'] for s in samples if s['split'] == split}
                  for split in ('train', 'validation', 'test')}
        assert all(not groups[a] & groups[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
        counts[source] = {split: sum(s['split'] == split for s in samples) for split in groups}
        assert counts[source] == {'train': 1988, 'validation': 735, 'test': 1456 if source == '1194m' else 2002}
        train_ids = {s['sample_id'] for s in samples if s['split'] == 'train'}
        for root in view_roots(source):
            check(root/'manifest.json')
            rows = read_json(root/'manifest.json')['samples']
            assert {s['sample_id'] for s in rows if s['split'] == 'train'} == train_ids
            for sample in rows:
                for field in ('image', 'label'):
                    check(root/sample[field], sample[field+'_sha256'])
    verification = read_json(OUT/'augmentation_1194m/verification.json')
    check(OUT/'augmentation_1194m/plan.json', verification['plan_sha256'])
    check(OUT/'augmentation_1194m/verification.json')
    for view, digest in verification['manifest_hashes'].items():
        check(OUT/f'augmentation_1194m/{view}/manifest.json', digest)
    references = {}
    for seed in config['seeds']:
        folder = REFERENCE/str(seed)
        metadata = read_json(folder/'provenance.json')
        assert metadata['training_source'] == 'diversified' and not metadata['target_data_used_for_training']
        for key in ('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'positive_weight', 'ransac',
                    'view_probabilities', 'occlusion_sampling', 'selection'):
            assert metadata['config'][key] == config[key], key
        assert metadata['config']['variants']['boundary_projection_relative'] == config['variants']['diversified']
        for path, digest in metadata['files'].items():
            check(ROOT/path, digest)
        for group in ('projection_files', 'surface_data_hashes', 'occlusion_sampling_files'):
            for path, digest in metadata['config'][group].items():
                check(ROOT/path, digest)
        for key, path in [('boundary', 'research/paf_generalization/boundary.py'),
                          ('ransac', 'paf_ring_detection/methods/ransac.py')]:
            check(ROOT/path, metadata['config'][key+'_implementation_sha256'])
        for name in ('best.pt', 'provenance.json', 'training_history.json', 'complete.json'):
            check(folder/name)
        check(folder/'best.pt', read_json(folder/'complete.json')['checkpoint_sha256'])
        references[str(seed)] = {'folder': str(folder.relative_to(ROOT)), 'checkpoint_sha256': sha(folder/'best.pt')}
    plan = {'config': config, 'counts': counts, 'files': files, 'references': references,
            'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
            'claim': '同一学習・推論パイプラインで学習データ源を比較。形状だけの因果効果は分離しない。'}
    path = OUT/'plan.json'
    if path.exists():
        assert read_json(path) == plan, '固定済み計画との不一致'
    else:
        write_json(path, plan)
    print('PREPARED', counts, len(files), flush=True)
    return plan


def freeze(folder, config, variant, seed):
    """学習元を正しく記録し、再利用と新規学習を共通の計画に結び付ける。"""
    metadata = {'config': config, 'variant': variant, 'seed': seed,
                'plan_sha256': sha(OUT/'plan.json'), 'training_source': variant,
                'target_data_used_for_training': variant == '1194m'}
    path = folder/'provenance.json'
    if path.exists():
        assert read_json(path) == metadata
    else:
        write_json(path, metadata)
    return metadata


def model_folder(source, seed):
    return REFERENCE/str(seed) if source == 'diversified' else OUT/f'runs/{source}/{seed}'


def evaluate(source, seed):
    """両CNNへ同じ画像・中心線・RANSAC・OpenCV乱数と一様画像棄却を適用する。"""
    config = read_json(CONFIG)
    folder = model_folder(source, seed)
    checkpoint = folder/'best.pt'
    digest = sha(checkpoint)
    assert read_json(folder/'complete.json')['checkpoint_sha256'] == digest
    model = TinyUNet(16).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state'])
    for dataset in config['sources']:
        destination = OUT/f'results/{source}/{seed}/test_{dataset}.json'
        if destination.exists():
            result = read_json(destination)
            assert result['checkpoint_sha256'] == digest and result['plan_sha256'] == sha(OUT/'plan.json')
            continue
        rows = []
        for sample in load_samples(BASE/dataset, 'test'):
            image, truth = image_and_truth(BASE/dataset, sample)
            ellipse = None
            if np.ptp(image) != 0:
                x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
                with torch.inference_mode():
                    probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                cv2.setRNGSeed(config['opencv_seed_per_image'])
                ellipse = boundary.boundary_ellipse(probability, config['ransac'])
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                'ellipse': ellipse, 'truth_ellipse': truth, 'ellipse_iou': iou, 'success': iou >= .8})
        write_json(destination, {'training_source': source, 'dataset': dataset, 'seed': seed,
            'checkpoint_sha256': digest, 'plan_sha256': sha(OUT/'plan.json'), 'rows': rows})
        print('EVAL', source, seed, dataset, sum(r['success'] for r in rows), len(rows), flush=True)


def main():
    global SOURCE
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'run'])
    parser.add_argument('--sources', nargs='+', choices=['1194m', 'diversified'], default=['1194m', 'diversified'])
    parser.add_argument('--seeds', nargs='+', type=int, default=read_json(CONFIG)['seeds'])
    args = parser.parse_args()
    prepare()
    if args.stage == 'prepare':
        return
    shared.CONFIG, shared.OUT = CONFIG, OUT
    shared.RingDataset, shared.objective, shared.freeze_run = training_dataset, boundary.boundary_objective, freeze
    for SOURCE in args.sources:
        for seed in args.seeds:
            if args.stage in ('train', 'run') and SOURCE == '1194m':
                SHARED_TRAIN(SOURCE, seed)
            if args.stage in ('evaluate', 'run'):
                evaluate(SOURCE, seed)


if __name__ == '__main__':
    main()
