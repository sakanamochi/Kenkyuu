"""固定分割・穴領域学習・四方式共通評価の接続を確認する。"""

import copy
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from paf_ring_detection.data import RingDataset, read_json, sample_image, write_image, write_json
from paf_ring_detection.evaluate import METHODS, checkpoint_path, evaluate
from paf_ring_detection.prepare import digest, prepare_training_data, verify_prepared
from paf_ring_detection.report import aggregate_scores, build_report
from paf_ring_detection.train import train


def test_fixed_splits_and_full_region_pipeline():
    # 検証用データ・重み・結果は処理終了時に全て片付ける。
    with tempfile.TemporaryDirectory(prefix='paf_pipeline_') as temporary:
        root = Path(temporary)
        config = read_json(Path(__file__).resolve().parents[1] / 'config/experiment.json')
        config.update(output=str(root / 'experiment'), input_size=64, training_seeds=[20260907])
        config['cnn'].update(epochs=1, base_channels=2, batch_size=8)
        config['sources'] = {name: {'path': str(root / name), 'split_seed': 7}
                             for name in ('1194m', 'diversified')}
        for name, source in config['sources'].items():
            folder = Path(source['path'])
            samples = []
            for index in range(16):
                sid = f'sample_{index:02d}'
                ellipse = ((32, 32), (32, 20), 15)
                image = np.empty((64, 64, 3), np.uint8)
                image[:] = (80 + index, 100 + index, 130 + index)
                cv2.ellipse(image, ellipse, (0, 0, 0), -1)
                points = cv2.ellipse2Poly((32, 32), (16, 10), 15, 0, 360, 5).tolist()
                write_image(folder / f'images/{sid}.png', image)
                write_json(folder / f'labels/{sid}.json', {'sample_id': sid, 'image_width': 64,
                           'image_height': 64, 'image_points': points})
                samples.append({'sample_id': sid, 'shape_id': sid, 'parameters': {'hole': index},
                    'conditions': {'camera_id': f'camera_{index // 2}'},
                    'image': f'images/{sid}.png', 'label': f'labels/{sid}.json'})
            write_json(folder / 'manifest.json', {'samples': samples})
        prepare_training_data(config)
        verify_prepared(config)
        manifests = {name: read_json(Path(config['output']) / 'datasets' / name / 'manifest.json')
                     for name in config['sources']}
        assert manifests['1194m']['base_counts']['train'] == manifests['diversified']['base_counts']['train']
        for name, manifest in manifests.items():
            groups = {}
            for sample in manifest['samples']:
                groups.setdefault(sample['group_id'], set()).add(sample['split'])
            assert all(len(splits) == 1 for splits in groups.values())
            folder = Path(config['output']) / 'datasets' / name
            data = RingDataset(folder, 'train', 64)
            for effect in ('clean', 'black_rectangle', 'sensor_whiteout', 'sensor_black_crush'):
                index = next(i for i, s in enumerate(data.samples) if s['augmentation']['name'] == effect)
                first, target, _ = data[index]
                assert target[0, 32, 32] == 1.0
                assert torch.equal(first, data[index][0])
                assert torch.equal(first[0], first[1]) and torch.equal(first[1], first[2])
            for split in ('validation', 'test'):
                validation = RingDataset(folder, split, 64)
                for sample in validation.samples:
                    image = sample_image(folder, sample)
                    assert np.array_equal(image[..., 0], image[..., 1])
                    assert np.array_equal(image[..., 1], image[..., 2])
                # 矩形の遮蔽方向を強度間で固定する。
                seeds = {s['augmentation']['seed'] for s in validation.samples
                         if s['conditions']['base_sample_id'] == validation.samples[0]['conditions']['base_sample_id']
                         and s['augmentation']['name'] == 'black_rectangle'}
                assert len(seeds) == 1
            current = copy.deepcopy(config)
            current.update(seed=20260907, training_source=name, dataset_manifest_sha256=digest(folder / 'manifest.json'))
            current['paths'] = {'training_dataset': str(folder),
                                'checkpoint': str(checkpoint_path(config, name, 20260907))}
            checkpoint = train(current)
            assert torch.load(checkpoint, map_location='cpu', weights_only=False)['target'] == 'hole_region'
        a, b = [m['samples'] for m in manifests.values()]
        assert [s['augmentation'] for s in a if s['split'] == 'train'] == [s['augmentation'] for s in b if s['split'] == 'train']
        evaluate(config)
        build_report(config)
        root_results = Path(config['output']) / 'results'
        summary = read_json(root_results / 'summary.json')
        assert summary['complete'] and not summary['limited']
        assert read_json(root_results / 'audit.json')['status'] == 'passed'
        for dataset in config['sources']:
            records = [read_json(p) for p in (root_results / dataset).glob('*.json')]
            assert {r['method'] for r in records} == set(METHODS)
            assert len({r['rows'][0]['sample_id'] for r in records}) == 1
            assert all(r['input_size'] == 64 for r in records)
            for record in records:
                assert all(row['success'] == (row['ellipse_iou'] >= config['success_iou']) for row in record['rows'])
        # 準備済み原画像の書換えを黙って使わないことを確認する。
        source = Path(config['output']) / 'datasets/1194m'
        sample = manifests['1194m']['samples'][0]
        image_path = (source / sample['image']).resolve()
        assert sample_image(source, sample).shape == (64, 64, 3)
        image_path.write_bytes(b'changed')
        with pytest.raises(ValueError, match='入力が変わって'):
            verify_prepared(config)


def test_seed_average_does_not_duplicate_geometry_or_pool_unequal_counts():
    # 幾何の一回評価とCNNのseed反復を区別し、平均率の分母を守る。
    base = {'dataset': '1194m', 'degradation': 'all', 'severity': None,
            'sample_count': 10, 'missed_count': 1}
    rows = [{**base, 'method': 'CNN', 'seed': i, 'success_rate': rate}
            for i, rate in enumerate((0.2, 0.5, 0.8))]
    rows.append({**base, 'method': 'Canny', 'seed': None, 'success_rate': 0.7})
    aggregate = aggregate_scores(rows)
    assert aggregate[0]['success_rate'] == pytest.approx(0.5)
    assert aggregate[0]['seed_count'] == 3
    assert aggregate[0]['sample_count_per_model'] == 10
    assert aggregate[1]['seed_count'] == 0
    assert aggregate[1]['success_rate'] == 0.7
    rows[1]['sample_count'] = 20
    with pytest.raises(ValueError, match='評価枚数'):
        aggregate_scores(rows)
