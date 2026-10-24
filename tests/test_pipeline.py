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
from paf_ring_detection.report import build_report
from paf_ring_detection.train import train


def test_fixed_splits_and_full_region_pipeline():
    # 検証用データ・重み・結果は処理終了時に全て片付ける。
    with tempfile.TemporaryDirectory(prefix='paf_pipeline_') as temporary:
        root = Path(temporary)
        config = read_json(Path(__file__).resolve().parents[1] / 'config/experiment.json')
        config.update(output=str(root / 'experiment'), input_size=64, training_seeds=[20260907])
        config['cnn'].update(epochs=1, base_channels=2, batch_size=8)
        config['sources'] = {name: {'path': str(root / name), 'split_seed': 7}
                             for name in ('m1194m', 'diversified')}
        for name, source in config['sources'].items():
            folder = Path(source['path'])
            samples = []
            for index in range(16):
                sid = f'sample_{index:02d}'
                ellipse = ((32, 32), (32, 20), 15)
                image = np.full((64, 64, 3), 100 + index, np.uint8)
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
        assert manifests['m1194m']['base_counts']['train'] == manifests['diversified']['base_counts']['train']
        for name, manifest in manifests.items():
            groups = {}
            for sample in manifest['samples']:
                groups.setdefault(sample['group_id'], set()).add(sample['split'])
            assert all(len(splits) == 1 for splits in groups.values())
            folder = Path(config['output']) / 'datasets' / name
            data = RingDataset(folder, 'train', 64)
            for effect in ('clean', 'occlusion', 'whiteout', 'black_crush'):
                index = next(i for i, s in enumerate(data.samples) if s['augmentation']['name'] == effect)
                first, target, _ = data[index]
                assert target[0, 32, 32] == 1.0
                assert torch.equal(first, data[index][0])
            current = copy.deepcopy(config)
            current.update(seed=20260907, training_source=name, dataset_manifest_sha256=digest(folder / 'manifest.json'))
            current['paths'] = {'training_dataset': str(folder),
                                'checkpoint': str(checkpoint_path(config, name, 20260907))}
            checkpoint = train(current)
            assert torch.load(checkpoint, map_location='cpu', weights_only=False)['target'] == 'hole_region'
        a, b = [m['samples'] for m in manifests.values()]
        assert [s['augmentation'] for s in a if s['split'] == 'train'] == [s['augmentation'] for s in b if s['split'] == 'train']
        evaluate(config, limit=1)
        build_report(config, limit=1)
        root_results = Path(config['output']) / 'checks/limit_1'
        summary = read_json(root_results / 'summary.json')
        assert summary['complete'] and summary['limited']
        for dataset in config['sources']:
            records = [read_json(p) for p in (root_results / dataset).glob('*.json')]
            assert {r['method'] for r in records} == set(METHODS)
            assert len({r['rows'][0]['sample_id'] for r in records}) == 1
            assert all(r['input_size'] == 64 for r in records)
        # 準備済み原画像の書換えを黙って使わないことを確認する。
        source = Path(config['output']) / 'datasets/m1194m'
        sample = manifests['m1194m']['samples'][0]
        image_path = (source / sample['image']).resolve()
        assert sample_image(source, sample).shape == (64, 64, 3)
        image_path.write_bytes(b'changed')
        with pytest.raises(ValueError, match='入力が変わって'):
            verify_prepared(config)
