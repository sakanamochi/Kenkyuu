"""投影を揃えた再描画・補間画像で、同じ内周境界CNNを学習する。"""

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, read_image
from research.paf_generalization import experiment, boundary, boundary_surface

DATA = ROOT/'output/paf_generalization_v2/native_framing'
MODE = 'native'
BASE_FREEZE = boundary.freeze_boundary


def prepare():
    """256pxの元画像を、再描画と同じ480pxの投影へ補間する。"""
    verification = read_json(DATA/'verification.json')
    assert verification['plan_sha256'] == experiment.sha(DATA/'plan.json')
    protocol = {'implementation_sha256': experiment.sha(Path(__file__)),
                'render_verification_sha256': experiment.sha(DATA/'verification.json'),
                'intermediate_size': 256, 'final_size': 480,
                'effect_order': 'apply_same_effect_at_480_after_projection_then_gray256'}
    path = DATA/'resampled_plan.json'
    if path.exists():
        assert read_json(path) == protocol
    else:
        write_json(path, protocol)
    for style in ('plain', 'surface'):
        source_root = experiment.BASE/'datasets/diversified' if style == 'plain' else boundary_surface.DATA
        source = {s['sample_id']: s for s in read_json(source_root/'manifest.json')['samples']}
        native = read_json(DATA/f'native_{style}/manifest.json')['samples']
        rows, seen = [], set()
        for sample in native:
            sid = sample['conditions']['base_sample_id']
            destination = DATA/f'resampled_{style}/images/{sid}.png'
            if sid not in seen:
                label = read_json(sample['label'])
                frame = label['framing']
                original = source[sample['sample_id']]
                image = read_image(source_root/original['image'])
                assert experiment.sha(source_root/original['image']) == original['image_sha256']
                small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
                scale = frame['scale']
                center, target = np.asarray(frame['source_center_480']), np.asarray(frame['center_480'])
                # OpenCV resizeの画素中心対応を含め、元480px座標と再描画を一致させる。
                offset = target-scale*center+scale*(480/256-1)/2
                matrix = np.array([[scale*480/256, 0, offset[0]],
                                   [0, scale*480/256, offset[1]]], np.float32)
                transformed = cv2.warpAffine(small, matrix, (480, 480), flags=cv2.INTER_LINEAR)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    assert np.array_equal(read_image(destination), transformed)
                else:
                    success, encoded = cv2.imencode('.png', transformed)
                    assert success
                    encoded.tofile(destination)
                seen.add(sid)
            rows.append({**sample, 'image': str(destination), 'image_sha256': experiment.sha(destination)})
        write_json(DATA/f'resampled_{style}/manifest.json', {'samples': rows})
    write_json(DATA/'paired_verification.json', {'n_shapes': verification['n_shapes'],
               'plan_sha256': experiment.sha(path),
               'manifests': {f'{mode}_{style}': experiment.sha(DATA/f'{mode}_{style}/manifest.json')
                             for mode in ('native', 'resampled') for style in ('plain', 'surface')}})
    preview()


def preview():
    """全形状の投影サイズの分位点から例を選び、学習入力の対応を可視化する。"""
    plan = sorted(read_json(DATA/'plan.json')['plan'], key=lambda r: r['major_axis_256'])
    original_root = experiment.BASE/'datasets/diversified'
    originals = {s['conditions']['base_sample_id']: s for s in read_json(original_root/'manifest.json')['samples']
                 if s['conditions']['degradation'] == 'clean'}
    canvas = np.zeros((4*260, 4*256, 3), np.uint8)
    rows = []
    for index, quantile in enumerate((.1, .4, .7, .95)):
        item = plan[round(quantile*(len(plan)-1))]
        sid = item['base_sample_id']
        label = read_json(DATA/f'labels/{sid}.json')
        inputs = [original_root/originals[sid]['image']]
        inputs += [DATA/f'{kind}/images/{sid}.png' for kind in ('resampled_plain', 'native_plain', 'native_surface')]
        for column, path in enumerate(inputs):
            image = cv2.resize(read_image(path), (256, 256), interpolation=cv2.INTER_AREA)
            if column:
                points = (np.asarray(label['image_points'])*256/480).round().astype(np.int32)
                cv2.polylines(image, [points], True, (0, 255, 0), 1, cv2.LINE_AA)
            canvas[index*260:index*260+256, column*256:(column+1)*256] = image
            title = ('source', 'interpolated', 'native', 'surface')[column]
            cv2.putText(canvas, f'{sid} {title}', (column*256+4, index*260+18),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        rows.append({'quantile': quantile, **item,
                     'image_hashes': {str(p.relative_to(ROOT)): experiment.sha(p) for p in inputs}})
    success, encoded = cv2.imencode('.png', canvas)
    assert success
    encoded.tofile(DATA/'projection_preview.png')
    write_json(DATA/'projection_preview.json', {'selection': 'all_602_shapes_major_axis_quantiles', 'rows': rows})


class FramedDataset(boundary.BoundaryDataset):
    """学習時は同じ構図の元材質と表面模様を等確率で使い、validationは元画像を使う。"""

    def __init__(self, dataset_dir, split, input_size, cache=False):
        active = Path(dataset_dir).name == 'diversified' and split == 'train'
        super().__init__(DATA/f'{MODE}_plain' if active else dataset_dir, split, input_size, cache)
        self.alternative = boundary.BoundaryDataset(DATA/f'{MODE}_surface', split, input_size, cache) if active else None
        if active:
            assert [s['sample_id'] for s in self.samples] == [s['sample_id'] for s in self.alternative.samples]
            self.rng = np.random.default_rng(torch.initial_seed()+5200)

    def __getitem__(self, index):
        if self.alternative is not None and self.rng.random() < .5:
            return self.alternative[index]
        return super().__getitem__(index)


def freeze(folder, config, variant, seed):
    """対応画像・投影・学習器を固定し、1194Mを学習に使わない。"""
    verified = read_json(DATA/'paired_verification.json')
    seen = set()
    for name, digest in verified['manifests'].items():
        path = DATA/name/'manifest.json'
        assert experiment.sha(path) == digest
        for sample in read_json(path)['samples']:
            for field in ('image', 'label'):
                p = Path(sample[field])
                if p not in seen:
                    assert experiment.sha(p) == sample[field+'_sha256']
                    seen.add(p)
    config = copy.deepcopy(config)
    config['framing_files'] = {str(p.relative_to(ROOT)): experiment.sha(p) for p in
                              (Path(__file__), Path(__file__).with_name('render_native.py'),
                               DATA/'plan.json', DATA/'verification.json', DATA/'resampled_plan.json',
                               DATA/'paired_verification.json')}
    return BASE_FREEZE(folder, config, variant, seed)


BASE_TRAIN = experiment.train_variant


def train(variant, seed):
    global MODE
    MODE = read_json(experiment.CONFIG)['variants'][variant]['input_mode']
    return BASE_TRAIN(variant, seed)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'prepare':
        prepare()
    else:
        experiment.CONFIG = Path(__file__).with_suffix('.json')
        experiment.OUT = ROOT/'output/paf_generalization_v2'
        experiment.RingDataset = FramedDataset
        experiment.objective = boundary.boundary_objective
        experiment.freeze_run = freeze
        boundary.freeze_boundary = freeze
        experiment.train_variant = train
        experiment.evaluate_variant = boundary.evaluate_boundary
        experiment.main()
