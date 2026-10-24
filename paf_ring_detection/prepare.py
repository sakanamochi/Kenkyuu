"""二つの学習元を固定分割し、同じ撮像効果を持つ参照manifestを作る。"""

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from paf_ring_detection.data import read_image, read_json, label_ellipse, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _split_groups(groups, settings, seed):
    """派生画像を作る前に、カメラまたは形状を分割する。"""
    groups = sorted(set(groups))
    random.Random(seed).shuffle(groups)
    train_end = int(len(groups) * settings['train_fraction'])
    validation_end = train_end + round(len(groups) * settings['validation_fraction'])
    return {group: ('train' if i < train_end else 'validation' if i < validation_end else 'test')
            for i, group in enumerate(groups)}


def _variants(split, settings, rng):
    """通常画像に加え、遮蔽・白飛び・黒つぶれを全分割へ含める。"""
    if split == 'train':
        return [('clean', 0.0)] + [(name, float(rng.uniform(settings['severity_min'], settings['severity_max'])))
                                   for name in settings['effects']]
    severities = settings['validation_severities'] if split == 'validation' else settings['test_severities']
    return [('clean', 0.0)] + [(name, severity) for name in settings['effects'] for severity in severities]


def prepare_training_data(config):
    """原画像を複製せず、学習・validation枚数を二つの学習元で揃える。"""
    selected, provenance = {}, {}
    for name, source in config['sources'].items():
        folder = Path(source['path'])
        manifest = read_json(folder / 'manifest.json')
        samples = manifest['samples']
        groups = [s['conditions']['camera_id'] if name == '1194m' else s['shape_id'] for s in samples]
        assignments = _split_groups(groups, config['split'], source['split_seed'])
        # データ破損と寸法重複は派生manifestを保存する前に検出する。
        if name == 'diversified':
            identities = [tuple(sorted(s['parameters'].items())) for s in samples]
            if len(set(identities)) != len(identities):
                raise ValueError('汎化データに同一寸法の重複があります')
        selected[name] = {split: [] for split in ('train', 'validation', 'test')}
        for sample, group in zip(samples, groups):
            image_path, label_path = folder / sample['image'], folder / sample['label']
            image = read_image(image_path)
            label = read_json(label_path)
            points = np.asarray(label['image_points'])
            if (image is None or image.shape[:2] != (label['image_height'], label['image_width'])
                    or label['image_height'] != label['image_width'] or not np.isfinite(points).all()
                    or points.min() < 0 or points.max() >= label['image_width']):
                raise ValueError(f'画像または正解が不正です: {sample["sample_id"]}')
            label_ellipse(label)
            hashes = {key: digest(folder / sample[key]) for key in ('image', 'label')}
            for key, value in hashes.items():
                if sample.get(key + '_sha256', value) != value:
                    raise ValueError(f'元データのhashが不一致です: {sample["sample_id"]}')
            selected[name][assignments[group]].append({**sample, 'group_id': group,
                'image_sha256': hashes['image'], 'label_sha256': hashes['label']})
        provenance[name] = digest(folder / 'manifest.json')

    counts = {split: min(len(parts[split]) for parts in selected.values()) for split in ('train', 'validation')}
    if min(counts.values()) < 1:
        raise ValueError('学習・validation画像が不足しています')
    manifests = {}
    for name, parts in selected.items():
        source = Path(config['sources'][name]['path'])
        output = Path(config['output']) / 'datasets' / name
        records, base_counts = [], {}
        for split, candidates in parts.items():
            random.Random(config['augmentation_seed']).shuffle(candidates)
            candidates = candidates[:counts[split]] if split in counts else candidates
            base_counts[split] = len(candidates)
            for index, sample in enumerate(candidates):
                # 学習元に依存しない乱数により、両CNNの劣化強度分布を揃える。
                rng = np.random.default_rng(config['augmentation_seed'] + index)
                for variant, (effect, severity) in enumerate(_variants(split, config['training_data'], rng)):
                    effect_seed = config['augmentation_seed'] + index * 100 + variant
                    conditions = {**sample['conditions'], 'degradation': effect, 'severity': severity,
                                  'base_sample_id': sample['sample_id']}
                    records.append({**sample, 'sample_id': f'{sample["sample_id"]}__{variant:02d}',
                        'split': split, 'image': Path(os.path.relpath(source / sample['image'], output)).as_posix(),
                        'label': Path(os.path.relpath(source / sample['label'], output)).as_posix(),
                        'conditions': conditions, 'augmentation': {'name': effect, 'severity': severity, 'seed': effect_seed}})
        value = {'source': name, 'source_manifest_sha256': provenance[name],
                 'split_seed': config['sources'][name]['split_seed'], 'base_counts': base_counts,
                 'counts': dict(Counter(s['split'] for s in records)), 'samples': records}
        path = output / 'manifest.json'
        if path.exists() and read_json(path) != value:
            raise ValueError(f'保存済み分割・入力と異なります。別のoutputを指定してください: {path}')
        write_json(path, value)
        manifests[name] = digest(path)
        print(name, '原画像', base_counts, '劣化込み', value['counts'], flush=True)
    write_json(Path(config['output']) / 'readiness.json', {
        'status': 'prepared', 'matched_base_counts': counts, 'source_manifests': provenance,
        'dataset_manifests': manifests,
        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()})


def verify_prepared(config):
    """準備後の設定・分割・原画像・正解の変更を学習と評価の前に検出する。"""
    root = Path(config['output'])
    readiness = read_json(root / 'readiness.json')
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()
    if config_hash != readiness['config_sha256']:
        raise ValueError('準備時と設定が異なります。別のoutputでprepareしてください')
    for name in config['sources']:
        folder = root / 'datasets' / name
        path = folder / 'manifest.json'
        if digest(path) != readiness['dataset_manifests'][name]:
            raise ValueError(f'準備後に分割が変わっています: {path}')
        checked = set()
        for sample in read_json(path)['samples']:
            for key in ('image', 'label'):
                target = (folder / sample[key]).resolve()
                if target not in checked:
                    if digest(target) != sample[key + '_sha256']:
                        raise ValueError(f'準備後に入力が変わっています: {target}')
                    checked.add(target)
