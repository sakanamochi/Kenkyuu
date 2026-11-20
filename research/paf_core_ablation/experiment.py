"""mainのリングCNNを保ち、投影・表面模様の2要因を対応比較する。"""

import argparse
import ast
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import (read_json, write_json, load_samples, sample_image,
                                     label_ellipse, scaled_ellipse)
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.train import _dice_loss
from research.paf_generalization.diagnose import sha, image_and_truth
from research.paf_core_ablation.main_compat import rasterize_ring_mask, probability_ellipse

OUT = ROOT/'output/paf_core_ablation_v1'
BASE = ROOT/'output/hole_region_comparison_v1/datasets'
CONFIG = Path(__file__).with_suffix('.json')
SURFACE = ROOT/'output/paf_generalization_v1/surface_counterfactual'
PROJECTION = ROOT/'output/paf_generalization_v2/projection_coverage'


def view_roots(settings, split):
    """検証時は元の学習元だけを使い、対応ビューは学習でのみ抽選する。"""
    original = BASE/settings['source']
    if split != 'train' or settings['source'] != 'diversified':
        return {(False, False): original}
    roots = {(False, False): original}
    if settings['surface']:
        roots[False, True] = SURFACE
    if settings['projection']:
        roots[True, False] = PROJECTION/'native_plain'
        if settings['surface']:
            roots[True, True] = PROJECTION/'native_surface'
    return roots


class FactorialDataset(Dataset):
    """二つの乱数を常に消費し、要因の有無で他方の抽選列が変わらないようにする。"""

    def __init__(self, settings, split, seed, cache=True):
        self.settings, self.split = settings, split
        self.roots = view_roots(settings, split)
        self.views = {key: load_samples(root, split) for key, root in self.roots.items()}
        self.samples = self.views[False, False]
        ids = [s['sample_id'] for s in self.samples]
        assert all([s['sample_id'] for s in values] == ids for values in self.views.values())
        self.rng = np.random.default_rng(seed+5200)
        self.images = {} if cache else None
        self.targets = {}

    def __len__(self):
        return len(self.samples)

    def get_view(self, key, index):
        """画像は8bitで保持し、教師はmainと同じfloat32の完全リングを保持する。"""
        root, sample = self.roots[key], self.views[key][index]
        image_key = key, index
        gray = self.images.get(image_key) if self.images is not None else None
        if gray is None:
            image = sample_image(root, sample)
            gray = cv2.resize(image[:, :, 0], (256, 256), interpolation=cv2.INTER_AREA)
            if self.images is not None:
                self.images[image_key] = gray
        label_path = root/sample['label']
        if label_path not in self.targets:
            label = read_json(label_path)
            ellipse = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
            self.targets[label_path] = rasterize_ring_mask(256, 256, ellipse, thickness=3, blur_sigma=.8)
        image = (torch.from_numpy(gray).float()/127.5-1)[None].expand(3, -1, -1)
        mask = torch.from_numpy(self.targets[label_path])[None]
        return image, mask, index

    def __getitem__(self, index):
        key = False, False
        if self.split == 'train':
            p, s = self.rng.random(2) < .5
            key = bool(p and self.settings['projection']), bool(s and self.settings['surface'])
        return self.get_view(key, index)


def prepare():
    """参照コード・入力実体・分割を照合し、共通の実験計画を一度だけ固定する。"""
    config = read_json(CONFIG)
    ref = OUT/'main_reference'
    inventory = read_json(ref/'inventory.json')
    assert inventory['main_commit'] == config['main_commit']
    for path, digest in inventory['files'].items():
        assert sha(ref/path) == digest, path

    def definitions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in
                ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    model = definitions(ROOT/'paf_ring_detection/methods/cnn.py')
    assert all(model[k] == v for k, v in definitions(ref/'paflab/model.py').items())
    reference_ransac = definitions(ref/'analysis/ellipse_ransac.py')
    current_ransac = definitions(ROOT/'paf_ring_detection/methods/ransac.py')
    for name in ('_point_distances', '_angular_coverage', '_ellipse_perimeter',
                 '_ransac_hypothesis_score', '_is_valid_ellipse', 'fit_ellipse_ransac'):
        assert reference_ransac[name] == current_ransac[name], name
    compatibility = definitions(Path(__file__).with_name('main_compat.py'))
    for name, path in [('rasterize_ring_mask', 'paflab/labels.py'),
                       ('probability_ellipse', 'paflab/evaluate_cnn.py')]:
        assert compatibility[name] == definitions(ref/path)[name], name
    paths = [Path(__file__), CONFIG, Path(__file__).with_name('main_compat.py'),
             ROOT/'paf_ring_detection/data.py', ROOT/'paf_ring_detection/effects.py',
             ROOT/'paf_ring_detection/methods/cnn.py', ROOT/'paf_ring_detection/methods/ransac.py',
             ROOT/'paf_ring_detection/train.py', ROOT/'paf_ring_detection/geometry.py',
             ROOT/'research/paf_generalization/diagnose.py', ref/'inventory.json']
    roots = {BASE/'1194m', BASE/'diversified', SURFACE,
             PROJECTION/'native_plain', PROJECTION/'native_surface'}
    checked = {}
    for root in sorted(roots):
        paths.append(root/'manifest.json')
        for sample in read_json(root/'manifest.json')['samples']:
            for field in ('image', 'label'):
                path = root/sample[field]
                if path not in checked:
                    checked[path] = sha(path)
                assert checked[path] == sample[field+'_sha256'], path
    plan = {'config': config, 'files': {str(p.relative_to(ROOT)): sha(p) for p in paths},
            'input_files': {str(p): v for p, v in checked.items()},
            'main_model_and_ring_and_ransac_ast_identical': True,
            'torch': torch.__version__, 'device': torch.cuda.get_device_name()}
    path = OUT/'plan.json'
    if path.exists():
        assert read_json(path) == plan, '固定計画との不一致。別実験として扱ってください。'
    else:
        write_json(path, plan)
    return config


def train(variant, seed, config):
    settings = config['variants'][variant]
    folder = OUT/'runs'/variant/str(seed)
    provenance = {'variant': variant, 'seed': seed, 'plan_sha256': sha(OUT/'plan.json'),
                  'training_source': settings['source'],
                  'target_data_used_for_training': settings['source'] == '1194m'}
    if (folder/'provenance.json').exists():
        assert read_json(folder/'provenance.json') == provenance
    else:
        write_json(folder/'provenance.json', provenance)
    if (folder/'complete.json').exists():
        assert read_json(folder/'complete.json')['checkpoint_sha256'] == sha(folder/'best.pt')
        return
    if (folder/'training_history.json').exists():
        raise RuntimeError('未完了学習の記録があります。実プロセスを確認し、上書きしないでください。')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = FactorialDataset(settings, 'train', seed)
    validation = FactorialDataset(settings, 'validation', seed)
    assert len(data) == 1988 and len(validation) == 735
    train_loader = DataLoader(data, batch_size=config['batch_size'], shuffle=True,
                              generator=torch.Generator().manual_seed(seed), pin_memory=True)
    val_loader = DataLoader(validation, batch_size=config['batch_size'], pin_memory=True)
    model = TinyUNet(16).cuda()
    assert sum(p.numel() for p in model.parameters()) == 482737
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['learning_rate']*.05)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['positive_weight']], device='cuda'))
    best, history = float('inf'), []
    print('TRAIN', variant, seed, flush=True)
    for epoch in range(1, config['epochs']+1):
        start = time.monotonic()
        totals = []
        for is_train, loader in ((True, train_loader), (False, val_loader)):
            model.train(is_train)
            total = 0.
            with torch.set_grad_enabled(is_train):
                for images, masks, _ in loader:
                    images, masks = images.cuda(), masks.cuda()
                    if is_train:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = bce(logits, masks)+config['dice_weight']*_dice_loss(logits, masks)
                    if is_train:
                        loss.backward()
                        optimizer.step()
                    total += float(loss.detach())*len(images)
            totals.append(total/len(loader.dataset))
        assert all(math.isfinite(value) for value in totals)
        row = {'epoch': epoch, 'train_loss': totals[0], 'validation_loss': totals[1],
               'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_seconds': time.monotonic()-start}
        if totals[1] < best:
            best = totals[1]
            torch.save({'model_state': model.state_dict(), 'base_channels': 16, 'epoch': epoch,
                        'validation_loss': best, 'provenance': provenance}, folder/'best.pt')
        history.append(row)
        write_json(folder/'training_history.json', history)
        scheduler.step()
        print(variant, seed, row, flush=True)
    write_json(folder/'complete.json', {'epochs': len(history), 'best_validation_loss': best,
                                       'checkpoint_sha256': sha(folder/'best.pt')})


def evaluate(variant, seed, config):
    """mainの閾値点群とRANSACを保ち、両対象の全条件を共通GTで評価する。"""
    folder = OUT/'runs'/variant/str(seed)
    checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['provenance'] == read_json(folder/'provenance.json')
    assert checkpoint['provenance']['plan_sha256'] == sha(OUT/'plan.json')
    assert read_json(folder/'complete.json')['checkpoint_sha256'] == sha(folder/'best.pt')
    model = TinyUNet(16).cuda().eval()
    model.load_state_dict(checkpoint['model_state'], strict=True)
    for source in ('1194m', 'diversified'):
        destination = folder/f'test_{source}.json'
        if destination.exists():
            assert read_json(destination)['checkpoint_sha256'] == sha(folder/'best.pt')
            continue
        rows = []
        for sample in load_samples(BASE/source, 'test'):
            image, truth = image_and_truth(BASE/source, sample)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            sample_seed = (config['cnn_ransac']['random_seed']+zlib.crc32(sample['sample_id'].encode('utf-8'))) % 2**32
            cv2.setRNGSeed(20261025)
            fit = probability_ellipse(probability, config['cnn_ransac'], random_seed=sample_seed)
            ellipse = fit['ellipse'] if fit else None
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                         'base_sample_id': sample['conditions']['base_sample_id'],
                         'degradation': sample['conditions']['degradation'],
                         'severity': sample['conditions']['severity'], 'truth_ellipse': truth,
                         'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
        write_json(destination, {'variant': variant, 'seed': seed, 'dataset': source,
                                'checkpoint_sha256': sha(folder/'best.pt'),
                                'provenance': checkpoint['provenance'], 'rows': rows})
        print('EVAL', variant, seed, source, sum(r['success'] for r in rows), len(rows), flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'run'])
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=[20260907])
    args = parser.parse_args()
    config = prepare()
    if args.stage == 'prepare':
        print('fixed main-compatible factorial plan', flush=True)
        return
    for seed in args.seeds:
        assert seed in config['seeds']
        for variant in args.variants or config['variants']:
            assert variant in config['variants']
            if args.stage in ('train', 'run'):
                train(variant, seed, config)
            if args.stage in ('evaluate', 'run'):
                evaluate(variant, seed, config)


if __name__ == '__main__':
    main()
