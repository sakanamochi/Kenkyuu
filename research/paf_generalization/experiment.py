"""空間的な汎化学習と微分可能な楕円制約の対照実験。"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import RingDataset, read_json, write_json, region_mask
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.methods.cnn_region import region_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.train import _dice_loss
from research.paf_generalization.diagnose import (BASE, OUT, backup, sha, image_and_truth)

CONFIG = Path(__file__).with_name('experiment.json')


def moments(probability):
    """面積重みの中心と共分散を計算する。座標は画素中心を[-1,1]へ写す。"""
    n = probability.shape[-1]
    coordinates = (torch.arange(n, device=probability.device, dtype=probability.dtype) + .5) * 2 / n - 1
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    weight = probability[:, 0]
    mass = weight.sum((1, 2)).clamp_min(1e-6)
    cx, cy = (weight * xx).sum((1, 2)) / mass, (weight * yy).sum((1, 2)) / mass
    dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    vx = (weight * dx.square()).sum((1, 2)) / mass + 1e-5
    vy = (weight * dy.square()).sum((1, 2)) / mass + 1e-5
    xy = (weight * dx * dy).sum((1, 2)) / mass
    return cx, cy, vx, vy, xy, dx, dy


def ellipse_projection(probability):
    """一様楕円領域の共分散=半軸二乗/4を用いて、滑らかな楕円を描画する。"""
    cx, cy, vx, vy, xy, dx, dy = moments(probability)
    determinant = (vx * vy - xy.square()).clamp_min(1e-9)
    # q=1が楕円境界。画素ごとの穴確率を5自由度の楕円へ射影する。
    q = (vy[:, None, None] * dx.square() + vx[:, None, None] * dy.square()
         - 2 * xy[:, None, None] * dx * dy) / (4 * determinant[:, None, None])
    rendered = torch.sigmoid(8 * (1-q))[:, None]
    return rendered, (cx, cy, vx, vy, xy)


def geometry_loss(logits, masks):
    """背景の微小確率を抑えた面積モーメントと、完全楕円の重なりを教師にする。"""
    prediction, values = ellipse_projection(torch.sigmoid(logits).pow(4))
    _, target = ellipse_projection(masks.pow(4))
    cx, cy, vx, vy, xy = values
    tx, ty, tvx, tvy, txy = target
    center = ((cx-tx).square() + (cy-ty).square()) / (tvx+tvy).clamp_min(1e-4)
    covariance = ((vx-tvx).square() + (vy-tvy).square() + 2*(xy-txy).square()) / (tvx+tvy).square().clamp_min(1e-8)
    overlap = 1 - (2*(prediction*masks).sum((1, 2, 3))+1) / (prediction.sum((1, 2, 3))+masks.sum((1, 2, 3))+1)
    # 初期の一様出力で極端な勾配を出さず、相対的な幾何誤差を減らす。
    return (torch.log1p(center) + torch.log1p(covariance) + overlap).mean()


def spatial_augment(images, masks, rng, config):
    """教師と画像へ同じ相似変換を行い、内周の位置・大きさを独立に変える。"""
    batch = len(images)
    center_x, center_y, *_ = moments(masks)
    scales = np.exp(rng.uniform(np.log(config['scale_range'][0]), np.log(config['scale_range'][1]), batch))
    target = rng.uniform(*config['center_range'], (batch, 2)) * 2 - 1
    active = rng.random(batch) < config['spatial_probability']
    matrices = np.zeros((batch, 2, 3), np.float32)
    for i in range(batch):
        # 教師を用いるのは学習時の切り出しだけ。推論時には画像全体を入力する。
        ys, xs = torch.where(masks[i, 0] > .5)
        cx, cy = float(center_x[i]), float(center_y[i])
        ex = max(abs((float(xs.min())+.5)/128-1-cx), abs((float(xs.max())+.5)/128-1-cx))
        ey = max(abs((float(ys.min())+.5)/128-1-cy), abs((float(ys.max())+.5)/128-1-cy))
        tx, ty = target[i]
        scale = min(scales[i], (.95-abs(tx))/max(ex, 1e-3), (.95-abs(ty))/max(ey, 1e-3))
        matrices[i, 0, 0] = matrices[i, 1, 1] = 1/scale
        matrices[i, 0, 2], matrices[i, 1, 2] = cx-tx/scale, cy-ty/scale
        if not active[i]:
            matrices[i] = np.array([[1, 0, 0], [0, 1, 0]])
    theta = torch.as_tensor(matrices, device=images.device)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    transformed_images = F.grid_sample(images+1, grid, align_corners=False) - 1
    transformed_masks = F.grid_sample(masks, grid, align_corners=False)
    return transformed_images, transformed_masks


def objective(logits, masks, settings, config, bce):
    loss = bce(logits, masks) + _dice_loss(logits, masks)
    if settings['geometry']:
        loss = loss + config['geometry_weight'] * geometry_loss(logits, masks)
    return loss


def freeze_run(folder, config, variant, seed):
    """設定・コード・分割を固定し、異なる内容で既存実験を上書きしない。"""
    paths = [Path(__file__), CONFIG, ROOT/'paf_ring_detection/data.py',
             ROOT/'paf_ring_detection/effects.py', ROOT/'paf_ring_detection/methods/cnn.py',
             ROOT/'paf_ring_detection/methods/cnn_region.py', ROOT/'paf_ring_detection/geometry.py']
    provenance = {'config': config, 'variant': variant, 'seed': seed,
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        'manifests': {name: sha(BASE/f'datasets/{name}/manifest.json') for name in ('1194m', 'diversified')},
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'training_source': 'diversified', 'target_data_used_for_training': False}
    path = folder / 'provenance.json'
    if path.exists():
        assert read_json(path) == provenance, '実験内容が異なるため別の保存先が必要です'
    else:
        write_json(path, provenance)
    return provenance


def train_variant(variant, seed):
    config = read_json(CONFIG)
    settings = config['variants'][variant]
    folder = OUT / 'runs' / variant / str(seed)
    provenance = freeze_run(folder, config, variant, seed)
    completed = folder / 'complete.json'
    if completed.exists():
        assert read_json(completed)['checkpoint_sha256'] == sha(folder/'best.pt')
        return
    if (folder/'training_history.json').exists():
        raise RuntimeError('未完了学習があります。別実験として保存するか実行状態を確認してください')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda')
    train_data = RingDataset(BASE/'datasets/diversified', 'train', 256, cache=True)
    validation = RingDataset(BASE/'datasets/diversified', 'validation', 256, cache=True)
    options = dict(batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, **options)
    val_loader = DataLoader(validation, shuffle=False, **options)
    model = TinyUNet(16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['positive_weight']], device=device))
    rng = np.random.default_rng(seed+4100)
    best, history = float('inf'), []
    print(f'TRAIN {variant} {seed}', flush=True)
    for epoch in range(1, config['epochs']+1):
        started = time.monotonic()
        model.train()
        total = 0.
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            if settings['spatial']:
                images, masks = spatial_augment(images, masks, rng, config)
            optimizer.zero_grad()
            logits = model(images)
            loss = objective(logits, masks, settings, config, bce)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
        model.eval()
        val = 0.
        with torch.inference_mode():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                val += float(objective(model(images), masks, settings, config, bce)) * len(images)
        row = {'epoch': epoch, 'train_loss': total/len(train_data),
               'validation_loss': val/len(validation), 'elapsed_seconds': time.monotonic()-started}
        assert math.isfinite(row['train_loss']) and math.isfinite(row['validation_loss'])
        history.append(row)
        if row['validation_loss'] < best:
            best = row['validation_loss']
            torch.save({'model_state': model.state_dict(), 'epoch': epoch,
                        'validation_loss': best, 'provenance': provenance}, folder/'best.pt')
        write_json(folder/'training_history.json', history)
        print(variant, seed, row, flush=True)
    write_json(completed, {'epochs': len(history), 'best_validation_loss': best,
                          'checkpoint_sha256': sha(folder/'best.pt')})


def moment_ellipse(probability):
    """学習時の幾何射影と同じ面積モーメントをOpenCV楕円へ変換する。"""
    with torch.inference_mode():
        _, values = ellipse_projection(torch.from_numpy(probability)[None, None].pow(4))
    cx, cy, vx, vy, xy = [float(v[0]) for v in values]
    eigenvalues, eigenvectors = np.linalg.eigh([[vx, xy], [xy, vy]])
    if min(eigenvalues) <= 0 or not np.isfinite(eigenvalues).all():
        return None
    direction = eigenvectors[:, 0]
    return (((cx+1)*128-.5, (cy+1)*128-.5), tuple((4*np.sqrt(eigenvalues)*128).tolist()),
            float(np.rad2deg(np.arctan2(direction[1], direction[0]))))


def evaluate_variant(variant, seed):
    config = read_json(CONFIG)
    folder = OUT/'runs'/variant/str(seed)
    provenance = freeze_run(folder, config, variant, seed)
    checkpoint = folder/'best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['provenance'] == provenance
    model = TinyUNet(16).cuda().eval()
    model.load_state_dict(saved['model_state'])
    for source in ('1194m', 'diversified'):
        path = folder/f'test_{source}.json'
        if path.exists():
            assert read_json(path)['checkpoint_sha256'] == sha(checkpoint)
            continue
        dataset = RingDataset(BASE/'datasets'/source, 'test', 256)
        rows = []
        for index, sample in enumerate(dataset.samples):
            image, truth = image_and_truth(dataset.root, sample)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            ellipse = region_ellipse(probability)
            metric = evaluate_ellipses(ellipse, truth, (256, 256)) if ellipse else {'ellipse_iou': 0.}
            mask = region_mask(256, truth) > .5
            binary = probability >= .5
            row = {'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                'base_sample_id': sample['conditions']['base_sample_id'],
                'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                'ellipse': ellipse, 'truth_ellipse': truth, **metric, 'success': metric['ellipse_iou'] >= .8,
                'mask_iou': float((binary & mask).sum()/max(1, (binary | mask).sum()))}
            if config['variants'][variant]['geometry']:
                projected = moment_ellipse(probability)
                projected_iou = evaluate_ellipses(projected, truth, (256, 256))['ellipse_iou'] if projected else 0.
                row.update(projected_ellipse=projected, projected_iou=projected_iou, projected_success=projected_iou >= .8)
            rows.append(row)
        write_json(path, {'dataset': source, 'variant': variant, 'seed': seed,
            'checkpoint_sha256': sha(checkpoint), 'provenance': provenance, 'rows': rows})
        print('EVAL', variant, seed, source, sum(r['success'] for r in rows), len(rows), flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    os.chdir(ROOT)
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'evaluate', 'run'])
    parser.add_argument('--variants', nargs='+', default=list(read_json(CONFIG)['variants']))
    parser.add_argument('--seeds', nargs='+', type=int, default=read_json(CONFIG)['seeds'])
    args = parser.parse_args()
    backup()
    for seed in args.seeds:
        for variant in args.variants:
            if args.stage in ('train', 'run'):
                train_variant(variant, seed)
            if args.stage in ('evaluate', 'run'):
                evaluate_variant(variant, seed)


if __name__ == '__main__':
    main()
