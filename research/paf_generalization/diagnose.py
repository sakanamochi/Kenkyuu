"""既存比較を保護し、寸法・明暗・拡大縮小への依存を再現可能に測る。"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import (read_json, write_json, load_samples,
    label_ellipse, scaled_ellipse, sample_image, region_mask)
from paf_ring_detection.methods.cnn import load_model
from paf_ring_detection.methods.cnn_region import region_ellipse
from paf_ring_detection.geometry import evaluate_ellipses

BASE = ROOT / 'output/hole_region_comparison_v1'
OUT = ROOT / 'output/paf_generalization_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup():
    """比較結果・重み・実装を一度保存し、原画像はhash付き参照で保持する。"""
    target = OUT / 'backup/baseline_20261025'
    if (target / 'inventory.json').exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    files = list(BASE.rglob('*'))
    files += [ROOT / 'run.py', ROOT / 'config/experiment.json',
              ROOT / '.codex/PROJECT_CONTEXT.md', ROOT / '.codex/RESEARCH_HANDOFF.md']
    files += list((ROOT / 'paf_ring_detection').rglob('*.py'))
    inventory = []
    for path in files:
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(ROOT)
        saved = target / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        digest = sha(path)
        assert sha(saved) == digest
        inventory.append({'path': relative.as_posix(), 'sha256': digest,
                          'bytes': path.stat().st_size})
    write_json(target / 'inventory.json', {'files': inventory,
        'raw_data_policy': '原画像は既存manifestのパスとSHA256で参照。原画像は変更しない。'})


def image_and_truth(folder, sample):
    image = cv2.resize(sample_image(folder, sample), (256, 256), interpolation=cv2.INTER_AREA)
    label = read_json(folder / sample['label'])
    truth = scaled_ellipse(label_ellipse(label), 256 / label['image_width'])
    return image, truth


def predict(model, images):
    tensor = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).to('cuda').float() / 127.5 - 1
    with torch.inference_mode():
        return torch.sigmoid(model(tensor))[:, 0].cpu().numpy()


def diagnostics():
    """正解は採点だけに使用し、入力変換・推定選択には使わない。"""
    torch.set_num_threads(4)
    distributions, rows = {}, []
    # 拡大縮小は画像中心に固定する。診断条件の最良値を検出器へ採用しない。
    transforms = [('identity', 1., False), ('scale_050', .5, False),
                  ('scale_075', .75, False), ('scale_125', 1.25, False),
                  ('scale_150', 1.5, False), ('invert', 1., True)]
    for source in ('diversified', '1194m'):
        folder = BASE / 'datasets' / source
        for split in ('train', 'validation', 'test'):
            samples = [s for s in load_samples(folder, split) if s['conditions']['degradation'] == 'clean']
            values = []
            for sample in samples:
                image, truth = image_and_truth(folder, sample)
                mask = region_mask(256, truth) > .5
                gray = image[:, :, 0]
                values.append({'id': sample['sample_id'], 'group': sample['group_id'],
                    'major': max(truth[1]), 'minor': min(truth[1]), 'cx': truth[0][0], 'cy': truth[0][1],
                    'interior_mean': float(gray[mask].mean()),
                    'image_mean': float(gray.mean())})
            distributions[f'{source}/{split}'] = {'n': len(values), 'rows': values,
                'quantiles': {key: np.percentile([v[key] for v in values], [0, 10, 50, 90, 100]).tolist()
                              for key in ('major', 'minor', 'cx', 'cy', 'interior_mean', 'image_mean')}}
        samples = [s for s in load_samples(folder, 'test') if s['conditions']['degradation'] == 'clean']
        loaded = [image_and_truth(folder, sample) for sample in samples]
        for seed in (20260907, 20260908, 20260909):
            model = load_model(BASE / f'models/diversified/{seed}/best.pt', torch.device('cuda'))
            for name, scale, invert in transforms:
                matrix = cv2.getRotationMatrix2D((127.5, 127.5), 0, scale)
                images = [cv2.warpAffine(im, matrix, (256, 256), flags=cv2.INTER_AREA)
                          if scale != 1 else im for im, _ in loaded]
                if invert:
                    images = [255 - im for im in images]
                for start in range(0, len(images), 16):
                    probabilities = (np.concatenate([predict(model, [im]) for im in images[start:start+16]])
                                     if name == 'identity' else predict(model, images[start:start+16]))
                    for j, probability in enumerate(probabilities, start):
                        ellipse = region_ellipse(probability)
                        if ellipse is not None and scale != 1:
                            ellipse = (((ellipse[0][0]-127.5)/scale+127.5,
                                        (ellipse[0][1]-127.5)/scale+127.5),
                                       (ellipse[1][0]/scale, ellipse[1][1]/scale), ellipse[2])
                        truth = loaded[j][1]
                        metric = evaluate_ellipses(ellipse, truth, (256, 256)) if ellipse else {'ellipse_iou': 0.}
                        rows.append({'dataset': source, 'seed': seed, 'transform': name,
                            'sample_id': samples[j]['sample_id'], 'group_id': samples[j]['group_id'],
                            'ellipse': ellipse, 'truth': truth, **metric,
                            'success': metric['ellipse_iou'] >= .8})
                print(source, seed, name, flush=True)
    summary = []
    for source in ('diversified', '1194m'):
        for name, _, _ in transforms:
            selected = [r for r in rows if r['dataset'] == source and r['transform'] == name]
            summary.append({'dataset': source, 'transform': name, 'n_predictions': len(selected),
                'success_rate': float(np.mean([r['success'] for r in selected])),
                'mean_iou': float(np.mean([r['ellipse_iou'] for r in selected]))})
    # 無変換の通常画像は既存の正式推定と全件照合する。
    for source in ('diversified', '1194m'):
        for seed in (20260907, 20260908, 20260909):
            saved = read_json(BASE / f'results/{source}/cnn_diversified_{seed}.json')
            lookup = {r['sample_id']: r for r in saved['rows']}
            for row in rows:
                if row['dataset'] == source and row['seed'] == seed and row['transform'] == 'identity':
                    assert row['ellipse_iou'] == lookup[row['sample_id']]['ellipse_iou']
    write_json(OUT / 'diagnosis.json', {'distributions': distributions, 'rows': rows,
        'summary': summary, 'identity_matches_baseline': True,
        'interpretation': '変換別の探索的感度診断。拡大時の画面切断・縮小時の解像度変化を含む。'})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    backup()
    diagnostics()
