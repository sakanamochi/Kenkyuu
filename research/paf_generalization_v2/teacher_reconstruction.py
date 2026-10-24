"""完全な境界教師を推定器へ渡し、教師表現と離散化による復元誤差を診断する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization import boundary
from research.paf_generalization.diagnose import BASE, sha, image_and_truth

OUT = ROOT/'output/paf_generalization_v2'


def protocol():
    """固定教師・固定推定器を使い、診断の入力と実装を保存する。"""
    paths = [Path(__file__), ROOT/'research/paf_generalization/boundary.py',
             ROOT/'research/paf_generalization/diagnose.py', ROOT/'paf_ring_detection/data.py',
             ROOT/'paf_ring_detection/effects.py', ROOT/'paf_ring_detection/geometry.py',
             ROOT/'paf_ring_detection/methods/ransac.py']
    return {'files': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'manifests': {s: sha(BASE/f'datasets/{s}/manifest.json') for s in ('1194m', 'diversified')},
        'ransac': read_json(ROOT/'research/paf_generalization/boundary.json')['ransac'],
        'opencv_seed_per_image': 20261025,
        'scope': '通常・半画面遮蔽の正解境界教師を予測確率の代わりに入力する診断。GTと既知の矩形を使用し、学習済みCNNの性能・運用可能な方式・厳密な性能上限ではない。'}


def summarize(rows):
    """同じ入力条件で、通常と矩形遮蔽を対象・視点別に集計する。"""
    summary = []
    for source in ('1194m', 'diversified'):
        tilts = [None]+sorted({r['tilt'] for r in rows if r['dataset'] == source and r['tilt'] is not None})
        for effect in ('clean', 'black_rectangle'):
            for tilt in tilts:
                group = [r for r in rows if r['dataset'] == source and r['degradation'] == effect
                         and (tilt is None or r['tilt'] == tilt)]
                if group:
                    summary.append({'dataset': source, 'degradation': effect, 'tilt': tilt, 'n': len(group),
                        'success': sum(r['success'] for r in group),
                        'mean_iou': float(np.mean([r['ellipse_iou'] for r in group]))})
    return summary


def run(audit=False):
    """監査時は教師から細線化・RANSAC・元GTのIoUまで再実行する。"""
    destination = OUT/'teacher_reconstruction.json'
    spec = protocol()
    if destination.exists() and not audit:
        assert read_json(destination)['protocol'] == spec
        return
    previous = read_json(destination) if audit else None
    if previous:
        assert previous['protocol'] == spec
    rows = []
    for source in ('1194m', 'diversified'):
        dataset = boundary.BoundaryDataset(BASE/'datasets'/source, 'test', 256, cache=False)
        for index, sample in enumerate(dataset.samples):
            condition = sample['conditions']
            if (condition['degradation'], condition['severity']) not in (('clean', 0.), ('black_rectangle', .5)):
                continue
            _, masks, _ = dataset[index]
            probability = boundary.boundary_target(masks[None])[0, 0].numpy()
            _, truth = image_and_truth(dataset.root, sample)
            points, _ = boundary.skeleton_points(probability)
            cv2.setRNGSeed(spec['opencv_seed_per_image'])
            ellipse = boundary.boundary_ellipse(probability, spec['ransac'])
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            # JSONと再実行の座標型を揃え、数値も成否も完全一致で検証する。
            encoded = [list(ellipse[0]), list(ellipse[1]), ellipse[2]] if ellipse else None
            rows.append({'dataset': source, 'sample_id': sample['sample_id'],
                'degradation': condition['degradation'], 'severity': condition['severity'],
                'tilt': condition.get('camera', {}).get('tilt_deg'), 'n_points': len(points),
                'ellipse': encoded, 'ellipse_iou': iou, 'success': iou >= .8})
        print(source, len(rows), flush=True)
    assert len(rows) == 532
    result = {'protocol': spec, 'rows': rows, 'summary': summarize(rows)}
    if audit:
        assert result == previous
        write_json(OUT/'teacher_reconstruction_audit.json', {'input_sha256': sha(destination),
            'recomputed_images': len(rows), 'scope': '元教師・細線化・RANSAC・元ラベルIoU・全集計を再計算して完全一致。'})
    else:
        write_json(destination, result)
    print({'audit': audit, 'images': len(rows)}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    run(parser.parse_args().audit)
