"""リング帯と中心線の違いを、同じmain推定器と固定した点数条件で診断する。"""

import argparse
import copy
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import sha
from research.paf_generalization.boundary import skeleton_points
from research.paf_core_ablation.experiment import OUT, BASE, CONFIG
from research.paf_core_ablation.main_compat import rasterize_ring_mask, probability_ellipse

MODES = {'band_45': (False, 45), 'band_12': (False, 12),
         'centerline_45': (True, 45), 'centerline_12': (True, 12)}


def predict(probability, settings, seed, mode):
    """閾値・重み・RANSACを保持し、渡す点集合と最小支持点数だけを切り分ける。"""
    centerline, minimum = MODES[mode]
    settings = copy.deepcopy(settings)
    settings['min_inliers'] = minimum
    if centerline:
        points, weights = skeleton_points(probability, settings['probability_threshold'])
        sparse = np.zeros_like(probability)
        if len(points):
            x, y = points.astype(int).T
            sparse[y, x] = weights
        probability = sparse
    cv2.setRNGSeed(20261025)
    fitted = probability_ellipse(probability, settings, random_seed=seed)
    return fitted['ellipse'] if fitted else None


def teacher_diagnosis():
    """通常GT教師で2×2を診断する。運用性能や学習器の上限とは呼ばない。"""
    config = read_json(CONFIG)
    previous = read_json(OUT/'teacher_diagnosis.json')
    lookup = {(r['dataset'], r['sample_id']): r for r in previous['rows']}
    rows = []
    for source in ('1194m', 'diversified'):
        samples = [s for s in load_samples(BASE/source, 'test') if s['conditions']['degradation'] == 'clean']
        for sample in samples:
            label = read_json(BASE/source/sample['label'])
            truth = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
            probability = rasterize_ring_mask(256, 256, truth, thickness=3, blur_sigma=.8)
            seed = (config['cnn_ransac']['random_seed']+zlib.crc32(sample['sample_id'].encode('utf-8'))) % 2**32
            for mode in MODES:
                ellipse = predict(probability, config['cnn_ransac'], seed, mode)
                iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                if mode == 'band_45':
                    assert abs(iou-lookup[source, sample['sample_id']]['ellipse_iou']) < 1e-10
                rows.append({'dataset': source, 'mode': mode, 'sample_id': sample['sample_id'],
                             'major_axis_px': max(truth[1]), 'truth_ellipse': truth,
                             'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
        print('point teacher diagnosis', source, flush=True)
    summaries = []
    for source in ('1194m', 'diversified'):
        for mode in MODES:
            for lower, upper in ((0, float('inf')), (0, 24), (24, 48), (48, float('inf'))):
                selected = [r for r in rows if r['dataset'] == source and r['mode'] == mode and lower <= r['major_axis_px'] < upper]
                if selected:
                    summaries.append({'dataset': source, 'mode': mode, 'major_min_inclusive': lower,
                                      'major_max_exclusive': upper if np.isfinite(upper) else None,
                                      'n_images': len(selected), 'success_rate': float(np.mean([r['success'] for r in selected]))})
    write_json(OUT/'point_teacher_diagnosis.json', {'plan_sha256': sha(OUT/'plan.json'),
        'implementation_sha256': sha(Path(__file__)),
        'skeleton_implementation_sha256': sha(ROOT/'research/paf_generalization/boundary.py'),
        'reference_teacher_diagnosis_sha256': sha(OUT/'teacher_diagnosis.json'),
        'rows': rows, 'summaries': summaries,
        'scope': 'GT教師の点集合・最小支持点数の対照診断。新たな閾値探索ではなく、mainの45点と既存境界推定器の12点を比較。モデルの運用性能や厳密な上限ではない。'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['teacher'])
    parser.parse_args()
    teacher_diagnosis()
