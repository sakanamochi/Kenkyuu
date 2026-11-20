"""mainの正しいリング教師を推定器へ渡し、投影サイズごとの幾何誤差を測る。"""

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
from research.paf_core_ablation.experiment import OUT, BASE, CONFIG
from research.paf_core_ablation.main_compat import rasterize_ring_mask, probability_ellipse


def main():
    config = read_json(CONFIG)
    rows = []
    for source in ('1194m', 'diversified'):
        samples = [s for s in load_samples(BASE/source, 'test') if s['conditions']['degradation'] == 'clean']
        for sample in samples:
            path = BASE/source/sample['label']
            label = read_json(path)
            assert sha(path) == sample['label_sha256']
            truth = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
            probability = rasterize_ring_mask(256, 256, truth, thickness=3, blur_sigma=.8)
            seed = (config['cnn_ransac']['random_seed']+zlib.crc32(sample['sample_id'].encode('utf-8'))) % 2**32
            cv2.setRNGSeed(20261025)
            fit = probability_ellipse(probability, config['cnn_ransac'], random_seed=seed)
            ellipse = fit['ellipse'] if fit else None
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'dataset': source, 'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                         'major_axis_px': max(truth[1]), 'minor_axis_px': min(truth[1]),
                         'threshold_point_count': int(np.count_nonzero(probability >= config['cnn_ransac']['probability_threshold'])),
                         'truth_ellipse': truth, 'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
        print('teacher diagnosis', source, sum(r['success'] for r in rows if r['dataset'] == source), len(samples), flush=True)
    summaries = []
    for source in ('1194m', 'diversified'):
        for lower, upper in ((0, float('inf')), (0, 24), (24, 48), (48, float('inf'))):
            selected = [r for r in rows if r['dataset'] == source and lower <= r['major_axis_px'] < upper]
            if selected:
                summaries.append({'dataset': source, 'major_min_inclusive': lower,
                                  'major_max_exclusive': upper if np.isfinite(upper) else None,
                                  'n_images': len(selected), 'success_rate': float(np.mean([r['success'] for r in selected])),
                                  'mean_iou': float(np.mean([r['ellipse_iou'] for r in selected]))})
    write_json(OUT/'teacher_diagnosis.json', {'plan_sha256': sha(OUT/'plan.json'),
        'implementation_sha256': sha(Path(__file__)), 'rows': rows, 'summaries': summaries,
        'scope': '通常testのGT楕円から作ったmain教師の直接入力。学習器の性能・運用可能な成績・厳密な上限ではない。閾値やモデル選択にGTを渡さない。'})


if __name__ == '__main__':
    main()
