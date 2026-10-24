"""境界の画素誤差が、既知の投影楕円のIoU判定へ与える感度を測る。"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE, sha


def main():
    rng = np.random.default_rng(2026102512)
    folder = BASE/'datasets/1194m'
    samples = [s for s in load_samples(folder, 'test') if s['conditions']['degradation'] == 'clean']
    rows = []
    for sample in samples:
        label = read_json(folder/sample['label'])
        truth = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
        angle = sample['conditions']['camera']['tilt_deg']
        for displacement in (-2., -1., -.5, .5, 1., 2.):
            # 半軸の一定変位は正確な法線オフセットではなく、輪郭位置誤差の感度例である。
            axes = tuple(a+2*displacement for a in truth[1])
            prediction = (truth[0], axes, truth[2]) if min(axes) > 0 else None
            iou = evaluate_ellipses(prediction, truth, (256, 256))['ellipse_iou'] if prediction else 0.
            rows.append({'sample_id': sample['sample_id'], 'tilt': angle, 'kind': 'semi_axis_offset',
                         'method': 'analytic', 'error_px': displacement, 'visible_fraction': 1., 'ellipse_iou': iou, 'success': iou >= .8})
        phase = np.linspace(0, 2*np.pi, 256, endpoint=False)
        rotation = np.deg2rad(truth[2])
        matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        full = np.stack([truth[1][0]/2*np.cos(phase), truth[1][1]/2*np.sin(phase)], axis=1)@matrix.T+truth[0]
        for fraction in (1., .5, .25):
            for deviation in (.25, .5, 1., 2.):
                # 正解上の連続弧へ独立な位置ノイズだけを与え、誤境界や画像認識の影響を含めない。
                for repetition in range(20):
                    start = int(rng.integers(0, len(full)))
                    indices = (np.arange(round(len(full)*fraction))+start) % len(full)
                    points = full[indices]+rng.normal(0, deviation, (len(indices), 2))
                    for method, fit in [('direct', cv2.fitEllipseDirect), ('standard', cv2.fitEllipse), ('ams', cv2.fitEllipseAMS)]:
                        cv2.setRNGSeed(20261025)
                        fitted = fit(points.astype(np.float32).reshape(-1, 1, 2))
                        iou = evaluate_ellipses(fitted, truth, (256, 256))['ellipse_iou']
                        rows.append({'sample_id': sample['sample_id'], 'tilt': angle, 'kind': 'noisy_correct_arc',
                                     'method': method, 'error_px': deviation, 'visible_fraction': fraction,
                                     'repetition': repetition, 'ellipse_iou': iou, 'success': iou >= .8})
    keys = sorted({(r['tilt'], r['kind'], r['method'], r['error_px'], r['visible_fraction']) for r in rows})
    summary = []
    for tilt, kind, method, error, fraction in keys:
        selected = [r for r in rows if (r['tilt'], r['kind'], r['method'], r['error_px'], r['visible_fraction']) == (tilt, kind, method, error, fraction)]
        summary.append({'tilt': tilt, 'kind': kind, 'method': method, 'error_px': error, 'visible_fraction': fraction,
                        'n_predictions': len(selected), 'n_images': len({r['sample_id'] for r in selected}),
                        'success_rate': float(np.mean([r['success'] for r in selected])),
                        'mean_iou': float(np.mean([r['ellipse_iou'] for r in selected]))})
    write_json(ROOT/'output/paf_generalization_v2/precision_sensitivity.json', {
        'implementation_sha256': sha(Path(__file__)), 'manifest_sha256': sha(folder/'manifest.json'),
        'seed': 2026102512, 'rows': rows, 'summary': summary,
        'interpretation': '正解楕円を使う画素精度の感度診断。連続弧は角度パラメータ上の割合であり、実際の矩形遮蔽率とは異なる。達成性能・検出器・厳密な上限とは扱わない。'})
    for row in summary:
        if row['tilt'] == 75 and row['kind'] == 'noisy_correct_arc' and row['visible_fraction'] == .5:
            print(row['method'], row['error_px'], round(row['success_rate']*100, 2))


if __name__ == '__main__':
    main()
