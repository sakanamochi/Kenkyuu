"""短い楕円弧の補完に必要な中心精度を、正解からの合成点群で診断する。"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE, sha
from research.paf_generalization_v2.center_geometry import fit_centered, fit_center_votes


def main():
    rng = np.random.default_rng(2026102523)
    directory = BASE/'datasets/1194m'
    rows = []
    for sample in load_samples(directory, 'test'):
        if sample['conditions']['degradation'] != 'clean':
            continue
        label = read_json(directory/sample['label'])
        truth = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
        angle = np.deg2rad(truth[2])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        phase = np.linspace(0, 2*np.pi, 256, endpoint=False)
        full = np.stack([truth[1][0]/2*np.cos(phase), truth[1][1]/2*np.sin(phase)], 1)@rotation.T+truth[0]
        for fraction in (.5, .25):
            for repeat in range(20):
                indices = (np.arange(round(len(full)*fraction))+rng.integers(0, len(full))) % len(full)
                points = full[indices]+rng.normal(0, 1, (len(indices), 2))
                center_noise = rng.normal(0, 1, 2)
                cases = [('standard', None, cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2)))]
                for sigma in (0., .5, 1., 2., 4.):
                    center = np.asarray(truth[0])+sigma*center_noise
                    cases.append(('known_noisy_center', sigma, fit_centered(points, center)))
                direction = np.asarray(truth[0])-points
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                angle_noise = rng.normal(0, 1, len(points))
                for sigma in (0., 2., 5., 10.):
                    delta = np.deg2rad(sigma)*angle_noise
                    dx, dy = direction.T
                    rotated = np.stack([np.cos(delta)*dx-np.sin(delta)*dy, np.sin(delta)*dx+np.cos(delta)*dy], 1)
                    cases.append(('noisy_center_direction', sigma, fit_center_votes(points, rotated)))
                for sigma in (1., 2., 5., 10.):
                    # CNNの方向誤差が弧全体で相関する場合も、同じ点群で診断する。
                    delta = np.deg2rad(sigma)*angle_noise[0]
                    dx, dy = direction.T
                    rotated = np.stack([np.cos(delta)*dx-np.sin(delta)*dy, np.sin(delta)*dx+np.cos(delta)*dy], 1)
                    cases.append(('shared_direction_bias', sigma, fit_center_votes(points, rotated)))
                for method, sigma, ellipse in cases:
                    iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                    rows.append({'sample_id': sample['sample_id'], 'tilt': sample['conditions']['camera']['tilt_deg'],
                                 'visible_fraction': fraction, 'repeat': repeat, 'position_sigma_px': 1.,
                                 'method': method, 'sigma': sigma, 'sigma_unit': 'degrees' if method in ('noisy_center_direction', 'shared_direction_bias') else 'pixels',
                                 'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
    summary = []
    for tilt in sorted({r['tilt'] for r in rows}):
        for fraction in (.5, .25):
            for method in ('standard', 'known_noisy_center', 'noisy_center_direction', 'shared_direction_bias'):
                for sigma in sorted({r['sigma'] for r in rows if r['method'] == method}, key=lambda v: v or 0):
                    selected = [r for r in rows if (r['tilt'], r['visible_fraction'], r['method'], r['sigma']) == (tilt, fraction, method, sigma)]
                    summary.append({'tilt': tilt, 'visible_fraction': fraction, 'method': method, 'sigma': sigma,
                                    'n': len(selected), 'success_rate': float(np.mean([r['success'] for r in selected]))})
    write_json(ROOT/'output/paf_generalization_v2/center_sensitivity.json', {
        'implementation_sha256': sha(Path(__file__)), 'geometry_sha256': sha(Path(__file__).with_name('center_geometry.py')),
        'manifest_sha256': sha(directory/'manifest.json'), 'seed': 2026102523,
        'rows': rows, 'summary': summary,
        'interpretation': '正解から作る点・中心・方向の合成感度診断。独立方向ノイズと弧全体に共通する角度バイアスを別に測る。実CNNの誤差分布・誤境界選択を表さず、検出性能やその厳密な上限ではない。'})
    for row in summary:
        if row['tilt'] == 75 and row['visible_fraction'] == .5:
            print(row, flush=True)


if __name__ == '__main__':
    main()
