"""位置と法線の線形拘束による楕円当てはめと、部分弧の感度診断。"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE, sha


def ellipse_normals(points, ellipse):
    """陰関数の勾配を画像座標へ戻し、開口から外向きの単位法線を返す。"""
    angle = np.deg2rad(ellipse[2])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = (np.asarray(points, dtype=np.float64)-ellipse[0])@rotation
    normal = (local/np.square(np.asarray(ellipse[1])/2))@rotation.T
    return normal/np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)


def fit_oriented(points, normals, weights=None):
    """点の通過拘束と接線の直交拘束を、同じ係数ベクトルへ当てはめる。"""
    points, normals = np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)
    if len(points) < 3 or len(normals) != len(points):
        return None
    # 相似正規化で数値条件を整え、法線角度を保つ。
    origin = points.mean(0)
    scale = max(float(np.sqrt(np.mean(np.sum((points-origin)**2, axis=1)))), 1.)
    x, y = ((points-origin)/scale).T
    nx, ny = normals.T
    position = np.stack([x*x, x*y, y*y, x, y, np.ones_like(x)], 1)
    tangent = np.stack([-2*ny*x, nx*x-ny*y, 2*nx*y, -ny, nx, np.zeros_like(x)], 1)
    # 各点の位置式・方向式を単位長にし、確率重みを両者へ等しく適用する。
    position /= np.maximum(np.linalg.norm(position, axis=1, keepdims=True), 1e-12)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    if weights is not None:
        w = np.sqrt(np.maximum(np.asarray(weights), 0))[:, None]
        position, tangent = position*w, tangent*w
    try:
        _, singular, vt = np.linalg.svd(np.concatenate([position, tangent]), full_matrices=False)
        if singular[-2] < 1e-9:
            return None
        a, b, c, d, e, f = vt[-1]
        q = np.array([[a, b/2], [b/2, c]])
        linear = np.array([d, e])
        center = -.5*np.linalg.solve(q, linear)
        value = -(f+.5*linear@center)
        if abs(value) < 1e-12:
            return None
        eigenvalues, eigenvectors = np.linalg.eigh(q/value)
        if min(eigenvalues) <= 0 or not np.isfinite(eigenvalues).all():
            return None
        axes = 2*scale/np.sqrt(eigenvalues)
        direction = eigenvectors[:, 0]
        return (tuple(center*scale+origin), tuple(axes), float(np.rad2deg(np.arctan2(direction[1], direction[0]))))
    except np.linalg.LinAlgError:
        return None


def main():
    """正解部分弧へ独立な位置・角度ノイズを加え、補助方向の必要精度を調べる。"""
    rng = np.random.default_rng(2026102517)
    directory = BASE/'datasets/1194m'
    rows = []
    samples = [s for s in load_samples(directory, 'test') if s['conditions']['degradation'] == 'clean']
    for sample in samples:
        label = read_json(directory/sample['label'])
        truth = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
        rotation = np.deg2rad(truth[2])
        matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        phase = np.linspace(0, 2*np.pi, 256, endpoint=False)
        full = np.stack([truth[1][0]/2*np.cos(phase), truth[1][1]/2*np.sin(phase)], 1)@matrix.T+truth[0]
        for fraction in (.5, .25):
            for repeat in range(20):
                indices = (np.arange(round(len(full)*fraction))+int(rng.integers(0, len(full)))) % len(full)
                ideal = full[indices]
                points = ideal+rng.normal(0, 1., ideal.shape)
                normals = ellipse_normals(ideal, truth)
                angle_noise = rng.normal(0, 1, len(points))
                cases = [('standard', None, cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2)))]
                for degrees in (0., 2., 5., 10.):
                    delta = np.deg2rad(degrees)*angle_noise
                    nx, ny = normals.T
                    noisy = np.stack([np.cos(delta)*nx-np.sin(delta)*ny, np.sin(delta)*nx+np.cos(delta)*ny], 1)
                    cases.append(('oriented', degrees, fit_oriented(points, noisy)))
                for method, degrees, ellipse in cases:
                    iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                    rows.append({'sample_id': sample['sample_id'], 'tilt': sample['conditions']['camera']['tilt_deg'],
                                 'visible_fraction': fraction, 'position_sigma_px': 1., 'normal_sigma_deg': degrees,
                                 'method': method, 'repeat': repeat, 'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
    summary = []
    for tilt in sorted({r['tilt'] for r in rows}):
        for fraction in (.5, .25):
            for method, degrees in [('standard', None)]+[('oriented', x) for x in (0., 2., 5., 10.)]:
                selected = [r for r in rows if (r['tilt'], r['visible_fraction'], r['method'], r['normal_sigma_deg']) == (tilt, fraction, method, degrees)]
                summary.append({'tilt': tilt, 'visible_fraction': fraction, 'method': method, 'normal_sigma_deg': degrees,
                                'n': len(selected), 'success_rate': float(np.mean([r['success'] for r in selected]))})
    write_json(ROOT/'output/paf_generalization_v2/normal_sensitivity.json', {
        'implementation_sha256': sha(Path(__file__)), 'manifest_sha256': sha(directory/'manifest.json'),
        'seed': 2026102517, 'rows': rows, 'summary': summary,
        'interpretation': '正解楕円から得る合成点・法線の感度診断。画像の認識性能ではなく、独立ノイズはCNNの系統誤差を含まない。角度上の連続弧であり実際の矩形遮蔽率とは異なる。'})
    for r in summary:
        if r['tilt'] == 75:
            print(r, flush=True)


if __name__ == '__main__':
    main()
