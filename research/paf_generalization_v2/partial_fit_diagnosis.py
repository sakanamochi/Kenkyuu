"""検出点を固定し、誤点除去と位置誤差除去による部分弧の復元を診断する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from paf_ring_detection.methods.ransac import _point_distances
from research.paf_generalization.boundary import skeleton_points
from research.paf_generalization.diagnose import BASE, sha, image_and_truth
from research.paf_generalization_v2.fitting import direct_fit

OUT = ROOT/'output/paf_generalization_v2'
METHODS = ('direct_all', 'oracle_near', 'oracle_exact')


def exact_positions(points, truth):
    """GT楕円の偏角を保って半径を正し、位置誤差だけを除く診断用座標を作る。"""
    center, axes, angle = truth
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = (np.asarray(points).reshape(-1, 2)-center)@rotation
    radius = np.asarray(axes)/2
    phase = np.arctan2(local[:, 1]/radius[1], local[:, 0]/radius[0])
    exact = np.stack([radius[0]*np.cos(phase), radius[1]*np.sin(phase)], 1)@rotation.T+center
    return exact


def fits(points, truth, settings):
    """同じ最低点数・有効楕円条件を使い、診断を検出性能と混同しない。"""
    points = np.asarray(points, np.float32).reshape(-1, 2)
    near = points[_point_distances(points, truth) <= 2.] if len(points) else points
    inputs = (points, near, exact_positions(near, truth))
    results = {}
    for method, selected in zip(METHODS, inputs):
        # 完全楕円点などでOpenCV内部の摂動が使われても、保存点から再現できる。
        cv2.setRNGSeed(20261025)
        ellipse = direct_fit(selected, (256, 256), settings)
        iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
        results[method] = {'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8}
    return len(near), results


def summarize(rows):
    """通常と半画面遮蔽、視点別に同じ画像の成否の組合せを数える。"""
    summaries = []
    for source in ('1194m', 'diversified'):
        tilts = [None]+sorted({r['tilt'] for r in rows if r['dataset'] == source and r['tilt'] is not None})
        for effect in ('clean', 'black_rectangle'):
            for tilt in tilts:
                group = [r for r in rows if r['dataset'] == source and r['degradation'] == effect
                         and (tilt is None or r['tilt'] == tilt)]
                if not group:
                    continue
                item = {'dataset': source, 'degradation': effect, 'tilt': tilt, 'n': len(group)}
                for method in ('reference',)+METHODS:
                    item[method+'_success'] = sum(r['fits'][method]['success'] for r in group)
                for ref in (False, True):
                    for near in (False, True):
                        item[f'reference_{ref}_near_{near}'] = sum(
                            r['fits']['reference']['success'] == ref and r['fits']['oracle_near']['success'] == near for r in group)
                summaries.append(item)
    return summaries


def main(variant, seed):
    folder = OUT/f'runs/{variant}/{seed}'
    destination = OUT/f'partial_fit_diagnosis/{variant}/{seed}.json'
    checkpoint = sha(folder/'best.pt')
    assert read_json(folder/'complete.json')['checkpoint_sha256'] == checkpoint
    settings = read_json(folder/'provenance.json')['config']['ransac']
    paths = [Path(__file__), ROOT/'research/paf_generalization/boundary.py',
             ROOT/'research/paf_generalization/diagnose.py', Path(__file__).with_name('fitting.py'),
             ROOT/'paf_ring_detection/methods/ransac.py', ROOT/'paf_ring_detection/methods/cnn.py',
             ROOT/'paf_ring_detection/data.py', ROOT/'paf_ring_detection/effects.py',
             ROOT/'paf_ring_detection/geometry.py']
    protocol = {'variant': variant, 'seed': seed, 'checkpoint_sha256': checkpoint,
        'files': {str(p.relative_to(ROOT)): sha(p) for p in paths}, 'ransac': settings, 'opencv_seed_per_fit': 20261025,
        'reference_hashes': {s: sha(folder/f'test_{s}.json') for s in ('1194m', 'diversified')},
        'manifests': {s: sha(BASE/f'datasets/{s}/manifest.json') for s in ('1194m', 'diversified')},
        'scope': '通常と半画面遮蔽。正解2px以内の点をDirectで当てはめ、同じ偏角の正確な点とも比較する。GT選別・位置修正は診断専用で、運用性能や厳密な上限ではない。保存済み標準推論と比較し推定器を選び直さない。'}
    if destination.exists():
        assert read_json(destination)['protocol'] == protocol
        return
    model = TinyUNet(16).cuda().eval()
    model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=False)['model_state'])
    rows = []
    for source in ('1194m', 'diversified'):
        root = BASE/'datasets'/source
        references = {r['sample_id']: r for r in read_json(folder/f'test_{source}.json')['rows']}
        for sample in load_samples(root, 'test'):
            condition = sample['conditions']
            if (condition['degradation'], condition['severity']) not in (('clean', 0.), ('black_rectangle', .5)):
                continue
            image, truth = image_and_truth(root, sample)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).cuda().float()[None]/127.5-1
            with torch.inference_mode():
                probability = torch.sigmoid(model(x)[0, 0]).cpu().numpy()
            points, _ = skeleton_points(probability)
            count, results = fits(points, truth, settings)
            ref = references[sample['sample_id']]
            results['reference'] = {k: ref[k] for k in ('ellipse', 'ellipse_iou', 'success')}
            rows.append({'dataset': source, 'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                'degradation': condition['degradation'], 'severity': condition['severity'],
                'tilt': condition.get('camera', {}).get('tilt_deg'), 'truth': truth,
                'points': points.tolist(), 'n_near': count, 'fits': results})
    assert len(rows) == 532
    write_json(destination, {'protocol': protocol, 'rows': rows, 'summary': summarize(rows)})
    print({'variant': variant, 'seed': seed, 'images': len(rows)}, flush=True)


def audit():
    """保存点から全当てはめと元GTのIoUを再計算する。CNNの再実行とは区別する。"""
    files = sorted((OUT/'partial_fit_diagnosis').glob('*/*.json'))
    total = 0
    samples = {s: {r['sample_id']: r for r in load_samples(BASE/'datasets'/s, 'test')
                  if (r['conditions']['degradation'], r['conditions']['severity']) in
                  (('clean', 0.), ('black_rectangle', .5))} for s in ('1194m', 'diversified')}
    for path in files:
        saved = read_json(path)
        protocol = saved['protocol']
        for name, digest in protocol['files'].items():
            assert sha(ROOT/name) == digest
        folder = OUT/f"runs/{protocol['variant']}/{protocol['seed']}"
        assert sha(folder/'best.pt') == protocol['checkpoint_sha256']
        references = {}
        for source in samples:
            assert sha(BASE/f'datasets/{source}/manifest.json') == protocol['manifests'][source]
            assert sha(folder/f'test_{source}.json') == protocol['reference_hashes'][source]
            references[source] = {r['sample_id']: r for r in read_json(folder/f'test_{source}.json')['rows']}
        assert len(saved['rows']) == 532
        assert {(r['dataset'], r['sample_id']) for r in saved['rows']} == {(s, k) for s, data in samples.items() for k in data}
        for row in saved['rows']:
            source, sid = row['dataset'], row['sample_id']
            sample = samples[source][sid]
            _, truth = image_and_truth(BASE/'datasets'/source, sample)
            assert np.allclose([*truth[0], *truth[1], truth[2]], [*row['truth'][0], *row['truth'][1], row['truth'][2]])
            assert row['group_id'] == sample['group_id']
            assert all(row[k] == sample['conditions'][k] for k in ('degradation', 'severity'))
            assert row['tilt'] == sample['conditions'].get('camera', {}).get('tilt_deg')
            count, results = fits(row['points'], truth, protocol['ransac'])
            assert count == row['n_near']
            ref = references[source][sid]
            results['reference'] = {k: ref[k] for k in ('ellipse', 'ellipse_iou', 'success')}
            for method, result in results.items():
                old = row['fits'][method]
                assert result['ellipse_iou'] == old['ellipse_iou'] and result['success'] == old['success']
                if result['ellipse'] is None:
                    assert old['ellipse'] is None
                else:
                    current, previous = result['ellipse'], old['ellipse']
                    assert np.allclose([*current[0], *current[1], current[2]], [*previous[0], *previous[1], previous[2]])
                ellipse = old['ellipse']
                ellipse = (tuple(ellipse[0]), tuple(ellipse[1]), ellipse[2]) if ellipse else None
                iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                assert iou == old['ellipse_iou'] and old['success'] == (iou >= .8)
                total += 1
        assert saved['summary'] == summarize(saved['rows'])
    write_json(OUT/'partial_fit_audit.json', {'files': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'images': total//4, 'fit_checks': total, 'scope': '保存点から診断当てはめと元GTのIoUを再計算。CNN推論の再実行ではない。'})
    print({'files': len(files), 'fit_checks': total}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'audit'])
    parser.add_argument('--variants', nargs='+', default=['boundary_projection_mix', 'boundary_projection_relative'])
    parser.add_argument('--seed', type=int, default=20260907)
    args = parser.parse_args()
    if args.stage == 'audit':
        audit()
    else:
        for variant in args.variants:
            main(variant, args.seed)
