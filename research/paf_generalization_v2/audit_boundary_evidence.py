"""境界証拠の診断を元の成否・全条件・集計・依存hashへ照合する。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha

OUT = ROOT/'output/paf_generalization_v2'


def main():
    cases, truths = {}, {}
    for source in ('1194m', 'diversified'):
        root = BASE/'datasets'/source
        for sample in load_samples(root, 'test'):
            if sample['conditions']['degradation'] != 'clean' and sample['conditions']['severity'] != .5:
                continue
            key = source, sample['sample_id']
            cases[key] = sample
            label = read_json(root/sample['label'])
            truths[key] = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    files, count = {}, 0
    for path in sorted((OUT/'boundary_evidence').glob('*/*.json')):
        saved = read_json(path)
        protocol = saved['protocol']
        for name, digest in protocol['files'].items():
            assert sha(ROOT/name) == digest
        variant, seed = protocol['variant'], protocol['seed']
        folder = (PREVIOUS if variant == 'boundary_surface' else OUT)/f'runs/{variant}/{seed}'
        assert sha(folder/'best.pt') == protocol['checkpoint_sha256']
        reference = {}
        for source in ('1194m', 'diversified'):
            assert sha(BASE/f'datasets/{source}/manifest.json') == protocol['input_manifests'][source]
            assert sha(folder/f'test_{source}.json') == protocol['test_files'][source]
            reference.update({(source, r['sample_id']): r for r in read_json(folder/f'test_{source}.json')['rows']})
        assert len(saved['rows']) == len(cases)
        assert {(r['dataset'], r['sample_id']) for r in saved['rows']} == set(cases)
        for row in saved['rows']:
            key = row['dataset'], row['sample_id']
            original = reference[key]
            e = original['ellipse']
            ellipse = (tuple(e[0]), tuple(e[1]), e[2]) if e else None
            iou = evaluate_ellipses(ellipse, truths[key], (256, 256))['ellipse_iou'] if ellipse else 0.
            assert row['ellipse_iou'] == original['ellipse_iou'] == iou
            assert row['success'] == original['success'] == (iou >= .8)
            condition = cases[key]['conditions']
            assert (row['degradation'], row['severity'], row['tilt']) == (
                condition['degradation'], condition['severity'], condition.get('camera', {}).get('tilt_deg'))
            n, k, visible = row['n_points'], row['n_near_points'], row['n_visible_queries']
            assert 0 <= k <= n and 0 <= visible <= protocol['curve_queries']
            assert row['point_precision'] == (k/n if n else None)
            assert (row['near_mean_distance_px'] is None) == (k == 0)
            if k:
                assert 0 <= row['near_mean_distance_px'] <= protocol['threshold_px']
            assert (row['weighted_precision'] is None) == (n == 0)
            if n:
                assert 0 <= row['weighted_precision'] <= 1
            if visible:
                recall = row['visible_recall']
                assert 0 <= recall <= 1 and abs(recall*visible-round(recall*visible)) < 1e-9
            else:
                assert row['visible_recall'] is None
            count += 1
        for item in saved['summary']:
            rows = [r for r in saved['rows'] if (r['dataset'], r['degradation']) == (item['dataset'], item['degradation'])
                    and (item['success'] is None or r['success'] == item['success'])
                    and (item['tilt'] is None or r['tilt'] == item['tilt'])]
            assert len(rows) == item['n']
            for field in ('n_points', 'n_near_points', 'point_precision', 'weighted_precision', 'visible_recall', 'near_mean_distance_px'):
                values = [r[field] for r in rows if r[field] is not None]
                assert item[field+'_n'] == len(values)
                assert item[field+'_mean'] == (float(np.mean(values)) if values else None)
                assert item[field+'_median'] == (float(np.median(values)) if values else None)
        files[str(path.relative_to(ROOT))] = sha(path)
    write_json(OUT/'boundary_evidence_audit.json', {'implementation_sha256': sha(Path(__file__)),
        'files': files, 'case_checks': count,
        'scope': '全条件の網羅、元ラベルからの保存推論IoU、点数からの精度、範囲、集計とhashを照合。CNN推論・検出点座標の再実行は行わない。'})
    print('boundary evidence cases checked', count, 'files', len(files))


if __name__ == '__main__':
    main()
