"""学習結果とは別に、当てはめと対応画像の診断結果を元ラベルから照合する。"""

import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha

OUT = ROOT/'output/paf_generalization_v2'


def truths(root, split):
    result = {}
    for sample in load_samples(root, split):
        label = read_json(root/sample['label'])
        result[sample['sample_id']] = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    return result


def check_rows(rows, lookup):
    for row in rows:
        truth = lookup[row['sample_id']]
        e = row['ellipse']
        ellipse = (tuple(e[0]), tuple(e[1]), e[2]) if e else None
        iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
        assert row['ellipse_iou'] == iou and row['success'] == (iou >= .8)
    return len(rows)


def main():
    audit = {'files': {}, 'prediction_checks': 0, 'selection_checks': []}
    for experiment in ('fitting', 'refinement', 'consensus'):
        directory = OUT/experiment
        if not (directory/'selection.json').exists():
            continue
        protocol = read_json(directory/'protocol.json')
        assert protocol['implementation_sha256'] == sha(ROOT/f'research/paf_generalization_v2/{experiment}.py')
        roots = {'source_validation': (BASE/'datasets/diversified', 'validation'),
                 'original_validation': (BASE/'datasets/diversified', 'validation'),
                 'surface_validation': (PREVIOUS/'surface_counterfactual', 'validation')}
        for source in ('1194m', 'diversified'):
            roots[('test_' if experiment == 'fitting' else '')+source] = (BASE/'datasets'/source, 'test')
        for name, (root, split) in roots.items():
            path = directory/f'{name}.json'
            if not path.exists():
                continue
            saved = read_json(path)
            assert saved['protocol'] == protocol
            assert saved['manifest_sha256'] == sha(root/'manifest.json')
            audit['prediction_checks'] += check_rows(saved['rows'], truths(root, split))
            audit['files'][str(path.relative_to(ROOT))] = sha(path)
            if 'rates' in saved:
                for method, rate in saved['rates'].items():
                    assert rate == float(np.mean([r['success'] for r in saved['rows'] if r['method'] == method]))
        selection = read_json(directory/'selection.json')
        names = ('source_validation', 'surface_validation') if experiment == 'fitting' else ('original_validation', 'surface_validation')
        selected_inputs = [read_json(directory/f'{name}.json')['rows'] for name in names]
        hashes = selection['selection_inputs'] if experiment == 'fitting' else selection['input_hashes']
        assert hashes == [sha(directory/f'{name}.json') for name in names]
        methods = protocol.get('methods', ['standard', 'ams'])
        scores = {method: float(np.mean([np.mean([r['success'] for r in rows if r['method'] == method]) for rows in selected_inputs])) for method in methods}
        key = 'selected_simple_method' if experiment == 'fitting' else 'selected'
        assert selection[key] == max(methods, key=scores.get)
        audit['selection_checks'].append({'experiment': experiment, 'selected': selection[key], 'scores': scores})
    path = OUT/'partial_resolution.json'
    saved = read_json(path)
    assert saved['implementation_sha256'] == sha(ROOT/'research/paf_generalization_v2/partial_resolution.py')
    native = PREVIOUS/'framing_counterfactual'
    assert saved['manifest_sha256'] == sha(native/'manifest.json')
    audit['prediction_checks'] += check_rows(saved['rows'], truths(native, 'validation'))
    for row in saved['rows']:
        seed = zlib.crc32(f"{row['sample_id']}:{row['degradation']}:{row['severity']}:20261025".encode())
        assert row['effect_seed'] == seed
    for summary in saved['summary']:
        rows = [r for r in saved['rows'] if (r['kind'], r['degradation'], r['severity']) == (summary['kind'], summary['degradation'], summary['severity'])]
        assert len(rows) == summary['n_images']
        assert float(np.mean([r['success'] for r in rows])) == summary['success_rate']
    audit['files'][str(path.relative_to(ROOT))] = sha(path)
    path = OUT/'normal_sensitivity.json'
    if path.exists():
        saved = read_json(path)
        assert saved['implementation_sha256'] == sha(ROOT/'research/paf_generalization_v2/normal_geometry.py')
        assert saved['manifest_sha256'] == sha(BASE/'datasets/1194m/manifest.json')
        audit['prediction_checks'] += check_rows(saved['rows'], truths(BASE/'datasets/1194m', 'test'))
        for summary in saved['summary']:
            keys = ('tilt', 'visible_fraction', 'method', 'normal_sigma_deg')
            rows = [r for r in saved['rows'] if all(r[k] == summary[k] for k in keys)]
            assert len(rows) == summary['n']
            assert float(np.mean([r['success'] for r in rows])) == summary['success_rate']
        audit['files'][str(path.relative_to(ROOT))] = sha(path)
    for selection_path in sorted((OUT/'direction_score').glob('*/selection.json')):
        directory = selection_path.parent
        protocol = read_json(directory/'protocol.json')
        assert protocol['implementation_sha256'] == sha(ROOT/'research/paf_generalization_v2/direction_score.py')
        assert protocol['normal_geometry_sha256'] == sha(ROOT/'research/paf_generalization_v2/normal_geometry.py')
        assert protocol['oriented_sha256'] == sha(ROOT/'research/paf_generalization_v2/oriented.py')
        assert protocol['checkpoint_sha256'] == sha(OUT/f"runs/boundary_normal_aux/{protocol['seed']}/best.pt")
        roots = {'original_validation': (BASE/'datasets/diversified', 'validation'),
                 'surface_validation': (PREVIOUS/'surface_counterfactual', 'validation'),
                 '1194m': (BASE/'datasets/1194m', 'test'), 'diversified': (BASE/'datasets/diversified', 'test')}
        for name, (root, split) in roots.items():
            path = directory/f'{name}.json'
            if not path.exists():
                continue
            saved = read_json(path)
            assert saved['protocol'] == protocol and saved['manifest_sha256'] == sha(root/'manifest.json')
            audit['prediction_checks'] += check_rows(saved['rows'], truths(root, split))
            for method, rate in saved['rates'].items():
                assert rate == float(np.mean([r['success'] for r in saved['rows'] if r['method'] == method]))
            audit['files'][str(path.relative_to(ROOT))] = sha(path)
        names = ('original_validation', 'surface_validation')
        selection = read_json(selection_path)
        assert selection['input_hashes'] == [sha(directory/f'{name}.json') for name in names]
        rates = {m: float(np.mean([read_json(directory/f'{name}.json')['rates'][m] for name in names])) for m in protocol['methods']}
        assert selection['rates'] == rates
        expected = 'direction' if rates['direction']-rates['standard'] >= protocol['minimum_gain'] else 'standard'
        assert selection['selected'] == expected
        audit['selection_checks'].append({'experiment': 'direction_score', 'seed': protocol['seed'], 'selected': expected, 'scores': rates})
    path = OUT/'center_sensitivity.json'
    if path.exists():
        saved = read_json(path)
        assert saved['implementation_sha256'] == sha(ROOT/'research/paf_generalization_v2/center_sensitivity.py')
        assert saved['geometry_sha256'] == sha(ROOT/'research/paf_generalization_v2/center_geometry.py')
        assert saved['manifest_sha256'] == sha(BASE/'datasets/1194m/manifest.json')
        audit['prediction_checks'] += check_rows(saved['rows'], truths(BASE/'datasets/1194m', 'test'))
        for summary in saved['summary']:
            keys = ('tilt', 'visible_fraction', 'method', 'sigma')
            rows = [r for r in saved['rows'] if all(r[k] == summary[k] for k in keys)]
            assert len(rows) == summary['n'] and float(np.mean([r['success'] for r in rows])) == summary['success_rate']
        audit['files'][str(path.relative_to(ROOT))] = sha(path)
    directory = OUT/'input_resolution'
    if (directory/'selection.json').exists():
        protocol = read_json(directory/'protocol.json')
        assert protocol['implementation_sha256'] == sha(ROOT/'research/paf_generalization_v2/input_resolution.py')
        roots = {'original_validation': (BASE/'datasets/diversified', 'validation'),
                 'surface_validation': (PREVIOUS/'surface_counterfactual', 'validation'),
                 '1194m': (BASE/'datasets/1194m', 'test'), 'diversified': (BASE/'datasets/diversified', 'test')}
        for name, (root, split) in roots.items():
            path = directory/f'{name}.json'
            if not path.exists():
                continue
            saved = read_json(path)
            assert saved['protocol'] == protocol and saved['manifest_sha256'] == sha(root/'manifest.json')
            audit['prediction_checks'] += check_rows(saved['rows'], truths(root, split))
            for size, rate in saved['rates'].items():
                assert rate == float(np.mean([r['success'] for r in saved['rows'] if r['input_size'] == int(size)]))
            audit['files'][str(path.relative_to(ROOT))] = sha(path)
            reference_path = OUT/f'direction_score/20260907/{name}.json'
            if split == 'validation' and reference_path.exists():
                reference = {r['sample_id']: r for r in read_json(reference_path)['rows'] if r['method'] == 'standard'}
                rows = [r for r in saved['rows'] if r['input_size'] == 256]
                audit.setdefault('resolution_baseline_comparisons', []).append({
                    'view': name, 'n': len(rows),
                    'iou_differences': sum(r['ellipse_iou'] != reference[r['sample_id']]['ellipse_iou'] for r in rows),
                    'success_differences': sum(r['success'] != reference[r['sample_id']]['success'] for r in rows)})
            if split == 'test':
                reference_path = OUT/f'runs/boundary_normal_aux/20260907/test_{name}.json'
                reference = {r['sample_id']: r for r in read_json(reference_path)['rows']}
                rows = [r for r in saved['rows'] if r['input_size'] == 256]
                audit.setdefault('resolution_baseline_comparisons', []).append({
                    'view': name, 'n': len(rows),
                    'iou_differences': sum(r['ellipse_iou'] != reference[r['sample_id']]['ellipse_iou'] for r in rows),
                    'success_differences': sum(r['success'] != reference[r['sample_id']]['success'] for r in rows)})
        names = ('original_validation', 'surface_validation')
        selection = read_json(directory/'selection.json')
        assert selection['input_hashes'] == [sha(directory/f'{name}.json') for name in names]
        rates = {str(size): float(np.mean([read_json(directory/f'{name}.json')['rates'][str(size)] for name in names])) for size in protocol['sizes']}
        assert selection['rates'] == rates
        best = max(protocol['sizes'], key=lambda size: rates[str(size)])
        expected = best if rates[str(best)]-rates['256'] >= protocol['minimum_source_gain'] else 256
        assert selection['selected_size'] == expected
        audit['selection_checks'].append({'experiment': 'input_resolution', 'selected': expected, 'scores': rates})
    for selection_path in sorted((OUT/'center_votes').glob('*/selection.json')):
        directory = selection_path.parent
        protocol = read_json(directory/'protocol.json')
        assert protocol['implementation_sha256'] == sha(ROOT/'research/paf_generalization_v2/center_votes.py')
        assert protocol['geometry_sha256'] == sha(ROOT/'research/paf_generalization_v2/center_geometry.py')
        roots = {name: (Path(item['root']), 'validation') for name, item in protocol['validation_views'].items()}
        roots.update({source: (BASE/'datasets'/source, 'test') for source in ('1194m', 'diversified')})
        for name, (root, split) in roots.items():
            path = directory/f'{name}.json'
            if not path.exists():
                continue
            saved = read_json(path)
            assert saved['protocol'] == protocol and saved['manifest_sha256'] == sha(root/'manifest.json')
            audit['prediction_checks'] += check_rows(saved['rows'], truths(root, split))
            for method, rate in saved['rates'].items():
                assert rate == float(np.mean([r['success'] for r in saved['rows'] if r['method'] == method]))
            audit['files'][str(path.relative_to(ROOT))] = sha(path)
        names = list(protocol['validation_views'])
        selection = read_json(selection_path)
        assert selection['input_hashes'] == [sha(directory/f'{name}.json') for name in names]
        rates = {m: float(np.mean([read_json(directory/f'{name}.json')['rates'][m] for name in names])) for m in protocol['methods']}
        assert selection['rates'] == rates
        expected = 'center_votes' if rates['center_votes']-rates['standard'] >= protocol['minimum_gain'] else 'standard'
        assert selection['selected'] == expected
        audit['selection_checks'].append({'experiment': 'center_votes', 'seed': protocol['seed'], 'selected': expected, 'scores': rates})
    directory = OUT/'consensus_projection'
    if (directory/'selection.json').exists():
        protocol = read_json(directory/'protocol.json')
        assert protocol['implementation_sha256'] == sha(ROOT/'research/paf_generalization_v2/consensus_projection.py')
        assert protocol['scorer_sha256'] == sha(ROOT/'research/paf_generalization_v2/consensus.py')
        assert protocol['checkpoint_sha256'] == sha(PREVIOUS/'runs/boundary_surface/20260907/best.pt')
        roots = {name: (Path(item['root']), 'validation') for name, item in protocol['validation_views'].items()}
        roots.update({source: (BASE/'datasets'/source, 'test') for source in ('1194m', 'diversified')})
        for name, (root, split) in roots.items():
            path = directory/f'{name}.json'
            if not path.exists():
                continue
            saved = read_json(path)
            assert saved['protocol'] == protocol and saved['manifest_sha256'] == sha(root/'manifest.json')
            if 'reused_file' in saved:
                previous = ROOT/saved['reused_file']
                assert sha(previous) == saved['reused_sha256']
                original = read_json(previous)
                assert original['protocol']['implementation_sha256'] == protocol['scorer_sha256']
                assert original['rows'] == saved['rows'] and original['rates'] == saved['rates']
            audit['prediction_checks'] += check_rows(saved['rows'], truths(root, split))
            for method, rate in saved['rates'].items():
                assert rate == float(np.mean([r['success'] for r in saved['rows'] if r['method'] == method]))
            audit['files'][str(path.relative_to(ROOT))] = sha(path)
        names = list(protocol['validation_views'])
        selection = read_json(directory/'selection.json')
        assert selection['input_hashes'] == [sha(directory/f'{name}.json') for name in names]
        rates = {m: float(np.mean([read_json(directory/f'{name}.json')['rates'][m] for name in names])) for m in protocol['methods']}
        assert selection['rates'] == rates
        expected = 'support' if rates['support']-rates['coverage'] >= protocol['minimum_gain'] else 'coverage'
        assert selection['selected'] == expected
        audit['selection_checks'].append({'experiment': 'consensus_projection', 'selected': expected, 'scores': rates})
    for path in sorted(OUT.glob('center_transfer_*.json')):
        saved = read_json(path)
        seed = int(path.stem.rsplit('_', 1)[1])
        assert saved['implementation_sha256'] == sha(ROOT/'research/paf_generalization_v2/center_transfer.py')
        assert saved['geometry_sha256'] == sha(ROOT/'research/paf_generalization_v2/center_geometry.py')
        assert saved['checkpoint_sha256'] == sha(OUT/f'runs/boundary_center_aux/{seed}/best.pt')
        for name, digest in saved['dependencies'].items():
            assert sha(ROOT/name) == digest
        for source in ('1194m', 'diversified'):
            root = BASE/'datasets'/source
            assert saved['input_manifests'][source] == sha(root/'manifest.json')
            lookup = truths(root, 'test')
            samples = {s['sample_id']: s for s in load_samples(root, 'test')
                       if s['conditions']['degradation'] == 'clean' or s['conditions']['severity'] == .5}
            rows = [r for r in saved['rows'] if r['dataset'] == source]
            assert len(rows) == 2*len(samples)
            assert {(r['sample_id'], r['kind']) for r in rows} == {
                (sid, kind) for sid in samples for kind in ('all_predicted_points', 'oracle_near_points')}
            for row in rows:
                condition = samples[row['sample_id']]['conditions']
                assert (row['degradation'], row['severity'], row['tilt']) == (
                    condition['degradation'], condition['severity'], condition.get('camera', {}).get('tilt_deg'))
                # 診断専用の正解中心と予測中心を混同せず、両方のIoUを独立に再計算する。
                for prefix in ('predicted', 'oracle'):
                    audit['prediction_checks'] += check_rows([{'sample_id': row['sample_id'],
                        'ellipse': row[prefix+'_center_ellipse'], 'ellipse_iou': row[prefix+'_iou'],
                        'success': row[prefix+'_success']}], lookup)
                if row['center'] is not None:
                    truth = lookup[row['sample_id']]
                    delta = np.asarray(row['center'])-np.asarray(truth[0])
                    angle = np.deg2rad(truth[2])
                    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                    assert np.isclose(row['center_error_px'], np.linalg.norm(delta))
                    assert np.isclose(row['center_error_semiaxis_units'], np.linalg.norm((delta@rotation)/(np.asarray(truth[1])/2)))
                if 'mean_absolute_angle_deg' in row:
                    assert 0 <= row['mean_absolute_angle_deg'] <= 180
                    assert -180 <= row['signed_mean_angle_deg'] <= 180
                    assert row['absolute_bias_deg'] == abs(row['signed_mean_angle_deg'])
        for summary in saved['summary']:
            rows = [r for r in saved['rows'] if all(r[k] == summary[k] for k in ('dataset', 'degradation', 'kind'))
                    and (summary['tilt'] is None or r['tilt'] == summary['tilt'])]
            assert len(rows) == summary['n']
            for field in ('predicted_success', 'oracle_success', 'center_error_px', 'center_error_semiaxis_units',
                          'mean_absolute_angle_deg', 'signed_mean_angle_deg', 'absolute_bias_deg'):
                values = [r[field] for r in rows if field in r]
                assert summary[field+'_n'] == len(values)
                assert summary[field] == (float(np.mean(values)) if values else None)
        audit['files'][str(path.relative_to(ROOT))] = sha(path)
    write_json(OUT/'diagnostics_audit.json', audit)
    print({'prediction_checks': audit['prediction_checks'], 'selections': audit['selection_checks']})


if __name__ == '__main__':
    main()
