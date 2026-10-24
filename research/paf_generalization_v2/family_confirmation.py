"""1194M追加確認と同じ固定候補を、既存の別形状族の対応画像で評価する。"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from research.paf_generalization.diagnose import sha
from research.paf_generalization.evaluate_holdout import DATA, verify as verify_original_data
from research.paf_generalization.report import paired_interval
from research.paf_generalization.report_holdout import paired_shape_difference
from research.paf_generalization_v2 import confirmation

OUT = ROOT/'output/paf_generalization_v2/family_confirmation'
PARENT = ROOT/'output/paf_generalization_v2/confirmation/frozen_candidate.json'


def cases():
    """元の96形状対・4劣化条件を保ち、共通評価器に必要な識別子を付ける。"""
    verify_original_data()
    samples = load_samples(DATA, 'test')
    assert len(samples) == 768 and len({s['group_id'] for s in samples}) == 96
    assert len({s['sample_id'] for s in samples}) == len(samples)
    groups = {}
    for sample in samples:
        groups.setdefault(sample['group_id'], set()).add((sample['view'], sample['conditions']['degradation']))
    expected = {(view, effect) for view in ('control', 'new_family')
                for effect in ('clean', 'black_rectangle', 'sensor_whiteout', 'sensor_black_crush')}
    assert all(group == expected for group in groups.values())
    return [{**s, 'base_sample_id': s['conditions']['base_sample_id'],
             'effect_seed': s['augmentation']['seed'] if s['augmentation'] else None} for s in samples]


def freeze():
    """1194M側で固定済みの候補・全3seed・推定器を、そのまま別形状にも適用する。"""
    path = OUT/'frozen_candidate.json'
    if path.exists():
        return verify()
    cases()
    fixed = copy.deepcopy(confirmation.verify())
    fixed.update(parent_candidate_sha256=sha(PARENT), data_manifest_sha256=sha(DATA/'manifest.json'),
                 scope='既存family_holdout_v1の再評価。初回研究で参照済みの形状集合であり、新しい盲検集合ではない。元構造と別構造の対応、全3seed、全画像分母を維持する。')
    for p in (Path(__file__), ROOT/'research/paf_generalization/report_holdout.py'):
        fixed['frozen_files'][str(p.relative_to(ROOT))] = sha(p)
    write_json(path, fixed)
    return verify()


def verify():
    fixed = read_json(OUT/'frozen_candidate.json')
    assert fixed['parent_candidate_sha256'] == sha(PARENT)
    assert fixed['data_manifest_sha256'] == sha(DATA/'manifest.json')
    parent = read_json(PARENT)
    assert fixed['jobs'] == parent['jobs'] and fixed['variant'] == parent['variant']
    for path, digest in fixed['frozen_files'].items():
        assert sha(ROOT/path) == digest, path
    for job in fixed['jobs']:
        if 'checkpoint' in job:
            path = ROOT/job['checkpoint']
            assert sha(path) == job['checkpoint_sha256']
            assert sha(path.with_name('provenance.json')) == job['provenance_sha256']
    return fixed


def common(stage):
    """推論と元ラベル照合を共通評価器に委ね、処理の複製による差を防ぐ。"""
    saved = confirmation.OUT, confirmation.DATA, confirmation.cases, confirmation.verify
    try:
        confirmation.OUT, confirmation.DATA = OUT, DATA
        confirmation.cases, confirmation.verify = cases, verify
        {'evaluate': confirmation.evaluate, 'report': confirmation.report}[stage]()
    finally:
        confirmation.OUT, confirmation.DATA, confirmation.cases, confirmation.verify = saved


def report():
    """全推定の照合後、形状族・対応ビュー・劣化条件別の成績を保存する。"""
    common('report')
    fixed = verify()
    lookup = {s['sample_id']: s for s in cases()}
    records = []
    for job in fixed['jobs']:
        seed = job['seed']
        path = OUT/f"results/{job['method']}_{seed if seed is not None else 'fixed'}.json"
        for row in read_json(path)['rows']:
            sample = lookup[row['sample_id']]
            records.append({**row, 'method': job['method'], 'seed': seed,
                            'family': sample['family'], 'view': sample['view']})
    summaries, differences = [], []
    for method in dict.fromkeys(r['method'] for r in records):
        method_rows = [r for r in records if r['method'] == method]
        differences.append({'method': method, **paired_shape_difference(method_rows)})
        for family in ['all']+sorted({r['family'] for r in method_rows}):
            for view in ('control', 'new_family'):
                for effect in ('all', 'clean', 'black_rectangle', 'sensor_whiteout', 'sensor_black_crush'):
                    rows = [r for r in method_rows if r['view'] == view and
                            (family == 'all' or r['family'] == family) and (effect == 'all' or r['degradation'] == effect)]
                    summaries.append({'method': method, 'family': family, 'view': view, 'degradation': effect,
                        'n_images': len({r['sample_id'] for r in rows}),
                        'success_rate': float(np.mean([r['success'] for r in rows])),
                        'per_seed': {str(seed): float(np.mean([r['success'] for r in rows if r['seed'] == seed]))
                                     for seed in dict.fromkeys(r['seed'] for r in rows)}})
    candidate = [r for r in records if r['method'] == fixed['variant'] and r['view'] == 'new_family']
    baseline = [r for r in records if r['method'] == 'boundary_surface' and r['view'] == 'new_family']
    comparison = paired_interval(candidate, baseline)
    comparison['uncertainty_scope'] = '固定3seedに対する形状群の再標本化。形状族全般・seed母集団・探索の多重性は含まない。'
    write_json(OUT/'family_summary.json', {'rows': summaries, 'paired_shape_differences': differences,
        'candidate_vs_boundary_surface': comparison, 'scope': fixed['scope'],
        'frozen_candidate_sha256': sha(OUT/'frozen_candidate.json'), 'audited_summary_sha256': sha(OUT/'summary.json')})
    print('family confirmation summaries verified', len(records), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify_data', 'freeze', 'evaluate', 'report'])
    args = parser.parse_args()
    if args.stage == 'verify_data':
        print('verified paired conditions', len(cases()))
    elif args.stage == 'freeze':
        freeze()
    elif args.stage == 'evaluate':
        common('evaluate')
    else:
        report()
