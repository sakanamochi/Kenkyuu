"""画像ごとの寸法抽選と、既存成果物を使う比較設計の回帰テスト。"""

from pathlib import Path

import pytest

from parametric_paf.random_study import PLAN, SOURCE_PLAN, identity, random_shapes, read, rebase_samples


def test_random_dimensions_are_unique_reproducible_and_outside_reserved_shapes():
    """検証・テストとの寸法漏洩を避け、各画像で別の寸法を学習する。"""
    protocol = read(PLAN / 'protocol.json')
    reserved = read(SOURCE_PLAN / 'shapes.json')
    shapes = random_shapes(protocol, reserved)
    assert shapes == random_shapes(protocol, reserved)
    identities = {identity(s) for s in shapes}
    assert len(shapes) == len(identities) == 1024
    assert not identities & {identity(s) for s in reserved}
    for shape in shapes:
        assert all(a <= shape['ratios'][key] <= b for key, (a, b) in protocol['shape_ranges'].items())
        p = shape['parameters']
        assert p['top_outer_diameter'] - 2 * p['wall_thickness'] == pytest.approx(1)
        assert p['bottom_outer_diameter'] >= p['top_outer_diameter']
        assert p['rib_count'] == 0


def test_random_seed_changes_dimensions_without_changing_budget():
    protocol = read(PLAN / 'protocol.json')
    a = random_shapes(protocol, [])
    b = random_shapes({**protocol, 'shape_seed': protocol['shape_seed'] + 1}, [])
    assert a != b
    assert len(a) == len(b) == 1024


def test_comparison_keeps_capture_appearance_and_training_budget():
    """主比較の変更要因を寸法のみに限定する。"""
    new = read(PLAN / 'protocol.json')
    old = read(SOURCE_PLAN / 'protocol.json')
    assert new['arms'] == ['single', 'random']
    for key in ('shape_ranges', 'training', 'camera', 'appearance_expanded', 'training_seeds'):
        assert new[key] == old[key]
    assert new['train_images_per_arm'] == old['train_images_per_arm']
    assert new['train_images_per_arm'] // new['training']['batch_size'] * new['training']['epochs'] == 768


def test_rebased_paths_refer_to_the_same_existing_files():
    """保存先を変えても再利用する画像・正解の参照先は変わらない。"""
    folder = SOURCE_PLAN
    destination = PLAN / 'nested'
    samples = [{'image': 'protocol.json', 'label': 'shapes.json', 'sample_id': 'example'}]
    rebased = rebase_samples(folder, samples, destination)[0]
    for key in ('image', 'label'):
        assert (destination / rebased[key]).resolve() == (folder / samples[0][key]).resolve()
    assert rebased['sample_id'] == 'example'


def test_report_keeps_missed_images_in_denominator_and_out_of_distance_mean():
    """未検出を0px誤差として平均せず、成功率の分母には残す。"""
    from parametric_paf.audit_random_study import summarize_group
    rows = [
        {'target_present': True, 'detected': True, 'strict_success': True, 'iou_success': True,
         'missed': False, 'wrong_output': False, 'contour_p95_px': 1.5, 'probability_full_contour_coverage': 1.0},
        {'target_present': True, 'detected': False, 'strict_success': False, 'iou_success': False,
         'missed': True, 'wrong_output': False, 'contour_p95_px': None, 'probability_full_contour_coverage': 0.0},
    ]
    result = summarize_group(rows)
    assert result['n_present'] == 2
    assert result['strict_successes'] == result['missed'] == 1
    assert result['n_detected_for_distance'] == 1
    assert result['mean_contour_p95_px_detected'] == 1.5
    assert result['mean_full_contour_coverage'] == 0.5


def test_absent_report_does_not_count_rejection_as_target_detection_success():
    """対象なしの棄却と、対象ありの内周検出成功を分けて集計する。"""
    from parametric_paf.audit_random_study import summarize_group
    result = summarize_group([{'target_present': False, 'detected': False, 'absent_false_positive': False},
                              {'target_present': False, 'detected': True, 'absent_false_positive': True}])
    assert result['n_absent'] == 2 and result['n_present'] == 0
    assert result['absent_false_positives'] == 1
    assert result['strict_successes'] == 0
    assert result['mean_contour_p95_px_detected'] is None
