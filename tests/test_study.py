"""新しい輪郭評価と、形状単位の実験分離を守る回帰テスト。"""

import numpy as np
import pytest

from parametric_paf.diagnosis_metrics import contour_metrics, ellipse_points, nearest_distances
from parametric_paf.evaluate_study import evaluate_prediction
from parametric_paf.study import PLAN, read


def test_contour_metric_matches_concentric_circle_distance():
    """同心円の半径差は、輪郭の双方向距離と一致する。"""
    truth=((100,100),(100,100),0)
    larger=((100,100),(106,106),0)
    result=contour_metrics(larger,truth)
    assert result['contour_mean_px']==pytest.approx(3,abs=1e-5)
    assert result['contour_p95_px']==pytest.approx(3,abs=1e-5)
    assert result['contour_p95_ratio']==pytest.approx(0.03,abs=1e-5)


def test_contour_metric_is_invariant_to_ellipse_axis_representation():
    """長短軸を交換して90度回転しても同じ輪郭になる。"""
    a=((110,120),(120,60),15)
    b=((110,120),(60,120),105)
    assert contour_metrics(a,b)['contour_p95_px']<1e-10


def test_iou_alone_can_accept_wrong_nearby_boundary():
    """面積IoUは成功でも、別境界に相当する3pxのずれは主指標で失敗にする。"""
    settings={'success_iou':0.8,'contour_p95_px':2}
    result=evaluate_prediction(((100,100),(106,106),0),((100,100),(100,100),0),(256,256),settings)
    assert result['iou_success'] is True
    assert result['strict_success'] is False
    assert result['wrong_output'] is True


def test_missing_prediction_and_absent_target_are_not_conflated():
    settings={'success_iou':0.8,'contour_p95_px':2}
    e=((100,100),(100,60),0)
    missing=evaluate_prediction(None,e,(256,256),settings)
    absent=evaluate_prediction(e,None,(256,256),settings)
    rejected=evaluate_prediction(None,None,(256,256),settings)
    assert missing['missed'] and not missing['wrong_output']
    assert absent['strict_success'] is None and absent['absent_false_positive']
    assert rejected['absent_false_positive'] is False


def test_empty_point_cloud_has_no_false_contour_support():
    p=ellipse_points(((100,100),(100,60),0))
    assert np.isinf(nearest_distances(p,np.empty((0,2)))).all()


def test_shape_splits_are_geometrically_disjoint_and_test_types_are_true():
    """IDの違いだけで同じ形状を未知扱いせず、補間と外挿を区別する。"""
    protocol=read(PLAN/'protocol.json'); shapes=read(PLAN/'shapes.json')
    identities=[(s['family'],tuple(sorted(s['ratios'].items()))) for s in shapes]
    assert len(identities)==len(set(identities))
    for shape in shapes:
        inside=all(bounds[0]<=shape['ratios'][k]<=bounds[1] for k,bounds in protocol['shape_ranges'].items())
        if shape['split'] in ('train','validation','test_interpolation'): assert inside
        elif shape['split']=='test_extrapolation': assert not inside
        else: assert shape['family']!='tapered'
        p=shape['parameters']
        assert p['top_outer_diameter']-2*p['wall_thickness']==pytest.approx(1)
        assert p['bottom_outer_diameter']>=p['top_outer_diameter']


def test_paired_training_conditions_have_balanced_shape_slots():
    conditions=read(PLAN/'conditions.json')['train']
    slots=[x['shape_slot'] for x in conditions]
    assert len({x['condition_id'] for x in conditions})==len(conditions)
    assert {slots.count(slot) for slot in set(slots)}=={64}
