"""上端外径D・独立穴径・固定板厚を守る形状定義の回帰テスト。"""

from collections import Counter
import itertools
import math

import pytest

from parametric_paf.paf_model import make_geometry


@pytest.mark.parametrize('bottom,height,hole', list(itertools.product((1.0, 3.0), (0.5, 3.0), (0.4, 0.95))))
def test_extreme_dimensions_make_closed_mesh_with_correct_rim(bottom, height, hole):
    """範囲の端点でも全ての辺が2面に接し、指定した径・全高になる。"""
    values = {'top_outer_diameter': 1, 'bottom_outer_diameter': bottom, 'height': height,
              'hole_diameter': hole, 'wall_thickness': 0.02}
    vertices, faces, rim, profile = make_geometry(values)
    edges = Counter(tuple(sorted((face[i], face[(i + 1) % len(face)])))
                    for face in faces for i in range(len(face)))
    assert set(edges.values()) == {2}
    assert len(vertices) - len(edges) + len(faces) == 2
    assert max(z for _, _, z in vertices) == 0
    assert min(z for _, _, z in vertices) == -height
    assert max(math.hypot(x, y) for x, y, _ in vertices) == pytest.approx(bottom / 2)
    assert all(math.hypot(vertices[i][0], vertices[i][1]) == pytest.approx(hole / 2) for i in rim)
    assert all(vertices[i][2] == 0 for i in rim)


def test_wall_thickness_is_normal_to_the_wall_and_scales_with_outer_diameter():
    """傾いた側壁と内壁の法線距離が、Dの2%で一定になる。"""
    for diameter in (1, 2):
        values = {'top_outer_diameter': diameter, 'bottom_outer_diameter': 3 * diameter,
                  'height': 0.5 * diameter, 'hole_diameter': 0.95 * diameter,
                  'wall_thickness': 0.02 * diameter}
        _, _, _, profile = make_geometry(values)
        slope = (values['bottom_outer_diameter'] - diameter) / (2 * values['height'])
        for radius, z in (profile[5], profile[6]):
            distance = abs(radius + slope * z - diameter / 2) / math.sqrt(1 + slope * slope)
            assert distance == pytest.approx(0.02 * diameter)
        assert profile[7][1] == pytest.approx(-0.02 * diameter)


def test_hole_diameter_does_not_change_outer_dimensions_or_wall_thickness():
    """穴径を変えたとき、側壁・全高・上下面の板厚は変わらない。"""
    values = {'top_outer_diameter': 1, 'bottom_outer_diameter': 2, 'height': 1.5,
              'hole_diameter': 0.4, 'wall_thickness': 0.02}
    _, _, _, a = make_geometry(values)
    _, _, _, b = make_geometry({**values, 'hole_diameter': 0.95})
    assert a[1:7] == b[1:7]
    assert a[0][0] == 0.2 and b[0][0] == 0.475
    assert a[7][1] == b[7][1] == -0.02
