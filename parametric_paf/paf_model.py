"""上端外径を基準に、穴径と板厚を独立に指定するPAFの回転断面。"""

import math


def make_geometry(parameters, segments=192):
    """上面・底面の板厚と側壁の法線方向厚さをそろえた閉じたメッシュを作る。"""
    diameter = float(parameters['top_outer_diameter'])
    bottom = float(parameters['bottom_outer_diameter'])
    height = float(parameters['height'])
    hole = float(parameters['hole_diameter'])
    thickness = float(parameters['wall_thickness'])
    if not all(math.isfinite(v) and v > 0 for v in (diameter, bottom, height, hole, thickness)):
        raise ValueError('寸法は有限の正数で指定してください')
    if bottom < diameter or not 0 < hole < diameter or height <= 2 * thickness:
        raise ValueError('下端径・穴径・高さの組合せが不正です')
    outer_top, outer_bottom, opening = diameter / 2, bottom / 2, hole / 2
    slope = (outer_bottom - outer_top) / height
    radial_offset = thickness * math.sqrt(1 + slope * slope)
    inner_top = outer_top + slope * thickness - radial_offset
    inner_bottom = outer_bottom - slope * thickness - radial_offset
    if inner_top <= opening or inner_bottom <= 0:
        raise ValueError('穴径と肉厚を両立できません')
    # 上面の開口径を独立に変え、上面・底面に厚さthicknessの板を残す。
    profile = [(opening, 0), (outer_top, 0), (outer_bottom, -height),
               (0, -height), (0, -height + thickness),
               (inner_bottom, -height + thickness), (inner_top, -thickness),
               (opening, -thickness)]
    vertices, loops = [], []
    for radius, z in profile:
        if radius == 0:
            loops.append([len(vertices)])
            vertices.append((0.0, 0.0, z))
        else:
            loops.append(list(range(len(vertices), len(vertices) + segments)))
            vertices.extend((radius * math.cos(2 * math.pi * i / segments),
                             radius * math.sin(2 * math.pi * i / segments), z) for i in range(segments))
    faces = []
    for index, current in enumerate(loops):
        following = loops[(index + 1) % len(loops)]
        if len(current) == len(following) == 1:
            continue
        for i in range(segments):
            j = (i + 1) % segments
            if len(current) == 1:
                faces.append((current[0], following[i], following[j]))
            elif len(following) == 1:
                faces.append((current[i], following[0], current[j]))
            else:
                faces.append((current[i], following[i], following[j], current[j]))
    return vertices, faces, loops[0], profile
