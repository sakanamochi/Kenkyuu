"""元モデルの軸方向断面と半径分布を数値確認する。"""

import json
import math

import bpy
from mathutils import Vector


obj = bpy.data.objects["PAF_BASE_01"]
world_vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]

sample_y = [-4.7, -3.5, -2.0, 0.0, 2.0, 4.0]
sections = []
for center_y in sample_y:
    radii = [
        math.hypot(vertex.x, vertex.z)
        for vertex in world_vertices
        if abs(vertex.y - center_y) <= 0.15
    ]
    radii.sort()
    if radii:
        sections.append(
            {
                "y": center_y,
                "count": len(radii),
                "r_min": round(radii[0], 3),
                "r_median": round(radii[len(radii) // 2], 3),
                "r_max": round(radii[-1], 3),
            }
        )

print(
    json.dumps(
        {
            "vertex_count": len(world_vertices),
            "y_min": min(vertex.y for vertex in world_vertices),
            "y_max": max(vertex.y for vertex in world_vertices),
            "sections": sections,
        },
        ensure_ascii=False,
    )
)
