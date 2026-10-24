"""多様形状PAFの閉メッシュ作成と法線・体積検査。"""

import bpy
import bmesh


def mesh_object(name, vertices, faces, material):
    """明示した頂点と面からメッシュを作る。"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    if mesh.validate():
        raise ValueError(f"不正なメッシュ: {name}")
    # 閉じた各部品の法線を外向きに統一し、穴・退化面を検出する。
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    if not all(edge.is_manifold for edge in bm.edges) or bm.calc_volume(signed=True) <= 0:
        bm.free()
        raise ValueError(f"閉じた立体になっていません: {name}")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj
