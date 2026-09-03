"""JSONパラメータから比較実験用のPAF型OBJを生成する。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Mesh:
    """OBJへ書き出すための最小メッシュ表現。"""

    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces_by_material: dict[str, list[tuple[int, ...]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_vertex(self, value: tuple[float, float, float]) -> int:
        self.vertices.append(value)
        return len(self.vertices)

    def add_face(self, material: str, *indices: int) -> None:
        self.faces_by_material[material].append(tuple(indices))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(config: dict) -> None:
    """破綻した形状を早い段階で止める。"""
    segments = int(config["segments"])
    dimensions = config["dimensions"]
    inner = float(dimensions["inner_radius"])
    lip_outer = float(dimensions["lip_outer_radius"])
    rear_inner = float(dimensions["rear_inner_radius"])
    outer = float(dimensions["outer_radius"])
    front_z = float(dimensions["front_z"])
    rear_z = float(dimensions["rear_z"])

    if segments < 32:
        raise ValueError("segmentsは32以上にしてください")
    if not 0 < inner < lip_outer < outer:
        raise ValueError("半径は inner < lip_outer < outer を満たす必要があります")
    if not inner <= rear_inner < outer:
        raise ValueError("rear_inner_radiusが外形の範囲外です")
    if not rear_z < front_z:
        raise ValueError("rear_zはfront_zより小さくしてください")
    for section in ("ribs", "latches", "guide_posts", "bolts"):
        if int(config[section]["count"]) < 0:
            raise ValueError(f"{section}.countは0以上にしてください")
    if set(config["materials"]) != {
        "AluminumBody",
        "AluminumRib",
        "PEEKLatch",
        "SteelPost",
    }:
        raise ValueError("materialsの4名称は変更せず、値だけを調整してください")


def _add_revolved_body(mesh: Mesh, config: dict) -> None:
    dimensions = config["dimensions"]
    segments = int(config["segments"])
    inner = float(dimensions["inner_radius"])
    lip_outer = float(dimensions["lip_outer_radius"])
    outer = float(dimensions["outer_radius"])
    rear_inner = float(dimensions["rear_inner_radius"])
    front_z = float(dimensions["front_z"])
    rear_z = float(dimensions["rear_z"])
    lip_thickness = float(dimensions["lip_thickness"])
    outer_rim_height = float(dimensions["outer_rim_height"])

    # 最初の輪が正解内周になる。順番はメタデータにも保存する。
    profile = [
        (inner, front_z),
        (lip_outer, front_z),
        (lip_outer, front_z - lip_thickness),
        (outer, rear_z + outer_rim_height),
        (outer, rear_z),
        (rear_inner, rear_z),
        (inner, front_z - lip_thickness),
    ]
    rings: list[list[int]] = []
    for radius, z_value in profile:
        ring = []
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            ring.append(
                mesh.add_vertex(
                    (radius * math.cos(angle), radius * math.sin(angle), z_value)
                )
            )
        rings.append(ring)

    for profile_index in range(len(rings)):
        next_profile = (profile_index + 1) % len(rings)
        for index in range(segments):
            next_index = (index + 1) % segments
            mesh.add_face(
                "AluminumBody",
                rings[profile_index][index],
                rings[profile_index][next_index],
                rings[next_profile][next_index],
                rings[next_profile][index],
            )


def _cone_z(radius: float, config: dict) -> float:
    dimensions = config["dimensions"]
    radius_0 = float(dimensions["lip_outer_radius"])
    radius_1 = float(dimensions["outer_radius"])
    z_0 = float(dimensions["front_z"]) - float(dimensions["lip_thickness"])
    z_1 = float(dimensions["rear_z"]) + float(dimensions["outer_rim_height"])
    ratio = (radius - radius_0) / (radius_1 - radius_0)
    return z_0 + ratio * (z_1 - z_0)


def _add_oriented_box(
    mesh: Mesh,
    *,
    angle: float,
    radius_inner: float,
    radius_outer: float,
    width: float,
    z_inner: float,
    z_outer: float,
    height: float,
    material: str,
) -> None:
    """半径方向へ向いた直方体または傾斜リブを追加する。"""
    radial = (math.cos(angle), math.sin(angle))
    tangent = (-math.sin(angle), math.cos(angle))
    bottom = []
    top = []
    for radius, z_value in ((radius_inner, z_inner), (radius_outer, z_outer)):
        for side in (-0.5, 0.5):
            x = radius * radial[0] + side * width * tangent[0]
            y = radius * radial[1] + side * width * tangent[1]
            bottom.append(mesh.add_vertex((x, y, z_value)))
            top.append(mesh.add_vertex((x, y, z_value + height)))

    # 頂点順は内側左、内側右、外側左、外側右。
    mesh.add_face(material, bottom[0], bottom[2], bottom[3], bottom[1])
    mesh.add_face(material, top[0], top[1], top[3], top[2])
    mesh.add_face(material, bottom[0], top[0], top[2], bottom[2])
    mesh.add_face(material, bottom[1], bottom[3], top[3], top[1])
    mesh.add_face(material, bottom[0], bottom[1], top[1], top[0])
    mesh.add_face(material, bottom[2], top[2], top[3], bottom[3])


def _add_ribs(mesh: Mesh, config: dict) -> None:
    settings = config["ribs"]
    count = int(settings["count"])
    inner = float(settings["inner_radius"])
    outer = float(settings["outer_radius"])
    for index in range(count):
        angle = 2.0 * math.pi * index / max(count, 1)
        _add_oriented_box(
            mesh,
            angle=angle,
            radius_inner=inner,
            radius_outer=outer,
            width=float(settings["width"]),
            z_inner=_cone_z(inner, config) + 0.02,
            z_outer=_cone_z(outer, config) + 0.02,
            height=float(settings["height"]),
            material="AluminumRib",
        )


def _add_latches(mesh: Mesh, config: dict) -> None:
    settings = config["latches"]
    count = int(settings["count"])
    radius = float(settings["radius"])
    radial_length = float(settings["radial_length"])
    height = float(settings["height"])
    # 外接箱中心によるモデル位置ずれを避け、上面だけをフランジから僅かに出す。
    z_value = float(config["dimensions"]["front_z"]) - height + 0.02
    offset = math.radians(float(settings.get("angle_offset_deg", 0.0)))
    for index in range(count):
        angle = offset + 2.0 * math.pi * index / max(count, 1)
        _add_oriented_box(
            mesh,
            angle=angle,
            radius_inner=radius - radial_length / 2.0,
            radius_outer=radius + radial_length / 2.0,
            width=float(settings["tangential_width"]),
            z_inner=z_value,
            z_outer=z_value,
            height=height,
            material="PEEKLatch",
        )


def _add_cylinder(
    mesh: Mesh,
    *,
    x_center: float,
    y_center: float,
    radius: float,
    z_min: float,
    z_max: float,
    segments: int,
    material: str,
) -> None:
    bottom = []
    top = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        x = x_center + radius * math.cos(angle)
        y = y_center + radius * math.sin(angle)
        bottom.append(mesh.add_vertex((x, y, z_min)))
        top.append(mesh.add_vertex((x, y, z_max)))
    for index in range(segments):
        next_index = (index + 1) % segments
        mesh.add_face(
            material,
            bottom[index],
            bottom[next_index],
            top[next_index],
            top[index],
        )
    mesh.add_face(material, *reversed(bottom))
    mesh.add_face(material, *top)


def _add_posts_and_bolts(mesh: Mesh, config: dict) -> None:
    posts = config["guide_posts"]
    post_count = int(posts["count"])
    post_offset = math.radians(float(posts.get("angle_offset_deg", 0.0)))
    post_orbit = float(posts["radius_from_center"])
    for index in range(post_count):
        angle = post_offset + 2.0 * math.pi * index / max(post_count, 1)
        _add_cylinder(
            mesh,
            x_center=post_orbit * math.cos(angle),
            y_center=post_orbit * math.sin(angle),
            radius=float(posts["post_radius"]),
            z_min=float(posts["z_min"]),
            z_max=float(posts["z_max"]),
            segments=int(posts["segments"]),
            material="SteelPost",
        )

    bolts = config["bolts"]
    bolt_count = int(bolts["count"])
    bolt_orbit = float(bolts["radius_from_center"])
    front_z = float(config["dimensions"]["front_z"])
    bolt_height = float(bolts["height"])
    for index in range(bolt_count):
        angle = 2.0 * math.pi * index / max(bolt_count, 1)
        _add_cylinder(
            mesh,
            x_center=bolt_orbit * math.cos(angle),
            y_center=bolt_orbit * math.sin(angle),
            radius=float(bolts["bolt_radius"]),
            z_min=front_z - bolt_height + 0.02,
            z_max=front_z + 0.02,
            segments=int(bolts["segments"]),
            material="SteelPost",
        )


def build_mesh(config: dict) -> Mesh:
    _validate(config)
    mesh = Mesh()
    _add_revolved_body(mesh, config)
    _add_ribs(mesh, config)
    _add_latches(mesh, config)
    _add_posts_and_bolts(mesh, config)
    return mesh


def _write_mtl(path: Path, materials: dict) -> None:
    lines = ["# パラメトリックPAF材質"]
    for name, values in materials.items():
        red, green, blue = values["base_color"]
        lines.extend(
            [
                "",
                f"newmtl {name}",
                f"Kd {red:.6f} {green:.6f} {blue:.6f}",
                "Ka 0.000000 0.000000 0.000000",
                "Ks 0.500000 0.500000 0.500000",
                "Ns 96.000000",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_obj(path: Path, mesh: Mesh, mtl_name: str) -> None:
    lines = [
        "# JSONパラメータから生成した比較実験用PAF型モデル",
        f"mtllib {mtl_name}",
        "o PAF_PARAMETRIC",
    ]
    lines.extend(
        f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in mesh.vertices
    )
    lines.append("s 1")
    for material, faces in mesh.faces_by_material.items():
        lines.append(f"usemtl {material}")
        lines.extend("f " + " ".join(map(str, face)) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(config_path: Path, output_path: Path) -> dict:
    config = _read_json(config_path)
    mesh = build_mesh(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = output_path.with_suffix(".mtl")
    metadata_path = output_path.with_suffix(".json")
    _write_mtl(mtl_path, config["materials"])
    _write_obj(output_path, mesh, mtl_path.name)

    dimensions = config["dimensions"]
    metadata = {
        "model_id": config["model_id"],
        "description": config["description"],
        "source_config": str(config_path),
        "obj": str(output_path),
        "vertex_count": len(mesh.vertices),
        "face_count": sum(len(faces) for faces in mesh.faces_by_material.values()),
        "target_ring": {
            "radius": float(dimensions["inner_radius"]),
            "z": float(dimensions["front_z"]),
            "vertex_count": int(config["segments"]),
            "obj_vertex_indices_1_based": list(range(1, int(config["segments"]) + 1)),
        },
        "parameters": config,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="PAF型モデルをOBJとして生成する")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = generate(args.config, args.output)
    print(
        f"Generated: {metadata['obj']} "
        f"({metadata['vertex_count']} vertices, {metadata['face_count']} faces)"
    )


if __name__ == "__main__":
    main()
