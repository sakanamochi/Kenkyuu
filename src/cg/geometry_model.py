"""多様形状PAFの回転断面モデル、寸法計画生成、生成済みCGの検証（Blender不要）。"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_geometry(
    parameters: dict[str, float],
    segments: int = 192,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]], list[int], list[tuple[float, float]]]:
    """上端外径を基準に、独立した穴径と法線方向一定肉厚を持つ閉メッシュを作る。"""
    diameter = float(parameters["top_outer_diameter"])
    bottom = float(parameters["bottom_outer_diameter"])
    height = float(parameters["height"])
    hole = float(parameters["hole_diameter"])
    thickness = float(parameters["wall_thickness"])

    if not all(math.isfinite(v) and v > 0.0 for v in (diameter, bottom, height, hole, thickness)):
        raise ValueError("寸法は有限の正数で指定してください。")
    if bottom < diameter or not (0.0 < hole < diameter) or height <= 2.0 * thickness:
        raise ValueError("下端径・穴径・高さの組合せが不正です。")

    outer_top, outer_bottom, opening = diameter / 2.0, bottom / 2.0, hole / 2.0
    slope = (outer_bottom - outer_top) / height
    radial_offset = thickness * math.sqrt(1.0 + slope * slope)
    inner_top = outer_top + slope * thickness - radial_offset
    inner_bottom = outer_bottom - slope * thickness - radial_offset
    if inner_top <= opening or inner_bottom <= 0.0:
        raise ValueError("穴径と肉厚を両立できません。")

    profile = [
        (opening, 0.0),
        (outer_top, 0.0),
        (outer_bottom, -height),
        (0.0, -height),
        (0.0, -height + thickness),
        (inner_bottom, -height + thickness),
        (inner_top, -thickness),
        (opening, -thickness),
    ]
    vertices: list[tuple[float, float, float]] = []
    loops: list[list[int]] = []
    for radius, z in profile:
        if radius == 0.0:
            loops.append([len(vertices)])
            vertices.append((0.0, 0.0, z))
        else:
            loops.append(list(range(len(vertices), len(vertices) + segments)))
            vertices.extend(
                (
                    radius * math.cos(2.0 * math.pi * i / segments),
                    radius * math.sin(2.0 * math.pi * i / segments),
                    z,
                )
                for i in range(segments)
            )

    faces: list[tuple[int, ...]] = []
    for index, current in enumerate(loops):
        following = loops[(index + 1) % len(loops)]
        if len(current) == 1 and len(following) == 1:
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


def build_dimensions(config: dict[str, Any], ratios: dict[str, float]) -> dict[str, float]:
    """上端外径Dと比率辞書から実寸法辞書を作成する。"""
    diameter = float(config["top_outer_diameter"])
    return {
        "top_outer_diameter": diameter,
        "bottom_outer_diameter": diameter * float(ratios["bottom_outer_diameter_ratio"]),
        "height": diameter * float(ratios["height_ratio"]),
        "hole_diameter": diameter * float(ratios["hole_diameter_ratio"]),
        "wall_thickness": diameter * float(config["wall_thickness_ratio"]),
    }


def generate_diversified_plan(
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """多様形状PAFの重複なし寸法・撮影条件計画を生成し、plan.jsonへ保存する。"""
    config = read_json(config_path)
    rng = random.Random(int(config["seed"]))
    samples: list[dict[str, Any]] = []
    identities: set[tuple[float, float, float]] = set()
    ratio_keys = ("bottom_outer_diameter_ratio", "height_ratio", "hole_diameter_ratio")

    while len(samples) < int(config["image_count"]):
        ratios = {key: round(rng.uniform(*config[key]), 6) for key in ratio_keys}
        identity = (ratios[ratio_keys[0]], ratios[ratio_keys[1]], ratios[ratio_keys[2]])
        if identity in identities:
            continue
        identities.add(identity)
        appearance = config["appearance"]
        conditions = {
            "camera_tilt": rng.uniform(*config["camera"]["tilt_deg"]),
            "camera_azimuth": rng.uniform(*config["camera"]["azimuth_deg"]),
            "brightness": rng.uniform(*appearance["brightness"]),
            "metallic": rng.uniform(*appearance["metallic"]),
            "roughness": rng.uniform(*appearance["roughness"]),
            "light_tilt": rng.uniform(*appearance["light_tilt_deg"]),
            "light_azimuth": rng.uniform(0.0, 360.0),
            "energy": rng.uniform(*appearance["energy"]),
            "exposure": rng.uniform(*appearance["exposure"]),
        }
        samples.append(
            {
                "sample_id": f"paf_{len(samples):04d}",
                "ratios": ratios,
                "parameters": build_dimensions(config, ratios),
                "conditions": conditions,
            }
        )

    plan_value = {"config_sha256": sha256_file(config_path), "samples": samples}
    destination = Path(output_dir) / "plan.json"
    if destination.exists() and read_json(destination) != plan_value:
        raise ValueError("既存のplan.jsonと計画が一致しません。別の出力先を指定してください。")
    write_json(destination, plan_value)
    return plan_value


def verify_diversified_dataset(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """生成済み多様形状CGの全画像・寸法・正解座標・SHA256を検査する。"""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    config = read_json(config_path)
    out = Path(output_dir)
    planned = read_json(out / "plan.json")
    manifest = read_json(out / "manifest.json")

    if len(manifest["samples"]) != int(config["image_count"]):
        raise ValueError("生成枚数が設定と一致しません。")

    ratio_keys = ("bottom_outer_diameter_ratio", "height_ratio", "hole_diameter_ratio")
    identities: set[tuple[float, ...]] = set()
    resolution = int(config["resolution"])

    for sample, expected in zip(manifest["samples"], planned["samples"]):
        if sample["sample_id"] != expected["sample_id"] or sample["parameters"] != expected["parameters"]:
            raise ValueError("寸法計画と生成データが一致しません。")
        ratios = sample["ratios"]
        identities.add(tuple(ratios[key] for key in ratio_keys))
        for key in ratio_keys:
            if not (config[key][0] <= ratios[key] <= config[key][1]):
                raise ValueError(f"寸法比が範囲外です: {key}={ratios[key]}")
        for key in ("image", "label"):
            if sha256_file(out / sample[key]) != sample[f"{key}_sha256"]:
                raise ValueError(f"{key}のSHA256が一致しません: {sample['sample_id']}")
        label = read_json(out / sample["label"])
        points = np.asarray(label["image_points"], dtype=np.float64)
        if not np.isfinite(points).all() or points.min() < 0 or points.max() >= resolution:
            raise ValueError(f"正解輪郭が画面外です: {sample['sample_id']}")
        image = cv2.imdecode(np.fromfile(out / sample["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (resolution, resolution, 3):
            raise ValueError(f"画像の読込または解像度が不正です: {sample['sample_id']}")
        dims = build_dimensions(config, ratios)
        if dims != label["parameters"] or dims != sample["parameters"]:
            raise ValueError(f"寸法記録が一致しません: {sample['sample_id']}")
        make_geometry(dims, int(config["segments"]))

    if len(identities) != int(config["image_count"]):
        raise ValueError("寸法の組合せに重複があります。")

    canvas = Image.new("RGB", (4 * 256, 4 * 294), "white")
    draw = ImageDraw.Draw(canvas)
    for index, sample in enumerate(manifest["samples"][:16]):
        img = Image.open(out / sample["image"]).convert("RGB").resize((256, 256))
        x, y = (index % 4) * 256, (index // 4) * 294
        canvas.paste(img, (x, y))
        ratios = sample["ratios"]
        draw.text((x + 3, y + 257), f"{sample['sample_id']}  D=1", fill="black")
        draw.text(
            (x + 3, y + 275),
            f"B={ratios[ratio_keys[0]]:.2f} H={ratios[ratio_keys[1]]:.2f} hole={ratios[ratio_keys[2]]:.2f}",
            fill="black",
        )
    canvas.save(out / "dataset_preview.png")

    result = {
        "status": "passed",
        "sample_count": len(manifest["samples"]),
        "unique_dimensions": len(identities),
        "manifest_sha256": sha256_file(out / "manifest.json"),
    }
    write_json(out / "verification.json", result)
    return result
