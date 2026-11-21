"""多様形状PAFの回転断面モデル、寸法計画生成、分割manifest作成、生成済みCG検証（Blender不要）。"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

RATIO_KEYS = ("bottom_outer_diameter_ratio", "height_ratio", "hole_diameter_ratio")
DEGRADATION_NAMES = ("black_rectangle", "sensor_whiteout", "sensor_black_crush")
VAL_SEVERITIES = (0.3, 0.6)
TEST_SEVERITIES = (0.25, 0.5, 0.75, 1.0)


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


def _assign_splits(image_count: int, splits_cfg: dict[str, int] | None) -> list[str]:
    if not splits_cfg:
        return ["train"] * image_count
    labels: list[str] = []
    for split_name in ("train", "validation", "test"):
        labels.extend([split_name] * int(splits_cfg.get(split_name, 0)))
    if len(labels) != image_count:
        raise ValueError(f"splits合計({len(labels)})とimage_count({image_count})が一致しません。")
    return labels


def generate_diversified_plan(
    config_path: Path,
    output_dir: Path,
    *,
    fixed_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    """多様形状PAF（または固定代表円錐）の寸法・撮影・投影・外観計画を生成しplan.jsonへ保存する。"""
    config = read_json(config_path)
    rng = random.Random(int(config["seed"]))
    image_count = int(config["image_count"])
    splits = _assign_splits(image_count, config.get("splits"))

    proj_cfg = config.get(
        "projection",
        {"major_px_256": [24.0, 180.0], "aspect_ratio": [0.1, 0.996], "center_range": [0.3, 0.7]},
    )
    surf_cfg = config.get(
        "surface",
        {
            "albedo_low": [0.05, 0.35],
            "albedo_high": [0.45, 0.85],
            "metallic": [0.0, 0.85],
            "roughness": [0.15, 0.7],
            "stripe_cycles": [12, 48],
            "noise_scale": [3.0, 14.0],
            "checker_scale": [6.0, 24.0],
        },
    )
    appearance = config["appearance"]
    log_min = math.log(float(proj_cfg["major_px_256"][0]))
    log_max = math.log(float(proj_cfg["major_px_256"][1]))

    samples: list[dict[str, Any]] = []
    identities: set[tuple[float, float, float]] = set()
    while len(samples) < image_count:
        drawn = {key: round(rng.uniform(*config[key]), 6) for key in RATIO_KEYS}
        identity = tuple(drawn[key] for key in RATIO_KEYS)
        if identity in identities:
            continue
        identities.add(identity)
        ratios = (
            {key: round(float(fixed_ratios[key]), 6) for key in RATIO_KEYS}
            if fixed_ratios is not None
            else drawn
        )
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
        projection = {
            "target_major_px_256": math.exp(rng.uniform(log_min, log_max)),
            "target_aspect_ratio": rng.uniform(*proj_cfg["aspect_ratio"]),
            "target_cx": rng.uniform(*proj_cfg["center_range"]),
            "target_cy": rng.uniform(*proj_cfg["center_range"]),
            "camera_azimuth": rng.uniform(*config["camera"]["azimuth_deg"]),
        }
        albedo_low = rng.uniform(*surf_cfg["albedo_low"])
        albedo_high = rng.uniform(*surf_cfg["albedo_high"])
        # 模様なし(albedo)は模様あり(stripe/noise/checker)と平均反射率・材質分布を一致させる。
        surface = {
            "albedo_low": albedo_low,
            "albedo_high": albedo_high,
            "albedo_mean": 0.5 * (albedo_low + albedo_high),
            "metallic": rng.uniform(*surf_cfg["metallic"]),
            "roughness": rng.uniform(*surf_cfg["roughness"]),
            "stripe_cycles": int(rng.randint(*surf_cfg["stripe_cycles"])),
            "stripe_phase": rng.uniform(0.0, 2.0 * math.pi),
            "noise_scale": rng.uniform(*surf_cfg["noise_scale"]),
            "checker_scale": rng.uniform(*surf_cfg["checker_scale"]),
        }
        samples.append(
            {
                "sample_id": f"paf_{len(samples):04d}",
                "split": splits[len(samples)],
                "ratios": ratios,
                "parameters": build_dimensions(config, ratios),
                "conditions": conditions,
                "projection": projection,
                "surface": surface,
            }
        )

    plan_value = {
        "config_sha256": sha256_file(config_path),
        "fixed_ratios": fixed_ratios,
        "samples": samples,
    }
    destination = Path(output_dir) / "plan.json"
    if destination.exists() and read_json(destination) != plan_value:
        raise ValueError("既存のplan.jsonと計画が一致しません。別の出力先を指定してください。")
    write_json(destination, plan_value)
    return plan_value


def build_split_manifests(
    dataset_root: Path,
    view_names: list[str],
    *,
    seed: int = 2026112102,
) -> dict[str, Path]:
    """描画済み各ビューに対して学習・検証・評価用の劣化展開manifest.jsonを作成する。"""
    root = Path(dataset_root).resolve()
    plan = read_json(root / "plan.json")
    samples = plan["samples"]

    # 劣化強度と乱数seedは全ビューで完全に共有し、条件間の揺らぎを防ぐ。
    rng = random.Random(seed)
    expansion: dict[str, list[tuple[str, float, int]]] = {}
    for sample in samples:
        sid = sample["sample_id"]
        split = sample["split"]
        if split == "train":
            rows = [("clean", 0.0, rng.randrange(1, 2**31 - 1))]
            for deg in DEGRADATION_NAMES:
                rows.append((deg, round(rng.uniform(0.1, 0.8), 6), rng.randrange(1, 2**31 - 1)))
        elif split == "validation":
            rows = [("clean", 0.0, rng.randrange(1, 2**31 - 1))]
            for deg in DEGRADATION_NAMES:
                for sev in VAL_SEVERITIES:
                    rows.append((deg, sev, rng.randrange(1, 2**31 - 1)))
        else:
            rows = [("clean", 0.0, rng.randrange(1, 2**31 - 1))]
            for deg in DEGRADATION_NAMES:
                for sev in TEST_SEVERITIES:
                    rows.append((deg, sev, rng.randrange(1, 2**31 - 1)))
        expansion[sid] = rows

    written: dict[str, Path] = {}
    for view in view_names:
        view_dir = root / view
        label_source = "projection_plain" if view.startswith("projection_") else "base"
        manifest_rows: list[dict[str, Any]] = []
        for sample in samples:
            sid = sample["sample_id"]
            image_path = view_dir / "images" / f"{sid}.png"
            if not image_path.exists():
                continue
            label_dir = root / label_source
            label_path = label_dir / "labels" / f"{sid}.json"
            if not label_path.exists():
                raise FileNotFoundError(f"対応ラベルがありません: {label_path}")
            rel_label = (
                f"labels/{sid}.json"
                if label_source == view
                else f"../{label_source}/labels/{sid}.json"
            )
            img_hash = sha256_file(image_path)
            lbl_hash = sha256_file(label_path)
            for idx, (deg, sev, aug_seed) in enumerate(expansion[sid]):
                manifest_rows.append(
                    {
                        "sample_id": f"{sid}__{idx:02d}",
                        "ratios": sample["ratios"],
                        "parameters": sample["parameters"],
                        "conditions": {
                            **sample["conditions"],
                            "degradation": deg,
                            "severity": sev,
                            "base_sample_id": sid,
                        },
                        "split": sample["split"],
                        "shape_id": sid,
                        "group_id": sid,
                        "image": f"images/{sid}.png",
                        "label": rel_label,
                        "image_sha256": img_hash,
                        "label_sha256": lbl_hash,
                        "augmentation": {"name": deg, "severity": sev, "seed": aug_seed},
                    }
                )
        dest = view_dir / "manifest.json"
        write_json(dest, {"view": view, "samples": manifest_rows})
        written[view] = dest
    return written


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

    identities: set[tuple[float, ...]] = set()
    resolution = int(config["resolution"])
    fixed = planned.get("fixed_ratios") is not None

    for sample, expected in zip(manifest["samples"], planned["samples"]):
        if sample["sample_id"] != expected["sample_id"] or sample["parameters"] != expected["parameters"]:
            raise ValueError("寸法計画と生成データが一致しません。")
        ratios = sample["ratios"]
        identities.add(tuple(ratios[key] for key in RATIO_KEYS))
        for key in RATIO_KEYS:
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

    if not fixed and len(identities) != int(config["image_count"]):
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
            f"B={ratios[RATIO_KEYS[0]]:.2f} H={ratios[RATIO_KEYS[1]]:.2f} hole={ratios[RATIO_KEYS[2]]:.2f}",
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
