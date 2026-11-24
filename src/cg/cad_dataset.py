"""外部CAD(.obj)の形状検査、全13条件評価manifest生成、生成済みデータセット検証（Blender不要）。"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from cg.geometry_model import DEGRADATION_NAMES, TEST_SEVERITIES, read_json, sha256_file, write_json


def inspect_cad_obj(obj_path: Path, *, min_ring_vertices: int = 16) -> dict[str, Any]:
    """Wavefront OBJの座標範囲・マテリアル・Z軸回りリング候補・推奨撮影距離を解析する。"""
    resolved = Path(obj_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"OBJファイルが見つかりません: {resolved}")

    groups: list[dict[str, Any]] = []
    materials: set[str] = set()
    all_vertices: list[tuple[float, float, float]] = []
    current_group = "(default)"
    current_vertices: list[tuple[float, float, float]] = []

    def _flush_group() -> None:
        if current_vertices:
            groups.append({"name": current_group, "vertices": list(current_vertices)})

    with resolved.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(("g ", "o ")):
                _flush_group()
                current_group = stripped[2:].strip() or "(unnamed)"
                current_vertices = []
            elif stripped.startswith("usemtl "):
                materials.add(stripped[7:].strip())
            elif stripped.startswith("v "):
                parts = stripped.split()
                if len(parts) >= 4:
                    pt = (float(parts[1]), float(parts[2]), float(parts[3]))
                    all_vertices.append(pt)
                    current_vertices.append(pt)
    _flush_group()

    if not all_vertices:
        raise ValueError(f"OBJに頂点が含まれていません: {resolved}")

    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    zs = [v[2] for v in all_vertices]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)
    outer_diameter = max(span_x, span_y)

    # Z軸回りの水平リング候補を抽出し、カメラ側(+Z)かつ小半径(内周)優先で上位を返す。
    buckets: dict[tuple[float, float], list[tuple[float, float, float]]] = defaultdict(list)
    for vx, vy, vz in all_vertices:
        radius = math.hypot(vx, vy)
        if radius <= 1e-4:
            continue
        buckets[(round(vz, 4), round(radius, 4))].append((vx, vy, vz))

    ring_candidates: list[dict[str, Any]] = []
    for (_, _), pts in buckets.items():
        if len(pts) < min_ring_vertices:
            continue
        z_vals = [p[2] for p in pts]
        r_vals = [math.hypot(p[0], p[1]) for p in pts]
        angles = sorted(math.degrees(math.atan2(p[1], p[0])) for p in pts)
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(angles[0] + 360.0 - angles[-1])
        ring_candidates.append(
            {
                "z": round(sum(z_vals) / len(z_vals), 6),
                "radius": round(sum(r_vals) / len(r_vals), 6),
                "diameter": round(2.0 * sum(r_vals) / len(r_vals), 6),
                "vertex_count": len(pts),
                "max_angle_gap_deg": round(max(gaps), 2),
                "z_spread": round(max(z_vals) - min(z_vals), 6),
                "r_spread": round(max(r_vals) - min(r_vals), 6),
            }
        )

    # 全周を均等に覆う（最大角度ギャップ <= 25度）リングを +Z 側・内周優先で並べる。
    ring_candidates.sort(key=lambda item: (-item["z"], item["radius"]))
    top_rings = [item for item in ring_candidates if item["max_angle_gap_deg"] <= 25.0][:10]

    # 1194M / 1666MA と整合する推奨カメラ距離・オフセット（外径基準）を算出する。
    rec_near = round(outer_diameter * 1.80, 3)
    rec_far = round(outer_diameter * 2.34, 3)
    rec_offset = round(outer_diameter * 0.068, 3)

    return {
        "model_path": str(resolved),
        "vertex_count": len(all_vertices),
        "groups": [{"name": g["name"], "vertex_count": len(g["vertices"])} for g in groups],
        "materials": sorted(materials),
        "bbox": {
            "x": [round(min(xs), 6), round(max(xs), 6)],
            "y": [round(min(ys), 6), round(max(ys), 6)],
            "z": [round(min(zs), 6), round(max(zs), 6)],
            "span": [round(span_x, 6), round(span_y, 6), round(span_z, 6)],
            "outer_diameter": round(outer_diameter, 6),
        },
        "top_ring_candidates": top_rings,
        "recommended_camera": {
            "distances": [rec_near, rec_far],
            "target_offset_x": rec_offset,
        },
    }


def build_cad_evaluation_manifest(
    dataset_dir: Path,
    *,
    source_name: str,
    seed: int = 2026102404,
) -> Path:
    """原CGのraw_manifest.jsonから、統一評価用の全13条件test manifest.jsonを作成する。"""
    root = Path(dataset_dir).resolve()
    raw_path = root / "raw_manifest.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"原CGのraw_manifest.jsonが見つかりません: {raw_path}")

    raw_manifest = read_json(raw_path)
    raw_samples = raw_manifest["samples"]
    if not raw_samples:
        raise ValueError(f"raw_manifest.jsonにサンプルがありません: {raw_path}")

    rng = random.Random(int(seed))
    expanded: list[dict[str, Any]] = []
    for sample in raw_samples:
        base_id = str(sample["sample_id"])
        image_rel = str(sample["image"])
        label_rel = str(sample["label"])
        image_path = root / image_rel
        label_path = root / label_rel
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"画像またはラベルが不足しています: {base_id}")

        img_hash = sha256_file(image_path)
        lbl_hash = sha256_file(label_path)
        base_conditions = dict(sample["conditions"])
        group_id = str(base_conditions.get("camera_id", base_id))

        rows: list[tuple[str, float]] = [("clean", 0.0)]
        for deg in DEGRADATION_NAMES:
            for sev in TEST_SEVERITIES:
                rows.append((deg, float(sev)))

        for idx, (deg, sev) in enumerate(rows):
            aug_seed = rng.randrange(1, 2**31 - 1)
            expanded.append(
                {
                    "sample_id": f"{base_id}__{idx:02d}",
                    "image": image_rel,
                    "label": label_rel,
                    "conditions": {
                        **base_conditions,
                        "degradation": deg,
                        "severity": sev,
                        "base_sample_id": base_id,
                    },
                    "group_id": group_id,
                    "image_sha256": img_hash,
                    "label_sha256": lbl_hash,
                    "split": "test",
                    "augmentation": {
                        "name": deg,
                        "severity": sev,
                        "seed": aug_seed,
                    },
                }
            )

    payload = {
        "source": source_name,
        "dataset_id": raw_manifest.get("dataset_id", source_name),
        "image_mode": "grayscale",
        "source_manifest_sha256": sha256_file(raw_path),
        "evaluation_manifest_seed": int(seed),
        "base_counts": {"train": 0, "validation": 0, "test": len(raw_samples)},
        "counts": {"train": 0, "validation": 0, "test": len(expanded)},
        "samples": expanded,
    }
    destination = root / "manifest.json"
    write_json(destination, payload)
    return destination


def verify_cad_dataset(config_path: Path, dataset_dir: Path) -> dict[str, Any]:
    """生成済みCAD評価データセットの全原画像・ラベル・画面内座標・ハッシュを検証しプレビューを作成する。"""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    config = read_json(config_path)
    root = Path(dataset_dir).resolve()
    raw_manifest = read_json(root / "raw_manifest.json")
    eval_manifest = read_json(root / "manifest.json")

    width = int(config["render"]["width"])
    height = int(config["render"]["height"])
    expected_verts = int(config["target_ring"]["expected_vertex_count"])

    checked_images: dict[str, str] = {}
    checked_labels: dict[str, str] = {}
    min_margin_px = float("inf")
    major_px_256_list: list[float] = []

    for sample in raw_manifest["samples"]:
        sid = sample["sample_id"]
        img_path = root / sample["image"]
        lbl_path = root / sample["label"]
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (height, width, 3):
            raise ValueError(f"画像の読込または解像度が不正です: {sid}")

        label = read_json(lbl_path)
        points = np.asarray(label["image_points"], dtype=np.float64)
        if len(points) != expected_verts:
            raise ValueError(f"正解頂点数が設定と一致しません: {sid} ({len(points)} != {expected_verts})")
        if not np.isfinite(points).all():
            raise ValueError(f"正解座標に非有限値が含まれます: {sid}")
        margin = float(
            min(
                points[:, 0].min(),
                points[:, 1].min(),
                width - 1.0 - points[:, 0].max(),
                height - 1.0 - points[:, 1].max(),
            )
        )
        if margin < 0.0:
            raise ValueError(f"正解内周エッジが画面外に出ています: {sid} (margin={margin:.2f}px)")
        min_margin_px = min(min_margin_px, margin)

        ellipse = cv2.fitEllipse(points.astype(np.float32))
        major_px_256_list.append(max(ellipse[1]) * (256.0 / float(width)))

        checked_images[sample["image"]] = sha256_file(img_path)
        checked_labels[sample["label"]] = sha256_file(lbl_path)

    # 評価manifestの全件ハッシュ整合と13条件展開を確認する。
    eval_samples = eval_manifest["samples"]
    expected_eval_count = len(raw_manifest["samples"]) * (1 + len(DEGRADATION_NAMES) * len(TEST_SEVERITIES))
    if len(eval_samples) != expected_eval_count:
        raise ValueError(f"評価サンプル数が13条件展開と一致しません: {len(eval_samples)} != {expected_eval_count}")
    for row in eval_samples:
        if row["split"] != "test":
            raise ValueError(f"未知CAD評価データにtest以外のsplitが混在しています: {row['sample_id']}")
        if checked_images.get(row["image"]) != row["image_sha256"]:
            raise ValueError(f"画像SHA256が一致しません: {row['sample_id']}")
        if checked_labels.get(row["label"]) != row["label_sha256"]:
            raise ValueError(f"ラベルSHA256が一致しません: {row['sample_id']}")

    # 仰角・距離を横断した16枚の正解重ね合わせプレビュー画像を生成する。
    raw_samples = raw_manifest["samples"]
    step = max(1, len(raw_samples) // 16)
    picked = [raw_samples[i] for i in range(0, len(raw_samples), step)][:16]
    cols = 4
    rows = max(1, math.ceil(len(picked) / cols))
    canvas = Image.new("RGB", (cols * 256, rows * 280), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, sample in enumerate(picked):
        bgr = cv2.imdecode(np.fromfile(root / sample["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        label = read_json(root / sample["label"])
        pts = np.asarray(label["image_points"], dtype=np.float32)
        ellipse = cv2.fitEllipse(pts)
        cv2.ellipse(bgr, ellipse, (0, 0, 255), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x, y = (idx % cols) * 256, (idx // cols) * 280
        canvas.paste(Image.fromarray(rgb), (x, y))
        cam = sample["conditions"]["camera"]
        light = sample["conditions"]["lighting"]
        caption = f"t={cam['tilt_deg']:.0f} a={cam['azimuth_deg']:.0f} L={light['tilt_deg']:.0f}/{light['azimuth_deg']:.0f}"
        draw.text((x + 4, y + 260), caption, fill="black")
    canvas.save(root / "dataset_preview.png")

    result = {
        "status": "passed",
        "dataset_id": raw_manifest.get("dataset_id"),
        "raw_image_count": len(raw_samples),
        "evaluation_sample_count": len(eval_samples),
        "min_ring_margin_px": round(min_margin_px, 2),
        "major_px_256_range": [round(min(major_px_256_list), 2), round(max(major_px_256_list), 2)],
        "manifest_sha256": sha256_file(root / "manifest.json"),
    }
    write_json(root / "verification.json", result)
    return result
