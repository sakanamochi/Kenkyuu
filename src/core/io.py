"""UTF-8 JSON入出力、Windows日本語パス対応の画像入出力、SHA256計算。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    """設定JSONを読み込み、設定辞書とプロジェクト基準ディレクトリを返す。"""
    resolved = Path(config_path).resolve()
    return read_json(resolved), resolved.parent.parent


def resolve_path(base_dir: Path, relative_or_absolute: str | Path) -> Path:
    """相対パスを基準ディレクトリから解決する。"""
    path = Path(relative_or_absolute)
    return path if path.is_absolute() else (base_dir / path).resolve()


def read_json(path: str | Path) -> Any:
    """UTF-8でJSONファイルを読み込む。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """UTF-8で整形済みJSONファイルを書き出す。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """日本語を含むWindowsパスでも安全に画像を読み込む。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {file_path}")
    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"画像をデコードできません: {file_path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """日本語を含むWindowsパスへ安全に画像を書き出す。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(target.suffix or ".png", image)
    if not ok:
        raise ValueError(f"画像をエンコードできません: {target}")
    encoded.tofile(target)


def sha256_file(path: str | Path) -> str:
    """ファイルのSHA256ハッシュを計算する。"""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """バイト列のSHA256ハッシュを計算する。"""
    return hashlib.sha256(data).hexdigest()


def load_manifest_samples(dataset_dir: Path, split: str | None = None) -> list[dict[str, Any]]:
    """manifest.jsonから指定splitのサンプル一覧を読み込む。"""
    manifest = read_json(Path(dataset_dir) / "manifest.json")
    samples = manifest["samples"]
    if split is None:
        return list(samples)
    return [sample for sample in samples if sample["split"] == split]


def load_evaluation_test_samples(
    config: dict[str, Any],
    data_root: Path,
    dataset_name: str,
) -> tuple[Path, list[dict[str, Any]]]:
    """評価用testサンプル一覧を返す。test_views指定時は形状IDごとに各外観ビューを均等・決定的に割当する。"""
    ds_cfg = config["datasets"][dataset_name]
    base_dir = (Path(data_root) / ds_cfg["base"]).resolve()
    test_views = ds_cfg.get("test_views")
    if not test_views:
        return base_dir, load_manifest_samples(base_dir, "test")

    base_samples = load_manifest_samples(base_dir, "test")
    views_cfg = ds_cfg.get("views", {})
    shape_ids = sorted({str(s.get("shape_id", s["group_id"])) for s in base_samples})
    shape_to_view = {sid: str(test_views[idx % len(test_views)]) for idx, sid in enumerate(shape_ids)}

    view_lookup: dict[str, tuple[Path, dict[str, dict[str, Any]]]] = {}
    for vname in set(shape_to_view.values()):
        vdir = (Path(data_root) / views_cfg[vname]).resolve()
        vrows = {row["sample_id"]: row for row in load_manifest_samples(vdir, "test")}
        view_lookup[vname] = (vdir, vrows)

    merged: list[dict[str, Any]] = []
    for base_row in base_samples:
        sid = str(base_row.get("shape_id", base_row["group_id"]))
        vname = shape_to_view[sid]
        if vname == "base":
            merged.append({**base_row, "eval_view": vname})
            continue
        vdir, vrows = view_lookup[vname]
        vrow = vrows.get(base_row["sample_id"])
        if vrow is None:
            raise FileNotFoundError(f"評価ビュー {vname} にサンプル {base_row['sample_id']} が存在しません。")
        img_abs = (vdir / vrow["image"]).resolve()
        lbl_abs = (vdir / vrow["label"]).resolve()
        merged.append(
            {
                **vrow,
                "image": str(img_abs),
                "label": str(lbl_abs),
                "eval_view": vname,
            }
        )
    return base_dir, merged
