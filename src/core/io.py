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
