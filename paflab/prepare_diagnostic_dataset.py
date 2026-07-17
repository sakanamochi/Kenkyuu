from __future__ import annotations

import argparse
import json
import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np

from paflab.camera_effects import EFFECTS
from paflab.image_io import imread, imwrite
from paflab.labels import fit_label_ellipse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="教授向け反証用の単純遮蔽・撮像劣化データを作る")
    parser.add_argument("--config", default="config/research_second_stage.json")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % (2**32)


def diagnostic_seed(
    experiment_id: str,
    base_sample_id: str,
    effect: str,
    severity: float,
) -> int:
    """黒矩形だけは強度間で同じ画面端を選び、進行方向を固定する。"""
    if effect == "black_rectangle":
        return stable_seed(experiment_id, base_sample_id, effect)
    return stable_seed(experiment_id, base_sample_id, effect, severity)


def main() -> None:
    args = parse_args()
    suite = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    base_config = json.loads(project_path(suite["base_config"]).read_text(encoding="utf-8"))
    source_dir = project_path(base_config["stress_dataset_dir"])
    output_dir = project_path(suite["diagnostics"]["dataset_dir"])
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    base_samples = [
        sample
        for sample in manifest["samples"]
        if sample["split"] == "test" and sample["conditions"]["degradation"] == "clean"
    ]
    samples = []

    for base_sample in base_samples:
        image = imread(source_dir / base_sample["image"], cv2.IMREAD_COLOR)
        label_source = source_dir / base_sample["label"]
        label = json.loads(label_source.read_text(encoding="utf-8"))
        ellipse = fit_label_ellipse(label)
        variants = [("clean", 0.0)] + [
            (effect, float(severity))
            for effect in suite["diagnostics"]["effects"]
            for severity in suite["diagnostics"]["severities"]
        ]
        for variant_index, (effect, severity) in enumerate(variants):
            seed = diagnostic_seed(
                suite["experiment_id"],
                base_sample["sample_id"],
                effect,
                severity,
            )
            rng = np.random.default_rng(seed)
            if effect == "clean":
                rendered = image.copy()
                metadata = {"definition": "未劣化"}
            elif effect == "black_rectangle":
                rendered, metadata = EFFECTS[effect](image, ellipse, severity, rng=rng)
            else:
                rendered, metadata = EFFECTS[effect](image, severity, rng=rng)
            sample_id = (
                f"{base_sample['conditions']['base_sample_id']}__diagnostic"
                f"__{effect}_s{round(severity * 1000):04d}_v{variant_index:02d}"
            )
            image_path = images_dir / f"{sample_id}.png"
            label_path = labels_dir / f"{sample_id}.json"
            imwrite(image_path, rendered)
            shutil.copyfile(label_source, label_path)
            conditions = {
                **base_sample["conditions"],
                "degradation": effect,
                "severity": severity,
                "diagnostic_metadata": metadata,
                "split": "diagnostic_test",
                "variant_seed": seed,
            }
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": "diagnostic_test",
                    "image": image_path.relative_to(output_dir).as_posix(),
                    "label": label_path.relative_to(output_dir).as_posix(),
                    "conditions": conditions,
                }
            )

    output_manifest = {
        "experiment_id": suite["experiment_id"],
        "source_dataset": str(source_dir),
        "sample_count": len(samples),
        "base_sample_count": len(base_samples),
        "effects": suite["diagnostics"]["effects"],
        "severities": suite["diagnostics"]["severities"],
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"dataset": str(output_dir), "samples": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
