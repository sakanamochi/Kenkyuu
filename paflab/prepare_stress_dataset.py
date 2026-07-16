from __future__ import annotations

import argparse
import json
import random
import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np

from paflab.degradations import apply_degradation
from paflab.image_io import imread, imwrite
from paflab.labels import fit_label_ellipse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % (2**32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blender画像から劣化強度付き研究データを生成する")
    parser.add_argument("--config", default="config/research_experiment.json")
    parser.add_argument("--limit-groups", type=int, default=None, help="スモーク試験用の群数上限")
    return parser.parse_args()


def assign_group_splits(groups: list[str], settings: dict, seed: int) -> dict[str, str]:
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, round(len(shuffled) * float(settings["train_fraction"])))
    validation_count = max(
        1, round(len(shuffled) * float(settings["validation_fraction"]))
    )
    if train_count + validation_count >= len(shuffled):
        validation_count = 1
        train_count = len(shuffled) - 2
    return {
        group: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, group in enumerate(shuffled)
    }


def variants_for_split(split: str, degradation_settings: dict, rng) -> list[tuple[str, float]]:
    if split == "train":
        variants = [("clean", 0.0)]
        types = degradation_settings["types"]
        for index in range(int(degradation_settings["train_variants_per_base"])):
            degradation = types[index % len(types)]
            severity = float(
                rng.uniform(
                    degradation_settings["train_severity_min"],
                    degradation_settings["train_severity_max"],
                )
            )
            variants.append((degradation, severity))
        return variants

    severity_key = "validation_severities" if split == "validation" else "test_severities"
    return [("clean", 0.0)] + [
        (degradation, float(severity))
        for degradation in degradation_settings["types"]
        for severity in degradation_settings[severity_key]
    ]


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = project_path(config["base_dataset_dir"])
    output_dir = project_path(config["stress_dataset_dir"])
    base_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    group_key = config["split"]["group_key"]
    groups = sorted({sample["conditions"][group_key] for sample in base_manifest["samples"]})
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]
    if len(groups) < 3:
        raise ValueError("train/validation/testには3群以上必要です")
    group_splits = assign_group_splits(groups, config["split"], int(config["seed"]))

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    for base_sample in base_manifest["samples"]:
        group = base_sample["conditions"][group_key]
        if group not in group_splits:
            continue
        split = group_splits[group]
        base_image = imread(base_dir / base_sample["image"], cv2.IMREAD_COLOR)
        if base_image is None:
            raise FileNotFoundError(base_dir / base_sample["image"])
        source_label = json.loads((base_dir / base_sample["label"]).read_text(encoding="utf-8"))
        ellipse = fit_label_ellipse(source_label)
        sample_rng = np.random.default_rng(stable_seed(config["seed"], base_sample["sample_id"]))

        for variant_index, (degradation, severity) in enumerate(
            variants_for_split(split, config["degradations"], sample_rng)
        ):
            variant_seed = stable_seed(config["seed"], base_sample["sample_id"], variant_index)
            variant_rng = np.random.default_rng(variant_seed)
            degraded, degradation_metadata = apply_degradation(
                base_image,
                degradation,
                severity,
                ellipse,
                rng=variant_rng,
            )
            severity_code = round(float(severity) * 1000)
            sample_id = (
                f"{base_sample['sample_id']}__{split[:2]}"
                f"__{degradation}_s{severity_code:04d}_v{variant_index:02d}"
            )
            image_path = images_dir / f"{sample_id}.png"
            label_path = labels_dir / f"{sample_id}.json"
            imwrite(image_path, degraded)
            shutil.copyfile(base_dir / base_sample["label"], label_path)
            conditions = {
                **base_sample["conditions"],
                **degradation_metadata,
                "split": split,
                "base_sample_id": base_sample["sample_id"],
                "variant_seed": variant_seed,
            }
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "image": image_path.relative_to(output_dir).as_posix(),
                    "label": label_path.relative_to(output_dir).as_posix(),
                    "conditions": conditions,
                }
            )

    split_counts = {
        split: sum(sample["split"] == split for sample in samples)
        for split in ("train", "validation", "test")
    }
    manifest = {
        "experiment_id": config["experiment_id"],
        "source_dataset": str(base_dir),
        "config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "split_policy": {
            **config["split"],
            "seed": config["seed"],
            "group_assignments": group_splits,
        },
        "split_counts": split_counts,
        "sample_count": len(samples),
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"dataset": str(output_dir), "split_counts": split_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
