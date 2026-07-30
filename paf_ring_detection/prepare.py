"""基礎CGから学習用・撮像診断用データを作る。"""

from __future__ import annotations

import random
import shutil
import zlib
from pathlib import Path

import numpy as np

from paf_ring_detection.data import (
    label_ellipse,
    read_image,
    read_json,
    write_image,
    write_json,
)
from paf_ring_detection.effects import DIAGNOSTIC_EFFECTS, training_effect


def _seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return zlib.crc32(text) % (2**32)


def _split_groups(groups: list[str], settings: dict, seed: int) -> dict[str, str]:
    groups = sorted(groups)
    random.Random(seed).shuffle(groups)
    train_end = round(len(groups) * settings["train_fraction"])
    validation_end = train_end + round(
        len(groups) * settings["validation_fraction"]
    )
    return {
        group: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, group in enumerate(groups)
    }


def _training_variants(split: str, settings: dict, rng) -> list[tuple[str, float]]:
    if split == "train":
        variants = [("clean", 0.0)]
        for index in range(settings["variants_per_image"]):
            name = settings["effects"][index % len(settings["effects"])]
            severity = rng.uniform(settings["severity_min"], settings["severity_max"])
            variants.append((name, float(severity)))
        return variants

    severities = (
        settings["validation_severities"]
        if split == "validation"
        else settings["test_severities"]
    )
    return [("clean", 0.0)] + [
        (name, float(severity))
        for name in settings["effects"]
        for severity in severities
    ]


def prepare_training_data(config: dict, limit_groups: int | None = None) -> None:
    """714枚の基礎CGをカメラ単位で分割し、劣化画像を作る。"""
    seed = int(config["seed"])
    source = Path(config["paths"]["base_dataset"])
    output = Path(config["paths"]["training_dataset"])
    manifest = read_json(source / "manifest.json")

    groups = sorted(
        {sample["conditions"]["camera_id"] for sample in manifest["samples"]}
    )
    if limit_groups:
        groups = groups[:limit_groups]
    splits = _split_groups(groups, config["split"], seed)

    samples = []
    for source_sample in manifest["samples"]:
        camera_id = source_sample["conditions"]["camera_id"]
        if camera_id not in splits:
            continue

        split = splits[camera_id]
        image = read_image(source / source_sample["image"])
        label_path = source / source_sample["label"]
        ellipse = label_ellipse(read_json(label_path))
        variant_rng = np.random.default_rng(_seed(seed, source_sample["sample_id"]))

        for index, (effect, severity) in enumerate(
            _training_variants(split, config["training_data"], variant_rng)
        ):
            rng_seed = _seed(seed, source_sample["sample_id"], index)
            rendered = training_effect(
                image,
                effect,
                severity,
                ellipse,
                np.random.default_rng(rng_seed),
            )
            sample_id = (
                f"{source_sample['sample_id']}__{split[:2]}"
                f"__{effect}_s{round(severity * 1000):04d}_v{index:02d}"
            )
            image_path = output / "images" / f"{sample_id}.png"
            label_output = output / "labels" / f"{sample_id}.json"
            write_image(image_path, rendered)
            label_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(label_path, label_output)

            conditions = {
                **source_sample["conditions"],
                "degradation": effect,
                "severity": severity,
                "base_sample_id": source_sample["sample_id"],
            }
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "image": image_path.relative_to(output).as_posix(),
                    "label": label_output.relative_to(output).as_posix(),
                    "conditions": conditions,
                }
            )

    write_json(
        output / "manifest.json",
        {
            "sample_count": len(samples),
            "group_assignments": splits,
            "samples": samples,
        },
    )
    print(f"学習データを作成しました: {output} ({len(samples)}枚)")


def prepare_diagnostic_data(config: dict) -> None:
    """testのclean画像から遮蔽・白飛び・黒つぶれ診断を作る。"""
    source = Path(config["paths"]["training_dataset"])
    output = Path(config["paths"]["diagnostic_dataset"])
    source_samples = [
        sample
        for sample in read_json(source / "manifest.json")["samples"]
        if sample["split"] == "test"
        and sample["conditions"]["degradation"] == "clean"
    ]

    settings = config["diagnostic_data"]
    variants = [("clean", 0.0)] + [
        (name, float(severity))
        for name in settings["effects"]
        for severity in settings["severities"]
    ]
    samples = []

    for source_sample in source_samples:
        image = read_image(source / source_sample["image"])
        label_path = source / source_sample["label"]
        base_id = source_sample["conditions"]["base_sample_id"]

        for index, (effect, severity) in enumerate(variants):
            # 黒矩形は強度が変わっても同じ側から伸ばす。
            seed_parts = (config["experiment_id"], base_id, effect)
            if effect != "black_rectangle":
                seed_parts += (severity,)
            rng = np.random.default_rng(_seed(*seed_parts))
            rendered = (
                image.copy()
                if effect == "clean"
                else DIAGNOSTIC_EFFECTS[effect](image, severity, rng)
            )
            sample_id = (
                f"{base_id}__diagnostic"
                f"__{effect}_s{round(severity * 1000):04d}_v{index:02d}"
            )
            image_path = output / "images" / f"{sample_id}.png"
            label_output = output / "labels" / f"{sample_id}.json"
            write_image(image_path, rendered)
            label_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(label_path, label_output)

            conditions = {
                **source_sample["conditions"],
                "degradation": effect,
                "severity": severity,
                "base_sample_id": base_id,
            }
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": "diagnostic_test",
                    "image": image_path.relative_to(output).as_posix(),
                    "label": label_output.relative_to(output).as_posix(),
                    "conditions": conditions,
                }
            )

    write_json(
        output / "manifest.json",
        {"sample_count": len(samples), "samples": samples},
    )
    print(f"診断データを作成しました: {output} ({len(samples)}枚)")
