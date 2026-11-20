"""旧ソースをimportせず、固定画像・重み・推定JSONから再現性を検証する。"""

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import time

import cv2
import numpy as np
import torch

from core.effects import apply_effect
from core.ellipse import label_ellipse, scale_ellipse
from core.files import read_json, sha
from core.images import prepare_image, read_image
from evaluation.metrics import ellipse_iou, ellipse_tuple
from inference.predictor import BoundaryPredictor


def verify_reference(reference_root, output, config, settings, device, input_files):
    """全testをモデルごとに再推論し、楕円座標と元正解からの成功判定を照合する。"""
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p.resolve()): sha(p) for p in input_files}
    cache = {}
    start = time.monotonic()

    def checked(path):
        path = path.resolve()
        if str(path) not in hashes:
            hashes[str(path)] = sha(path)
        return path

    def external(pattern, **values):
        return checked(reference_root / pattern.format(**values))

    for dataset in config["sources"]:
        manifest = external(config["manifest"], dataset=dataset)
        samples = [s for s in read_json(manifest)["samples"] if s["split"] == "test"]
        cache[dataset] = []
        originals = {}
        for sample in samples:
            image_path = checked(manifest.parent / sample["image"])
            label_path = checked(manifest.parent / sample["label"])
            if hashes[str(image_path)] != sample["image_sha256"] or hashes[str(label_path)] != sample["label_sha256"]:
                raise ValueError(f"固定入力のhash不一致: {sample['sample_id']}")
            key = (image_path, label_path)
            if key not in originals:
                image = read_image(image_path)
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                label = read_json(label_path)
                truth = scale_ellipse(label_ellipse(label), 256 / label["image_width"])
                originals[key] = image, truth
            image, truth = originals[key]
            effect = sample["augmentation"]
            degraded = apply_effect(image, effect["name"], effect["severity"], effect["seed"])
            gray = prepare_image(degraded)
            shared = np.repeat(gray[:, :, None], 3, axis=2)
            cache[dataset].append((sample, gray, truth, hashlib.sha256(shared.tobytes()).hexdigest()))
        print(f"INPUTS {dataset}: {len(cache[dataset])}", flush=True)

    summaries = []
    failures = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for source in config["sources"]:
            for seed in config["seeds"]:
                checkpoint = external(config["checkpoints"][source], seed=seed)
                completion = checked(checkpoint.with_name("complete.json"))
                if read_json(completion)["checkpoint_sha256"] != hashes[str(checkpoint)]:
                    raise ValueError(f"重みのhash不一致: {checkpoint}")
                detector = BoundaryPredictor(checkpoint, device, settings)
                for dataset in config["sources"]:
                    result_path = external(config["results"], source=source, seed=seed, dataset=dataset)
                    reference = read_json(result_path)
                    if reference["checkpoint_sha256"] != detector.checkpoint_sha256:
                        raise ValueError(f"参照結果と重みが不一致: {result_path}")
                    # 条件ごとの参照先から、採用仕様の推定値だけを読み込む。
                    expected = {row["sample_id"]: row for row in reference["rows"]
                                if row["degradation"] != "sensor_whiteout"}
                    for severity in (0.25, 0.5, 0.75, 1.0):
                        path = external(config["whiteout_results"][str(seed)], dataset=dataset, severity=severity)
                        for row in read_json(path)["rows"]:
                            if row["method"] == source:
                                expected[row["sample_id"]] = row
                    actual_ids = {sample["sample_id"] for sample, *_ in cache[dataset]}
                    if set(expected) != actual_ids:
                        raise ValueError("参照結果と固定testのsample IDが一致しません。")
                    counts = defaultdict(lambda: {"count": 0, "successes": 0, "reference_successes": 0,
                                                   "ellipse_mismatches": 0, "iou_mismatches": 0,
                                                   "maximum_ellipse_error": 0.0})
                    for index, (sample, gray, truth, image_hash) in enumerate(cache[dataset], start=1):
                        ref = expected[sample["sample_id"]]
                        if "image_sha256" in ref and ref["image_sha256"] != image_hash:
                            raise ValueError(f"参照結果と入力画素が不一致: {sample['sample_id']}")
                        if not np.allclose(np.r_[truth[0], truth[1], truth[2]],
                                           np.r_[ref["truth_ellipse"][0], ref["truth_ellipse"][1], ref["truth_ellipse"][2]],
                                           rtol=0, atol=1e-10):
                            raise ValueError("元ラベルから求めた正解楕円が参照結果と不一致です。")
                        result = detector.predict(gray)
                        reference_ellipse = ellipse_tuple(ref["ellipse"])
                        same = result.ellipse_256 == reference_ellipse
                        error = 0.0
                        if result.ellipse_256 is not None and reference_ellipse is not None:
                            a, b = result.ellipse_256, reference_ellipse
                            error = float(np.max(np.abs(np.r_[a[0], a[1], a[2]] - np.r_[b[0], b[1], b[2]])))
                        iou = ellipse_iou(result.ellipse_256, truth)
                        success = iou >= config["success_iou"]
                        metric_same = abs(iou - ref["ellipse_iou"]) <= 1e-10 and success == ref["success"]
                        conditions = sample["conditions"]
                        for key in (("all", None), (conditions["degradation"], conditions["severity"])):
                            item = counts[key]
                            item["count"] += 1
                            item["successes"] += int(success)
                            item["reference_successes"] += int(ref["success"])
                            item["ellipse_mismatches"] += int(not same)
                            item["iou_mismatches"] += int(not metric_same)
                            item["maximum_ellipse_error"] = max(item["maximum_ellipse_error"], error)
                        row = {"source": source, "seed": seed, "dataset": dataset,
                               "degradation": conditions["degradation"], "severity": conditions["severity"],
                               "sample_id": sample["sample_id"], "ellipse": result.ellipse_256,
                               "iou": iou, "success": success, "status": result.status,
                               "ellipse_exact_match": same, "metric_match": metric_same,
                               "image_sha256": image_hash}
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if not same or not metric_same:
                            failures.append({**row, "reference_ellipse": reference_ellipse})
                        if index % 250 == 0:
                            print(f"VERIFY {source}/{seed}/{dataset}: {index}/{len(cache[dataset])}", flush=True)
                    for (effect, severity), values in counts.items():
                        summaries.append({"source": source, "seed": seed, "dataset": dataset,
                                          "degradation": effect, "severity": severity, **values})
                    print(f"DONE {source}/{seed}/{dataset}; mismatches={len(failures)}", flush=True)
                del detector
    report = {"schema": 1, "scope": "指定CNN・採用13条件の全test再推論",
              "effective_config": config,
              "ellipse_settings": asdict(settings), "torch": str(torch.__version__), "opencv": cv2.__version__,
              "numpy": np.__version__, "device": device, "seconds": time.monotonic() - start,
              "summaries": summaries, "failures": failures, "files_sha256": hashes,
              "predictions_sha256": sha(output / "predictions.jsonl")}
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)}件で参照結果との不一致があります。verification.jsonを確認してください。")
    print(f"VERIFIED {sum(x['count'] for x in summaries if x['degradation'] == 'all')} predictions", flush=True)
