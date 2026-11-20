"""学習入力のhash・分割・教師を確認する。重みの更新は行わない。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.files import read_json, sha
from training.data import dataset_from_config
from training.objective import boundary_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training_data.json")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    files = {str(config_path): sha(config_path), str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    for path in sorted((ROOT / "src").rglob("*.py")):
        files[str(path)] = sha(path)
    summaries = []
    for source, settings in config["sources"].items():
        base = (config_path.parent / settings["base"]).resolve()
        samples = read_json(base / "manifest.json")["samples"]
        groups = {split: {s["group_id"] for s in samples if s["split"] == split}
                  for split in ("train", "validation", "test")}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if groups[a] & groups[b]:
                raise ValueError(f"分割群の重複: {source} / {a} / {b}")
        for split in ("train", "validation"):
            data = dataset_from_config(config_path, source, split, args.seed, cache=False)
            # 選ばれなかったビューも含め、使用可能な全画像・ラベルを検証する。
            for view in {v.root: v for v in data.views}.values():
                manifest = view.root / "manifest.json"
                files[str(manifest)] = sha(manifest)
                for sample in view.samples:
                    for field in ("image", "label"):
                        path = (view.root / sample[field]).resolve()
                        if str(path) not in files:
                            files[str(path)] = sha(path)
                        if files[str(path)] != sample[field + "_sha256"]:
                            raise ValueError(f"入力hashの不一致: {path}")
            digest = hashlib.sha256()
            hidden = 0
            for index in range(len(data)):
                image, masks, _ = data[index]
                target = boundary_target(masks[None])
                if not torch.isfinite(image).all() or not torch.isfinite(target).all():
                    raise ValueError(f"非有限入力または教師: {source}/{split}/{index}")
                if image.shape != (3, 256, 256) or masks.shape != (2, 256, 256):
                    raise ValueError("入力または教師の形状が不正です。")
                for tensor in (image, masks, target):
                    digest.update(tensor.numpy().tobytes())
                hidden += int(target.sum() == 0)
                if (index + 1) % 500 == 0:
                    print(f"CHECK {source}/{split}: {index + 1}/{len(data)}", flush=True)
            summaries.append({"source": source, "split": split, "samples": len(data),
                              "empty_boundary_targets": hidden, "tensor_sequence_sha256": digest.hexdigest(),
                              "views": [str(v.root) for v in data.views], "split_groups_disjoint": True})
            print(f"DONE {source}/{split}: {len(data)}", flush=True)
    # 実行中に入力・実装が変わっていないことも確認する。
    if any(sha(Path(path)) != digest for path, digest in files.items()):
        raise ValueError("確認中に入力または実装が変更されました。")
    report = {"seed": args.seed, "config": config, "summaries": summaries, "files_sha256": files,
              "scope": "入力・分割・教師の確認。再学習や精度の検証ではない。"}
    (args.output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
