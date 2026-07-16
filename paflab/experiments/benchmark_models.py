from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from paflab.model import TinyUNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model, tensor, device: torch.device, warmup: int, repeats: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        synchronize(device)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(tensor)
            synchronize(device)
            durations.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.mean(durations),
        "p95_ms": sorted(durations)[int(0.95 * (len(durations) - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny U-Net単画像推論速度を測る")
    parser.add_argument(
        "--root", default="output/experiments/paf_second_stage_v1/model_ablation"
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    root = PROJECT_ROOT / args.root
    rows = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for width in (2, 4, 8, 16):
        run = root / f"c{width:02d}_s{args.seed}"
        checkpoint_path = run / "cnn_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for device in devices:
            model = TinyUNet(width)
            model.load_state_dict(checkpoint["model_state"])
            model.to(device).eval()
            tensor = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
            metrics = measure(model, tensor, device, warmup=10, repeats=args.repeats)
            rows.append(
                {
                    "base_channels": width,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "checkpoint_mib": checkpoint_path.stat().st_size / (1024**2),
                    "device": str(device),
                    "input_shape": "1x3x256x256",
                    "repeats": args.repeats,
                    **metrics,
                }
            )
    with (root / "latency.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "latency.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
