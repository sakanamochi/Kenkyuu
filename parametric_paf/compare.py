"""既存の3検出器を再利用し、学習せず形状別に比較する。"""

import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, read_image, write_image, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import load_model
from paf_ring_detection.methods.cnn_ransac import detect_from_probability
from paf_ring_detection.methods.canny_contour_ransac import detect_canny_contour_ransac
from paf_ring_detection.methods.zhang2019 import detect_zhang_arc_candidates
from paf_ring_detection.methods.zhang2019_paf import score_zhang_candidates_for_paf, select_zhang_inner_boundary

LABELS = {"cnn": "CNN + weighted RANSAC", "zhang": "Zhang 2019型（再現実装）",
          "canny": "Canny + 輪郭別RANSAC"}


def summarize(rows, manifest):
    """成功数と基準モデルとの差を、同一撮影条件で集計する。"""
    summaries = []
    model_names = list(dict.fromkeys(row["model"] for row in rows))
    for name in model_names:
        for method in LABELS:
            subset = [r for r in rows if r["model"] == name and r["method"] == method]
            base = {r["condition_id"]: r for r in rows if r["model"] == "baseline" and r["method"] == method}
            summaries.append({"model": name, "method": method, "n": len(subset),
                              "successes": sum(r["success"] for r in subset),
                              "detected": sum(r["detected"] for r in subset),
                              "success_rate": sum(r["success"] for r in subset)/len(subset),
                              "wins_vs_baseline": sum(r["success"] and not base[r["condition_id"]]["success"] for r in subset),
                              "losses_vs_baseline": sum(not r["success"] and base[r["condition_id"]]["success"] for r in subset)})
    lines = ["# パラメトリックPAFの初期比較", "", "各モデル48条件（視点4傾斜×2方位×照明3傾斜×2方位）。",
             "黒背景、Sun光源。追加劣化・再学習・モデル別の閾値調整なし。",
             "3方式すべて同じ256×256画像を入力し、塗りつぶした楕円領域のIoU ≥ 0.80を成功とする。",
             "標準評価のZhang型は元解像度で処理するため、入力解像度が異なる成功率とは直接比較しない。", "",
             "|モデル|CNN|Zhang 2019型（再現実装）|Canny|", "|---|---:|---:|---:|"]
    for name in model_names:
        values = [next(s for s in summaries if s["model"] == name and s["method"] == m) for m in LABELS]
        lines.append("|" + name + "|" + "|".join(f"{s['successes']}/{s['n']} ({100*s['success_rate']:.1f}%)" for s in values) + "|")
    lines += ["", "reference_1194Mは元CADの参照結果。簡易モデルとは材質・細部も異なる。",
              "darkは色の変更であり形状変化とは分けて読む。wide_topは上端外径1.4 m。",
              "baselineとの差が各変更の比較。学習に未使用の形状への初期診断であり、未知PAF全般への性能保証ではない。",
              "テーパの自己遮蔽があっても正解は完全な上端内周。IoU基準だけでは近接した別輪郭との取り違えを完全には区別しない。",
              "", "## 視点傾斜別の成功数（各12条件）", "", "|モデル|方式|10°|30°|50°|70°|", "|---|---|---:|---:|---:|---:|"]
    for name in model_names:
        for method in LABELS:
            counts = [sum(r["success"] for r in rows if r["model"] == name and r["method"] == method and r["camera_tilt"] == tilt)
                      for tilt in manifest["settings"]["camera_tilts"]]
            lines.append(f"|{name}|{LABELS[method]}|" + "|".join(map(str, counts)) + "|")
    lines += ["", "examples/はカメラ方位0°・照明傾斜60°／方位0°を固定した検出例。成否による選別なし。",
              "各列はCNN、Zhang型、Canny。緑線は正解、赤線は推定、白文字はIoU。",
              "同条件の乱数seedを全形状で共有。results.csvが全画像の記録、summary.jsonが対応ありの改善・悪化数。"]
    return summaries, "\n".join(lines) + "\n"


def main():
    torch.set_num_threads(4)
    config = read_json(ROOT / "config/experiment.json")
    folder = HERE / "output/comparison"
    manifest = read_json(folder / "manifest.json")
    checkpoint = ROOT / config["paths"]["checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    size = config["input_size"]
    zs = config["zhang2019"]
    rows = []
    start = time.perf_counter()
    for index, sample in enumerate(manifest["samples"]):
        source = read_image(folder / sample["image"])
        image = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
        truth = scaled_ellipse(label_ellipse(sample), size/source.shape[1])
        seed = (config["seed"] + zlib.crc32(sample["condition_id"].encode())) % (2**32)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(rgb).float().unsqueeze(0).to(device)/127.5-1
        with torch.inference_mode():
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        cnn = detect_from_probability(probability, config["ransac"], seed)
        candidates, stages = detect_zhang_arc_candidates(image, zs["preprocess"], zs["detector"])
        candidates = score_zhang_candidates_for_paf(candidates, stages, zs["paf_postprocess"]["candidate_validation"])
        zhang = select_zhang_inner_boundary(candidates, zs["paf_postprocess"]["selector"])
        canny, _ = detect_canny_contour_ransac(image, config["canny_contour_ransac"], config["ransac"], seed)
        panels = []
        for method, result in zip(LABELS, (cnn, zhang, canny)):
            ellipse = result["ellipse"] if result else None
            metrics = evaluate_ellipses(ellipse, truth, image.shape) if ellipse is not None else {}
            row = {k: sample[k] for k in ("model", "condition_id", "camera_tilt", "camera_azimuth", "light_tilt", "light_azimuth")}
            row.update(method=method, detected=ellipse is not None,
                       success=metrics.get("ellipse_iou", 0) >= config["success_iou"],
                       ellipse_iou=metrics.get("ellipse_iou"), center_error_px=metrics.get("center_error_px"),
                       estimated_ellipse=json.dumps(ellipse) if ellipse is not None else "")
            rows.append(row)
            panel = image.copy()
            cv2.ellipse(panel, truth, (0, 255, 0), 1, cv2.LINE_AA)
            if ellipse is not None:
                cv2.ellipse(panel, ellipse, (0, 0, 255), 1, cv2.LINE_AA)
            caption = f"{method}: {metrics.get('ellipse_iou', 0):.2f}" if ellipse is not None else f"{method}: none"
            cv2.putText(panel, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        if sample["camera_azimuth"] == 0 and sample["light_tilt"] == 60 and sample["light_azimuth"] == 0:
            write_image(folder / "examples" / f"{sample['model']}_{sample['condition_id']}.png", np.hstack(panels))
        if (index+1) % 12 == 0:
            print(f"{index+1}/{len(manifest['samples'])}: {sample['model']} ({time.perf_counter()-start:.1f}s)", flush=True)
    with (folder / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries, text = summarize(rows, manifest)
    (folder / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (folder / "comparison.md").write_text(text, encoding="utf-8")
    provenance = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "experiment_config": config, "input_size_all_methods": size, "device": str(device),
                  "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()}
    (folder / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
