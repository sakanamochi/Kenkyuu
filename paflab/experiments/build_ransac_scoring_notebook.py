from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/experiments/paf_second_stage_v1/ransac_scoring_ablation"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def build_notebook():
    cells = [
        markdown(
            """
# RANSAC周長補正アブレーション

## tl;dr

- CNN確率点群では周長補正を外すと、validation成功率が **69.4%から75.8%**、testが **51.4%から55.5%**、OODが **74.0%から79.8%**、撮像診断が **57.2%から61.3%**へ改善した。
- 問題のEarth例はIoU **0.035から0.854**へ改善し、極小楕円の選択が解消された。
- Cannyベースラインでは補正なしが一貫して優れるわけではない。clean testでは内周prior込みで **87.5%から84.8%へ低下**し、OODでは **54.2%から55.8%へ改善**した。
- したがってCNNと古典法は同じRANSACスコアを共有せず、CNNは総支持量、古典法は現行周長補正と内周ペアpriorを基本線とする。
"""
        ),
        markdown(
            """
## Context & Methods

### Key Assumptions

- 楕円成功は既存研究環境と同じ `IoU >= 0.80`。
- CNN比較では同じ画像についてCNN推論を一度だけ行い、同じ閾値点群・同じRANSAC乱数列へ異なるスコアを適用する。
- 周長補正指数はvalidationで `1.0 / 0.5 / 0.0` を比較し、CNNは最良だった `0.0` をtestへ固定した。
- 古典法はclean validationで候補生成スコアと最終選択規則を比較し、test/OODは現行 `1.0` と補正なし `0.0` を確認した。

スコアは `weighted support / perimeter ** power * angular coverage`。`power=1.0` が現行、`0.0` が周長補正なし。
"""
        ),
        code(
            """
from pathlib import Path
from collections import defaultdict
from math import comb
import csv
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse
from paflab.image_io import imread

ROOT = Path.cwd()
OUTPUT = ROOT / "output/experiments/paf_second_stage_v1/ransac_scoring_ablation"
assert (ROOT / "paflab").exists(), "Kenkyuuルートから実行してください"

font_path = Path(r"C:\Windows\Fonts\meiryo.ttc")
if font_path.exists():
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

def read_rows(name):
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))

def as_bool(value):
    return str(value).lower() == "true"

def exact_paired_pvalue(wins, regressions):
    discordant = wins + regressions
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(wins, regressions) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)
"""
        ),
        markdown("## Data"),
        code(
            """
cnn_scopes = ["validation", "test", "ood", "diagnostic"]
classic_scopes = ["validation_clean", "test_clean", "ood"]
cnn_rows = {scope: read_rows(f"cnn_{scope}.csv") for scope in cnn_scopes}
classic_rows = {scope: read_rows(f"classic_{scope}.csv") for scope in classic_scopes}

input_counts = {
    "cnn": {scope: len({row["sample_id"] for row in rows}) for scope, rows in cnn_rows.items()},
    "classic": {scope: len({row["sample_id"] for row in rows}) for scope, rows in classic_rows.items()},
}
print(json.dumps(input_counts, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """
def paired_summary(rows):
    paired = defaultdict(dict)
    for row in rows:
        paired[row["sample_id"]][row["score_mode"]] = row
    pairs = [value for value in paired.values() if "density" in value and "support" in value]
    stable_success = sum(as_bool(x["density"]["success"]) and as_bool(x["support"]["success"]) for x in pairs)
    wins = sum(not as_bool(x["density"]["success"]) and as_bool(x["support"]["success"]) for x in pairs)
    regressions = sum(as_bool(x["density"]["success"]) and not as_bool(x["support"]["success"]) for x in pairs)
    stable_failure = len(pairs) - stable_success - wins - regressions
    return {
        "sample_count": len(pairs),
        "stable_success": stable_success,
        "improved": wins,
        "regressed": regressions,
        "stable_failure": stable_failure,
        "paired_exact_p": exact_paired_pvalue(wins, regressions),
    }

paired = {scope: paired_summary(rows) for scope, rows in cnn_rows.items()}
print(json.dumps(paired, ensure_ascii=False, indent=2))
"""
        ),
        markdown("## Results"),
        code(
            """
def success_rate(rows, mode, rule=None):
    selected = [
        row for row in rows
        if row["score_mode"] == mode and (rule is None or row["selection_rule"] == rule)
    ]
    return sum(as_bool(row["success"]) for row in selected) / len(selected)

cnn_rates = {
    scope: {mode: success_rate(rows, mode) for mode in ("density", "support")}
    for scope, rows in cnn_rows.items()
}
classic_rates = {
    scope: {
        f"{mode}_{rule}": success_rate(rows, mode, rule)
        for mode in ("density", "support")
        for rule in ("quality", "inner_pair")
    }
    for scope, rows in classic_rows.items()
}
print("CNN", json.dumps(cnn_rates, ensure_ascii=False, indent=2))
print("Classic", json.dumps(classic_rates, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """
def grouped_rates(rows, group_key):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["score_mode"] in ("density", "support"):
            grouped[row[group_key]][row["score_mode"]].append(as_bool(row["success"]))
    return {
        group: {mode: sum(values) / len(values) for mode, values in modes.items()}
        for group, modes in grouped.items()
    }

breakdowns = {
    "test_by_degradation": grouped_rates(cnn_rows["test"], "degradation"),
    "ood_by_background": grouped_rates(cnn_rows["ood"], "background"),
    "diagnostic_by_effect": grouped_rates(cnn_rows["diagnostic"], "degradation"),
}
print(json.dumps(breakdowns, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """
earth_id = "camera_t067_a045_d039.0_o00__light_t020_a060_e01.5__bg_earth"
earth_rows = {
    row["score_mode"]: row
    for row in cnn_rows["ood"]
    if row["sample_id"] == earth_id
}
ood_manifest = json.loads(
    (ROOT / "output/datasets/research_ood_base_v1/manifest.json").read_text(encoding="utf-8")
)
earth_sample = next(sample for sample in ood_manifest["samples"] if sample["sample_id"] == earth_id)
earth_image = imread(
    ROOT / "output/datasets/research_ood_base_v1" / earth_sample["image"],
    cv2.IMREAD_COLOR,
)
earth_image = cv2.cvtColor(cv2.resize(earth_image, (256, 256)), cv2.COLOR_BGR2RGB)
earth_label = json.loads(
    (ROOT / "output/datasets/research_ood_base_v1" / earth_sample["label"]).read_text(encoding="utf-8")
)
gt_points = np.asarray(earth_label["image_points"], dtype=np.float32) * (256 / earth_label["image_width"])
gt = cv2.fitEllipse(gt_points.reshape(-1, 1, 2))

def row_ellipse(row):
    return (
        (float(row["predicted_center_x"]), float(row["predicted_center_y"])),
        (float(row["predicted_axis_1"]), float(row["predicted_axis_2"])),
        float(row["predicted_angle_deg"]),
    )
"""
        ),
        code(
            """
blue = "#2563eb"
orange = "#d97706"
gray = "#64748b"
dark = "#172033"

fig = plt.figure(figsize=(16, 11), facecolor="white")
grid = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

# A: CNN success rates
ax = fig.add_subplot(grid[0, 0])
labels = ["Validation", "Test", "OOD", "撮像診断"]
scopes = ["validation", "test", "ood", "diagnostic"]
x = np.arange(len(scopes)); width = 0.34
current = [cnn_rates[s]["density"] * 100 for s in scopes]
new = [cnn_rates[s]["support"] * 100 for s in scopes]
bars1 = ax.bar(x - width/2, current, width, color="white", edgecolor=blue, hatch="//", label="現行: 周長補正1.0")
bars2 = ax.bar(x + width/2, new, width, color=orange, edgecolor=orange, label="CNN案: 補正なし")
ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
ax.set_xticks(x, labels); ax.set_ylim(0, 100); ax.set_ylabel("成功率 (%)")
ax.set_title("A. CNN + RANSAC 成功率", loc="left", color=dark)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.2)

# B: paired transitions
ax = fig.add_subplot(grid[0, 1])
wins = [paired[s]["improved"] for s in scopes]
losses = [-paired[s]["regressed"] for s in scopes]
ax.barh(labels, wins, color=blue, label="失敗→成功")
ax.barh(labels, losses, color="white", edgecolor=orange, hatch="//", label="成功→失敗")
for i, (win, loss) in enumerate(zip(wins, losses)):
    ax.text(win + 2, i, str(win), va="center", fontsize=9)
    ax.text(loss - 2, i, str(-loss), va="center", ha="right", fontsize=9)
ax.axvline(0, color=dark, linewidth=1)
ax.set_xlabel("pairedサンプル数（右が改善）")
ax.set_title("B. CNN方式変更による勝敗", loc="left", color=dark)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="x", alpha=0.2)

# C: classic baseline
ax = fig.add_subplot(grid[1, 0])
classic_labels = ["Val clean", "Test clean", "OOD"]
classic_scope_order = ["validation_clean", "test_clean", "ood"]
series = [
    ("density_quality", "現行・品質順位", blue, "//", "white"),
    ("density_inner_pair", "現行・内周prior", blue, None, blue),
    ("support_quality", "補正なし・品質順位", orange, "//", "white"),
    ("support_inner_pair", "補正なし・内周prior", orange, None, orange),
]
x = np.arange(3); width = 0.19
for index, (key, label, color, hatch, face) in enumerate(series):
    values = [classic_rates[s][key] * 100 for s in classic_scope_order]
    bars = ax.bar(x + (index - 1.5) * width, values, width, label=label, color=face, edgecolor=color, hatch=hatch)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, rotation=90)
ax.set_xticks(x, classic_labels); ax.set_ylim(0, 100); ax.set_ylabel("成功率 (%)")
ax.set_title("C. Cannyベースライン 成功率", loc="left", color=dark)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.2)

# D: Earth example
ax = fig.add_subplot(grid[1, 1]); ax.axis("off")
ax.set_title("D. Earth背景の代表例", loc="left", color=dark, pad=8)
for index, mode in enumerate(("density", "support")):
    inset = ax.inset_axes([index * 0.51, 0.08, 0.48, 0.84])
    inset.imshow(earth_image)
    pred = row_ellipse(earth_rows[mode])
    inset.add_patch(Ellipse(gt[0], gt[1][0], gt[1][1], angle=gt[2], fill=False, edgecolor="#00a6d6", linestyle="--", linewidth=2))
    inset.add_patch(Ellipse(pred[0], pred[1][0], pred[1][1], angle=pred[2], fill=False, edgecolor=blue if mode == "support" else orange, linewidth=2.5))
    inset.set_title(("現行" if mode == "density" else "補正なし") + f"  IoU {float(earth_rows[mode]['ellipse_iou']):.3f}", fontsize=10)
    inset.set_xticks([]); inset.set_yticks([])

fig.suptitle("RANSAC周長補正アブレーション", fontsize=20, color=dark, y=0.98)
fig.text(0.5, 0.015, "楕円成功: IoU ≥ 0.80。CNN比較は同一確率マップ・同一RANSAC乱数列。水色破線は正解楕円。", ha="center", fontsize=10, color=gray)
figure_path = OUTPUT / "ransac-scoring-ablation-summary.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()
print(figure_path)
"""
        ),
        code(
            """
summary = {
    "question": "CNNとCannyベースラインでRANSACの周長補正を分けるべきか",
    "cnn_rates": cnn_rates,
    "cnn_paired": paired,
    "cnn_breakdowns": breakdowns,
    "classic_rates": classic_rates,
    "earth_example": {
        mode: {
            "ellipse_iou": float(row["ellipse_iou"]),
            "axis_1": float(row["predicted_axis_1"]),
            "axis_2": float(row["predicted_axis_2"]),
        }
        for mode, row in earth_rows.items()
    },
    "recommendation": {
        "cnn": "perimeter_power=0.0",
        "classic": "現行perimeter_power=1.0を維持し、内周ペアpriorを使用。OOD向け変更は別validationで再選定する。",
    },
    "caveat": "現行CNN test再計算は保存済み結果と境界上の1枚だけ成否不一致。方式間比較は同一再計算内のpaired値を使用。",
}
(OUTPUT / "ransac_scoring_ablation_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary["recommendation"], ensure_ascii=False, indent=2))
"""
        ),
        markdown(
            """
## Takeaways

1. CNN確率点群では、周長で割ることが「同じ意味クラスの中から極小仮説を選ぶ」副作用を持つ。総支持量へ切り替える根拠はvalidation・test・OOD・撮像診断で一貫している。
2. 改善は特にEarth背景、黒つぶれ、遮蔽で大きい。一方、clean・whiteoutでは少数の回帰があるため、採用時も条件別結果を併記する。
3. Cannyでは小さい楕円を無条件に優遇するのではなく、同心二重輪郭のペアから小さい側を選ぶpriorが主要因。周長補正を外す判断はデータ領域によって反転するため、現時点では変更しない。
4. 次の改善候補は、CNN用と古典法用のRANSAC設定を設定ファイル上でも分離し、CNN側に候補信頼度・棄却規則を追加すること。
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (paflab)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
    )
    return notebook


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    notebook_path = OUTPUT / "ransac_scoring_ablation.ipynb"
    notebook = build_notebook()
    nbformat.write(notebook, notebook_path)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    status = {
        "notebook": notebook_path.relative_to(ROOT).as_posix(),
        "executed_cell_count": sum(
            cell.cell_type == "code" and cell.get("execution_count") is not None
            for cell in notebook.cells
        ),
        "code_cell_count": sum(cell.cell_type == "code" for cell in notebook.cells),
        "status": "executed_successfully",
    }
    (OUTPUT / "notebook_execution.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
