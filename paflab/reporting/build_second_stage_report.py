from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/experiments/paf_second_stage_v1"
REPORT = OUTPUT / "report"
METHOD_LABELS = {
    "zhang2019_arc_reproduction": "Zhang 2019型（再現実装）",
    "cnn_ransac_support": "CNN + weighted RANSAC",
    "contour_fit": "Contour fit",
    "canny_ransac_inner_pair": "Canny + 輪郭別RANSAC",
    "cnn_ransac": "CNN + weighted RANSAC",
}
PRIMARY_METHODS = {"zhang2019_arc_reproduction", "cnn_ransac_support"}
EFFECT_LABELS = {
    "clean": "clean",
    "black_rectangle": "黒矩形遮蔽",
    "sensor_whiteout": "センサ白飛びproxy",
    "sensor_black_crush": "センサ黒つぶれproxy",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sql_project(datasets: dict[str, list[dict]], sql_path: Path) -> dict[str, list[dict]]:
    """レポート表示データを実行済みSQL viewから取得する。"""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    for name, rows in datasets.items():
        columns = list(rows[0])
        types = []
        for column in columns:
            value = next((row[column] for row in rows if row.get(column) is not None), None)
            types.append("REAL" if isinstance(value, (int, float)) else "TEXT")
        definition = ", ".join(
            f'"{column}" {kind}' for column, kind in zip(columns, types)
        )
        connection.execute(f'CREATE TABLE "source_{name}" ({definition})')
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f'INSERT INTO "source_{name}" VALUES ({placeholders})',
            [[row.get(column) for column in columns] for row in rows],
        )
    connection.executescript(sql_path.read_text(encoding="utf-8"))
    projected = {
        name: [
            dict(row)
            for row in connection.execute(f'SELECT * FROM "report_{name}"')
        ]
        for name in datasets
    }
    connection.close()
    return projected


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    artifact = load_json(
        ROOT / "output/experiments/paf_robustness_v3/report/artifact.json"
    )
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    title = "PAF内周リング検出：CG耐性・OOD・モデル最小化検証"
    manifest["title"] = title
    manifest["description"] = (
        "v3耐性比較を保持し、未知姿勢・未知照明・Earth/Moon背景、モデル幅、"
        "RANSAC内周選択、カメラ効果proxyを追加検証した技術レポート。"
    )
    manifest["generatedAt"] = generated
    snapshot["generatedAt"] = generated
    manifest["blocks"][0]["body"] = f"# {title}\n\n技術レビュー用・2026-07-16"

    summary = load_json(OUTPUT / "second_stage_summary.json")
    validation = load_json(OUTPUT / "validation_report.json")
    if validation["status"] != "ready_to_share":
        raise RuntimeError("第2段階の結果監査が未通過です")
    model_rows = read_csv(OUTPUT / "model_ablation_aggregate.csv")
    ood_background = read_csv(OUTPUT / "ood_by_background.csv")
    ood_tilt = read_csv(OUTPUT / "ood_by_camera_tilt.csv")
    diagnostic = read_csv(OUTPUT / "diagnostic_curves.csv")
    latency = read_csv(OUTPUT / "model_ablation/latency.csv")

    for row in model_rows:
        for key, value in list(row.items()):
            if key in ("base_channels", "parameter_count", "seed_count"):
                row[key] = int(float(value))
            elif value != "":
                row[key] = float(value)
    performance_long = []
    for row in model_rows:
        for evaluation, label in (
            ("validation_success_rate", "validation（clean+劣化）"),
            ("ood_success_rate", "OOD clean"),
        ):
            performance_long.append(
                {
                    "base_channels": row["base_channels"],
                    "parameter_count": row["parameter_count"],
                    "evaluation": label,
                    "mean_success_rate": row[f"mean_{evaluation}"],
                    "minimum_seed_rate": row[f"min_{evaluation}"],
                    "maximum_seed_rate": row[f"max_{evaluation}"],
                    "seed_count": row["seed_count"],
                    "mean_best_epoch": row["mean_best_epoch"],
                    "mean_loss_gap_at_best": row["mean_loss_gap_at_best"],
                    "mean_post_best_worsening": row["mean_post_best_worsening"],
                }
            )

    def typed(rows: list[dict], numeric: set[str]) -> list[dict]:
        output = []
        for original in rows:
            row = dict(original)
            row["method_label"] = METHOD_LABELS.get(row.get("method"), row.get("method"))
            if "degradation" in row:
                row["effect_label"] = EFFECT_LABELS.get(
                    row["degradation"], row["degradation"]
                )
            for key in numeric:
                if row.get(key, "") != "":
                    row[key] = float(row[key])
            output.append(row)
        return output

    ood_background = typed(
        ood_background,
        {"sample_count", "camera_cluster_count", "success_count", "success_rate", "cluster_ci95_low", "cluster_ci95_high"},
    )
    ood_tilt = typed(
        ood_tilt,
        {"camera_tilt_deg", "sample_count", "camera_cluster_count", "success_count", "success_rate", "cluster_ci95_low", "cluster_ci95_high"},
    )
    diagnostic = typed(
        diagnostic,
        {"severity", "sample_count", "camera_cluster_count", "success_count", "success_rate", "cluster_ci95_low", "cluster_ci95_high"},
    )
    ood_background = [
        row for row in ood_background if row["method"] in PRIMARY_METHODS
    ]
    ood_tilt = [row for row in ood_tilt if row["method"] in PRIMARY_METHODS]
    diagnostic = [row for row in diagnostic if row["method"] in PRIMARY_METHODS]
    latency = typed(
        latency,
        {"base_channels", "parameter_count", "checkpoint_mib", "repeats", "median_ms", "mean_ms", "p95_ms"},
    )
    black_rectangle = [
        row for row in diagnostic if row["degradation"] in ("clean", "black_rectangle")
    ]
    cnn_effects = [
        row for row in diagnostic if row["method"] == "cnn_ransac_support"
    ]

    method_rates = {}
    for method, relative in {
        "zhang2019_arc_reproduction": "output/datasets/research_ood_base_v1/results/zhang2019_arc_ood/summary.json",
        "cnn_ransac_support": "output/datasets/research_ood_base_v1/results/cnn_ransac_support_ood/summary.json",
    }.items():
        method_rates[method] = load_json(ROOT / relative)["top1_match_rate"]
    minimum_width = summary["model_ablation"]["decision"][
        "minimum_noninferior_base_channels"
    ]
    minimum_model = next(row for row in model_rows if row["base_channels"] == minimum_width)
    overfit = summary["model_ablation"]["overfitting"]
    headlines = [
        {
            "id": "second_stage",
            "cnn_ood_rate": method_rates["cnn_ransac_support"],
            "best_classic_ood_rate": method_rates["zhang2019_arc_reproduction"],
            "minimum_width": minimum_width,
            "minimum_parameters": minimum_model["parameter_count"],
        }
    ]

    questions = [
        {"priority": 1, "question": "train/testの漏洩はないか", "answer": "camera_id単位で分離。OOD 480枚とdiagnostic 1,456枚は既存CNNへ未使用。", "remaining": "実写・形状個体は未分離"},
        {"priority": 2, "question": "未知角度・未知照明に一般化するか", "answer": "10–50°は100%、67°は67.7%、82°は0%。未知照明8条件は70.0–76.7%。", "remaining": "82°近傍の訓練追加と連続角度評価"},
        {"priority": 3, "question": "CNNは本当に必要か", "answer": f"OODはCNN {method_rates['cnn_ransac_support']:.1%}、Zhang 2019型 {method_rates['zhang2019_arc_reproduction']:.1%}。黒矩形遮蔽曲線も同一条件で比較。", "remaining": "実機計算量・実写で再確認"},
        {"priority": 4, "question": "RANSACは何を選んでいるか", "answer": "従来はcoverage×inlier ratio÷distance。内外の意味はない。CAD priorでclean 76.8→86.6%。", "remaining": "背景・劣化下では改善幅を再評価"},
        {"priority": 5, "question": "RANSAC乱数で結果が変わらないか", "answer": "劣化・背景の対比較は元画像IDでpaired seed化し、112組すべてが17条件で同じseed keyを共有。", "remaining": "複数RANSAC seedによるアルゴリズム分散評価"},
        {"priority": 6, "question": "過学習ではないか", "answer": "3 seed×4幅で再学習し、最良epoch・loss gap・OODを併記。", "remaining": "形状単位OODと実写が最重要"},
        {"priority": 7, "question": "白飛び・黒つぶれは物理的か", "answer": "現段階は線形露光、full-well clip、shot/read noise、black-level、量子化のproxy。", "remaining": "実機応答・HDR・PSF校正が必要"},
        {"priority": 8, "question": "Earth/Moon背景の影響は", "answer": "背景別・camera cluster 95%区間を独立集計。EarthでCNN性能低下。", "remaining": "実写テクスチャと大気・雲・位相の追加"},
        {"priority": 9, "question": "別PAF形状へ一般化するか", "answer": "未検証。現在は1 CADのみ。", "remaining": "リング径、厚み、支持構造を変えたCAD family split"},
        {"priority": 10, "question": "失敗を検知して棄却できるか", "answer": "RANSAC score/inlier/coverageを保存済み。", "remaining": "信頼度校正とrisk-coverage曲線"},
        {"priority": 11, "question": "新規性は何か", "answer": "CNN+RANSAC単体は既報があるため主張しない。PAF固有の意味選択、既知太陽方向条件付け、形状family OODを候補化。", "remaining": "系統的文献調査とアブレーション"},
    ]
    (OUTPUT / "professor_questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_datasets = sql_project(
        {
            "model_performance": performance_long,
            "ood_background": ood_background,
            "ood_tilt": ood_tilt,
            "black_rectangle": black_rectangle,
            "cnn_effects": cnn_effects,
            "model_latency": latency,
            "professor_questions": questions,
        },
        ROOT / "analysis/second_stage_report_source.sql",
    )
    snapshot["datasets"].update(report_datasets)

    new_sources = [
        {"id": "second_stage_report_query", "label": "Second-stage SQLite report query", "path": "analysis/second_stage_report_source.sql", "description": "監査済み第2段階集計をレポート表示粒度へ射影・整列した実行済みSQL。"},
        {"id": "second_stage_summary", "label": "Second-stage reviewed aggregates", "path": "output/experiments/paf_second_stage_v1/second_stage_summary.json", "description": "モデル幅、OOD、diagnosticの再現可能な集計定義。"},
        {"id": "zhang_2019_reference", "label": "Zhang et al. docking-ring detector", "path": "https://doi.org/10.3390/s19235243", "description": "主古典ベースラインの元論文。実装は著者コード移植ではなく再現実装。"},
        {"id": "ood_tilt_results", "label": "OOD pose results", "path": "output/experiments/paf_second_stage_v1/ood_by_camera_tilt.csv", "description": "未知姿勢480枚のcamera cluster集計。"},
        {"id": "ood_background_results", "label": "OOD background results", "path": "output/experiments/paf_second_stage_v1/ood_by_background.csv", "description": "space/Earth/Moon背景各160枚のcamera cluster集計。"},
        {"id": "diagnostic_results", "label": "Camera-effect diagnostic curves", "path": "output/experiments/paf_second_stage_v1/diagnostic_curves.csv", "description": "黒矩形遮蔽と物理motivated camera-effect proxyの強度曲線。"},
        {"id": "model_latency", "label": "Model latency benchmark", "path": "output/experiments/paf_second_stage_v1/model_ablation/latency.csv", "description": "256x256、batch 1、同一PCでのCPU/GPU推論時間。"},
        {"id": "camera_effects_code", "label": "Camera-effect implementation", "path": "paflab/camera_effects.py", "description": "露光、飽和、shot/read noise、black-level、量子化proxyの実装。"},
        {"id": "professor_questions", "label": "Professor-question audit", "path": "output/experiments/paf_second_stage_v1/professor_questions.json", "description": "想定質問、現時点の回答、残る検証を分離した監査表。"},
        {"id": "second_stage_validation", "label": "Second-stage validation report", "path": "output/experiments/paf_second_stage_v1/validation_report.json", "description": "サンプル数、条件バランス、方式間ID、12 run、112×17 paired groupの検算。"},
    ]
    existing_source_ids = {source["id"] for source in manifest["sources"]}
    manifest["sources"].extend(
        source for source in new_sources if source["id"] not in existing_source_ids
    )
    artifact["sources"] = list(manifest["sources"])

    manifest["charts"].extend(
        [
            {"id": "second_model_size", "title": "CNN規模と成功率", "subtitle": "各点3 seed平均。帯の代わりにtooltipでseed最小・最大を示す。", "showDescription": True, "type": "line", "dataset": "model_performance", "sourceId": "second_stage_summary", "encodings": {"x": {"field": "parameter_count", "type": "quantitative", "label": "パラメータ数"}, "y": {"field": "mean_success_rate", "type": "quantitative", "label": "平均成功率", "format": "percent"}, "color": {"field": "evaluation", "type": "nominal", "label": "評価"}, "tooltip": [{"field": "base_channels", "label": "base channels", "format": "number"}, {"field": "minimum_seed_rate", "label": "seed最小", "format": "percent"}, {"field": "maximum_seed_rate", "label": "seed最大", "format": "percent"}, {"field": "seed_count", "label": "seed数", "format": "number"}]}},
            {"id": "second_ood_tilt", "title": "未知カメラ傾斜角と成功率", "subtitle": "10–50°は補間、67°・82°は訓練範囲外寄りの性能境界。", "showDescription": True, "type": "line", "dataset": "ood_tilt", "sourceId": "ood_tilt_results", "encodings": {"x": {"field": "camera_tilt_deg", "type": "quantitative", "label": "カメラ傾斜角（deg）"}, "y": {"field": "success_rate", "type": "quantitative", "label": "成功率", "format": "percent"}, "color": {"field": "method_label", "type": "nominal", "label": "方式"}, "tooltip": [{"field": "sample_count", "label": "画像数", "format": "number"}, {"field": "cluster_ci95_low", "label": "95% CI下限", "format": "percent"}, {"field": "cluster_ci95_high", "label": "95% CI上限", "format": "percent"}]}},
            {"id": "second_ood_background", "title": "背景別OOD成功率", "subtitle": "space、procedural Earth、procedural Moonを各160枚で比較。", "showDescription": True, "type": "bar", "dataset": "ood_background", "sourceId": "ood_background_results", "encodings": {"x": {"field": "background", "type": "nominal", "label": "背景"}, "y": {"field": "success_rate", "type": "quantitative", "label": "成功率", "format": "percent"}, "color": {"field": "method_label", "type": "nominal", "label": "方式"}, "tooltip": [{"field": "sample_count", "label": "画像数", "format": "number"}, {"field": "cluster_ci95_low", "label": "95% CI下限", "format": "percent"}, {"field": "cluster_ci95_high", "label": "95% CI上限", "format": "percent"}]}},
            {"id": "second_black_rectangle", "title": "黒矩形遮蔽の強度と成功率", "subtitle": "物体形状をなぞらない単純な全高黒帯でCNNの構造補完を診断。", "showDescription": True, "type": "line", "dataset": "black_rectangle", "sourceId": "diagnostic_results", "encodings": {"x": {"field": "severity", "type": "quantitative", "label": "遮蔽幅 / 画像幅"}, "y": {"field": "success_rate", "type": "quantitative", "label": "成功率", "format": "percent"}, "color": {"field": "method_label", "type": "nominal", "label": "方式"}, "tooltip": [{"field": "sample_count", "label": "画像数", "format": "number"}, {"field": "cluster_ci95_low", "label": "95% CI下限", "format": "percent"}, {"field": "cluster_ci95_high", "label": "95% CI上限", "format": "percent"}]}},
            {"id": "second_cnn_effects", "title": "CNNのカメラ効果proxy耐性", "subtitle": "白飛び・黒つぶれは実機校正前のphysics-motivated proxy。", "showDescription": True, "type": "line", "dataset": "cnn_effects", "sourceId": "diagnostic_results", "encodings": {"x": {"field": "severity", "type": "quantitative", "label": "効果強度"}, "y": {"field": "success_rate", "type": "quantitative", "label": "成功率", "format": "percent"}, "color": {"field": "effect_label", "type": "nominal", "label": "効果"}, "tooltip": [{"field": "sample_count", "label": "画像数", "format": "number"}, {"field": "cluster_ci95_low", "label": "95% CI下限", "format": "percent"}, {"field": "cluster_ci95_high", "label": "95% CI上限", "format": "percent"}]}},
        ]
    )
    manifest["tables"].extend(
        [
            {"id": "second_model_table", "title": "モデル幅アブレーション", "subtitle": "3 seedの平均・範囲。validationはclean+劣化、OODはclean未知条件。", "showDescription": True, "dataset": "model_performance", "sourceId": "second_stage_summary", "columns": [{"field": "base_channels", "label": "base ch", "format": "number"}, {"field": "parameter_count", "label": "パラメータ", "format": "number"}, {"field": "evaluation", "label": "評価"}, {"field": "mean_success_rate", "label": "平均成功率", "format": "percent"}, {"field": "minimum_seed_rate", "label": "seed最小", "format": "percent"}, {"field": "maximum_seed_rate", "label": "seed最大", "format": "percent"}, {"field": "mean_best_epoch", "label": "平均best epoch", "format": "number"}, {"field": "mean_loss_gap_at_best", "label": "平均val-train loss gap", "format": "number"}, {"field": "mean_post_best_worsening", "label": "best後の平均悪化", "format": "number"}], "defaultSort": {"field": "parameter_count", "direction": "asc"}},
            {"id": "second_latency", "title": "Tiny U-Net単画像推論時間", "subtitle": "batch 1、256×256、同一PC、30反復。RANSACと画像I/Oは含まない。", "showDescription": True, "dataset": "model_latency", "sourceId": "model_latency", "columns": [{"field": "base_channels", "label": "base ch", "format": "number"}, {"field": "parameter_count", "label": "パラメータ", "format": "number"}, {"field": "checkpoint_mib", "label": "checkpoint MiB", "format": "number"}, {"field": "device", "label": "device"}, {"field": "median_ms", "label": "median ms", "format": "number"}, {"field": "p95_ms", "label": "p95 ms", "format": "number"}], "defaultSort": {"field": "parameter_count", "direction": "asc"}},
            {"id": "second_questions", "title": "指導教員から想定される質問", "subtitle": "現在の回答と、回答を確証へ変える次実験を分離。", "showDescription": True, "dataset": "professor_questions", "sourceId": "professor_questions", "columns": [{"field": "priority", "label": "優先", "format": "number"}, {"field": "question", "label": "質問"}, {"field": "answer", "label": "現在の回答"}, {"field": "remaining", "label": "残る検証"}], "defaultSort": {"field": "priority", "direction": "asc"}},
        ]
    )
    second_stage_component_ids = {
        "second_model_size",
        "second_ood_tilt",
        "second_ood_background",
        "second_black_rectangle",
        "second_cnn_effects",
        "second_model_table",
        "second_latency",
        "second_questions",
    }
    for component in [*manifest["charts"], *manifest["tables"]]:
        if component["id"] in second_stage_component_ids:
            component["sourceId"] = "second_stage_report_query"

    second_blocks = [
        {"id": "second_conclusion", "type": "markdown", "sourceId": "second_stage_summary", "body": f"## 第2段階の結論\n\nOOD 480枚ではCNNが{method_rates['cnn_ransac_support']:.1%}、Zhang 2019型（再現実装）が{method_rates['zhang2019_arc_reproduction']:.1%}。固定した3ポイント基準での最小採用幅は{minimum_width}。Zhang型はCanny後の分断弧を統合する宇宙機ドッキングリング向け古典法であり、単純な全エッジRANSACより妥当な主比較である。既存CNNは未知照明には安定したが、姿勢外挿と背景で明確な性能境界が出た。"},
        {"id": "second_model_finding", "type": "markdown", "sourceId": "second_stage_summary", "body": f"## どこまでCNNを小さくできるか\n\n幅ごとに3 seed・18 epochで再学習した。採用規則は、幅16平均に対してvalidationとOODの両方が3ポイント以内となる最小幅であり、後から結果に合わせて閾値を動かさない。{overfit['run_count']} runのbest epoch中央値は{overfit['median_best_epoch']:.1f}、最終epochがbestのrunは{overfit['best_at_final_epoch_count']}件、best後のvalidation loss最大悪化は{overfit['maximum_post_best_worsening']:.3f}。強い終盤過学習の有無と、小型幅の未学習・seed不安定性を分けて判断する。"},
        {"id": "second_model_chart", "type": "chart", "chartId": "second_model_size"},
        {"id": "second_model_table_block", "type": "table", "tableId": "second_model_table"},
        {"id": "second_latency_block", "type": "table", "tableId": "second_latency"},
        {"id": "second_ood_finding", "type": "markdown", "body": "## 未知姿勢・照明・背景\n\n学習に無い角度10/30/50/67/82°、未知の太陽方向・強度8条件、space/Earth/Moon背景を直交配置した。未知照明だけでなく、どの姿勢・背景で壊れるかを層別して報告する。"},
        {"id": "second_ood_tilt_chart", "type": "chart", "chartId": "second_ood_tilt"},
        {"id": "second_ood_background_chart", "type": "chart", "chartId": "second_ood_background"},
        {"id": "second_diagnostic_finding", "type": "markdown", "sourceId": "diagnostic_results", "body": "## 遮蔽とカメラ効果を分離する\n\nCNNの構造補完能力は、形状をなぞらない全高黒矩形で診断する。白飛び・黒つぶれは露光・飽和・ノイズ・量子化に基づくproxyとして実装した。"},
        {"id": "second_black_chart", "type": "chart", "chartId": "second_black_rectangle"},
        {"id": "second_effect_chart", "type": "chart", "chartId": "second_cnn_effects"},
        {"id": "second_physics_limits", "type": "markdown", "sourceId": "camera_effects_code", "body": "## 光学・センサモデルの解釈限界\n\n現在の効果は実機物理量へ校正されていない。白飛びは線形露光→full-well clip→shot noise→bloom、黒つぶれは低露光→read noise→black-level clip→量子化である。実センサ性能の主張にはHDR線形出力、PSF、応答曲線、full-well/read-noise校正が必要。"},
        {"id": "second_validation", "type": "markdown", "sourceId": "second_stage_validation", "body": "## 第2段階の結果監査\n\nOOD 480枚、診断1,456枚、全方式のサンプルID、4幅×3 seed×2評価を再集計した。診断は112個の元画像それぞれにcleanと3効果×4強度の13条件があり、同一組では共通のRANSAC seed keyを使う。劣化用サンプルIDの違いによる乱数交絡を除いた。"},
        {"id": "second_questions_intro", "type": "markdown", "body": "## 想定質問と未回答\n\n「答えられること」と「次の検証が必要なこと」を分離した。未回答を隠さず、次の実験設計へ直結させる。"},
        {"id": "second_questions_table", "type": "table", "tableId": "second_questions"},
        {"id": "second_positioning", "type": "markdown", "body": "## 先行研究との位置づけと新規性候補\n\n合成→実写のdomain gapは[SPEED+](https://arxiv.org/abs/2110.03101)が明確に示し、Earth背景を含むレンダリングは[URSO](https://arxiv.org/abs/1907.04298)、texture randomizationは[Parkら](https://arxiv.org/abs/1909.00392)に先行例がある。CNNでサンプリングを誘導するRANSACも[Neural-Guided RANSAC](https://arxiv.org/abs/1905.04132)や[Generalized Differentiable RANSAC](https://openaccess.thecvf.com/content/ICCV2023/html/Wei_Generalized_Differentiable_RANSAC_ICCV_2023_paper.html)があるため、CNN+RANSAC自体を新規性とはしない。ドッキングリングの幾何選択には[2026年のellipse detector](https://www.mdpi.com/1424-8220/26/2/396)もある。\n\n有望な独自軸は、(1) PAF内周という意味選択をsemantic CNNと明示的CAD priorで比較すること、(2)軌道・姿勢から既知の太陽方向をCNNへ条件付けし、画像だけのモデルとの照明OOD差を検証すること、(3)リング径・厚み・支持構造を変えたPAF family splitで形状一般化を測ること。既知太陽方向案は今回の限定検索では直接同一構成を確認できなかったが、現時点では新規性候補であり、網羅的調査前に新規性を断定しない。"},
        {"id": "second_recommendation", "type": "markdown", "body": "## 次に優先する研究\n\n1. 82°近傍とEarth背景を独立因子として学習追加し、性能回復量を測る。\n2. 既知太陽ベクトルをFiLMまたは特徴マップで条件付けし、画像のみCNNと同パラメータ規模で比較する。\n3. 複数PAF CAD familyを生成し、形状holdoutを最終testにする。\n4. 実機カメラのHDR応答・PSF・ノイズを校正し、sim-to-real testを固定する。\n5. RANSAC scoreからrisk-coverage曲線を作り、失敗棄却を評価する。"},
    ]
    insert_at = next(
        i for i, block in enumerate(manifest["blocks"]) if block["id"] == "audit_table"
    )
    manifest["blocks"][insert_at:insert_at] = second_blocks

    chart_map = {
        chart["id"]: {
            "dataset": chart["dataset"],
            "sourceId": chart.get("sourceId"),
            "question": chart["title"],
        }
        for chart in manifest["charts"]
    }
    (REPORT / "chart_map.json").write_text(
        json.dumps(chart_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(REPORT / "artifact.json")


if __name__ == "__main__":
    main()
