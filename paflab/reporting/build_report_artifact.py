from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_LABELS = {
    "zhang2019_arc_reproduction": "Zhang 2019型（再現実装）",
    "contour_fit": "Contour fit",
    "canny_ransac": "Canny + RANSAC",
    "cnn_ransac": "CNN + weighted RANSAC",
}
DEGRADATION_LABELS = {
    "occlusion": "遮蔽",
    "whiteout": "白飛び",
    "black_crush": "黒つぶれ",
}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最終実験から技術レポート用artifact.jsonを生成する")
    parser.add_argument("--config", default="config/research_experiment.json")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def relative_source(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sql_project_rows(
    overall_rows: list[dict], curve_rows: list[dict], aggregate_rows: list[dict], sql_path: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    """監査済み集計を実際のSQLクエリでレポート表示粒度へ射影する。"""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE source_overall (
        method TEXT, method_label TEXT, sample_count INTEGER, detected_count INTEGER,
        success_count INTEGER, success_rate REAL, detection_rate REAL, success_definition TEXT
        )"""
    )
    connection.executemany(
        "INSERT INTO source_overall VALUES (:method, :method_label, :sample_count, :detected_count, "
        ":success_count, :success_rate, :detection_rate, :success_definition)",
        overall_rows,
    )
    connection.execute(
        """CREATE TABLE source_curves (
        method TEXT, method_label TEXT, degradation TEXT, degradation_label TEXT,
        severity REAL, sample_count INTEGER, camera_cluster_count INTEGER,
        detection_rate REAL, success_rate REAL, cluster_ci95_low REAL,
        cluster_ci95_high REAL, mean_ellipse_iou REAL, median_ellipse_iou REAL
        )"""
    )
    connection.executemany(
        "INSERT INTO source_curves VALUES (:method, :method_label, :degradation, "
        ":degradation_label, :severity, :sample_count, :camera_cluster_count, :detection_rate, "
        ":success_rate, :cluster_ci95_low, :cluster_ci95_high, :mean_ellipse_iou, "
        ":median_ellipse_iou)",
        curve_rows,
    )
    connection.execute(
        """CREATE TABLE source_aggregate (
        method TEXT, method_label TEXT, degradation TEXT, degradation_label TEXT,
        robustness_auc REAL, critical_severity_50 REAL
        )"""
    )
    connection.executemany(
        "INSERT INTO source_aggregate VALUES (:method, :method_label, :degradation, "
        ":degradation_label, :robustness_auc, :critical_severity_50)",
        aggregate_rows,
    )
    connection.executescript(sql_path.read_text(encoding="utf-8"))

    def rows(view: str) -> list[dict]:
        return [dict(row) for row in connection.execute(f"SELECT * FROM {view}")]

    result = rows("report_overall"), rows("report_curves"), rows("report_aggregate")
    connection.close()
    return result


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_json(config_path)
    dataset_dir = project_path(config["stress_dataset_dir"])
    artifacts_dir = project_path(config["artifacts_dir"])
    report_dir = artifacts_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    robustness_path = artifacts_dir / "robustness_curves.csv"
    robustness_summary_path = artifacts_dir / "robustness_summary.json"
    validation_path = artifacts_dir / "validation_report.json"
    validation = load_json(validation_path)
    if validation["status"] != "ready_to_share":
        raise RuntimeError("検証未通過のためレポートを生成できません")

    curve_rows = []
    for row in load_csv(robustness_path):
        curve_rows.append(
            {
                "method": row["method"],
                "method_label": METHOD_LABELS[row["method"]],
                "degradation": row["degradation"],
                "degradation_label": DEGRADATION_LABELS[row["degradation"]],
                "severity": float(row["severity"]),
                "sample_count": int(row["sample_count"]),
                "camera_cluster_count": int(row["camera_cluster_count"]),
                "detection_rate": float(row["detection_rate"]),
                "success_rate": float(row["success_rate"]),
                "cluster_ci95_low": float(row["cluster_bootstrap_ci95_low"]),
                "cluster_ci95_high": float(row["cluster_bootstrap_ci95_high"]),
                "mean_ellipse_iou": (
                    float(row["mean_ellipse_iou"]) if row["mean_ellipse_iou"] else None
                ),
                "median_ellipse_iou": (
                    float(row["median_ellipse_iou"]) if row["median_ellipse_iou"] else None
                ),
            }
        )

    overall_rows = []
    for method in config["evaluation"]["methods"]:
        summary_path = dataset_dir / "results" / method / "summary.json"
        summary = load_json(summary_path)
        overall_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "sample_count": int(summary["sample_count"]),
                "detected_count": int(summary["detected_count"]),
                "success_count": int(summary["top1_match_count"]),
                "success_rate": float(summary["top1_match_rate"]),
                "detection_rate": float(summary["detected_count"]) / float(summary["sample_count"]),
                "success_definition": "top-1 ellipse IoU >= 0.80",
            }
        )

    robustness_summary = load_json(robustness_summary_path)
    aggregate_rows = [
        {
            **row,
            "method_label": METHOD_LABELS[row["method"]],
            "degradation_label": DEGRADATION_LABELS[row["degradation"]],
        }
        for row in robustness_summary["aggregate"]
    ]
    report_sql_path = PROJECT_ROOT / "analysis" / "report_source.sql"
    overall_rows, curve_rows, aggregate_rows = sql_project_rows(
        overall_rows, curve_rows, aggregate_rows, report_sql_path
    )
    report_data = {
        "experiment_id": config["experiment_id"],
        "overall": overall_rows,
        "curves": curve_rows,
        "aggregate": aggregate_rows,
        "validation": validation,
    }
    report_data_path = report_dir / "report_data.json"
    report_data_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    title = "PAF内周リング楕円検出：CG耐性検証 v3"
    cnn = next(row for row in overall_rows if row["method"] == "cnn_ransac")
    zhang = next(
        row
        for row in overall_rows
        if row["method"] == "zhang2019_arc_reproduction"
    )

    sources = [
        {
            "id": "report_query",
            "label": "SQLite report projection query",
            "path": relative_source(report_sql_path),
            "description": "監査済み集計をレポート表示粒度へ射影・整列した実行済みSQL。",
        },
        {
            "id": "report_data",
            "label": "v3 reviewed report data",
            "path": relative_source(report_data_path),
            "description": "監査済みの方式別集計、強度曲線、頑健性AUC。",
        },
        {
            "id": "robustness_curves",
            "label": "v3 robustness curves",
            "path": relative_source(robustness_path),
            "description": "各強度112画像・16カメラクラスタで集計した成功率と95%区間。",
        },
        {
            "id": "validation_report",
            "label": "v3 validation report",
            "path": relative_source(validation_path),
            "description": "ID、条件、IoU範囲、再集計、分割重複を検算した品質監査。",
        },
        {
            "id": "experiment_config",
            "label": "v3 experiment configuration",
            "path": relative_source(config_path),
            "description": "劣化強度、学習条件、RANSAC条件、成功判定の固定設定。",
        },
        {
            "id": "zhang_2019_reference",
            "label": "Zhang et al. docking-ring detector",
            "path": "https://doi.org/10.3390/s19235243",
            "description": "主古典ベースラインの元論文。著者コード移植ではなく再現実装。",
        },
    ]

    cards = []
    for row in overall_rows:
        cards.append(
            {
                "id": f"overall_{row['method']}",
                "dataset": "overall",
                "filter": {"method": row["method"]},
                "sourceId": "report_query",
                "description": "全19条件を等重みで集計。完全遮蔽・完全白飛びも含む。",
                "metrics": [
                    {"label": f"{row['method_label']} 成功率", "field": "success_rate", "format": "percent"},
                    {"label": "成功数", "field": "success_count", "format": "number"},
                    {"label": "評価数", "field": "sample_count", "format": "number"},
                ],
            }
        )

    charts = [
        {
            "id": "overall_success",
            "title": "方式別の全条件成功率",
            "subtitle": "成功はtop-1楕円IoU 0.80以上。各方式2,128画像。",
            "showDescription": True,
            "type": "bar",
            "dataset": "overall",
            "sourceId": "report_query",
            "encodings": {
                "x": {"field": "method_label", "type": "nominal", "label": "方式"},
                "y": {"field": "success_rate", "type": "quantitative", "label": "成功率", "format": "percent"},
                "tooltip": [
                    {"field": "success_count", "label": "成功数", "format": "number"},
                    {"field": "sample_count", "label": "評価数", "format": "number"},
                    {"field": "detection_rate", "label": "検出率", "format": "percent"},
                ],
            },
        }
    ]
    for degradation in config["degradations"]["types"]:
        charts.append(
            {
                "id": f"curve_{degradation}",
                "title": f"{DEGRADATION_LABELS[degradation]}強度と成功率",
                "subtitle": "各点112画像・16カメラ条件。線色は検出方式。",
                "showDescription": True,
                "type": "line",
                "dataset": f"curve_{degradation}",
                "sourceId": "report_query",
                "encodings": {
                    "x": {"field": "severity", "type": "quantitative", "label": "劣化強度"},
                    "y": {"field": "success_rate", "type": "quantitative", "label": "成功率", "format": "percent"},
                    "color": {"field": "method_label", "type": "nominal", "label": "方式"},
                    "tooltip": [
                        {"field": "sample_count", "label": "画像数", "format": "number"},
                        {"field": "camera_cluster_count", "label": "カメラ条件数", "format": "number"},
                        {"field": "cluster_ci95_low", "label": "95% CI 下限", "format": "percent"},
                        {"field": "cluster_ci95_high", "label": "95% CI 上限", "format": "percent"},
                        {"field": "detection_rate", "label": "検出率", "format": "percent"},
                        {"field": "mean_ellipse_iou", "label": "平均IoU", "format": "number"},
                    ],
                },
            }
        )

    tables = [
        {
            "id": "robustness_audit",
            "title": "頑健性AUCと成功率50%限界強度",
            "subtitle": "AUCは強度0〜1の成功率曲線を台形積分。高いほど頑健。",
            "showDescription": True,
            "dataset": "aggregate",
            "sourceId": "report_query",
            "columns": [
                {"field": "method_label", "label": "方式"},
                {"field": "degradation_label", "label": "劣化"},
                {"field": "robustness_auc", "label": "頑健性AUC", "format": "number"},
                {"field": "critical_severity_50", "label": "50%限界強度", "format": "number"},
            ],
            "defaultSort": {"field": "robustness_auc", "direction": "desc"},
        }
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "report_data",
            "body": (
                "## Technical summary\n\n"
                f"Blender 5.2で生成したPAF CGを用い、CNNが出力する内周リング確率を重み付きRANSACへ接続した。"
                f"カメラ条件を分離したtest 2,128画像で、CNN方式の成功率は **{cnn['success_rate']:.1%}**。"
                f"宇宙機ドッキングリング向け古典法を参考にしたZhang 2019型（再現実装）の **{zhang['success_rate']:.1%}** を上回った。"
                "現段階で確認できたのはCG内の頑健性であり、実写一般化やCNN+RANSAC自体の新規性を示したものではない。"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [f"overall_{method}" for method in config["evaluation"]["methods"]],
        },
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "report_data",
            "body": (
                "## Key findings\n\n"
                "- 主比較はZhang 2019型（再現実装）とCNN + weighted RANSACである。\n"
                "- Zhang型はCanny後の分断弧を統合し、実エッジ支持で候補楕円を検証する。\n"
                "- 完全遮蔽と完全白飛びでは全方式0%。情報が消失した入力で成功を捏造していない。\n"
                "- 旧Canny全点・輪郭別・内周prior方式は補足アブレーションとして別保存する。"
            ),
        },
        {"id": "overall_chart", "type": "chart", "chartId": "overall_success"},
        {
            "id": "occlusion_finding",
            "type": "markdown",
            "sourceId": "robustness_curves",
            "body": (
                "### 遮蔽\n\nCNN方式は強度0.4まで成功率50%超を維持するが、0.6で急落する。"
                "実機で遮蔽境界を扱うなら、単一楕円の成否だけでなく可視弧長と予測信頼度を併記すべきである。"
            ),
        },
        {"id": "occlusion_chart", "type": "chart", "chartId": "curve_occlusion"},
        {
            "id": "whiteout_finding",
            "type": "markdown",
            "sourceId": "robustness_curves",
            "body": (
                "### 白飛び\n\n局所飽和とブルームではCNN方式が0.9まで高い成功率を維持した。"
                "ただし1.0は画像全体を白へ置換するため全方式0%となる。"
            ),
        },
        {"id": "whiteout_chart", "type": "chart", "chartId": "curve_whiteout"},
        {
            "id": "black_crush_finding",
            "type": "markdown",
            "sourceId": "robustness_curves",
            "body": (
                "### 黒つぶれ\n\nCNN方式の50%限界強度は0.519。強度1.0も完全黒画像ではなく、"
                "黒レベル閾値0.58・γ2.8の暗部圧縮であるため、成功率5.4%が残る。"
            ),
        },
        {"id": "black_crush_chart", "type": "chart", "chartId": "curve_black_crush"},
        {"id": "audit_table", "type": "table", "tableId": "robustness_audit"},
        {
            "id": "scope_metrics",
            "type": "markdown",
            "sourceId": "experiment_config",
            "body": (
                "## Scope, data, and metric definitions\n\n"
                "- 基礎CG: 102カメラ条件 × 7照明条件 = 714画像（480×480）。\n"
                "- 分割: camera_id単位でtrain 71、validation 15、test 16条件。照明違いの兄弟画像を別splitへ跨がせない。\n"
                "- test: clean + 3劣化 × 6強度 = 19条件、各112画像、計2,128画像。\n"
                "- 成功: top-1推定楕円と正解楕円のIoUが0.80以上。\n"
                "- 頑健性AUC: 強度0〜1の成功率曲線を台形積分した正規化面積。\n"
                "- 95%区間: testの16カメラ条件をクラスタとする4,000回ブートストラップ。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "experiment_config",
            "body": (
                "## Methodology and experiment details\n\n"
                "Blenderの3D内周頂点を画像へ投影し、評価用の正解楕円を生成した。Tiny U-Net（約48万パラメータ）を12 epoch学習し、"
                "リング尤度上の点を確率重み付きRANSACへ渡した。モデル選択はvalidation lossのみ、testは最終評価に限定した。"
                "学習劣化は0.10〜0.80、testには未学習の0.90と1.00を含めた。"
            ),
        },
        {
            "id": "validation",
            "type": "markdown",
            "sourceId": "validation_report",
            "body": (
                "## Validation and robustness checks\n\n"
                "2,128件×3方式について、manifestとのID完全一致、重複なし、劣化条件一致、IoU範囲、summary再計算、"
                "強度グリッド、画像・ラベル存在、split非重複を検証し、すべて通過した。"
                "予備v1は白飛び定義、v2は局所マスク境界の正解形状漏洩を検出したため主結果から除外し、漏洩のないv3だけを採用した。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations and uncertainty\n\n"
                "- 単一CGモデル・単一レンダラの結果で、材質、レンズ、センサノイズ、実写への一般化は未検証。\n"
                "- 教師マスクは不可視部を含む完全楕円であり、可視エッジ検出というより構造補完を学ぶ。\n"
                "- 劣化強度は制御可能な画像変換で、露光量やセンサ黒レベルの校正値ではない。\n"
                "- 16カメラクラスタの区間推定は行ったが、独立な3D形状は1種類だけである。\n"
                "- CNN閾値・RANSAC設定の網羅的なハイパーパラメータ探索は未実施。"
            ),
        },
        {
            "id": "novelty_next",
            "type": "markdown",
            "body": (
                "## Novelty candidates and recommended next steps\n\n"
                "CNN特徴とRANSACの結合自体は既存研究があり、ここを新規性とは主張しにくい。"
                "[CNNによるRANSAC前処理（CVPR 2017）](https://openaccess.thecvf.com/content_cvpr_2017/html/Morley_Improving_RANSAC-Based_Segmentation_CVPR_2017_paper.html)や"
                "[Neural-Guided RANSAC（ICCV 2019）](https://openaccess.thecvf.com/content_ICCV_2019/html/Brachmann_Neural-Guided_RANSAC_Learning_Where_to_Sample_Model_Hypotheses_ICCV_2019_paper.html)を踏まえると、"
                "研究価値を出しやすい軸は次の通り。\n\n"
                "1. **PAF固有の動作限界マップ**: 遮蔽率・飽和率・暗部圧縮と幾何誤差／失敗確率を校正し、運用可能域を定量化する。\n"
                "2. **可視弧と不可視形状の分離**: 可視エッジ分割、楕円補完、信頼度校正を別タスクとして学習し、幻覚的補完を検知する。\n"
                "3. **sim-to-realの因果的検証**: 実写の露光・遮蔽条件を校正し、どのCG因子が実写改善へ寄与するかアブレーションする。\n"
                "4. **幾何制約付き学習**: 楕円残差やRANSAC合意度を損失へ組み込み、単純なセグメンテーションとの差を検証する。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- 実機カメラの露光・ゲイン・黒レベルを、今回の強度0〜1へどう対応付けるか。\n"
                "- 異なるPAF形状、背景、材質、レンズ歪みでも同じ限界強度が再現するか。\n"
                "- 完全楕円補完の信頼度をどう校正し、安全に棄却するか。\n"
                "- Event cameraや時系列情報との比較で、RGB方式が優位／不利になる領域はどこか。"
            ),
        },
        {
            "id": "reproducibility",
            "type": "markdown",
            "body": (
                "## Reproducibility\n\n"
                "`python run_research.py`でCG生成、劣化データ作成、学習、3方式評価、集計、監査を順に実行する。"
                "固定seedは20260716。Blender 5.2のbackground実行とbpy APIを使用する。"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "PAF内周リング検出の遮蔽・白飛び・黒つぶれ耐性をCGで検証した技術レポート。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "overall": overall_rows,
                "aggregate": aggregate_rows,
                **{
                    f"curve_{degradation}": [
                        row for row in curve_rows if row["degradation"] == degradation
                    ]
                    for degradation in config["degradations"]["types"]
                },
            },
        },
        "sources": sources,
    }
    artifact_path = report_dir / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
