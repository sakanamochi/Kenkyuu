# PAF内周検出システム

多様な合成PAF画像で学習した単一境界CNN（Tiny U-Net16＋形態学的スケルトン＋確率重み付きRANSAC）を中心に、1194M学習CNN・Canny＋輪郭別RANSAC・Zhang 2019型（再現実装）の推論・評価・集計・学習およびCG原画像再生成をまとめた最小構成です。

## フォルダ構成と役割

直接実行するスクリプトはすべて `scripts/` に集約し、本体は役割ごとに `src/` 直下へ分離しています。

```text
Kenkyuu/
  scripts/                  # ユーザーが直接実行するスクリプト（全6本）
    predict.py              # 推論用：画像1枚を検出（多様形状CNN / 1194M CNN / Canny / Zhang 2019型）
    train.py                # 学習用：Tiny U-Net16を可視内周境界教師で学習
    evaluate.py             # 評価用：4方式×2評価集合×13条件の評価＋確定参照との全件照合
    report.py               # 集計用：REPORT.md・summary.json・CSV・比較図の生成
    render_cg.py            # CG生成用：Blenderによる1194M／多様形状の原CG再生成と検証
    ablation.py             # 要因分解用：現実PAF範囲のCG生成・外観/投影/遮蔽/形状アブレーション実行・集計
  configs/                  # 設定ファイル
    experiment.json         # 学習・推論・評価・幾何比較器・レポート図の正本設定
    render_1194m.json       # 1194M原CG（714枚）のBlender描画設定
    render_diversified.json # 多様形状PAF（アダプタ部＋固定円筒部・756形状）の寸法・Blender描画設定
  src/                      # 役割別に分離した本体パッケージ
    defaults.py             # 基準パスと既定設定パスの定義
    core/                   # 基本システム共通部品（io / geometry / effects / model / ransac）
    inference/              # 推論処理（predictor / canny / zhang2019）※学習へ非依存
    training/               # 学習処理（dataset / trainer）※推論へ非依存
    evaluation/             # 評価・集計処理（evaluator / reporter）
    cg/                     # CG原画像再生成処理（geometry_model / blender_common / render_1194m / render_diversified）
  assets/1194M_3/           # 1194MのCAD素材（OBJ・MTL）
  data/                     # 全データセット統一配置（1194m / diversified / diversified_limited_size / single_cone）
  models/                   # 全学習済み重み統一配置（1194m / diversified / diversified_limited_size / ablation）
  references/               # 確定参照推定結果（再現性照合用）
  outputs/                  # 評価・推論・アブレーションの予測・レポート・図出力先
  tests/                    # 単体・結合・層分離テスト（test_paf_system.py）
  研究関連書類/             # 発表資料・先行研究（ローカル保持）
  .codex/                   # 研究仕様と引継ぎ資料（PROJECT_CONTEXT.md / RESEARCH_HANDOFF.md）
```

## 設定の変更場所

| 変えたい内容 | 編集する場所 | 主な項目 |
|---|---|---|
| 学習条件 | `configs/experiment.json` の `training`・`seeds`・`input_size` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `positive_weight`, `base_channels`, `presets.{1194m,diversified}.view_slots`（実行時は `scripts/train.py` の `--source`, `--seeds`, `--epochs`, `--output` で上書き可） |
| 推論・RANSAC・幾何比較器 | `configs/experiment.json` の `probability_threshold`, `ransac_cnn`, `ransac_geometry`, `canny_contour_ransac`, `zhang2019` | 確率閾値、RANSAC反復数・最小支持点数、Canny閾値、Zhang設定 |
| 評価条件（白飛び強度等） | `configs/experiment.json` の `evaluation`・`success_iou` | 評価対象方式・データセット、評価時の白飛び最大内部強度（2.0）、成功IoU閾値（0.8） |
| レポート比較図の体裁 | `configs/experiment.json` の `report.figures` | `dpi`, `format`（`png`/`svg`）, `font_family`, `font_size`, `overall_figsize`, `degradation_figsize`, `colors`, `markers` |
| CG原画像の寸法・撮影条件 | `configs/render_1194m.json` / `configs/render_diversified.json` | カメラ角度・距離・照明・寸法比範囲・生成枚数 |

## 実行コマンド

ルートディレクトリから既存の `.venv` で実行します。

### 1. 推論（画像1枚）

```powershell
# 多様形状学習CNN（既定）で推論
.venv/Scripts/python.exe -X utf8 -B scripts/predict.py --image data/1194m/base/images/camera_t000_a000_d044.0_o00__light_t000_a000_e03.0.png --output outputs/predict_example

# 方式切替（cnn_1194m / canny / zhang2019）
.venv/Scripts/python.exe -X utf8 -B scripts/predict.py --method cnn_1194m --seed 20260907 --image <画像> --output outputs/predict_1194m
.venv/Scripts/python.exe -X utf8 -B scripts/predict.py --method canny --image <画像> --output outputs/predict_canny
.venv/Scripts/python.exe -X utf8 -B scripts/predict.py --method zhang2019 --image <画像> --output outputs/predict_zhang

# 実画像など非正方形画像は黒余白を付けて正方形化（--pad-square）
.venv/Scripts/python.exe -X utf8 -B scripts/predict.py --image <実画像> --pad-square --output outputs/predict_real
```

### 2. 評価と集計・図生成

```powershell
# 全4方式×2データセット×13条件（白飛び最大内部強度2.0）の評価と確定参照照合
.venv/Scripts/python.exe -u -X utf8 -B scripts/evaluate.py --verify-reference

# REPORT.md・summary.json・CSV・比較図（overall.png / degradation.png）の生成
.venv/Scripts/python.exe -X utf8 -B scripts/report.py
```

### 3. 学習

```powershell
# 多様形状CNN（既定1seed: 20260907、4ビュー・開口幅遮蔽・60epoch）を学習
.venv/Scripts/python.exe -u -X utf8 -B scripts/train.py --source diversified

# 現実PAF範囲CNN（既定1seed: 20260907）を学習
.venv/Scripts/python.exe -u -X utf8 -B scripts/train.py --source diversified_limited_size

# 明示的に3seedで学習したい場合
.venv/Scripts/python.exe -u -X utf8 -B scripts/train.py --source 1194m --seeds 20260907 20260908 20260909
```

※ GPU学習は決定的アルゴリズムを強制していないため、再学習時の重みは保存重みとビット単位では一致しません。

### 4. CG原画像の再生成（Blender使用）

```powershell
# 1194Mの原CGを生成（少数確認は --limit 4 を指定）
.venv/Scripts/python.exe -u -X utf8 -B scripts/render_cg.py 1194m

# 多様形状の原CGを生成（plan → render → verify）
.venv/Scripts/python.exe -X utf8 -B scripts/render_cg.py diversified --stage plan
.venv/Scripts/python.exe -u -X utf8 -B scripts/render_cg.py diversified --stage render
.venv/Scripts/python.exe -X utf8 -B scripts/render_cg.py diversified --stage verify
```

### 5. テスト

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -v -p no:cacheprovider tests/test_paf_system.py
```

## 比較手法の出典

- **Zhang 2019型（再現実装）**：Limin Zhang, Wang Pan, Xianghua Ma, "Real-Time Docking Ring Detection Based on the Geometrical Shape for an On-Orbit Spacecraft," *Sensors*, 19(23), 5243, 2019, DOI `10.3390/s19235243`。著者コードの移植ではなく、論文の処理構成を参考にしたPython再現実装へPAF固有の全周エッジ支持・内包内周選択を追加したものです。
