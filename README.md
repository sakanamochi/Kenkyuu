# PAF内周リング検出・CGロバスト性研究環境

Blender 5.2でPAF（Payload Attach Fitting）のCGと3D形状由来の正解楕円を生成し、遮蔽・白飛び・黒つぶれに対する内周リング検出性能を比較する再現可能な研究環境です。

現在の主比較は、次の2方式を同じ画像・正解楕円・IoU基準で評価します。

1. `zhang2019_arc_reproduction`: Canny → 弧抽出・分類 → 分断弧の統合 → 楕円推定・検証
2. `cnn_ransac_support`: Tiny U-Netのリング尤度 → 確率重み付きRANSAC → インライア再フィット

成功判定はtop-1推定楕円と正解楕円のIoUが0.80以上です。

Zhang型の正規出典:

> Limin Zhang, Wang Pan, Xianghua Ma, “Real-Time Docking Ring Detection
> Based on the Geometrical Shape for an On-Orbit Spacecraft,” Sensors,
> 19(23), 5243, 2019.  
> https://doi.org/10.3390/s19235243

本実装は著者コードの移植ではなく、論文の処理構成を参考にした再現実装です。
詳細な設計判断と引き継ぎ情報は `.codex/PROJECT_CONTEXT.md` にあります。

## 現在の主比較結果

| 評価 | Zhang 2019型（再現実装） | CNN + weighted RANSAC |
|---|---:|---:|
| OOD 480枚 | 250 / 480（52.1%） | **383 / 480（79.8%）** |
| 撮像診断1,904枚 | 668 / 1,904（35.1%） | **1,167 / 1,904（61.3%）** |

## 進捗報告での扱い

研究環境の現在の主比較はZhang型とCNNですが、第1回進捗報告では開発時系列に沿って
Canny + 輪郭別RANSACとCNNの初期比較を示します。遮蔽で輪郭が分断される問題を説明し、
Zhang型は「次回までに導入予定」とします。地球・月背景や未知角度別の詳細結果は第1回には
含めず、データ分割について「学習・テストで異なるカメラ条件を使用」とだけ示します。

第1回用素材は
`output/presentation_assets/progress_meeting_1_v1/`、
生成コードは
`paflab/reporting/build_first_progress_figures.py`
です。

## 第2段階検証

`paf_second_stage_v1` は、v3で同じ角度・照明カテゴリを共有していた限界を補う独立評価です。

- OOD CG: 学習に無いカメラ傾斜 `10, 30, 50, 67, 82°`、未知照明8条件、`space / Earth / Moon` 背景、計480枚
- 撮像診断: clean 112枚と、黒矩形遮蔽・センサ白飛びproxy・センサ黒つぶれproxyの4強度、計1,456枚
- モデル規模: base channels `2, 4, 8, 16` × 3 seed × 18 epoch
- 古典法: 宇宙機ドッキングリング向けZhang 2019型を主ベースラインとして再現
- RANSAC監査: Canny + 輪郭別RANSACの候補生成・内周priorを監査
- 不確実性: camera_idクラスタbootstrap 5,000回。診断効果と背景の比較は同じ元画像で同じRANSAC乱数列を使うpaired評価

主要結果:

| 評価 | Zhang 2019型（再現実装） | CNN + weighted RANSAC |
|---|---:|---:|
| OOD 480枚 | 52.1% | **79.8%** |
| 撮像診断1,904枚 | 35.1% | **61.3%** |

Canny + 輪郭別RANSACは内周priorを含む単一方式として扱います。Zhang型は分断弧を統合できるため、遮蔽を含む主古典比較として
より妥当です。

モデル幅3 seed平均では、幅8（121,177パラメータ）がvalidation 74.6%、幅16（482,737パラメータ）が77.9%です。事前に固定した「幅16からvalidation/OODとも3ポイント以内」という基準では差が3.27ポイントのため幅16を採用し、幅8は容量優先の近似解とします。幅16の単画像CNN推論中央値は、このPCでCPU 9.35 ms、CUDA 1.21 msです（RANSACとI/Oを除く）。

一括再現:

```powershell
.venv\Scripts\python.exe run_second_stage.py
```

既存レンダリング・学習済み重み・Zhang型結果を再利用する場合:

```powershell
.venv\Scripts\python.exe run_second_stage.py --skip-render --skip-prepare --skip-models --skip-classic
```

第2段階の正規集計は `output/experiments/paf_second_stage_v1/second_stage_summary.json`、統合技術レポートは `output/experiments/paf_second_stage_v1/report/report.html` に出力します。

実験内容を一目で確認するためのまとめ画像は `output/experiments/paf_second_stage_v1/figures/` に出力します。

- `experiment-overview-poster.png`: 研究フロー、代表入力、出力楕円、主要結果
- `angle-input-output-grid.png`: 代表角度ごとの入力、CNN確率、CNN楕円、Zhang型楕円
- `background-input-output-grid.png`: space / Earth / Moon背景の対応比較
- `diagnostic-input-output-grid.png`: clean、遮蔽、白飛び、黒つぶれ、フレアの対応比較
- `quantitative-experiment-summary.png`: モデル規模、Zhang/CNN主比較、角度性能、撮像効果の定量図

集計済み結果と学習済み重みから画像だけを再生成する場合:

```powershell
.venv\Scripts\python.exe -m paflab.reporting.build_summary_figures
```

## 単体実験画像GUI

データセット内の任意の1枚を選び、効果と強度を指定して、入力から検出結果までの
中間画像をまとめて取得できます。

```powershell
.venv\Scripts\python.exe run_experiment_gui.py
```

GUIは次を個別PNGと一覧用 `overview.png`、設定・評価値を含む `metadata.json` として
`output/gui_exports/` 以下へ保存します。

- Canny + 輪郭別RANSAC、Zhang 2019型（再現実装）、CNNをチェックボックスで個別選択
- 選択した各手法のIoUと成功・失敗判定をGUI内へ表示
- 元画像、劣化適用画像
- grayscale、Gaussian平滑化、Cannyエッジ、分離輪郭
- Canny + 輪郭別RANSACの候補・結果
- Zhang 2019型（再現実装）の弧・候補・結果
- CNN入力、リング尤度、閾値点群、weighted RANSACインライア・結果

撮像診断用の単純黒矩形は左右いずれかの画面端から伸び、同一元画像では強度間で
進行方向を固定します。センサ黒つぶれproxyは暗い宇宙背景で50%以降が全面黒に
近づきすぎないよう、最大露光低下を2.5段、黒レベルを最大0.025、量子化を最低6 bitに
抑えた緩やかな曲線です。

撮像診断1,904枚の定量結果まで新しい効果定義で更新する場合:

```powershell
.venv\Scripts\python.exe run_second_stage.py --skip-render --skip-models
```

## 文献ベースライン比較

主比較は次の2方式です。

1. `zhang2019_arc_reproduction`: Zhang et al. (2019)を参考にした弧統合型ベースライン
2. `cnn_ransac`: CNNリング尤度 + 確率重み付きRANSAC

Zhang型再現実装は、Canny、勾配象限による弧分割、異なる象限の3弧統合、
`fitEllipseDirect`、エッジ密度・角度被覆・内外輝度による検証で構成します。
著者コードの移植ではないため、成果物では必ず「再現実装」と表記します。

旧Contour/Canny方式を含むアブレーションも再生成する場合:

```powershell
.venv\Scripts\python.exe run_literature_comparison.py --include-ablations
```

一括比較:

```powershell
.venv\Scripts\python.exe run_literature_comparison.py
```

既存結果から集計画像だけを再生成:

```powershell
.venv\Scripts\python.exe run_literature_comparison.py --figures-only
```

結果は `output/experiments/paf_second_stage_v1/literature_baselines/` に出力します。

- `comparison-summary.csv/json`: データセット・方式別の成功数と成功率
- `comparison-stratified.csv`: 角度、背景、劣化種別、強度ごとの層別成功率
- `comparison-audit.json`: 方式間で評価sample IDが完全一致することの監査結果
- `method-success-rate-comparison.png`: 全方式の定量比較
- `ood-input-intermediate-output-grid.png`: 代表角度・背景の入力、中間表現、出力楕円
- `diagnostic-input-intermediate-output-grid.png`: 遮蔽・白飛び・黒つぶれ等の対応比較

### 任意: AAMED

一般楕円検出器との外部比較用に、公式 `pyAAMED` がimport可能な場合だけ
`aamed`方式を追加できます。GPL-2.0の第三者コードは本リポジトリへ複製せず、
[公式AAMED](https://github.com/Li-Zhaoxi/AAMED)を別ディレクトリでビルドしてください。

```powershell
# pyAAMEDをビルドしてimport可能にした後
.venv\Scripts\python.exe run_literature_comparison.py --with-aamed
```

AAMEDが無い通常環境でも、Zhang型とCNNの主比較は完結します。

## RANSAC周長補正アブレーション

CNN確率点群とCanny輪郭方式では候補点の意味が異なるため、RANSAC仮説スコアの周長補正を別々に比較しています。`perimeter_power=1.0`が従来の支持密度、`0.0`が周長で割らない総支持量です。比較結果を受け、CNNでは`perimeter_power=0.0`を採用設定として`config/research_experiment.json`へ反映しています。Zhang型はRANSACではなく弧統合後の最小二乗楕円と候補検証を使うため、このアブレーションの対象外です。

- CNN: validation 69.4% → 75.8%、test 51.4% → 55.5%、OOD 74.0% → 79.8%
- Canny + 内周prior: clean testでは87.5% → 84.8%、OODでは54.2% → 55.8%であり、補正なしは一貫して優位ではない
- 実行済みノートブック: `output/experiments/paf_second_stage_v1/ransac_scoring_ablation/ransac_scoring_ablation.ipynb`
- 比較図: `output/experiments/paf_second_stage_v1/ransac_scoring_ablation/ransac-scoring-ablation-summary.png`

再計算例:

```powershell
.venv\Scripts\python.exe -m paflab.experiments.compare_ransac_scoring --task cnn --scope validation
.venv\Scripts\python.exe -m paflab.experiments.compare_ransac_scoring --task classic --scope validation_clean
.venv\Scripts\python.exe -m paflab.experiments.build_ransac_scoring_notebook
```

## 設計

```text
config/research_base.json
        │
        ▼
Blender 5.2 / blender/generate_dataset.py
        │  714基礎CG + 3D投影GT
        ▼
paflab/prepare_stress_dataset.py
        │  camera_id単位split + 制御劣化
        ├───────────────┐
        ▼               ▼
paflab/train_cnn.py     古典法評価
        │               analysis/evaluate_dataset.py
        ▼               │
paflab/evaluate_cnn.py  │
        └───────┬───────┘
                ▼
paflab/reporting/summarize_robustness.py
        │  強度曲線・AUC・カメラクラスタ95%区間
        ▼
paflab/reporting/validate_results.py
        │  ID・条件・集計・分割の検算
        ▼
paflab/reporting/build_report_artifact.py
        │
        ▼
artifact.json / report.html
```

処理を小さなCLIモジュールへ分離し、データ生成、学習、方式追加、評価、報告を独立に交換できるようにしています。画像入出力はWindowsの日本語パスへ対応しています。

## セットアップ

PowerShellでリポジトリ直下から実行します。

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

RTX 4070 TiなどNVIDIA GPUを使う場合は、CUDA版PyTorchを追加します。

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

GPU確認:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 一括再現

Blender 5.2の既定パスは `C:/Program Files/Blender Foundation/Blender 5.2/blender.exe` です。

```powershell
.venv\Scripts\python.exe run_research.py
```

一括処理は次を順に実行します。

1. Blender background + `bpy`による基礎CG生成
2. camera_id単位のtrain / validation / test分割と劣化画像生成
3. CNN学習
4. Zhang 2019型（再現実装）とCNN方式のtest評価
5. 頑健性集計、方式比較、結果監査
6. レポート用 `artifact.json` と `report_data.json` の更新

既存成果を再利用する場合:

```powershell
.venv\Scripts\python.exe run_research.py --skip-render --skip-prepare --skip-train
```

RANSAC評価は計算量が大きいため、`analysis/evaluate_dataset.py` は途中結果からの再開に対応しています。

## 個別実行

```powershell
# 劣化データセット作成
.venv\Scripts\python.exe -m paflab.prepare_stress_dataset --config config/research_experiment.json

# CNN学習・評価
.venv\Scripts\python.exe -m paflab.train_cnn --config config/research_experiment.json
.venv\Scripts\python.exe -m paflab.evaluate_cnn --config config/research_experiment.json

# 強度曲線と品質監査
.venv\Scripts\python.exe -m paflab.reporting.summarize_robustness --config config/research_experiment.json
.venv\Scripts\python.exe -m paflab.reporting.validate_results --config config/research_experiment.json

# レポートの正規データを更新
.venv\Scripts\python.exe -m paflab.reporting.build_report_artifact --config config/research_experiment.json
```

## データと分割

基礎データは102カメラ条件×7照明条件、計714画像です。正解は `GT_INNER_RING` の3D頂点をカメラ画像へ投影して生成します。

`paf_robustness_v3` はcamera_idをグループとして分割します。

| split | カメラ条件 | 画像数 | 用途 |
|---|---:|---:|---|
| train | 71 | 1,988 | clean + 3劣化（強度0.10〜0.80） |
| validation | 15 | 735 | モデル選択（強度0.30、0.60） |
| test | 16 | 2,128 | clean + 3劣化×6強度 |

同じカメラ条件の照明違い・劣化違いが別splitへ漏れないため、画像単位のランダム分割より厳しい評価です。

test強度は `0.20, 0.40, 0.60, 0.80, 0.90, 1.00` です。0.90と1.00は学習範囲外です。

## 劣化定義

- `occlusion`: 正解楕円角に沿う連続区間を画像端まで黒く遮蔽。強度1.0は全面黒。
- `whiteout`: 全体露光増加と、連続区間の局所飽和・ブルーム。強度1.0は全面白。
- `black_crush`: 黒レベル閾値上昇とγ変換による暗部圧縮。強度1.0は閾値0.58・γ2.8で、全面黒とは限らない。

劣化画像に正解楕円形状の外周境界が残らないよう、局所マスクは画像端まで伸ばしています。
現在は定義監査済みの `paf_robustness_v3` だけを使用し、旧v1・v2の生成物は削除済みです。

## 主要ファイル

```text
config/research_base.json              Blender基礎CG条件
config/research_experiment.json        分割・劣化・CNN・RANSAC・評価条件
blender/generate_dataset.py            Blender/bpyデータ生成
paflab/degradations.py                 劣化モデル
paflab/model.py                        Tiny U-Net
paflab/train_cnn.py                    学習とvalidation選択
paflab/evaluate_cnn.py                 CNN + weighted RANSAC評価
paflab/reporting/summarize_robustness.py  曲線、AUC、クラスタ区間
paflab/reporting/validate_results.py      共有前の結果監査
paflab/reporting/build_report_artifact.py レポート正規データ生成
analysis/evaluate_dataset.py           古典法評価
analysis/report_source.sql              レポート表示用SQL
tests/test_degradations.py              劣化定義の回帰テスト
config/research_ood_base.json           未知姿勢・照明・背景のBlender条件
config/research_second_stage.json       第2段階のデータ・モデル幅・seed条件
paflab/camera_effects.py                撮像効果proxy
paflab/experiment_images.py             単体画像の劣化・中間処理・検出結果出力
paflab/experiment_gui.py                単体実験画像GUI
paflab/experiments/run_model_ablation.py       4幅×3 seed学習・評価
paflab/experiments/analyze_ransac_selection.py 候補順位とoracle上限の監査
paflab/reporting/summarize_second_stage.py クラスタ区間・モデル規模・診断集計
paflab/reporting/validate_second_stage.py  第2段階のID・条件・paired群監査
run_second_stage.py                      第2段階の一括再現
run_experiment_gui.py                    単体実験画像GUI起動
```

## 出力

```text
output/datasets/research_base_v1/       714基礎CG・GT・manifest・blend
output/datasets/paf_robustness_v3/      学習/評価画像、ラベル、方式別結果
output/experiments/paf_robustness_v3/   CNN重み、履歴、曲線、監査
output/experiments/paf_robustness_v3/report/
  artifact.json                         検証済み正規レポート入力
  report_data.json                      レポート表示用の監査済み集計
  report.html                           自己完結の対話的技術レポート
```

## 評価上の限界

- CG形状は1種類で、実写一般化は未検証です。
- 教師マスクは遮蔽された不可視部も含む完全楕円です。現在のCNNは「可視エッジだけの検出」ではなく構造補完も学習します。
- 劣化強度は画像変換の制御量で、実機の露光・センサ黒レベルへは未校正です。
- 95%区間は16カメラ条件をクラスタとして推定していますが、独立形状数の不足は補えません。

次の研究段階では、実写校正、複数PAF形状、可視弧と補完形状の分離、信頼度校正、幾何制約付き学習を優先します。
