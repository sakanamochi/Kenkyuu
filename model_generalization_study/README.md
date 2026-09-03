# CNNモデル依存性の予備評価

既存の研究環境を変更せず、次の2点だけを独立して試すための専用ディレクトリです。

1. PAFらしい形状をJSONパラメータからOBJとして生成する
2. 1194M_3だけで学習した既存CNNを、未学習の比較モデルで評価する

## 比較モデルの位置づけ

最初の比較モデル `comparison_a_sparse_rib` は、実在機の忠実な再現ではありません。
PAFに共通すると考えられる「円形内周、円錐台状の外殻、放射状リブ、ラッチ、
内側ガイドポスト」を持つ合成モデルです。

公平な初回比較のため、1194M_3と次を揃えています。

- 検出対象の内周半径: 5.41875
- カメラ20条件、照明8条件、背景3条件（計480枚）
- 画像寸法、CNN入力寸法、学習済み重み、weighted RANSAC設定

一方、外径、奥行き、リブ数、ラッチ数、ガイドポスト数と配置、材質色は変更しています。
したがって、主に周辺形状の変更に対するCNNのモデル依存性を見る予備実験です。

## 構成

```text
model_generalization_study/
  config/
    experiment.json             評価対象と既存環境への参照
    experiment_comparison_c.json
    model_comparison_a.json     形状パラメータ
    render_comparison_a.json    1194M_3 OOD評価と同じ撮像条件
    render_comparison_c.json    比較Cの撮像条件と基準モデル共通の中心座標
  blender/                      Blender MCPで比較Cを制作した段階別スクリプト
  generate_parametric_paf.py    OBJ・MTL・生成メタデータを作成
  evaluate.py                   既存CNN + weighted RANSACで評価
  report.py                     対応あり比較と図表を作成
  run.py                        専用実験の実行入口
  tests/                        生成形状と正解リングの検査
  output/                       生成モデル・画像・結果（実行時に作成）
```

既存の `assets/`、`output/`、`paf_ring_detection/` は読み取るだけで、
この実験の生成物はすべて本ディレクトリの `output/` 以下へ保存します。

## 実行

プロジェクトルートから実行します。

```powershell
.venv\Scripts\python.exe model_generalization_study\run.py
```

段階別にも実行できます。

```powershell
# OBJ生成だけ
.venv\Scripts\python.exe model_generalization_study\run.py --stages model

# 先頭5条件で描画・評価の動作確認
.venv\Scripts\python.exe model_generalization_study\run.py `
  --stages model render evaluate report --limit-render 5 --limit-evaluate 5

# 生成器の単体検査
.venv\Scripts\python.exe -m unittest discover `
  -s model_generalization_study\tests -v
```

## 主な出力

```text
output/generated_models/comparison_a.obj
output/generated_models/comparison_a.mtl
output/generated_models/comparison_a.json
output/datasets/comparison_a/
output/results/comparison_a_cnn.csv
output/results/summary.json
output/results/model_dependency_comparison.png
output/mcp_model/comparison_c_from_1194M_3.blend
output/mcp_model/comparison_c_from_1194M_3.obj
output/datasets/comparison_c/
output/results_comparison_c/summary.json
```

`summary.json` には成功率差だけでなく、同じ撮像条件ごとの勝敗を使った
McNemarの正確検定と、成功率差の対応ありブートストラップ95%信頼区間も保存します。

## 初回評価結果（2026-07-30）

480組の同一撮像条件で、1194M_3だけを学習したCNNを再学習せず比較しました。

| 方式 | 1194M_3 | 比較A | 成功率差 |
|---|---:|---:|---:|
| CNN + weighted RANSAC | 384 / 480（80.0%） | 210 / 480（43.8%） | −36.3 pt |
| Zhang 2019型（再現実装） | 265 / 480（55.2%） | 99 / 480（20.6%） | −34.6 pt |

- CNNの対応あり95%信頼区間: −40.6〜−32.1ポイント
- Zhang 2019型（再現実装）の対応あり95%信頼区間: −39.4〜−29.8ポイント
- Zhang 2019型（再現実装）は348 / 480枚で楕円候補を生成できなかった
- Zhang型の候補中に正解があった99枚は最終選択でもすべて成功したため、
  主な低下要因は内周選択ではなく弧からの候補生成段階だった

両データセットの正解楕円は全480組でIoU 1.000でした。内周リングの画面上位置と
大きさを実質同一にした後でも両方式の性能が低下しました。この条件ではCNNの
モデル依存性を支持しますが、「古典方式なら形状変更に強い」という仮説は
Zhang 2019型（再現実装）については支持されませんでした。ただし、比較Aは
合成モデル1種類だけなので、一般的な結論には複数形状と実在機CADでの追試が必要です。

## Blender MCP高詳細モデルでの再評価（比較C、2026-07-30）

比較Aとは別に、ユーザー作成の `1194M_3.obj` を変更せず基礎形状として保持し、
Blender MCP経由で外周レール、機器ポッド、センサー、配線、パネル、ラッチなどを
追加した比較Cを制作しました。元形状は `.blend` 内の非表示コレクションにも保存し、
比較用の追加形状は編集可能な別コレクションへ分離しています。

内周リング形状、カメラ20条件、照明8条件、背景3条件は基準データと共通です。
付加物による外接境界中心の変化を撮像差へ混入させないため、比較Cの生成時には
1194M_3と同じモデル中心 `[0.0, 0.0, 0.3177365]` を明示しています。正解楕円の
対応あり幾何監査は全480組で平均・最小IoUともに1.000でした。

| 方式 | 1194M_3 | 比較C | 成功率差 |
|---|---:|---:|---:|
| CNN + weighted RANSAC | 384 / 480（80.0%） | 337 / 480（70.2%） | −9.8 pt |
| Zhang 2019型（再現実装） | 265 / 480（55.2%） | 56 / 480（11.7%） | −43.5 pt |

- CNNの対応あり95%信頼区間: −12.5〜−7.3ポイント、McNemar正確検定
  `p = 1.42e-14`
- Zhang 2019型（再現実装）の対応あり95%信頼区間:
  −48.1〜−39.0ポイント、McNemar正確検定 `p = 6.29e-59`
- CNNの差は主にカメラ傾斜67度で発生し、1194M_3の96 / 96に対して比較Cは
  49 / 96だった
- カメラ傾斜82度は両モデルともCNN成功0 / 96であり、モデル差とは別の共通限界である
- Zhang 2019型（再現実装）は比較Cで232 / 480枚に楕円候補を出したが、
  最終成功は56枚だった

比較CでもCNNのモデル依存性は統計的に確認されました。一方、成功率の絶対値はCNNが
比較Cでも70.2%を維持し、Zhang 2019型（再現実装）は11.7%まで低下しました。
したがって今回の1モデル比較は「CNNは未知モデルで低下する」を支持しますが、
「古典手法はモデル変更に強い」は支持しません。比較Cは実在する別機体CADではなく、
複数の外観要素を同時に追加した研究用モデルであるため、部位ごとの因果特定には
1因子ずつ除去するアブレーションが必要です。

比較Cだけを再実行する場合は、次を使います。

```powershell
.venv\Scripts\python.exe model_generalization_study\run.py `
  --experiment-config model_generalization_study/config/experiment_comparison_c.json `
  --stages render evaluate report --methods cnn zhang2019
```

## パラメトリック生成で変更できるもの

`config/model_comparison_a.json` の次の値を変更すると、別形状を生成できます。

- 内周・外周半径、前面・後面位置、内周リップ厚
- リブ数、幅、高さ、半径方向の範囲
- ラッチ数と寸法
- ガイドポスト数、配置半径、寸法
- 円周分割数と材質色

初回評価では複数パラメータを同時に変えて「モデルが変わると落ちるか」を確認します。
どの形状要因が効いたかを特定するには、次段階で1因子ずつ変えたアブレーションが必要です。
