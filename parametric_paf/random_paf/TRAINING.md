# 保存済み1,024画像の学習と独立評価

実験ID：`paf_outer_d_training_v2`。元画像は`paf_random_outer_d_v2`。上端外径D、下端外径1D〜3D、全高0.5D〜3D、穴径0.4D〜0.95D、肉厚0.02Dの単一PAFモデルを使う。

学習・最終評価・全770推定の整合性検査は完了。[結果](../output/paf_outer_d_training_v2/report/REPORT.md) と [推定例](../output/paf_outer_d_training_v2/report/predictions.png) を参照する。CNN3seedの平均主成功率は17.32%で、学習seed間のばらつきと12epochでの収束の限界を含めて解釈する。

## 分割と学習

元画像1,024枚をsplit_seed=2026102403で固定分割し、train716枚・validation154枚・test154枚とする。1画像ごとに寸法が異なるため、形状IDと寸法の組合せを区分間で共有しない。元データと元manifestは保持し、学習用manifestから同じ画像・ラベルを相対パスで参照する。

Tiny U-Net（base_channels=16）、入力256、12epoch、batch16、AdamW（学習率0.001、weight_decay0.0001）、BCE正例重み12＋Dice重み1を使用する。各540更新。学習seedは20260907/20260908/20260909の3本で、全てランダム初期化から学習する。1194Mや別仕様の学習済み重みは初期値に使わない。

各モデルはvalidation lossが最小のepochを採用する。testは学習・checkpoint選択・閾値選定に使わない。3seed全てを報告し、testの成績でseedを選び直さない。

## 評価

CNN3モデル、既存のCanny＋輪郭別RANSAC、Zhang 2019型（再現実装）を同じtest154枚で評価する。幾何2手法は学習seedを持たず、各1回の固定推定を使う。Canny閾値30/90と既存設定を固定し、このデータでの再調整は行わない。

主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px。IoUのみの成功、未検出、誤楕円も保存する。全方式256入力とし、画像ごとのOpenCV・NumPy乱数を同じ規則で固定する。test前にvalidation17画像を独立2プロセスで推論し、確率マップ（CNN）・推定楕円・指標の一致を確認する。

この評価は指定した寸法・描画分布内の未知寸法に対する推定性能である。1形状1画像であり、同じ形状を多視点で反復した評価ではない。単一寸法学習との比較や対象なし評価は含まないため、形状多様化の因果効果や対象有無判定は、この結果だけで結論しない。

## 構成・実行

`../train_random_paf.py`が分割・入力検査・凍結・学習・再現性検査・最終評価・集計を実行する。学習器、CNN構造、検出器、共通評価は既存実装を呼び出す。画像生成や新しい形状の追加は含まない。設定の正本は`training.json`。

```powershell
& .venv/Scripts/python.exe -u -X utf8 -B parametric_paf/train_random_paf.py run
```

出力は`../output/paf_outer_d_training_v2/`。`dataset/manifest.json`に分割、`freeze.json`にコード・入力・設定・環境のhash、`models/<seed>/`に採用重みと学習履歴、`reproducibility/`に独立プロセスでの推論記録、`evaluation/`に全画像の推定と判定、`report/REPORT.md`と`summary.json`に結果と整合性検査の根拠を保存する。

再開時は保存済みの入力・完了履歴・原記録の対応を確認して再利用する。途中の学習重みを完成モデルとして扱わず、完成した重みを上書きしない。`records.jsonl`は画像単位の研究原記録として保存する。
