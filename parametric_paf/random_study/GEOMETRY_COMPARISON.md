# ランダム寸法CNNと幾何手法の追加比較

実験ID：`paf_random_geometry_v1`。ユーザー指定の比較対象は既存のCanny＋輪郭別RANSACとZhang 2019型（再現実装）。CNNの学習や形状数比較は追加しない。

全件評価・集計・整合性検査は完了。[比較結果](../output/paf_random_geometry_v1/report/REPORT.md) を参照する。主テストはランダム寸法CNN平均54.92%、Canny71.35%、Zhang 2019型（再現実装）69.27%。

## 比較条件

- CNNの独立評価と同じ画像・ラベルを使用し、全方式を256×256入力に統一する。
- 主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px。共通評価関数を使う。
- 幾何のCanny閾値は既存validationの3候補比較で選定済みの30/90。その他も保存済み設定を使い、テストを使う再調整はしない。
- CNNはsingle/randomの各3学習seedの保存済み原記録を参照する。幾何は学習seedを持たず、各画像のNumPy・OpenCV乱数をCNNと同じ規則で固定して1回ずつ評価する。同じ幾何推定を3回分の独立反復として数えない。
- 幾何2手法とも、固定したvalidation17画像を独立2プロセスで推論し、楕円と評価値の完全一致を確認してから全件評価する。検査した環境と画像の確認であり、異なる環境での一致を保証しない。
- 対象は共通validation192枚、対象あり独立テスト1,584枚、対象なし64枚。各幾何手法2,224推定、計4,448推定を保存する。
- CNNとの差は同じ形状・撮像条件で対応を取り、CNN学習seedごとに形状平均差を保存する。対象なしの誤検出は別評価とする。

これはCNNの独立評価後に依頼された追加比較である。幾何設定の選定はvalidationだけで完了しており、この追加比較の結果を選定や調整には使わない。

## 実装と解釈

`../compare_geometry_random.py`が固定記録・推論再現性検査・全件評価・集計を担当する。既存の`run_geometry`と共通評価を呼び出し、学習器・検出器・既存CNN結果を変更しない。固定記録は`geometry_comparison.json`、出力は`../output/paf_random_geometry_v1/`。

Cannyは同心・相似な内外周候補対の事前情報を使う統制方式であり、先行研究の再現実装ではない。Zhang 2019型（再現実装）はZhang, Pan, Ma, Sensors 2019, DOI `10.3390/s19235243`の処理構成を参考にした実装とPAF固有後処理であり、著者コードの移植・完全再現ではない。両者との結果を幾何手法全般の性能に読み替えない。

## 実行

リポジトリ直下、既存venvで実行する。

```powershell
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/compare_geometry_random.py freeze
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/compare_geometry_random.py repro --method canny
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/compare_geometry_random.py evaluate --method canny
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/compare_geometry_random.py repro --method zhang
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/compare_geometry_random.py evaluate --method zhang
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/compare_geometry_random.py report
```

全件完了後、`report/REPORT.md`に比較表、`summary.json`に固定記録・原記録のhash、`shape_results.csv`に形状別の成功・未検出・誤楕円、`cnn_minus_geometry.csv`に対応差を保存する。`records.jsonl`は画像単位の研究原記録として残す。
