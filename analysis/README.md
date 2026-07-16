# `analysis` モジュール構成

このディレクトリには検出アルゴリズム本体と、方式共通の評価だけを置く。

- `ellipse_baseline.py`: Canny前処理、輪郭楕円、正解楕円、IoU評価
- `ellipse_ransac.py`: 点群・輪郭ごとのRANSAC楕円推定
- `zhang_arc_detector.py`: Zhang et al. (2019)型の弧統合ベースライン
- `evaluate_dataset.py`: 古典方式を同一形式でデータセット評価するCLI
- `compare_methods.py`: 方式別CSVの単純比較
- `aamed_adapter.py`: 任意の外部AAMEDラッパー

単画像pilot専用CLIや採用しなかった方式専用CLIは置かない。
発表図、レポート、学習処理は `paflab/` 側で管理する。
