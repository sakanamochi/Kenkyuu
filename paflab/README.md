# `paflab` モジュール構成

現行の研究処理は、役割ごとに次のように分ける。

## データ生成・前処理

- `prepare_stress_dataset.py`: 基礎CGから学習・検証・テスト用の制御劣化データを作る
- `prepare_diagnostic_dataset.py`: 黒矩形、白飛び、黒つぶれ、フレア診断画像を作る
- `dataset.py`: PyTorch Dataset
- `degradations.py`: 初期ロバスト性実験の劣化モデル
- `camera_effects.py`: 撮像効果proxy
- `labels.py`: 内周リング教師マスク
- `image_io.py`: Windowsの日本語パス対応画像入出力

## CNN

- `model.py`: Tiny U-Net
- `train_cnn.py`: 学習とvalidationによるモデル選択
- `evaluate_cnn.py`: CNN尤度とweighted RANSACによる楕円評価

## 集計・品質確認・図版

これらはすべて `reporting/` に置く。

- `summarize_robustness.py`, `summarize_second_stage.py`: 統計集計
- `validate_results.py`, `validate_second_stage.py`: 共有前の整合性検査
- `build_report_artifact.py`, `build_second_stage_report.py`: 技術レポート生成
- `build_summary_figures.py`: 第2段階の研究結果図
- `build_first_progress_figures.py`: 第1回進捗報告用の素材図
- `build_literature_comparison.py`: Zhang型とCNNの文献ベースライン比較図

## 補足実験

`experiments/` は主パイプラインから分離したアブレーション専用である。

- CNNモデル幅・seed比較
- CNN RANSACの周長補正比較
- 旧Canny方式の内周候補選択監査
- 推論速度計測

通常の研究実行では、ルートの `run_research.py`、`run_second_stage.py`、
`run_literature_comparison.py` を入口として使用し、個別モジュールを直接組み合わせない。
