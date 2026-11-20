# 白飛び強度の追加評価

既存の境界CNN 2方式と幾何2方式について、露光増加・飽和・局所的なにじみの強度範囲を広げた耐性評価を行う。CNNはseed20260907の採用済み重みを固定する。再学習や既存比較の上書きは行わない。

## 構成

- `effect.py`：表示強度0～1を、指定した最大内部強度まで拡張する。最大内部強度1では既存`effects.py`と全画素が一致する。
- `config.json`：検証する最大内部強度、成功率低下の基準、テスト強度、学習seedを固定する。
- `run.py`：検証集合で強度を選択し、その後に別のテスト集合を4方式で評価する。画像・正解・重み・設定の対応を記録する。
- `report.py`：検証結果とテスト結果を元正解から照合し、CSV、JSON、PNG、報告文書を保存する。

## 実行

リポジトリ直下で以下を実行する。

```powershell
.venv/Scripts/python.exe research/paf_whiteout_strength/run.py calibrate
.venv/Scripts/python.exe research/paf_whiteout_strength/run.py test
.venv/Scripts/python.exe research/paf_whiteout_strength/report.py
```

保存先は`output/paf_whiteout_strength_v1/`。採用する最大内部強度は`calibration.json`の`selected_maximum_strength`。検証用画像は各105枚、テスト原画像は1194M 112枚・多様形状154枚。強度ごとに全画像を評価し、未検出も失敗に含める。

## 比較の範囲

検証用画像における4方式平均の低下量から、十分な劣化が生じる最小の候補強度を選ぶ。方式の順位や方式間の差を最大化する選択は行わない。検証集合・テスト集合とも研究で参照済みの画像であり、新規盲検ではない。強度選択にテスト結果は使用しない。

今回のCNN結果は1seedであり、既存発表資料の3seed平均とは区別する。従来範囲で学習したモデルの追加耐性評価で、強めた白飛びによる再学習の効果は未検証。光源位置やレンズ内反射の物理モデルを導入したものではない。

Zhang方式は「Zhang 2019型（再現実装）」と表記する。正規の出典はZhang, Pan, Ma, Sensors 2019, DOI 10.3390/s19235243。

## 3seed平均の追加評価

`three_seed.py`は、最大内部強度2.0と入力画像を固定し、学習seed20260907・20260908・20260909の平均を求める。既存seed07・幾何方式の推定はhashを照合して再利用し、seed08・09のCNN推定だけを追加する。再学習や強度の再選択は行わない。

```powershell
.venv/Scripts/python.exe research/paf_whiteout_strength/three_seed.py evaluate
.venv/Scripts/python.exe research/paf_whiteout_strength/three_seed.py report
```

保存先は`output/paf_whiteout_strength_3seed_v1/`。1seedの保存先は維持する。`scores_mean.csv`に全強度の平均・最小最大、`scores_by_seed.csv`に各seedの成功数・成功率・平均IoUを保存する。グラフの帯は3seedの最小～最大であり、信頼区間ではない。幾何方式は固定1回分を表示し、同一結果の複製を独立した3反復とは扱わない。
