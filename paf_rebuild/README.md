# PAF内周検出

保存済みTiny U-Net16の境界確率から中心線を抽出し、重み付きRANSACでPAF内周楕円を求めます。1194M学習・多様形状学習の両モデルに同じ入口を使えます。学習入力・教師・損失も実装済みです。学習ループと幾何比較器は未実装です。

## ファイルの役割

|場所|役割|
|---|---|
|`scripts/predict.py`|画像1枚の推論|
|`scripts/check_training_data.py`|学習前の入力・教師・分割・hash確認|
|`scripts/verify_reference.py`|保存済み評価結果との照合|
|`configs/predict.json`|重み・計算装置・RANSAC設定|
|`configs/training_data.json`|学習用ビュー・白飛び倍率|
|`configs/reference.json`|照合する画像・重み・結果の参照先|
|`data/`|分割manifest・画像・正解ラベル|
|`models/`|両学習元×3seedの重み|
|`references/`|固定比較の保存済み推定結果|
|`src/core/`|モデル、画像変換、撮像効果、楕円計算、ファイル読込|
|`src/inference/`|重み読込と推論|
|`src/training/`|学習入力・可視境界教師・損失|
|`src/evaluation/`|楕円IoUと保存結果の照合|
|`tests/`|入力・推論・学習教師の単体テスト|
|`docs/TRAINING_DATA.md`|学習仕様|
|`docs/HANDOFF.md`|検証結果・制約・次の作業|
|`requirements*.txt`|実行環境の依存パッケージ|

旧`research/`・`paf_ring_detection/`はimportしません。単体推論に必要なのはPython環境・入力画像・重み・推論設定です。照合と学習入力確認では、それぞれの設定が参照するデータも必要です。

実行時に必要なファイルはこのフォルダ内に配置します。`data/`・`models/`・`references/`・`outputs/`はGit管理外です。別の場所へ配置する場合は、コードと一緒にこれらも保存してください。入力とモデルの照合用hashは`data/inventory.json`にあります。

## 推論の実行

`Kenkyuu`直下から既存venvを使う例です。`input.png`を入力画像へ置き換えてください。

```powershell
.venv/Scripts/python.exe -X utf8 -B paf_rebuild/scripts/predict.py --image input.png --output paf_rebuild/outputs/prediction_001
```

入力は空でない正方形uint8のグレーまたは3ch BGR画像です。モノクロ256px・同一輝度3ch・[-1,1]へ変換します。透明度付き画像や長方形画像は受け付けません。

重みは`configs/predict.json`の`checkpoint`または`--checkpoint`で指定します。既定は多様形状モデルseed20260907です。`device`は`auto`・`cpu`・`cuda`で、`--device`で変更できます。`auto`はCUDAが使えればGPU、それ以外はCPUです。

設定内の相対パスは設定ファイル基準、コマンド引数の相対パスは実行場所基準です。入口は自身の場所から`src/`と既定設定を見つけます。

幾何の既定値は確率0.5、RANSAC450仮説、距離2px、最小支持12点、最小軸6px、最小軸比0.1、最大軸は画像対角長の2倍、角度72bin、再推定3回、周長補正なしです。点群抽選seed20260716、画像ごとのOpenCV seed20261025を使います。

## 推論出力

出力先には新しいフォルダを指定します。既存フォルダは上書きしません。

|ファイル|内容|
|---|---|
|`probability.npy`|256×256のfloat32境界確率|
|`probability.png`|確率0〜1を黒〜白へ写した8bit画像|
|`overlay.png`|元画像座標で楕円を重ねた画像|
|`result.json`|楕円、検出状態、入力・重みのパスとhash、epoch、計算装置、幾何設定|

JSONの`ellipse`は元画像座標、`ellipse_256`は256px座標です。形式は`[[中心x, 中心y], [軸1の全長, 軸2の全長], 角度（度）]`です。

一様画像は`insufficient_image_evidence`で確率ファイルを出しません。支持された楕円がなければ`no_supported_prediction`、あれば`detected`です。検出状態と正解IoUによる成功判定は別です。

## 学習入力と検証

学習仕様は[TRAINING_DATA.md](docs/TRAINING_DATA.md)、確認済み結果と未検証範囲は[HANDOFF.md](docs/HANDOFF.md)を参照してください。

学習元ごとの入力・教師・分割・hashを確認するコマンドです。全5,446例を処理して`verification.json`を保存します。重みの更新は行いません。

```powershell
.venv/Scripts/python.exe -u -X utf8 -B paf_rebuild/scripts/check_training_data.py --output paf_rebuild/outputs/training_data_check_001
```

単体テストは次のコマンドで実行します。

```powershell
.venv/Scripts/python.exe -X utf8 -B -m unittest discover -s paf_rebuild/tests -v
```

保存済み6モデルを、白飛びを強化した発表用13条件で再推論するコマンドです。`--seeds 20260907`を加えると、そのseedの両モデルだけを照合します。

```powershell
.venv/Scripts/python.exe -u -X utf8 -B paf_rebuild/scripts/verify_reference.py --output paf_rebuild/outputs/reference_check
```

`predictions.jsonl`に推定、`verification.json`に楕円・IoU・成功判定の不一致件数と使用ファイルhashを保存します。不一致があれば終了コードを失敗とします。いずれの確認コマンドも、新しい出力フォルダを指定してください。
