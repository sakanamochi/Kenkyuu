# PAF内周検出

直接実行するPythonファイルは`scripts/`、変更する設定は`configs/`にまとめています。システム本体は`src/paf/`にあります。

## 現在できること

保存済みTiny U-Net16の重みを指定し、正方形画像1枚の内周境界確率を出力します。1194M学習・多様形状学習の両モデルに同じ入口を使えます。現在の出力は境界確率であり、楕円の検出結果ではありません。楕円推定、性能評価、学習処理は未実装です。

このフォルダ内のコードは旧`research/`・`paf_ring_detection/`をimportしません。必要なものはPython環境、入力画像、指定した重みだけです。初期設定は既存の多様形状モデルseed20260907を参照します。旧実験の監査JSONや元ソースは実行時に読み込みません。

## ファイルの役割

|場所|役割|
|---|---|
|`scripts/predict.py`|利用者が直接実行する推論入口|
|`configs/predict.json`|モデル重みと計算装置の指定|
|`src/paf/core/model.py`|学習・推論で共用するTiny U-Net16|
|`src/paf/core/images.py`|画像読込、モノクロ化、256px化、正規化|
|`src/paf/inference/predictor.py`|重み読込と境界確率の推定|
|`tests/test_core.py`|入力仕様とモデル構造の確認|
|`docs/HANDOFF.md`|現在の仕様、確認結果、次の作業|
|`requirements*.txt`|実行に必要なPythonパッケージ|

学習実装の配置は`src/paf/training/`、直接操作する学習入口は`scripts/train.py`とします。これらは学習を構築する段階で作成します。

## 推論の実行

プロジェクト直下の`Kenkyuu`から、既存venvを使う例です。`input.png`はご自身の入力画像へ置き換えます。

```powershell
.venv/Scripts/python.exe -X utf8 -B paf_rebuild/scripts/predict.py --image input.png --output paf_rebuild/outputs/example
```

このフォルダを別の場所へ配置した場合は、そのPython環境で`scripts/predict.py`を実行します。入口はカレントディレクトリにかかわらず自身の`src/`と既定設定を見つけます。

モデルを変更する場合は`configs/predict.json`の`checkpoint`を編集するか、`--checkpoint`で指定します。設定内の相対パスは設定ファイルの場所を基準に解決します。コマンド引数の相対パスは実行場所が基準です。推論時の設定は、学習設定や過去の実験名から暗黙に決めません。

`device`は`auto`、`cpu`、`cuda`です。`auto`はCUDAを利用できればGPU、それ以外ではCPUを使います。実行時は`--device cpu`等で指定を変更できます。既存結果との照合には同じ計算装置と環境を使います。

入力はuint8の正方形グレースケール画像または3ch BGR画像です。透明度付き画像・長方形画像の自動補正は行いません。

## 出力

|ファイル|内容|
|---|---|
|`probability.npy`|256×256のfloat32境界確率。数値比較に使う|
|`probability.png`|確率0〜1を黒〜白へ写した表示用画像|
|`result.json`|入力・モデルのパスとhash、採用epoch、計算装置、出力状態|

出力先には新しいフォルダ名を指定します。存在するフォルダは上書きしません。一様画像は`insufficient_image_evidence`とし、result.jsonだけを保存します。境界確率の出力成功は、楕円検出の成功を意味しません。

## 検証

```powershell
.venv/Scripts/python.exe -X utf8 -B -m unittest discover -s paf_rebuild/tests -v
```

構成・入力仕様の検証と、旧実装との数値照合は別に扱います。比較範囲と未検証事項は`docs/HANDOFF.md`を参照してください。
