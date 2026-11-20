# 出力の見方

検出した位置を見るには[overlay.png](ellipse_example/overlay.png)、全体の成功率を見るには[REPORT.md](reference_verification/REPORT.md)を開いてください。

## 画像1枚の推論結果

`ellipse_example/`は、多様形状で学習済みのseed20260907モデルを1194Mの通常画像1枚へ適用した例です。

|ファイル|見る内容|
|---|---|
|[overlay.png](ellipse_example/overlay.png)|紫の線が推定したPAF内周楕円。正解線の重ね描きではない|
|[probability.png](ellipse_example/probability.png)|CNNの境界確率。白いほど内周境界らしいと推定した画素|
|[result.json](ellipse_example/result.json)|推定楕円の中心・軸長・角度、使用したモデル、検出状態|
|`probability.npy`|境界確率の数値配列。Pythonで分析する場合に使う|

`result.json`の`ellipse`は元画像座標、`ellipse_256`はCNN入力の256px座標です。軸長は半径ではなく全長です。`detected`は楕円が得られたことを表し、正解とのIoUによる成功判定ではありません。

## 全体の成績

`reference_verification/REPORT.md`に、1194M学習と多様形状学習の各モデルを、両test集合に適用した成功率をまとめています。`diversified`は多様形状を意味します。成功条件は楕円内部のIoUが0.8以上、CNNの成績は3seedの平均です。

「基本条件」は元の白飛び条件、「発表条件」は白飛び4強度を強化条件へ置き換えた結果です。今回新しく学習した成績ではなく、保存済みモデルを新構成で再推論した結果です。

細かい条件別集計は各`seed*/verification.json`、1画像ごとの推定は`seed*/predictions.jsonl`、全体の機械読込用集計は`summary.json`にあります。

## 入力の確認記録

|フォルダ|確認していること|
|---|---|
|`training_data_check/`|学習画像と教師の対応、分割、ファイルhash|
|`standalone_training_check/`|`paf_rebuild/data/`の入力が既存教師と一致し、単独で読み込めること|

これらは学習曲線やモデル精度ではありません。通常、検出結果を見るだけなら開く必要はありません。

## 多様形状学習の実装状況

多様形状の画像・ラベル、保存済み3seedのモデル、入力生成・教師生成・損失関数は揃っています。学習済み重みは`../models/diversified/`、学習データは`../data/diversified/`です。

重みを更新する学習ループと`train.py`は未実装です。新しい形状のCG生成も未実装です。そのため、この構成で一から学習した学習曲線・checkpointはまだ出力されません。現在の仕様と次の作業は[HANDOFF.md](../docs/HANDOFF.md)、操作方法は[README.md](../README.md)を参照してください。
