# 学習元を比較する共通境界CNN

本書は外観変更ありの共通条件比較を記す。現在の1194M指定条件は[外観変更なし・投影拡張あり](NO_SURFACE.md)であり、本書の結果は固定対照として保持する。両学習元に採用済み多様形状モデルの基礎を使い、Canny＋輪郭別RANSACとZhang 2019型（再現実装）も同じtest画像で比較する。

研究の問いは「特定機体の詳細CADを使わず、合成PAFでの学習から未知機体の内周を欠損に強く検出できるか」。学習データ源には形状・元の撮影・材質の差が含まれるので、形状多様性だけの因果効果とはしない。

## 共通仕様

|項目|両CNNの共通仕様|
|---|---|
|CNN|Tiny U-Net16、ランダム初期化、482,737パラメータ|
|入力|480画像へ劣化を適用、グレー化して同じ輝度を3chへ複製、256へ縮小|
|教師|穴領域から約2pxの境界を生成し、既知の矩形遮蔽部分を除く。露光後の可視性は判定しない|
|投影・外観|元投影/拡張投影×元材質/拡張外観の4ビューを各25%で抽選|
|投影拡張|内周長径24〜180px、中心0.3〜0.7、軸比0.1〜1.0、画像端12pxの内周余白。機体全体の切り出しは許容|
|外観拡張|ノイズ・縞・格子の同一パラメータ分布。色の明暗も変わる。元の粗さ・金属度・照明は保持|
|矩形遮蔽|学習の矩形例だけ、開口横幅を基準にオンライン再抽選|
|学習数|497原画像、1epoch 1,988例。4ビューでepochの例数は増やさない|
|validation|それぞれの学習元の105原画像・735例。元投影・元外観のみ、固定劣化|
|学習|60epoch、batch16、AdamW、一定lr=0.001、weight decay=0.0001|
|損失|正例重み12のBCE＋Dice|
|seed|20260907 / 20260908 / 20260909|
|重み選択|学習元のvalidation loss最小。testでは選ばない|
|推論|確率0.5→形態学的中心線→確率重み付きRANSAC、最小支持12点|
|RANSAC|450仮説、距離2px、最小軸6px、最小軸比0.1、最大軸=画像対角×2、72角度bin、再推定3回、周長補正0|
|乱数|RANSAC seed20260716、各画像でOpenCV seed20261025|
|一様画像|未検出とし、分母に含める|

1194Mの拡張CGはtrain497原画像からだけ生成する。多様形状のtrainの投影・外観パラメータ列を同じ順番で対応させる。1194Mの物体座標は上端面を原点、上端外径を1として模様の空間尺度を揃える。原CADの単位のまま模様だけが細かくなることを避ける。元の照明・材質と元投影は各学習元のデータに従う。

多様形状の既存3seed重みは、設定・実装・入力hash、共通入口と従来入口の入力・教師・抽選状態の一致を確認して再利用する。1194Mはこの共通仕様で3seedを学習する。両者の評価は新しい共通入口で実行する。

## 実装と実行

- `render_1194m.py`：既存のCAD読込・投影・模様生成を使って1194Mの不足する3ビューを描画する。testを読み込んで学習画像に変換しない。
- `experiment.py/json`：学習元の4ビューを指定する薄いデータアダプタ。教師・遮蔽・Tiny U-Net・学習ループ・RANSACは既存実装を共有する。固定済み入力・重みを上書きしない。
- `report.py`：全画像のIoUを元GTから再計算し、全3seed・最小validation重み・全条件の順位と同率を照合する。
- `check_inputs.py`：共通入口の入力・教師・抽選状態の一致と、497組の拡張パラメータ・投影誤差を確認する。
- `figures.py`：通常と全劣化強度の比較図を作る。帯は固定3seedの最小〜最大で、信頼区間ではない。
- `PRESENTATION_CHANGES.md`：次の発表用の比較方法と変更理由。

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python-exit-code 1 --python research/paf_fair_comparison/render_1194m.py
.venv/Scripts/python.exe -u -X utf8 research/paf_fair_comparison/experiment.py run
.venv/Scripts/python.exe -X utf8 research/paf_fair_comparison/check_inputs.py
.venv/Scripts/python.exe -X utf8 research/paf_fair_comparison/report.py
.venv/Scripts/python.exe -X utf8 research/paf_fair_comparison/figures.py
```

`experiment.py prepare/train/evaluate`で段階を指定できる。`--sources 1194m`、`--seeds 20260907`で対象を絞れる。多様形状側の`train`は固定済みの既存重みを検証して参照する。途中までの学習履歴があるrunを暗黙に再学習しない。

## 結果と保全

全6モデルと全27,664推定の照合が完了。全13条件の成功率（IoU≥0.8、CNNは3seed平均）は次のとおり。

|評価集合|1194M学習CNN|多様形状学習CNN|Canny|Zhang 2019型（再現実装）|
|---|---:|---:|---:|---:|
|1194M|81.64%|71.34%|49.18%|46.91%|
|多様形状|56.78%|74.84%|53.00%|61.24%|

1194M通常は100.00%対99.11%。多様形状では全体で1194M学習CNNがCannyを上回り、半画面遮蔽では両幾何を上回るので、幾何を一括りにした順位にはならない。多様形状の平均IoUでは1194M学習CNNがZhangも上回る。全強度と指標を明示した結論は[確定報告](../../output/paf_fair_comparison_v1/REPORT.md)を参照する。

正本は`output/paf_fair_comparison_v1/`。`plan.json`に共通条件・入力/実装hash・分割と枚数・参照モデルを保存する。`runs/1194m/{seed}/`が新しい1194M学習重みと全履歴、`results/{学習元}/{seed}/test_{評価元}.json`が両testの全推定。集計は`REPORT.md`・`summary.json`・`scores.csv`、照合は`audit.json`。

1194M test112原画像×13条件=1,456例、多様形状test154原画像×13条件=2,002例。13条件は通常＋遮蔽/白飛び/黒つぶれ各0.25/0.5/0.75/1.0。成功はIoU≥0.8。失敗・未検出を除外しない。CNNは3seed平均と各seedを報告し、モデル統合ではない。

以前のmain構成1194M参考値と要因対照は、`output/paf_core_ablation_v2/`とその`backup/clean_gap_diagnosed_20261025/`に保全する。公平比較のバックアップは別の`output/paf_fair_comparison_v1/backup/`で管理する。

全て黒背景・単一開口CG。testは探索で参照済みで盲検ではない。期待した順位になるように条件や重みを選び直さない。幾何の正規出典はZhang, Pan, Ma, Sensors 2019, DOI `10.3390/s19235243`であり、Zhang 2019型は論文構成を参考にした再現実装である。
