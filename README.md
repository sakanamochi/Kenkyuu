# PAF内周リング検出

PAFのCG画像から内周楕円を検出し、次の3方式を比較する個人研究用コードです。

- 標準Canny + 輪郭分離 + 共通RANSAC
- Zhang 2019型（再現実装）
- CNNリング尤度 + weighted RANSAC

成功条件は、推定楕円とCG正解楕円のIoUが0.80以上です。

## ファイル構成

```text
run.py                         唯一の実行入口
config/
  experiment.json             学習・検出・評価条件
  render_base.json            学習用CG条件
  render_ood.json             OOD用CG条件
assets/1194M_3/               PAFのOBJ・MTL
blender/generate_dataset.py    CGと正解内周頂点を生成
paf_ring_detection/
  data.py                      画像、ラベル、PyTorch Dataset
  effects.py                   遮蔽・白飛び・黒つぶれ
  prepare.py                   学習・診断データ作成
  geometry.py                  楕円IoUなどの共通計算
  train.py                     CNN学習
  evaluate.py                  3方式の共通評価
  report.py                    summary.jsonと比較図
  methods/
    cnn.py                     Tiny U-Net
    ransac.py                  weighted RANSAC
    canny_contour_ransac.py    標準Canny・輪郭分離・内周選択
    cnn_ransac.py              CNNとRANSACの接続
    zhang2019.py               Zhang 2019型（再現実装）の論文コア
    zhang2019_paf.py           PAF固有の候補評価・内周選択
tests/                         研究処理を壊していないか確認
output/                        データ・重み・結果（Git管理外）
```

## セットアップ

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

## 実行

すべてを最初から実行します。

```powershell
.venv\Scripts\python.exe run.py
```

必要な段階だけ実行することもできます。

```powershell
# 既存データと学習済み重みから評価・集計
.venv\Scripts\python.exe run.py --stages evaluate report

# 5枚ずつで評価処理を確認
.venv\Scripts\python.exe run.py --stages evaluate --limit 5

# 1 epochだけ学習を確認
.venv\Scripts\python.exe run.py --stages train --epochs 1
```

処理順は次の5段階だけです。

```text
render → prepare → train → evaluate → report
```

### Canny統制方式

Gaussian平滑化と標準Cannyでエッジを抽出し、連結輪郭ごとに分離して
`methods/ransac.py` の共通RANSACへ渡します。PAFの既知形状として、
同心・相似な内外周候補対の小さい側だけを内周として選びます。
この方式は先行研究の再現ではなく、CNNによるエッジ抽出の寄与を見る統制比較です。

## テスト

```powershell
.venv\Scripts\python.exe -m pytest -q
```

テストは研究の妥当性そのものではなく、コード整理によって次を壊していないか確認します。

- 撮像効果の向きと強度
- 楕円IoU
- weighted RANSAC
- CNNの入出力
- Zhang型の象限・凸性、弧ペア制約、中心整合、中心光線距離
- Zhang型の論文コアとPAF固有後処理の分離

## 出力

通常評価では大量の中間画像やサンプル別JSONを保存しません。

```text
output/results/
  ood/
    canny_contour_ransac.csv / .json
    zhang2019.csv / .json
    cnn_ransac.csv / .json
  diagnostic/
    canny_contour_ransac.csv / .json
    zhang2019.csv / .json
    cnn_ransac.csv / .json
  summary.json
  comparison.png
  diagnostic_by_severity.png
```

## Zhang型の出典

Limin Zhang, Wang Pan, Xianghua Ma,
“Real-Time Docking Ring Detection Based on the Geometrical Shape for an
On-Orbit Spacecraft,” Sensors, 19(23), 5243, 2019.
DOI: `10.3390/s19235243`

`methods/zhang2019.py` は、勾配象限と凸性による弧選別、隣接象限の
位置・接線制約、三弧の中心整合、最小二乗楕円、弧上点の中心光線距離による
適合度検証を、論文の処理構成に沿って実装したPython再現実装です。
全周エッジ支持と、内包関係があれば最も内側を、
なければ全周支持率最大を選ぶ最終選択は論文コアではないため、
`methods/zhang2019_paf.py` にPAF固有後処理として分離しています。
著者コードの移植ではないため、成果物では必ず
「Zhang 2019型（再現実装）」と表記します。
