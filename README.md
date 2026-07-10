# PAFリング楕円検出 基礎実験システム

Blenderで条件付きCG画像と3Dモデル由来の正解楕円を生成し、画像全体を対象とした古典法の楕円検出を同一条件で評価するための基礎システムです。

## 現在できること

- 複数のカメラ視点と照明条件をJSONで指定
- BlenderによるCG画像の一括生成
- `GT_INNER_RING`の3D頂点を画像座標へ投影
- Canny、輪郭抽出、楕円フィットによる全画像評価
- Canny輪郭ごとのRANSAC楕円推定
- 面積や画像内位置を使わない楕円候補の順位付け
- 全候補、トップ1、正解楕円のJSON・画像保存
- 2方式の中心、長短軸、角度、IoUのCSV比較

## 初期実験を実行する

AnacondaのPythonから次を実行します。

```powershell
C:\Users\hito\anaconda3\python.exe -B run_experiment.py
```

既定設定では、カメラと照明をPAF座標系で独立に変化させた126条件を生成して評価します。

- カメラ軸傾き：`0、15、30、45、60、75°`
- カメラ方位：`0、90、180、270°`
- 照明軸傾き：`0、15、30、45、60、75°`
- 照明方位：`0°`固定
- カメラ距離：`41`固定
- ライト：太陽角0°の`SUN`による平行光線

カメラ傾き0°では周囲方位が同一位置になるため、方位0°だけを生成します。カメラは21位置、照明は6条件、合計126条件です。OBJローカルのPAF軸`+Z`はBlenderワールド`+Z`へ正規化されます。

レンダリング済みデータだけを再評価する場合：

```powershell
C:\Users\hito\anaconda3\python.exe -B run_experiment.py --evaluate-only
```

CG生成だけを行う場合：

```powershell
C:\Users\hito\anaconda3\python.exe -B run_experiment.py --render-only
```

楕円検出の各画像処理段階を確認する場合：

```powershell
C:\Users\hito\anaconda3\python.exe -B analysis\export_detection_steps.py
```

元画像、グレースケール、平滑化、Canny、輪郭、全楕円候補、トップ1、正解比較が`output/detection_steps/`へ保存されます。

## 設定ファイル

- `config/factorial_experiment.json`: カメラ21位置×照明6条件の主実験
- `config/azimuth_360_experiment.json`: カメラ基準30°刻み360°照明実験
- `config/pilot_experiment.json`: 旧3視点×2照明パイロット実験
- `config/baseline.json`: Canny、RANSAC、候補評価、成功判定IoU

正解対象は、PAF内周面のカメラ側エッジです。

```json
"target_ring": {
  "z": 4.73574,
  "radius": 5.41875,
  "expected_vertex_count": 88
}
```

## 出力

既定実験では`output/datasets/factorial_v1/`以下に生成されます。

```text
images/                  CG画像
labels/                  3D頂点から投影した正解点
results/contour_fit/     Canny+fitEllipseの結果
results/canny_ransac/    Canny+RANSACの結果
results/comparison.csv   2方式の条件別比較
results/comparison.json  2方式の全体集計
manifest.json            画像・ラベル・条件の対応表
dataset_preview.blend    正解頂点グループを確認できるシーン
```

## 評価上の注意

正解ラベルは評価のみに使用し、検出候補の選択には使用しません。現在の成功判定はトップ1候補と正解楕円のIoUが0.8以上かどうかです。この閾値は`config/baseline.json`で変更できます。

現在は次の2方式を同じ画像・正解ラベルで比較します。

1. `contour_fit`: Canny → 輪郭 → 全輪郭点で`fitEllipse`
2. `canny_ransac`: Canny → 輪郭候補 → RANSAC → インライアで再フィット

RANSAC本体は点群と任意の重みを受け取ります。将来CNNを接続するときは、CNNのリング確率を点の重みとして渡せる構造です。
