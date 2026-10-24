# ランダムPAFデータセット

上端外径Dを基準に、PAFの寸法を画像ごとに独立一様抽選して生成する。

|項目|仕様|
|---|---|
|上端外径D|1mで正規化|
|下端外径|1D〜3D|
|全高|0.5D〜3D|
|上端の穴径|0.4D〜0.95D|
|肉厚|0.02D固定|

穴径と肉厚は独立したパラメータ。上面は円形の開口を持つ環状板で、上面・底面の板厚は0.02D、側壁は断面の法線方向で0.02D。底面は閉じる。全高は上端から下端までで、別の台座やリブは付けない。

## 構成

- `../paf_model.py`：寸法から回転断面と頂点・面を作る。Blenderに依存しない。
- `../random_dataset.py`：寸法抽選、Blender描画、正解投影、全画像検査、一覧画像の作成。
- `config.json`：寸法範囲・固定肉厚・乱数seed・画像枚数・通常の撮影条件。
- `../output/paf_random_outer_d_v2/`：生成したPNG、正解JSON、manifest、寸法計画、確認用画像、サンプルのBlenderモデル。

1,024画像に1,024通りの寸法を使う。形状の再抽選はデータ生成時だけ行い、寸法と撮影条件を保存する。学習・検出性能評価はこの生成処理には含まない。

## 撮影と正解

黒背景、480×480画像、焦点距離55mm。視点5〜70°と方位0〜360°、材質・照明はconfig.jsonの通常範囲から抽選する。全体が画面内に入るようカメラ距離を合わせ、本体中央を注視する。上端の穴の縁192点を正解としてBlenderから投影する。正解は画像と同じ座標系のimage_pointsに保存する。

## 実行

リポジトリ直下から実行する。

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python-exit-code 1 --python parametric_paf/random_dataset.py -- render
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/random_dataset.py verify
```

`manifest.json`は1画像ごとの寸法・撮影条件・画像/ラベルのhashを保持する。`verification.json`は全件の読込・寸法範囲・正解・hashの検査結果。`dataset_preview.png`は実際に保存した先頭16画像を固定抽出した一覧で、Bは下端外径/D、Hは全高/D、holeは穴径/Dを表す。画像は形状だけでなく撮影条件も異なる。

[生成データ](../output/paf_random_outer_d_v2/) ／ [実データの一覧](../output/paf_random_outer_d_v2/dataset_preview.png)

## 学習と検証

二つの学習元の穴領域CNNと幾何2方式の比較は[ルートREADME](../../README.md)を参照する。`run.py prepare`で多様形状を716/154/154に固定分割し、学習716枚から497枚、validation154枚から105枚を固定抽出して1194M側と枚数を揃える。test154枚は全て残す。両CNNの学習には通常画像に加え遮蔽・白飛び・黒つぶれを含める。
