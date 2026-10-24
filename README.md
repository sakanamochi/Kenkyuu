# PAF内周検出

作業ブランチ：`codex/hole-region-comparison`。

「機体ごとの詳細CADに頼らず、合成形状を多様化して学習すれば、未知形状のPAF内周を検出しやすくなるか」を調べる。比較対象は次の4つ。

- 穴領域CNN（1194M学習）
- 穴領域CNN（多様形状学習）
- Canny＋輪郭別RANSAC
- Zhang 2019型（再現実装）

学習・validation・testは全てモノクロ256入力。通常・矩形遮蔽・白飛び・黒つぶれを含める。成功条件はIoU≥0.8とし、未検出も分母に含める。

## 構成

```text
run.py                       render / prepare / train / evaluate / report
config/experiment.json       学習元・劣化・学習・幾何条件
config/render_base.json      1194Mの描画設定
assets/1194M_3/              1194MのOBJ・MTL
blender/generate_dataset.py   1194Mの描画・正解投影
parametric_paf/
  paf_model.py               多様形状の寸法とメッシュ定義
  generate.py                閉メッシュの作成・検査
  random_dataset.py          多様形状の描画・全件検査
  random_paf/config.json     多様形状の数値仕様
paf_ring_detection/
  data.py / effects.py       穴領域教師・撮像効果・共通モノクロ入力
  prepare.py                 固定分割・枚数統制・入力と実装のhash
  train.py                   共通CNN学習
  evaluate.py / report.py    共通評価・集計・全件照合
  geometry.py                楕円IoUなどの共通計算
  methods/                   Tiny U-Net・穴領域楕円化・幾何2方式
.codex/                      仕様と引継ぎ資料
研究関連書類/                研究資料（ローカル保存）
output/                      データ参照・重み・結果（ローカル保存）
```

## 実行

既存の`.venv`を使う。新規環境はPython 3.10でvenvを作り、`requirements.txt`、対応するCUDA環境では`requirements-cuda.txt`をインストールする。

```powershell
# 原画像がない場合の描画
.venv/Scripts/python.exe -X utf8 -B run.py render

# 固定分割と撮像条件・実装を保存
.venv/Scripts/python.exe -X utf8 -B run.py prepare

# 二つの学習元×3seedを初期化から60epoch学習
.venv/Scripts/python.exe -u -X utf8 -B run.py train

# 二つのtest集合を全4方式で評価・照合・集計
.venv/Scripts/python.exe -u -X utf8 -B run.py evaluate
.venv/Scripts/python.exe -X utf8 -B run.py report
```

一部を学習する場合は`train --source 1194m --seeds 20260907`。重み・履歴は同じ保存先で更新する。`prepare`は現在の設定とコードを固定するため、コードを編集した場合は再実行する。設定は`--config`でも指定できる。描画先は各描画設定に従う。

少数画像の動作確認は`evaluate --methods canny zhang2019 --limit 2`、集計は`report --limit 2`。確認用出力は`checks/limit_2/`へ分離し、正式評価には混ぜない。

## データと条件

1194Mの原画像は714枚、多様形状は1,024枚。1194Mはカメラ、多様形状は寸法単位で分割してから劣化を適用する。学習・validationの枚数と劣化強度を両学習元で揃える。

多様形状のtrain/validationでは、穴の投影長径が画像幅の0.06〜0.65となるよう撮影距離を抽選する。カメラ姿勢・形状・外観の抽選は保ち、穴の全周を画面内に残して本体の画面外を許容する。比較用の両test集合は固定する。

|学習元|学習の原画像／劣化込み|validationの原画像／劣化込み|testの原画像／劣化込み|
|---|---:|---:|---:|
|1194M|497／1,988|105／735|112／1,456|
|多様形状|497／1,988|105／735|154／2,002|

矩形遮蔽は画像端から幅×強度を覆い、強度間で左右を固定する。白飛びは露光増加・飽和・明部のブルーム・ショットノイズ、黒つぶれは低露光・読み出しノイズ・黒レベル・6〜8bit量子化を近似する。学習強度は0.1〜0.8、validationは0.3/0.6、testは0.25/0.5/0.75/1.0。

原画像の複製を作らず、manifestの固定seedで480画像へ劣化を適用する。共通読込処理でグレー化し、同じ輝度を3chへ複製して256へ縮小する。CNNと幾何へ同じ画像を渡す。固定した学習・validation入力はRAMに保持する。

CNNは両方ともTiny U-Net16、60epoch、batch16、同じAdamWとBCE＋Dice、seed20260907/20260908/20260909。多様形状CNNには1194MのCAD・画像・重みを使わず、両CNNをランダム初期化する。完全な穴領域を教師とし、確率0.5以上の最大領域外周へ楕円を当てはめる。重みはvalidation loss最小で選ぶ。

全方式の成功条件はIoU≥0.8。CNNは全3seedと平均、幾何は各画像一回の成績を報告する。幾何のCanny閾値30/90と既存RANSACは固定する。

## 現在の状態

多様形状の投影サイズ拡張後の3seed・60epoch学習、二つの固定test集合での4方式比較、全27,664推定の照合が完了。全8,904入力のモノクロ照合と32件の回帰テストが合格。多様形状CNNの成功率は1194M上15.00%、多様形状上45.99%で、同じtestでの拡張前21.38%/71.36%から低下した。詳細・条件別成績・変更前後の根拠は`results/REPORT.md`と`size_ablation.json`を参照する。

## 保存先

`output/hole_region_comparison_v1/`内の同じファイルを更新する。

- `datasets/{1194m,diversified}/manifest.json`：原画像参照・分割・劣化条件・hash。
- `readiness.json`・`provenance.json`：準備時の設定・入力・実装・実行環境。
- `models/{学習元}/{seed}/best.pt`・`training_history.json`：採用重みと全epoch履歴。
- `results/{評価集合}/{方式}_{seedまたはfixed}.json`：全画像の推定楕円・正解・指標。
- `results/REPORT.md`・`summary.json`・`scores.csv`：二つの比較、条件・強度・seed別の成績。
- `results/audit.json`：全epoch・最小validation重み・全推定の再計算照合。
- `results/size_ablation.json`：同じtestに対する撮影サイズ拡張前後の集計と入力同一性の根拠。

[比較結果](output/hole_region_comparison_v1/results/REPORT.md)と[現在の状態・次の手順](.codex/RESEARCH_HANDOFF.md)を参照する。原素材と研究関連書類はローカルに保持する。Git管理外の画像・重み・結果は別端末へ別途コピーする。

1194M testは同機体の撮影条件分割、多様形状testは同じ形状族内の未知寸法。両原データには撮影・材質・投影サイズの差もあるため、形状多様化の単独効果や実画像への汎化はこの比較だけで断定しない。

## 検証と出典

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q -p no:cacheprovider
```

形状・幾何検出・撮像効果・全分割のモノクロ入力・固定分割・短い学習から全方式評価と照合までを確認する。検証用画像・重み・結果はテスト終了時に削除する。

Zhang, Pan, Ma, Sensors 2019, DOI `10.3390/s19235243`。Zhang 2019型（再現実装）は論文の構成を参考にし、PAF固有の候補評価を分離した実装である。著者コードの移植・論文性能の完全再現ではない。詳しくは[実装仕様](.codex/PROJECT_CONTEXT.md)を参照する。
