# PAF内周検出

作業ブランチ：`codex/hole-region-comparison`。

研究の問いは「機体ごとの詳細CADに頼らず、合成形状を多様化して学習すれば、未知形状のPAF内周を検出しやすくなるか」。比較対象は次の4つに限定する。

- 穴領域CNN（1194M学習）：`assets/1194M_3/`のCADから生成した画像で学習する比較基準。
- 穴領域CNN（汎化形状学習）：[指定範囲の多様形状](parametric_paf/random_paf/README.md)で学習する。
- Canny＋輪郭別RANSAC。
- Zhang 2019型（再現実装）。

両CNNとも通常画像・遮蔽・白飛び・黒つぶれを用い、同じTiny U-Net16・穴領域教師・最適化・後処理を使う。汎化形状学習には1194MのCAD・画像・重みを使わず、両CNNをランダム初期化する。背景は黒。

## 構成

```text
run.py                       render / prepare / train / evaluate / report
config/experiment.json       二つの学習元・劣化条件・学習条件・幾何設定
config/render_base.json      1194Mの描画設定
assets/1194M_3/              1194MのOBJ・MTL
blender/generate_dataset.py   1194Mの描画・正解投影
parametric_paf/
  paf_model.py               多様形状の寸法とメッシュ定義
  generate.py                閉メッシュの作成・検査
  random_dataset.py          多様形状の描画・全件検査
  random_paf/config.json     多様形状の数値仕様
paf_ring_detection/
  data.py / effects.py       穴領域教師と固定seedの撮像効果
  prepare.py                 分割・枚数統制・入力hash検査
  train.py                   共通の穴領域CNN学習
  evaluate.py / report.py    共通評価・推定原記録・条件別CSV集計
  geometry.py                IoUと双方向輪郭距離
  methods/                   Tiny U-Net・穴領域楕円化・幾何2方式とその部品
.codex/                      研究仕様と引継ぎ資料
研究関連書類/                研究資料（ローカル保存）
output/                      データ・重み・結果（ローカル保存）
```

## 実行

既存の`.venv`を使う。新しい環境の場合はPython 3.10のvenvを作り、`requirements.txt`をインストールする。対応するCUDA環境では`requirements-cuda.txt`も使う。

原画像がない場合の描画：

```powershell
.venv/Scripts/python.exe -X utf8 -B run.py render
```

現在の原画像は1194M714枚・多様形状1,024枚で、次の準備処理を実行済み。

```powershell
.venv/Scripts/python.exe -X utf8 -B run.py prepare
```

学習・評価・集計は段階ごとに実行する。

```powershell
# 二つの学習元×3seedを、それぞれ初期化から60epoch学習
.venv/Scripts/python.exe -u -X utf8 -B run.py train

# 全モデルと幾何2方式を同じtest画像で評価し、条件別に集計
.venv/Scripts/python.exe -u -X utf8 -B run.py evaluate
.venv/Scripts/python.exe -X utf8 -B run.py report
```

一部だけ学習する場合は`train --source 1194m --seeds 20260907`のように指定する。少数画像の動作確認は`evaluate --methods canny zhang2019 --limit 2`で実行できる。少数評価は`checks/limit_2/`へ保存し、正式結果には混ぜない。`report --limit 2`でその範囲を集計する。

設定変更は`--config`で別のJSONを渡せる。既存重みの上書きは拒否するため、別実験は設定の`output`を新しい保存先にする。再描画の保存先は各描画設定に従う。

## 分割と比較条件

1194Mはカメラ単位、多様形状は寸法単位で分割してから劣化を適用する。派生画像が別splitへ混ざることを防ぐ。通常画像1枚と3種類の劣化画像を学習に用い、両学習元の原画像数・劣化強度・乱数条件を揃える。

|学習元|学習の原画像／劣化込み|validationの原画像／劣化込み|testの原画像／劣化込み|
|---|---:|---:|---:|
|1194M|497／1,988|105／735|112／2,128|
|汎化形状|497／1,988|105／735|154／2,926|

劣化は[既存の画像変換](paf_ring_detection/effects.py)による合成であり、物理センサの完全な再現ではない。原画像を複製せず、manifestの固定seedで480画像へ適用してから256入力へ縮小する。固定した学習・validation入力はRAMに保持し、epochごとの同じ変換を省く。教師は不可視部分を含む完全な穴領域。推論は確率0.5以上の最大領域外周へ楕円を当てはめる。checkpointはvalidation loss最小で選び、testで選び直さない。

全方式の入力は256。主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px。IoU単独成功・未検出・誤楕円も記録する。CNNは3seedを全て報告し、幾何は各画像1回だけ評価する。幾何のCanny閾値30/90は固定し、testで調整しない。

両test集合に対し、CNNの両学習元と幾何2方式を評価する。1194Mのtestは1194M学習CNNにとって未知機体ではない。多様形状のtestは学習寸法と非重複だが、同じ合成形状族・寸法分布内である。両データセットは撮影・材質・投影サイズの分布も異なるため、枚数と劣化条件の統制だけでは形状多様化の単独効果を確定できない。

既存の画像は方式検討で参照済みであり、現在のtestは参照比較に用いる。主仮説の最終検証には、撮影条件の統制と未使用の未知形状評価集合を比較前に確定する必要がある。

## 保存先と現在の状態

新しい比較の保存先は`output/hole_region_comparison_v1/`。

- `datasets/{1194m,diversified}/manifest.json`：原画像参照・固定分割・劣化条件・hash。
- `readiness.json`：準備時の設定と入力manifestのhash。
- `models/{学習元}/{seed}/best.pt`・`training_history.json`：学習時に保存する。
- `results/{評価集合}/{方式}_{seedまたはfixed}.json`：全画像の推定楕円・正解・指標。
- `results/summary.json`・`scores.csv`・`REPORT.md`：二つの比較、CNNのseed平均、劣化種類・強度別の成績。不足結果も明記する。
- `results/audit.json`：全epoch・最小validation checkpoint・全推定の再計算照合。

現在の実施対象は、CNN二系統×3seedの60epoch学習と、1194M・多様形状の二つのtest集合における4方式比較。進行と完了状態はlogs/と学習履歴・評価原記録を照合する。既存の通常条件の穴領域CNN重みは、劣化込みの学習済みモデルの代わりには扱わない。

研究資料・既存データ・重み・結果は`研究関連書類/`、`output/`、`parametric_paf/output/`に保持する。これらはGit管理外なので別端末へは別途コピーが必要。通常条件での穴領域CNNの根拠は[保存済み報告](parametric_paf/output/paf_convergence_v1/report/REPORT.md)、現在の方針と次の作業は[引継ぎ資料](.codex/RESEARCH_HANDOFF.md)を参照する。過去結果の再現用コードは`codex/minimal-research-environment`の`1fdfbca`に保存されている。

## 検証と出典

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q -p no:cacheprovider
```

形状の閉鎖性、幾何検出、劣化、穴領域教師、splitの非重複、短い学習から4方式共通評価までの接続を確認する。検証用画像・重み・結果はテスト終了時に削除する。

Zhang, Pan, Ma, Sensors 2019, DOI `10.3390/s19235243`。Zhang 2019型（再現実装）は論文の弧抽出・統合・楕円推定を参考にし、PAF固有の候補評価を分離した実装であり、著者コードの移植や論文性能の完全再現ではない。詳しい相違は[実装仕様](.codex/PROJECT_CONTEXT.md)を参照する。
