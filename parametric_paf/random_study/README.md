現行の生成仕様は [上端外径D基準のランダムPAF](../random_paf/README.md) を参照する。このフォルダーは保存済みデータの参照用であり、現行仕様の実行入口ではない。

# ランダム寸法によるPAF内周検出の検証

実験ID：`paf_random_dimensions_v1`。目的は、機体ごとの詳細CADを学習に使わず、合成PAFの寸法を多様化することで未知形状の上端内周を検出しやすくなるかを検証すること。

学習・独立評価・集計・整合性検査は完了。結果は [検証結果の要約](../output/paf_random_dimensions_v1/report/FINDINGS.md) と [seed別の表](../output/paf_random_dimensions_v1/report/REPORT.md) を参照する。主テストでの改善と、基準寸法の低下・対象なし誤検出を区別して扱う。

既存幾何2手法との同一画像での追加比較も完了している。[比較仕様](GEOMETRY_COMPARISON.md) と [CNN・幾何比較結果](../output/paf_random_geometry_v1/report/REPORT.md) を参照する。CNN内の形状多様化効果と、幾何手法に対する性能差は区別する。

## 必要な比較

|条件|学習寸法|画像・学習量|
|---|---|---|
|single：対照|基準合成形状1個|1,024枚、12epoch、768更新|
|random：主方式|画像ごとに独立抽選した1,024形状、各1枚|同上|

外観拡張は両条件共通。同じ画像番号の視点・距離・色・材質・照明・露光と内周投影を対応させる。Tiny U-Net・損失・optimizer・学習率・教師マスク・後処理も同じ。学習seedは共通の20260907/20260908/20260909。各条件3モデル、計6モデルを評価する。

singleは`paf_shape_appearance_v1`のB条件を使用する。画像・学習設定・学習履歴・採用重みの整合性を既存の凍結記録と照合し、保存済み3モデルを再利用する。randomだけを同じ学習器で新規学習する。1194MのCAD・画像・重みは両条件の学習・validationに使用しない。

固定16形状との比較、形状数比較、CNN構造比較、幾何方式の追加実装・再調整は行わない。従来の2×2実験は予備実験として独立した実験IDで保存される。本実験だけで幾何手法への優越は主張しない。

## ランダム寸法と保存方法

上端内径Dを基準とし、下端外径/Dを1.35〜1.90、高さ/Dを0.15〜0.45、半径方向肉厚/Dを0.04〜0.12から独立一様抽選する。shape seedは2026102401。小数6桁で保存し、同一寸法の重複と検証・テスト寸法との一致を禁止する。内径は生成時D=1m、内壁は同じテーパ、底面あり、リブなし。下端円筒高さは上端外径の10%で連動する。

画像ごとに新しい寸法のメッシュを生成して事前描画し、画像・正解・寸法・撮像条件・hashを保存する。学習中やepochごとの再抽選は行わない。3学習seedも同じ保存済みデータを使う。実在PAFの寸法分布を推定したものではなく、定義した合成形状族の検証である。

## 調整用データと独立評価

全モデルで共通のvalidation 8形状×24条件＝192枚を使用し、validation loss最小のcheckpointを採用する。validationはモデル選択用であり、最終結果に代用しない。テスト形状・撮像条件は保存済みの未評価計画を使用し、randomの学習前に新実験の計画として凍結する。

|区分|形状・画像数|役割|
|---|---|---|
|補間：主テスト|未学習12形状×48条件＝576枚|範囲内の未知寸法に対する主仮説|
|外挿|未学習12形状×48条件＝576枚|範囲外寸法での限界|
|撮像ストレス|上記30形状×16条件＝480枚|傾斜75/80°・露光±1・横位置ずれ±0.1の複合条件|
|single_shape_monitor|単一寸法側の学習形状1個×未使用48条件|基準寸法での性能低下の補足|
|対象なし|黒画像1枚＋穴なし円盤63枚|対象なし誤検出の補足。独立64形状とは扱わない|


## 判定と再現性

主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px（256入力）。IoUのみ、距離、未検出、誤楕円も記録する。主テストで3学習seedすべてのrandom−singleが正、平均10ポイント以上を主仮説支持の事前基準とする。monitorの低下5ポイント以内、対象なし誤検出増加5ポイント以内を補足の判断幅とし、超過は交換条件として明記する。運用・安全の許容値ではない。

同じ形状・撮像条件の対応差を形状ごとに集計し、seed別の値を示す。画像数を独立形状数に数え替えない。形状の抽選自体は1反復であり、学習seed3本と混同しない。最終テストを使う再調整・checkpoint選び直し・自動規模拡大は行わない。改善がなくても結果と適用範囲を報告する。

推論はOpenCVの乱数も画像単位で固定し、NumPyのRANSAC seedだけに依存しない。single/randomの各3モデルについて、固定したvalidation17画像を独立2プロセスで推論し、確率マップ・推定楕円・評価値の完全一致を確認してからテストに進む。これは記録した環境と検査画像の再現性であり、異なるGPU・ライブラリ版での一致を保証しない。

## 構成・実行

`../random_study.py`が計画・描画・入力検査・凍結・学習・推論・集計を担当し、既存の生成器、Tiny U-Net学習器、CNN＋RANSAC、共通輪郭評価を呼び出す。既存の標準実験コードと保存済み重みは変更しない。

リポジトリ直下で実行する。

```powershell
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/random_study.py run
```

段階別にも実行できる。描画段階だけはBlenderのPythonで実行する。

```powershell
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/random_study.py check
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/random_study.py train --seed 20260907
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/random_study.py repro --arm random
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/random_study.py summarize
```

全評価・集計の完了後、入力・学習ログ・全評価原記録・判定・集計の対応を照合する。

```powershell
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/audit_random_study.py
```

`audit_random_study.py`は学習・推論を変更せず、全18評価区分の整合性検査と、IoU成功・未検出・誤楕円・輪郭距離・支持被覆の断面族別／視点別集計を行う。読みやすい結果要約は`report/FINDINGS.md`、検査結果は`report/audit.json`、集計は`population_results.csv`と`view_results.csv`に保存する。監査コード自体のhashも記録する。

`plan.json`に寸法と撮像条件、`configs/`に3学習設定、`freeze.json`に入力・コード・再利用重みのhashと実行環境を保存する。データ・重み・評価・ログは`../output/paf_random_dimensions_v1/`、最終集計は同フォルダの`report/REPORT.md`。`records.jsonl`は再開可能な画像単位の研究原記録として保存する。既存重みは上書きせず、完了していない学習重みを完成モデルとして扱わない。
