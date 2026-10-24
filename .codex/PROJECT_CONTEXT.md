# PAF内周検出：実装仕様と出典

研究方針・現在の状態・次の具体的作業は[RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)を正とする。

## 主比較

研究の問いは「機体ごとの詳細CADに頼らず、合成形状を多様化して学習すれば、未知形状のPAF内周を検出しやすくなるか」。比較は穴領域CNN（1194M学習）、穴領域CNN（汎化形状学習）、Canny＋輪郭別RANSAC、Zhang 2019型（再現実装）の4つ。

1194MのCAD・派生画像は1194M学習CNNの比較基準にのみ使う。汎化形状学習CNNには混ぜず、両CNNはランダム初期化する。方式・形状族・背景・検討要素を独断で増やさない。現在はシミュレーションを対象とする。

## データと穴領域CNN

- 実行入口：`run.py`。正本設定：`config/experiment.json`。
- 1194M：`assets/1194M_3/`、描画設定`config/render_base.json`、原画像`output/datasets/base_cg/`714枚。
- 汎化形状：`parametric_paf/random_paf/config.json`、原画像`parametric_paf/output/paf_random_outer_d_v2/`1,024枚。
- 上端外径D=1mで正規化し、下端外径1D〜3D、全高0.5D〜3D、穴径0.4D〜0.95Dを独立一様抽選する。肉厚0.02D固定。上面は開口を持つ環状板、底面は閉じる。台座・リブは付けない。
- 形状定義`paf_model.py`、閉メッシュ検査`generate.py`、描画・全件検査`random_dataset.py`。480黒背景、上端穴縁192点を正解として投影する。
- 1194Mはカメラ単位、多様形状は寸法単位で70/15/15に固定分割する。多様形状は716/154/154。劣化適用前に分割する。
- 両学習元は原画像の学習497枚・validation105枚に揃える。学習は通常＋3劣化で各1,988枚。validationは通常＋3劣化×2強度で各735枚。学習時の劣化強度と効果seedは学習元間で共通にする。
- testは1194M112枚、多様形状154枚を保持し、通常＋3劣化×6強度で各2,128/2,926枚。同じtest画像を4方式へ渡す。
- 劣化は既存の扇形遮蔽・局所飽和／ブルーム・暗部階調変換。物理的なセンサ特性の完全再現ではない。原画像に固定seedで適用し、画像複製を避ける。
- Tiny U-Net16、入力256、60epoch、batch16、AdamWとBCE＋Dice。学習seed20260907/20260908/20260909を両学習元で共通にする。
- 教師は不可視部分を含む完全な穴内部。確率0.5以上の最大領域外周へ`fitEllipse`を適用する。正解による候補選択はしない。
- checkpointはvalidation loss最小。新比較の出力は`output/hole_region_comparison_v1/`に分ける。入力・分割・設定の変更はhashで検出し、既存重みの上書きは拒否する。

## 幾何と評価

Cannyは標準Canny＋輪郭分離＋共通RANSAC＋同心候補対の内周選択。Zhang 2019型（再現実装）は弧抽出・三弧統合・楕円推定＋PAF固有の候補評価。Canny閾値30/90、RANSACの`perimeter_power=0.0`を固定する。

全方式の入力は256。主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px。距離は各楕円720点間の双方向最近傍距離から計算する。IoU単独成功、未検出、誤楕円も分けて全画像を分母に報告する。幾何結果をCNNのseed数だけ複製しない。

現在のtestは参照済みのデータに由来する参照比較である。1194Mのtestは同機体学習CNNにとって未知形状ではなく、多様形状のtestも同じ合成形状族内の未知寸法である。両原データの撮影・材質・投影サイズ分布には差があるため、枚数・劣化条件を揃えただけでは形状多様化単独の因果効果を主張しない。

## 穴領域方式を採用する根拠

通常条件の方式確定後に生成した154画像で、穴領域CNN60epochの主成功率は3seed平均94.16%、Canny75.97%、Zhang 2019型（再現実装）81.82%。全seedが幾何を上回った。教師と後処理の組合せを選ぶ根拠であり、劣化込み学習・単一CAD対多様形状の比較結果ではない。

正本は`parametric_paf/output/paf_convergence_v1/report/REPORT.md`、`summary.json`、`audit.json`、`report_verification.json`と推定原記録。監査は全2,618推定を元ラベルから再計算して合格。過去結果の再現用コードはブランチ`codex/minimal-research-environment`、commit`1fdfbca`。新しい比較の設定・分割と区別する。

## 劣化込み学習と二つの比較：確認済み結果

両学習元×3seedの計6モデルをランダム初期化から60epoch学習済み。入力256、主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px。下表は通常・遮蔽・白飛び・黒つぶれを全て含み、CNNは3seed平均、幾何は一回の評価。

|方式|1194M test（2,128枚）|多様形状 test（2,926枚）|
|---|---:|---:|
|穴領域CNN（1194M学習）|34.26%|0.09%|
|穴領域CNN（汎化形状学習）|1.75%|56.38%|
|Canny＋輪郭別RANSAC|16.73%|25.32%|
|Zhang 2019型（再現実装）|14.99%|26.49%|

通常画像だけの主成功率は、1194M学習CNNが1194M上97.32%・多様形状上0.43%、多様形状学習CNNが1194M上8.63%・多様形状上96.10%。各CNNは自分の学習元に対応するtestでは幾何を上回るが、学習元をまたぐ適用では成績が大きく低下する。今回の条件で、多様形状学習による1194Mへの十分な汎化が確認できたとは言えない。

正本は`output/hole_region_comparison_v1/results/REPORT.md`、`summary.json`、`scores.csv`、各推定JSON、`audit.json`。全16設定・40,432推定を元ラベルから再計算して一致。全6学習の60epoch履歴・最小validation checkpoint・重みの有限性・固定した入力と実装のhashを照合済み。回帰テスト30件が合格し、検証用の画像・重み・結果は全て削除済み。

採用epochは1194M学習のseed20260907/20260908/20260909が58/60/59、多様形状学習が55/57/43。60epochまで実行したことだけで、十分な収束を断定しない。GPU演算による完全な学習軌跡の再現性は別途保証していない。

## Zhang型の正規出典

Limin Zhang, Wang Pan, Xianghua Ma,
“Real-Time Docking Ring Detection Based on the Geometrical Shape for an On-Orbit Spacecraft,” Sensors, 19(23), 5243, 2019.

- DOI: https://doi.org/10.3390/s19235243
- Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/

対象は軌道上宇宙機のドッキングリング。投影楕円となる内周を分断弧の統合で検出する問題として比較に用いる。

`paf_ring_detection/methods/zhang2019.py`は論文の処理構成を参考にしたPython再現実装であり、著者コードの移植ではない。少なくとも以下の相違がある。

- OpenCVのCanny・Sobel・`fitEllipseDirect`を使用。
- 計算量を固定するため弧数・組合せ数を制限。
- 平行弦中点法は離散弧上の補間と`fitLine`で数値実装。
- `zhang2019_paf.py`で全周エッジ支持・角度被覆を検証。
- 内包関係があれば最も内側、なければ全周支持率最大を選ぶPAF固有の最終選択を追加。

成果物では必ず「Zhang 2019型（再現実装）」と表記し、論文そのものの性能として結果を引用しない。
