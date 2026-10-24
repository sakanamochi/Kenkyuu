# PAF内周検出：実装仕様と出典

現在の作業状態・研究結果・次の手順は[RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)を正とする。

## 研究の主比較

問いは「機体ごとの詳細CADに頼らず、合成形状を多様化して学習すれば、未知形状のPAF内周を検出しやすくなるか」。比較対象は穴領域CNN（1194M学習）、穴領域CNN（多様形状学習）、Canny＋輪郭別RANSAC、Zhang 2019型（再現実装）の4つ。1194Mと多様形状の二つのtest集合に全方式を適用する。

1194MのCAD・画像は1194M学習CNNの比較基準に使う。多様形状学習CNNのtrain・validation・初期化には混ぜない。両CNNは同じTiny U-Net16をランダム初期化して学習する。方式・形状族・背景を独断で増やさない。シミュレーションを対象とする。

## データと学習

- 実行入口は`run.py`、正本設定は`config/experiment.json`。比較の保存先は`output/hole_region_comparison_v1/`に統一し、同じファイルを更新する。
- 1194Mの原画像は`output/datasets/base_cg/`714枚。CADは`assets/1194M_3/`、描画設定は`config/render_base.json`。
- 多様形状の原画像は`parametric_paf/output/paf_random_outer_d_v2/`1,024枚。正本仕様は`parametric_paf/random_paf/config.json`。
- 上端外径D=1m、下端外径1D〜3D、全高0.5D〜3D、穴径0.4D〜0.95Dを独立一様抽選する。肉厚0.02D固定。開口を持つ環状の上板・閉じた底・法線方向に一定厚の側壁からなる。台座やリブを付けない。
- 多様形状のtrain/validationは、穴長径が画像幅の0.06〜0.65となるよう独立乱数で撮影距離を調節する。カメラ姿勢と開口への方向は固定し、穴の全周と余白を画面内へ残す。本体の画面外は許容する。testは全体構図のまま固定する。
- 1194Mはカメラ、多様形状は寸法単位で分割する。両学習元の学習原画像497枚・validation105枚を揃える。testは1194M112枚、多様形状154枚。
- 各分割へ通常・矩形遮蔽・白飛び・黒つぶれを含める。学習強度0.1〜0.8、validation強度0.3/0.6、test強度0.25/0.5/0.75/1.0。学習各1,988枚、validation各735枚、testは1,456/2,002枚。
- 矩形遮蔽は画像端から画像幅×強度を黒くし、同じ原画像では強度間で左右を固定する。白飛びは線形露光の増加・飽和・明部のブルーム・ショットノイズ、黒つぶれは低露光・読み出しノイズ・黒レベル・6〜8bit量子化を近似する。
- `effects.py`は正解形状を使わずに劣化を適用する。`data.py`の共通読込処理で480画像へ劣化を適用した後、グレースケールへ変換して同じ輝度を3chへ複製し、256へ縮小する。学習・validation・testと幾何2方式で同じ画像を共有する。
- Tiny U-Net16、60epoch、batch16、AdamW・BCE＋Dice。seed20260907/20260908/20260909を両学習元で共通にする。固定した学習・validation入力はRAMへ保持する。
- 教師は不可視部分を含む完全な穴内部。推論は確率0.5以上の最大領域外周へ楕円を当てはめる。採用重みはvalidation loss最小とし、testで選び直さない。

## 幾何と評価

全方式の入力はモノクロ256。成功条件はIoU≥0.8のみ。未検出と誤楕円も全画像を分母に数える。CNNは全3seedとその平均、幾何は各画像一回の成績を報告する。

Cannyは標準Canny・輪郭分離・共通RANSAC・同心候補対の内周選択。Zhang 2019型（再現実装）は弧抽出・三弧統合・楕円推定とPAF固有の候補評価。Canny閾値30/90、RANSACの`perimeter_power=0.0`を固定し、testで調整しない。

保存するのは原画像参照のmanifest、重み・全epoch履歴、全画像の推定JSON、条件・強度・seed別CSVと集計JSON・比較表・照合結果。準備時に固定した設定・実装・入力hashと最小validation checkpointを確認し、推定指標を元ラベルから全件再計算する。

1194Mのtestは同機体の撮影条件分割、多様形状のtestは同じ合成形状族内の未知寸法。両原データには撮影・材質・投影サイズの差もあるため、この比較だけで形状多様化の単独因果効果を断定しない。既存画像は参照比較に使い、未使用の最終未知形状集合・実画像への汎化・対象なし判定は未検証。

## 確認済みの現行結果

多様形状CNNは、穴の投影サイズを拡げたtrain/validationで3seed・各60epoch学習済み。1194M学習CNN・幾何と二つのtest集合は固定する。全方式モノクロ256、IoU≥0.8、通常と3劣化を含み、CNNは3seed平均。

|方式|1194M test（1,456枚）|多様形状 test（2,002枚）|
|---|---:|---:|
|穴領域CNN（1194M学習）|67.26%|0.93%|
|穴領域CNN（多様形状学習）|15.00%|45.99%|
|Canny＋輪郭別RANSAC|49.18%|53.00%|
|Zhang 2019型（再現実装）|46.91%|61.24%|

多様形状CNNの同じtestでの成功率は、1194Mで21.38%→15.00%、多様形状で71.36%→45.99%。通常画像は1194M36.61%→22.62%、多様形状97.62%→93.72%。合成validationの長径100px以上は99.38%だが、今回のサイズ拡張条件だけでは1194Mへの転移改善は確認できていない。独立testとvalidationの確認を区別する。

正本は`output/hole_region_comparison_v1/results/REPORT.md`、`summary.json`、`scores.csv`、全推定JSON、`audit.json`。前後比較と入力固定の根拠は`size_ablation.json`。全16設定・27,664推定の元ラベルからの再計算照合、全6モデルの60epochと最小validation重み・有限性、全8,904入力のモノクロ照合、回帰テスト32件が合格。採用epochは1194M55/51/44、多様形状32/49/33。詳細な制約と次の手順は引継ぎ資料を参照する。

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
