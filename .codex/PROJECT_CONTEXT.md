# PAF内周検出：実装仕様と出典

研究方針と現在の作業状態は[RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)を参照する。

## 現在のデータ生成

- 仕様：`parametric_paf/random_paf/config.json`、説明：同フォルダのREADME。
- 上端外径D=1mで正規化。下端外径1D〜3D、全高0.5D〜3D、上端穴径0.4D〜0.95Dを独立一様抽選し、肉厚0.02Dを固定する。
- 上面は開口を持つ環状板、底面は閉じる。上面・底面の板厚と側壁の法線方向厚さを0.02Dとする。別の台座やリブは付けない。
- 形状定義：`parametric_paf/paf_model.py`、生成・検査：`parametric_paf/random_dataset.py`。
- 画像ごとに別寸法の1,024画像を生成済み。出力：`parametric_paf/output/paf_random_outer_d_v2/`。検査結果は`verification.json`、実データ一覧は`dataset_preview.png`。
- 黒背景480入力で描画し、モデル全体が画面内に入るよう距離を合わせる。上端の穴の縁192点を正解として保存する。
- 現行仕様の学習・独立評価は`parametric_paf/train_random_paf.py`で完了。元の1,024枚を学習716・モデル選択154・最終評価154に固定分割し、12epoch・3seedで学習した。出力は`parametric_paf/output/paf_outer_d_training_v2/`、仕様は`random_paf/TRAINING.md`。
- 最終評価の主成功率はCNNが37.01%/0.65%/14.29%（平均17.32%）、Canny＋輪郭別RANSAC72.08%、Zhang 2019型（再現実装）87.66%。全770推定を元ラベルから再計算して照合済み。CNNの学習seed間のばらつきは大きく、12epochで十分に収束したとは断定しない。別設定の過去の成績と直接比較しない。
- 学習延長と穴領域教師の検証は`parametric_paf/convergence_study.py`で完了。仕様は`random_paf/CONVERGENCE.md`、出力は`output/paf_convergence_v1/`。同じ716枚・3seed・既存Tiny U-Netで各60epoch。内周線方式の参照成功率は平均45.24%で幾何に届かず、穴領域予測＋最大連結領域外周への楕円当てはめを検証した。
- 方式確定後の新規154画像で、元12epochモデル23.81%、内周線60epoch51.30%、穴領域60epoch94.16%、Canny75.97%、Zhang 2019型（再現実装）81.82%。CNNは3seed平均。穴領域方式は全seedで幾何を上回り、現条件の主候補。全2,618推定を元ラベルから再計算して照合済み。教師に伴う後処理変更を含む比較であり、教師だけの因果効果・画像数追加の効果・任意の暗い円への汎用性は未検証。

## 検出器・評価

- CNN：Tiny U-Netの内周確率マップ＋重み付きRANSAC。
- 穴領域CNN：同じTiny U-Netで穴内部全体を教師にし、確率0.5以上の最大連結領域外周に楕円を当てはめる。学習率・損失・画像分割・形状分布を固定した検証では、この構成が内周線方式を上回る。
- Canny統制：標準Canny＋輪郭分離＋共通RANSAC＋同心候補対の内周選択。
- Zhang 2019型（再現実装）：弧抽出・三弧統合・楕円推定＋PAF固有の候補評価。
- 共通評価は`paf_ring_detection/geometry.py`。IoU≥0.8と双方向輪郭距離p95≤2px（256入力）を区別して扱う。異なる入力解像度の結果を直接比較しない。
- CNN教師は不可視部分を含む完全な内周リング。RANSACは総支持量（perimeter_power=0.0）。

## Zhang型の正規出典

Limin Zhang, Wang Pan, Xianghua Ma,
“Real-Time Docking Ring Detection Based on the Geometrical Shape for an
On-Orbit Spacecraft,” Sensors, 19(23), 5243, 2019.

- DOI: https://doi.org/10.3390/s19235243
- Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/

対象は軌道上宇宙機のドッキングリングであり、PAFそのものではないが、
「宇宙機画像から投影楕円となるリング内周を、分断弧の統合で検出する」という問題設定が
本研究の古典比較として最も近い。

`paf_ring_detection/methods/zhang2019.py` は論文の処理順を参考にした
Python再現実装であり、
著者コードの逐語的移植や完全再現ではない。論文との差分は少なくとも以下を含む。

- OpenCVによるCanny・Sobel・`fitEllipseDirect`を使用
- 計算量を固定するため弧数・組合せ数を制限
- 平行弦中点法は離散弧上の補間と`fitLine`で数値実装
- `zhang2019_paf.py`でPAF画像向けに全周エッジ支持と角度被覆を検証
- 同ファイルで候補間に内包関係があれば最も内側を選び、
  内包関係がなければ全周支持率最大を選ぶ最終選択を追加

したがって論文そのものの性能として結果を引用してはならない。

