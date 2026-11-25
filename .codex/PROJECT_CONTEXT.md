# PAF内周検出：実装仕様と出典

現在の作業状態・研究方針・実験結果・次の手順は [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) を正とする。

## 研究の問いと構成

問いは「国際標準クランプバンド規格（937 / 1194 / 1666 / 2624 mm 等）に代表される現実のPAF寸法・構造に基づき合成データを設計すれば、特定機体の詳細CADを学習に使わずに、幾何手法を上回る劣化耐性と未知PAFへの汎化を両立できるか」。

比較対象は境界CNN（1194M学習）、境界CNN（多様形状学習）、Canny＋輪郭別RANSAC、Zhang 2019型（再現実装）の4方式。1194MのCAD・画像は1194M学習CNNの比較基準および未知評価にのみ用い、多様形状学習CNNのtrain・validation・初期化には混ぜない。両CNNは同一の Tiny U-Net16（482,737パラメータ）をランダム初期化する。

## 現行システムとデータ仕様

- 直接実行スクリプト（`scripts/`）：`predict.py`・`train.py`・`evaluate.py`・`report.py`・`render_cg.py`・`ablation.py`。
- 正本設定（`configs/`）：`experiment.json`（学習・推論・評価・幾何・図設定）、`render_1194m.json`、`render_diversified.json`。
- 基本運用は固定1seed（`20260907`）。3seed（`20260907`・`20260908`・`20260909`）は明示指定時のみ実行する。
- 1194M原画像は `assets/1194M_3/1194M_3.obj` から生成した714枚（train 497 / val 105 / test 112）。
- 多様形状PAFの現行形状モデル（`src/cg/geometry_model.py`、`configs/render_diversified.json`）：
  - **下部円筒部（固定・コップ状）**：下端外径 $B = 1.0\text{ m}$ 固定、円筒高さ $H_{\mathrm{cyl}} = 0.25 B = 0.25\text{ m}$ 固定、平底で閉じる。
  - **上部アダプタ部（可変・円錐台、わずかな逆テーパ含む）**：上端外径比 $T/B \in [0.35, 1.05]$、アダプタ高さ比 $H/B \in [0.10, 0.36]$、穴径比 $\text{hole}/T \in [0.85, 0.96]$ を独立一様抽選する。
  - **肉厚と内部空間**：法線方向肉厚 $t = 0.012 T$。上端リング幅 $(T - \text{hole})/2 > t$ および内壁非干渉を検証し、上端開口から下部円筒底面まで一続きの内部空間を成す10点回転断面の閉メッシュを生成する。
  - 出力先は `data/diversified/`（756形状：train 497 / val 105 / test 154、8ビュー：元構図／拡張投影 × 無地／放射ストライプ／ノイズ／チェッカー）。評価用テスト集合は `configs/experiment.json` の `test_views` により test 154形状へ4外観（無地・放射ストライプ・ノイズ・チェッカー）を均等割当した2,002件を用いる。学習重み正本は `models/diversified/20260907/best.pt`。
- 撮像劣化は通常・矩形遮蔽・白飛び・黒つぶれ。480px原画像へ劣化適用後、グレースケール3ch複製・256px縮小する。評価は全13条件（白飛び最大内部強度2.0）、成功判定は塗りつぶし楕円 IoU ≥ 0.8（未検出・誤楕円を分母に含む）。
- 境界CNNの教師は矩形遮蔽のみを除いた可視内周境界。損失は正例重み12のBCE＋Dice、60epoch、batch 16、AdamW。推論は確率0.5以上で二値化・形態学的スケルトン細線化・確率重み付きRANSACで楕円を推定し、外周と内周が同時反応した際には第1候補内部の残余点群へ残差RANSACを適用して完全内包かつ十分な全周支持を持つ最内周を選ぶ（`prefer_inner_ellipse: true`）。

## Zhang型の正規出典

Limin Zhang, Wang Pan, Xianghua Ma,
“Real-Time Docking Ring Detection Based on the Geometrical Shape for an On-Orbit Spacecraft,” *Sensors*, 19(23), 5243, 2019.

- DOI: https://doi.org/10.3390/s19235243
- Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/

対象は軌道上宇宙機のドッキングリング。投影楕円となる内周を分断弧の統合で検出する問題として比較に用いる。

`src/inference/zhang2019.py` は論文の処理構成を参考にしたPython再現実装であり、著者コードの移植ではない。少なくとも以下の相違がある。

- OpenCVのCanny・Sobel・`fitEllipseDirect`を使用。
- 計算量を固定するため弧数・組合せ数を制限。
- 平行弦中点法は離散弧上の補間と`fitLine`で数値実装。
- 全周エッジ支持・角度被覆を検証し、内包関係があれば最も内側、なければ全周支持率最大を選ぶPAF固有の最終選択を追加。

成果物では必ず「Zhang 2019型（再現実装）」と表記し、論文そのものの性能として結果を引用しない。
