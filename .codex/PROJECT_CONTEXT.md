# PAF内周検出：実装仕様と出典

研究方針・実験計画・作業状態は [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) を参照する。
実行方法はルートおよび `parametric_paf/` のREADMEを参照する。

## 構成

- 標準実験の入口：`run.py`、設定：`config/experiment.json`
- パラメトリックPAFの生成・形状比較：`parametric_paf/`
- CNN：Tiny U-Netの内周確率マップ＋重み付きRANSAC
- Zhang 2019型（再現実装）：弧抽出・三弧統合・楕円推定＋PAF固有の候補評価と内周選択
- Canny統制：標準Canny＋輪郭分離＋共通RANSAC＋同心候補対の内周選択
- 共通評価：`paf_ring_detection/geometry.py`、標準集計：`paf_ring_detection/evaluate.py`
- 標準出力：`output/results/`、形状比較出力：`parametric_paf/output/comparison/`
- 評価背景：黒背景

## 評価・学習仕様

- 正解はPAF上端内周の投影楕円。
- 成功条件は塗りつぶした楕円領域のIoUが0.80以上。
- CNN教師マスクは不可視部分を含む完全な内周リング。
- RANSAC仮説スコアは総支持量を使用（`perimeter_power=0.0`）。
- 形状比較では全方式256×256入力。標準評価ではZhang型が元画像解像度、CNNとCannyが設定の入力解像度を使用する。異なる入力解像度の結果を直接比較しない。
- Canny方式はCNN抽出の寄与を調べる統制であり、先行研究の再現実装ではない。

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

