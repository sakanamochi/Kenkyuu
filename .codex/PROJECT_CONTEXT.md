# PAF内周リング検出研究: Codex引き継ぎコンテキスト

更新日: 2026-07-17

## 進捗報告の時系列

研究環境の現在の主比較はZhang型とCNNだが、第1回進捗報告では開発時系列を優先する。

### 第1回で示す内容

- CG評価環境、714枚の内訳、学習・評価データの構成
- Canny + 輪郭別RANSAC（内周priorを含む）とCNN方式
- RANSACとCNN学習の要点
- 単純黒矩形、白飛びproxy、黒つぶれproxyの初期結果
- Canny方式では遮蔽で分断された左右の弧を統合できないこと
- 次回までにZhang et al. (2019)型をベースラインとして導入する予定

第1回では地球・月背景、未知角度別の詳細結果、Zhang型の実験結果は扱わない。
未知条件については「学習用とテスト用で異なるカメラ条件を使用した」と簡潔に示す。
発表用生成コード `paflab/reporting/build_first_progress_figures.py` もこの時系列に合わせる。

### 第2回で示す内容

- Zhang 2019型（再現実装）へのベースライン切替
- 未知姿勢・未知照明・背景変更を含むOOD評価
- CNN規模比較、RANSAC候補選択監査、追加診断

## 現在の主比較

主古典ベースラインは `zhang2019_arc_reproduction`、学習方式はCNNリング尤度と
weighted RANSACの組合せである。

```text
Zhang 2019型（再現実装）
入力 → Canny → 勾配方向・凸性による弧抽出 → 複数弧のグループ化
     → 最小二乗楕円 → 実エッジ支持による検証 → 内周候補

CNN方式
入力 → Tiny U-Netリング尤度 → 閾値点群 → weighted RANSAC → 内周楕円
```

主比較に使う結果:

| 評価 | Zhang型 | CNN |
|---|---|---|
| OOD | `zhang2019_arc_ood` | `cnn_ransac_support_ood` |
| 撮像診断 | `zhang2019_arc_diagnostic` | `cnn_ransac_support_diagnostic` |

成功条件は推定楕円とCG正解内周楕円のIoUが0.80以上。

## Zhang型の正規出典

Limin Zhang, Wang Pan, Xianghua Ma,
“Real-Time Docking Ring Detection Based on the Geometrical Shape for an
On-Orbit Spacecraft,” Sensors, 19(23), 5243, 2019.

- DOI: https://doi.org/10.3390/s19235243
- Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC6928708/

対象は軌道上宇宙機のドッキングリングであり、PAFそのものではないが、
「宇宙機画像から投影楕円となるリング内周を、分断弧の統合で検出する」という問題設定が
本研究の古典比較として最も近い。

`analysis/zhang_arc_detector.py` は論文の処理順を参考にしたPython再現実装であり、
著者コードの逐語的移植や完全再現ではない。論文との差分は少なくとも以下を含む。

- OpenCVによるCanny・Sobel・`fitEllipseDirect`を使用
- 計算量を固定するため弧数・組合せ数を制限
- PAF画像向けに実エッジ支持、角度被覆、内外輝度を得点化
- 同心候補が存在する場合だけ内側を選ぶ最終選択を追加

したがって論文そのものの性能として結果を引用してはならない。

## 補足・アブレーションへ移した方式

- `kojima2021_fornaciari_reproduction`: Kojima et al. (2021) がPAFの内外周楕円検出に
  採用したFornaciari et al. (2014)の再現実装。勾配方向2群と凸性から4種類の弧を作り、
  3弧を統合して楕円方程式への適合率で検証する。原論文の分解1次元Hough投票は
  `fitEllipseDirect` に置換した。複数候補からPAF内周を選ぶ同心・相似ペアpriorは
  本評価用の追加処理であり、Kojima論文に記載された自動選択処理ではない。
- `canny_global_ransac`: Canny全エッジ点を単一RANSACへ入力。
  遮蔽で分断された弧は同時に扱えるが、外周・溝・付属部品も同時に入り、診断1,904枚で4.0%。
- `canny_ransac_inner_pair`: Canny輪郭ごとにRANSACを適用し、PAF同心二重輪郭の
  内周priorで候補を選ぶ。表示名は「Canny + 輪郭別RANSAC」に統一する。
- `contour_fit`: 輪郭ごとのOpenCV楕円フィット。最小統制。

これらの集計結果は研究判断の記録として残す。専用コードは現行判断の再検証に必要なものだけを
`paflab/experiments/` に残す。現在の研究成果物ではZhang型とCNNを主比較とするが、
第1回進捗報告だけは上記の時系列に従いCanny方式とCNNを表示する。

Kojima採用法の同一条件での初回評価は、OOD 480枚で71/480（14.8%）、撮像診断
1,456枚で126/1,456（8.7%）。同じIoU 0.80基準のZhang 2019型（再現実装）は
それぞれ250/480（52.1%）、556/1,456（38.2%）だった。これは再現実装間の比較であり、
原論文が報告した性能同士の比較ではない。

## 重要な過去判断

- CNNのRANSAC仮説スコアは、周長で割ると小楕円を優遇するため
  `perimeter_power=0.0`（総支持量）を採用。
- CNN教師マスクは遮蔽部分を含む完全なPAF内周リング。
- 推論尤度が遮蔽部で弧になることは矛盾ではない。画素損失と局所証拠により、
  不可視部分の確率が可視部分より低くなるため。
- CG正解はPAF内周輪郭。CAD頂点投影の詳細は発表本文では省略可能。
- 714枚は102カメラ条件×7照明条件。

## 再生成入口

```powershell
# Zhang型とCNNの文献ベースライン主比較
.venv\Scripts\python.exe run_literature_comparison.py

# 旧方式を含む監査図も必要な場合
.venv\Scripts\python.exe run_literature_comparison.py --include-ablations

# 第2段階集計と発表用画像
.venv\Scripts\python.exe -m paflab.reporting.summarize_second_stage
.venv\Scripts\python.exe -m paflab.reporting.build_summary_figures

# 第1回進捗報告用の図版（Canny + 輪郭別RANSACとCNN）
.venv\Scripts\python.exe -m paflab.reporting.build_first_progress_figures
```
