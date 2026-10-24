# 学習回数と穴領域教師の検証

設定は`convergence.json`、実行は`../convergence_study.py`、図表作成は`../report_convergence.py`。出力は`../output/paf_convergence_v1/`。元の12epoch実験と画像は上書きしない。

元と同じ716/154/154分割、3seed、Tiny U-Net、256入力、batch16、AdamW、学習率0.001、BCE正例重み12＋Dice重み1で60epoch学習する。各2,700更新。validation loss最小のcheckpointを採用し、全seedを報告する。学習初期値は同じseedから作るが、GPUの非決定的演算による元履歴との差も保存する。

内周線教師では元と同じ重み付きRANSACを使う。元のtest154枚は改善比較と条件分岐の参照評価とする。3seedの平均主成功数が最良幾何手法135/154を厳密に超えなければ、穴領域教師も同じ設定・60epochで学習する。

穴領域教師は正解楕円の内部全体を塗りつぶす。推論では確率0.5以上の最大連結領域を選び、その外周に楕円を当てはめる。正解による候補選択や閾値調整は行わない。教師表現に伴う後処理変更を含む方式比較である。

方式確定後、元と同じ生成設定でseed2026102411の新規154画像を生成する。全元画像との寸法非重複を検査し、元12epochモデル・延長モデル・必要なら穴領域モデル・既存幾何2手法を共通の画像で確認する。新しい形状・撮影条件・幾何方式は追加しない。

主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px。独立2プロセスでの検証17画像の推論一致を調べ、全評価原記録を元ラベルから再計算する。60epochの終了だけで十分な収束を断定しない。

```powershell
& .venv/Scripts/python.exe -u -X utf8 -B parametric_paf/convergence_study.py run
& .venv/Scripts/python.exe -X utf8 -B parametric_paf/report_convergence.py
```

`report/REPORT.md`に比較表、`learning_curves.png`に全学習履歴、`predictions.png`に新規評価の先頭3画像の推定を保存する。緑は正解、赤は推定。`decision.json`は条件分岐の根拠、`evaluation/`は画像ごとの原記録、`report/audit.json`と`report_verification.json`は照合結果を保持する。

この検証は指定範囲の合成PAF画像を対象とする。実画像への汎化、形状多様化の因果効果、対象なし判定は別の未検証事項である。Zhang 2019型（再現実装）は論文構成を参考にした実装であり、著者コードの完全再現ではない。
