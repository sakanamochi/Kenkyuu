# 形状・外観多様化の2×2実験

実験ID：`paf_shape_appearance_v1`。研究上の根拠は [PRETRAINING_REPORT.md](PRETRAINING_REPORT.md)。学習開始前の検査を終えてから、各arm・seedを明示して学習する。

## 比較条件

|arm|学習形状|外観・照明|画像枚数|
|---|---|---|---:|
|A|単一：train_00|基本分布|1,024|
|B|単一：train_00|拡張分布|1,024|
|C|16形状、各64枚|基本分布|1,024|
|D|16形状、各64枚|拡張分布|1,024|

主比較はD−B、補助比較はC−A。同じ画像番号はカメラ位置・内周の投影寸法が対応し、A/CとB/Dでは材質・照明・露光も一致する。追加の遮蔽・画像劣化・オンライン拡張はこの実験では使わない。

モデルは既存Tiny U-Net、base_channels=16。各armを共通の3 seed（20260907/20260908/20260909）でランダム初期化する。元1194M重みは引き継がない。12epoch、batch=16、各768 optimizer更新、AdamW、損失と学習率はprotocol.jsonに固定。3 seed間の結論を確認するまで学習枚数や形状数を拡大しない。

全armで同一の未学習validation 8形状×24条件＝192枚を用い、validation loss最小のcheckpointを保存する。検証画像はexpanded分布で共通。これは形状外validationによるモデル選択であり、最終テストではない。学習画像と検証画像は形状ID・寸法ベクトルの両方で分離する。

## 形状族と統制

上端内径を1mに固定し、下端外径/内径1.35〜1.90、高さ/内径0.15〜0.45、半径方向肉厚/内径0.04〜0.12を変える。これらは合成実験の範囲であり、実在PAFの寸法統計に基づく範囲ではない。

上端外径=1+2×肉厚。下端円筒高さは上端外径の10%で連動し、内壁は外壁と同じテーパで、底面は高さに連動する。リブは0本で固定。したがって肉厚・高さ変更は陰影や下部構造も含む総合的な形状操作であり、局所輪郭だけの独立操作とは主張しない。新しい実験に不要な既存生成器のパラメータ独立化は行っていない。

学習族は滑らかなテーパ。カメラは内周中心を注視し、傾斜5〜70°、方位0〜360°、距離/内径3.2〜4.2、焦点距離55mm、描画480×480、全検出器256×256入力。黒背景のみ。正解は上端内周の完全な投影楕円。

## 凍結したテスト

|区分|形状数|条件|解釈|
|---|---:|---|---|
|test_interpolation|12|形状ごと共通48条件|同じ形状族・定義範囲内の新しい寸法組合せ|
|test_extrapolation|12|同上|下端径・高さ・肉厚のいずれかが訓練範囲外|
|test_structure|6|同上|円筒＋フランジ3、段付き断面3。訓練とは異なる合成断面|
|各区分の_capture|同じ30形状|別の16条件|傾斜75/80°、露光±1、横方向の画像位置ずれ±0.1を含む複合ストレス|
|known_shape_monitor|学習16形状|未使用の共通48条件|既知形状の性能低下を調べる補助。未知形状テストとは別集計|
|absent_test|黒背景1種・穴なし円盤1形状|黒画像1枚＋円盤63条件|対象なしの誤検出を別評価。64個の独立形状とは扱わない|

テスト形状・乱数・条件は学習前に固定するが、学習前には推論評価しない。構造族の描画QAは別の開発形状を用いる。これらの合成断面は実在する別PAFのCADではなく、独立した実機構造への一般化を保証しない。1194Mと既存8モデルの比較結果は開発資料に限定する。

画像位置ずれはBlenderカメラの水平shiftを画像幅比±0.1に設定する。撮像条件の補足は複合ストレスであり、個々の要因の因果寄与を分離した実験ではない。背景は維持する。

## 指標・判断

主成功はIoU≥0.8かつ双方向輪郭距離p95≤2px。互換IoU成功、距離の平均・p95・長径比、未検出、誤楕円、確率マップの完全内周支持も保存する。主指標の閾値は開発データで固定し、テストで変更しない。

差は同じ形状・撮像条件で対応を取り、形状別成功率の平均差を各学習seedで報告する。学習seedは3本をそのまま示す。画像を独立した形状として数えない。信頼区間を追加する場合は条件の対応を維持した形状単位の再標本化を用い、12形状・6形状などの少数形状による限界を明記する。構造族は種類ごとの結果も示し、補間・外挿・構造・撮像ストレスを一つの平均にしない。

小規模実験の継続基準：validationにおいて3 seed全てでD−Bが正、平均で10ポイント以上改善。最終的に有効性を主張する条件として、既知形状monitorの低下と誤楕円出力増加は5ポイント以内を目安とする。10ポイントは10画像あたり1画像分の改善を求める研究上の判断幅であり、運用・安全の許容値ではない。これは事前に固定した判断基準で、統計的な検出力を保証しない。

小規模実験のvalidation判断後、最終テストを一度評価する。最終テストを使って調整や規模拡大を繰り返さない。既知形状monitorも最終評価に位置づけ、結果でモデルを選び直さない。DとBが同等なら形状多様化の必要性を取り下げ、外観拡張のみで十分な範囲や残る失敗原因を検討する。

## ファイル構成

- `study.py`：計画・設定の生成、学習前検査、凍結、明示指定したarm/seedの学習。
- `render_study.py`：計画を読み、既存生成器を再利用してCG・正解・manifestを生成。別断面は同ファイルの小さな形状生成関数に限定。
- `tune_geometry.py`：validationだけでCanny閾値を選択。
- `evaluate_study.py`：既存CNN・幾何実装を呼び出し、固定した指標で評価。
- `render_absent_study.py`：対象なし補足の描画。
- `diagnose.py` / `diagnosis_metrics.py` / `render_diagnostics.py` / `summarize_diagnosis.py`：既存CGの再現用診断。oracleは学習評価に混ぜない。
- `study/protocol.json` / `shapes.json` / `conditions.json` / `configs/` / `freeze.json`：Git管理する実験定義とhash。画像・重み・結果は `parametric_paf/output/` にローカル保存。

既存の標準実験、生成器、学習器、検出器は変更せず、新しい実験の入口を追加して接続している。学習重み・historyはarm/seedごとの別フォルダに保存し、既存1194M重みを上書きしない。

## 実行（リポジトリ直下）

現在の入力検査のみ：

```powershell
& .venv/Scripts/python.exe -X utf8 parametric_paf/study.py check
```

次に開始する学習1本（このコマンドは学習する）：

```powershell
& .venv/Scripts/python.exe -X utf8 parametric_paf/study.py train --arm A --seed 20260907
```

他のarmとseedも同じ形式で実行し、計12本を比較する。既存checkpointがある場合は上書きを拒否する。途中失敗時の自動再開は実装していないため、途中の重みを完成結果に混ぜず、入力・設定を保持して新しい実験出力先で扱う。

再現描画と検証用調整（未生成の保存先で実行。選択済み設定を上書きしない）：

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python-exit-code 1 --python parametric_paf/render_study.py -- --stage data
& .venv/Scripts/python.exe -X utf8 parametric_paf/tune_geometry.py
```

独立テストの描画・評価（小規模学習とvalidation判断が終わってから）：

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python-exit-code 1 --python parametric_paf/render_study.py -- --stage test
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python-exit-code 1 --python parametric_paf/render_absent_study.py
& .venv/Scripts/python.exe -X utf8 parametric_paf/evaluate_study.py --dataset parametric_paf/output/paf_shape_appearance_v1/test --checkpoint parametric_paf/output/paf_shape_appearance_v1/models/A/20260907/cnn_best.pt --output parametric_paf/output/paf_shape_appearance_v1/evaluation/A/20260907/test
```

同じcheckpointでdatasetを`absent_test`に切り替え、出力先も`absent`に分ける。学習後のvalidationも同じ評価入口でdatasetを`validation`にして測定できる。独立評価では`--limit`を付けない。全てのarm/seedで同じ凍結テストを使う。パスの変更は対応する出力先と記録を明示する。

既存CGの原因診断の再現は `python parametric_paf/diagnose.py --output <新しい保存先>`。陰影補足と分類の実験ID・ソースはPRETRAINING_REPORT.mdを参照する。
