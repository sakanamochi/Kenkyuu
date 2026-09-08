# 簡易PAFのパラメトリック生成

形状・外観多様化の2×2学習実験は [study/README.md](study/README.md)、学習前の診断結果と研究根拠は [study/PRETRAINING_REPORT.md](study/PRETRAINING_REPORT.md) を参照してください。以下は既存モデル生成と初期比較の実行方法です。

モデル生成と形状別の検出比較を行う独立した構成です。
実在するPAFの忠実な再現ではありません。寸法単位はmです。

- `generate.py`: 本体・リブ生成、OBJ・Blender形式の保存、プレビュー描画。
- `config.json`: 共通寸法と各モデルの変更値。
- `output/<モデル名>/`: model.blend、model.obj、model.mtl、preview.png、metadata.json。

## 寸法

|設定名|図との対応|意味|
|---|---|---|
|top_outer_diameter|A|上端外径|
|bottom_outer_diameter|B|下端外径。上端外径以上|
|taper_height|C|テーパ部分の高さ|
|wall_thickness|D|半径方向の肉厚。内径は上端外径−2×肉厚|
|color|色|Base Color（線形RGB、0〜1）|
|rib_count|リブ|等間隔のリブ本数。0でなし|

内壁は外壁と同じ傾斜で、半径方向の肉厚を一定にしています。
下端円筒の高さは上端外径の10%、リブ幅は2.5%、突出量は3.5%に固定。
リブは本体とは別の閉じたメッシュで、根元を外壁に少し埋め込みます。
製造用単一ソリッドではなく画像生成用です。
下端には上下を閉じた円筒を別メッシュで配置し、その上面（z = −taper_height）が
PAF開口部の底面になります。上端は開口しており、穴は下まで貫通しません。
金属度0.25、粗さ0.40を共通にしています。

検出対象は上端内周円です。中心を原点、軸を+Z、本体を−Z側に配置します。
内周192点をmetadata.jsonに保存し、Blenderではtarget_inner_rim頂点グループに登録します。
OBJはZ-upです。既存のOBJ読込設定と接続する際は軸の対応を確認してください。

## 実行（リポジトリ直下のPowerShell）

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python-exit-code 1 --python parametric_paf/generate.py
```

Blender同梱Pythonで動作し、追加pipインストールは不要です。
既存venvのPythonで直接実行するスクリプトではありません。
各モデルはdefaultsに個別設定を上書きして生成します。同名出力は再実行で上書きします。

付属設定は基準、下端外径変更、高さ変更、肉厚変更、リブ追加、色変更の6例です。
共通の正投影カメラと柔らかい照明によるプレビューは形状確認用で、研究評価用画像ではありません。
極端に大きい寸法ではカメラ範囲の調整が必要です。
ランダムな大量生成・学習データへの接続は含みません。

## 既存検出器との初期比較

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --python-exit-code 1 --python parametric_paf/render_comparison.py
& .venv/Scripts/python.exe -X utf8 parametric_paf/compare.py
```

- `comparison.json`: 共通の視点・照明・画像サイズ。
- `render_comparison.py`: 元1194Mと6例に上端外径変更を加えた計8モデルを描画。
- `compare.py`: 既存のCNN・Zhang 2019型（再現実装）・Canny検出器を呼び出して集計。
- `output/comparison/comparison.md`: 成功率と視点傾斜別の表。
- 同フォルダの`results.csv`: 全条件の成否・IoU・推定楕円。
- `summary.json`: 各変更による基準モデルからの改善・悪化数。
- `provenance.json`: 使用した重みのSHA256、検出設定、入力サイズ。
- `examples/`: 撮影条件で固定抽出した、正解緑線・推定赤線の比較画像。

基準設定は各48条件、合計384画像です。色変更は形状変更と分けて解釈します。
学習・閾値調整は行わず、既存のoutput/models/cnn_best.ptとconfig/experiment.jsonを使用。
全方式に同じ256×256画像を渡します。入力解像度が異なる評価結果とは直接比較できません。
黒背景とSun照明を使用し、正投影のプレビューとは異なり透視投影で評価します。
上端内周中心を注視し、内径に比例して距離を調整して、形状間の投影内周を一致させます。
元1194Mは形状比を維持して縮尺と原点をそろえ、既存の材質設定を適用。
この参照と簡易モデルの差は形状だけでなく材質や細部も含むため、変更効果はbaselineとの差を見ます。
単一の格子条件・乱数での予備診断です。未知形状一般への汎化や多様化学習の効果は未検証です。
