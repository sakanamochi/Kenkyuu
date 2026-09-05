# 簡易PAFのパラメトリック生成

既存の学習・評価コードから独立した、モデル生成だけの試作です。
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
上下の開口に蓋はありません。金属度0.25、粗さ0.40を共通にしています。

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
CNN評価・ランダムな大量生成・学習データへの接続は含みません。
