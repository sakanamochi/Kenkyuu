# PAF内周検出：研究方針と引継ぎ

最終更新：2026-11-24（JST）

## 研究モチベーションと問い

- **対象と課題**：能動的デブリ除去（ADR）の対象である非協力なロケット上段には、衛星分離後のPAF（Payload Attach Fitting）内周リングが共通して残存し、接近誘導・相対姿勢推定・把持の指標となる。単眼カメラは小型・低電力で実用的だが、宇宙特有の強い白飛び・黒つぶれ・部分遮蔽が生じる。
- **既存手法の二律背反**：
  - 幾何手法（Canny＋輪郭別RANSAC、Zhang 2019型（再現実装））は特定機体CADを必要としないが、部分遮蔽や明暗劣化、リブ等の不要エッジに弱い（確認済み：半画面遮蔽0〜9%、全13条件43〜53%）。
  - 特定機体CAD学習CNNは劣化耐性が高い一方、実運用では①過去の多国・多世代ロケット上段で詳細CADや正確な表面仕様が入手困難または実物と乖離する、②マルチターゲットADRで機体ごとに詳細CAD構築・再学習を行うコストが大きい、という制約がある。
- **研究の問い**：国際標準クランプバンド規格（937 / 1194 / 1666 / 2624 mm 等）に代表される現実のPAF寸法・構造に基づき合成データを体系的に設計すれば、特定機体の詳細CADを学習に使わずに、幾何手法を上回る劣化耐性と未知PAFクラスへの汎化を両立できるか。また、どの設計要素（形状構造・投影・外観/テクスチャ・遮蔽）がどれだけ汎化に寄与するかを定量的に解明する。

## 現実PAF寸法の調査結果（確認済み事実）

主要ロケットの公式 User's Manual（MHI H-IIA User's Manual Table 4.6-1、Arianespace Ariane 5/6 & Soyuz User's Manuals / Beyond Gravity PAS、ULA Atlas V User's Guide）および保有図面・CAD（`assets/1194M_3/1194M_3.obj`）から整理した代表寸法比：

- H-IIA 1194M（図面単体）：$T = 1215\text{ mm}$、$B \approx 2190\text{–}2266\text{ mm}$（$T/B \approx 0.54\text{–}0.55$）、$H = 480\text{ mm}$（$H/T = 0.395$）、$\text{hole} = 1084\text{ mm}$（$\text{hole}/T = 0.892$）。保有CAD `1194M_3.obj` は下部スカート込みで $T/B = 0.550$、全高比 $0.736 T$（下部スカート高 $\approx 0.19 B$）、$\text{hole}/T = 0.892$。
- H-IIA 1666MA / 1666LS-H：$T = 1666\text{ mm}$、$H = 480\text{ mm}$（$H/T = 0.288$）、$T/B \approx 0.74\text{–}0.76$。
- H-IIA 937LS-H：$T = 945\text{ mm}$、$H = 630\text{ mm}$（$H/T = 0.667$）、$T/B \approx 0.42\text{–}0.43$。
- Atlas V A937 / B1194 / D1666：分離リング単体で $H/T = 0.430 / 0.209 / 0.198$、B1194＋LVAで $H/T = 0.481\text{–}0.816$。
- Ariane 5/6 & Beyond Gravity PAS 937S / 1194VS / 1666MVS / PLA 2624：PAF単体 $H/T \approx 0.07\text{–}0.48$、LVA込み $H/T \approx 0.39\text{–}0.84$、$T/B \approx 0.46\text{–}1.00$。

## 現行の多様形状PAFモデル仕様（アダプタ部＋固定円筒部）

PAFを**上部の可変アダプタ部（円錐台）**と**下部の固定円筒部（コップ状スカート）**に分離し、上端開口から下部円筒底面までが一続きの内部空間を成す10点回転断面メッシュとして生成する（実装：`src/cg/geometry_model.py`、`src/cg/render_diversified.py`、正本設定：`configs/render_diversified.json`）。

- **下部円筒部（固定）**：
  - 下端（円筒）外径：$B = 1.0\text{ m}$ 固定
  - 円筒部高さ：$H_{\mathrm{cyl}} = 0.25 B = 0.25\text{ m}$ 固定（下端は平底で閉じる）
- **上部アダプタ部（可変・独立一様抽選）**：
  - 上端外径比：$T/B \in [0.40, 1.00]$
  - アダプタ高さ比：$H/T \in [0.20, 0.80]$（上端径 $T$ 基準で定義し、$T/B$ に依存した異常な細長・極薄化を防ぐ）
  - 内周穴径比：$\text{hole}/T \in [0.75, 0.95]$
- **肉厚と幾何制約**：
  - 法線方向肉厚：$t = 0.02 T$（上端外径 $T$ 基準）
  - 上端リング幅 $w_{\mathrm{rim}} = (T - \text{hole})/2 > t$ を `make_geometry()` で明示検証する（$t = 0.02 T$ と $\text{hole}/T \le 0.95$ により最小でも $w_{\mathrm{rim}} = 0.025 T > 0.02 T$ を全範囲で保証）。
  - アダプタ内壁と円筒内壁の交点 $z_{\mathrm{joint}} = -H - t \cdot s / (\sqrt{1+s^2} + 1)$ は $s = 0$（直円筒）でも連続かつ常に底面より上方に収まる。
- **確認済みの検証根拠（2026-11-24）**：
  - 回帰テスト `tests/test_paf_system.py` 全10件合格（端8点の10点断面生成、上端リング幅 $\le t$ の拒否、`plan.json` 生成、外部CAD検査・13条件manifest構築・`1666ma` データセット検証、層分離、Tiny U-Net16 482,737パラメータ、撮像劣化、RANSAC/スケルトン、幾何2方式、統一推論、1epoch学習動作確認）。
  - Blender 5.2.0 LTS による実描画検証（`scripts/render_cg.py diversified --stage render --limit 4 --output outputs/paf_cylinder_preview/cg_samples`）を実行し、端8点すべての閉多様体・正体積検証の通過、4形状のCG描画・`sample_model.blend` 保存、正解内周エッジの一致を目視確認済み（プレビュー：`outputs/paf_cylinder_preview/cross_sections.png`、`outputs/paf_cylinder_preview/cg_samples_grid.png`）。

## 未知実機PAF CADモデル評価パイプライン（1666MA・今後の追加モデル共通）

任意の外部CAD（`.obj`）を評価用データセット化する汎用パイプライン（`src/cg/cad_dataset.py`、`src/cg/render_1194m.py`、`scripts/render_cg.py cad`）を備え、`scripts/evaluate.py`・`scripts/report.py` は `configs/experiment.json` の `datasets` に登録された任意のデータセット名で評価・集計できる。

- **汎用CAD登録手順**：
  1. `scripts/render_cg.py cad --stage inspect --model <OBJ>` で軸範囲・マテリアル・上端内周リング候補（`z`・`radius`・`vertex_count`）・推奨カメラ距離を確認する。
  2. `configs/render_<name>.json` を作成し、`configs/experiment.json` の `datasets` に登録する。
  3. `scripts/render_cg.py cad --stage render / manifest / verify --render-config configs/render_<name>.json` で原CG描画・全13条件 `test` manifest生成・全画像および画面内整合検証（`dataset_preview.png` 出力）を行う。
- **1666MA（確認済み事実）**：
  - 素材：`assets/1666MA/1666MA.obj`（$B = 2.222\text{ m}$、$T = 1.666\text{ m}$、$T/B = 0.750$、アダプタ高 $H = 0.480\text{ m}$・$H/T = 0.288$、下部スカート高 $0.382\text{ m}$、上端内周穴径 $1.540\text{ m}$・$\text{hole}/T = 0.924$、正解内周リング $z = 0.48\text{ m}, r = 0.77\text{ m}, 35\text{頂点}$）。
  - 設定・出力：`configs/render_1666ma.json`、`configs/experiment.json`（`datasets.1666ma`）、`data/1666ma/base/`。
  - 検証結果：原CG 252枚（仰角 `[0, 20, 40, 60, 75]` × 方位 `[0, 90]` × 距離 `[4.0, 5.2]` × ターゲット2種 × 7照明）と全13条件評価manifest（3,276件、全件 `split="test"`）を生成・検証済み。全画像で正解内周エッジが画面内に収まり（最小余白 `46.72 px`、256px換算内周長径 `[119.65, 169.01] px`）、`data/1666ma/base/dataset_preview.png` で正解楕円の一致を目視確認済み。

## 現実PAF範囲（単一円錐台版）での確認済み要因分解結果

`data/diversified_limited_size/`（単一円錐台・現実PAF範囲756形状）で実施した体系的要因分解の確認済み結果（統一評価条件：モノクロ256×256、塗りつぶし楕円 IoU ≥ 0.8、全13条件・白飛び最大内部強度2.0、出力正本：`outputs/ablation/`）。

- **外観・テクスチャ分解（Group B、seed 20260907）**：
  - `B1_plain`（無地）：1194M **36.06%** / 現実PAF **75.92%**
  - `B2_albedo`（明暗のみ、模様なし）：1194M **35.44%**（`-0.62 pp`） / 現実PAF **75.97%** → 明暗変化のみでは未知1194Mへの転移は改善しない。
  - `B4_noise`（ノイズ）：1194M **47.80%**（`+11.74 pp`） / 現実PAF **75.52%**
  - `B3_stripe`（放射ストライプ）：1194M **57.42%**（`+21.36 pp`） / 現実PAF **74.93%**
  - `B5_checker`（チェッカー）：1194M **62.98%**（`+26.92 pp`） / 現実PAF **75.17%**
  - `B6_mixed`（3種混合）：1194M **65.45%**（`+29.39 pp`） / 現実PAF **75.77%**
- **主要設計要素の寄与分解（Group C、seed 20260907）**：
  - `C0` 元構図のみ（固定遮蔽）：1194M **31.80%** / 現実PAF **74.73%**
  - `C1` ＋投影のみ（固定遮蔽）：1194M **34.07%** / 現実PAF **76.22%**
  - `C2` ＋3種模様のみ（固定遮蔽）：1194M **58.10%** / 現実PAF **74.53%**
  - `C3` ＋投影＋3種模様（固定遮蔽）：1194M **64.49%** / 現実PAF **75.12%**
  - `B6_mixed` ＋開口幅オンライン遮蔽：1194M **65.45%**（半画面遮蔽 60.71→66.07%） / 現実PAF **75.77%**（半画面遮蔽 85.06→90.91%）
- **3seed平均比較（20260907 / 20260908 / 20260909）**：
  - `1194M学習CNN（3seed）`：1194M **79.97%** / 現実PAF **71.83%**
  - `現実PAF・3種混合CNN（B6_mixed・3seed）`：1194M **64.54%**（通常 88.69%、白飛1.0 **72.92%**、黒潰1.0 **27.08%**） / 現実PAF **75.77%**（通常 **100.00%**、遮蔽0.5 **91.99%**）
  - `Canny＋輪郭別RANSAC`：1194M **46.09%** / 現実PAF **52.85%**
  - `Zhang 2019型（再現実装）`：1194M **42.99%** / 現実PAF **48.90%**
- **仮説（未検証）**：単一円錐台版の `B6_mixed` で1194M通常失敗（仰角20°付近の外周誤認）が生じた一因は、低背形状の浅い底面反射による内周コントラスト低下と考えられる。今回の下部固定円筒部（深さ $+0.25 B$）の一体空間構造により開口内部の奥行きと暗さが増し、仰角20°付近の外周誤認が軽減される可能性がある（未検証）。

## 現行システム構成（Kenkyuu/ 直下）

- 実行スクリプト（`scripts/`）：`predict.py`、`train.py`、`evaluate.py`、`report.py`、`render_cg.py`、`ablation.py`
- 設定正本（`configs/`）：`experiment.json`、`render_1194m.json`、`render_1666ma.json`、`render_diversified.json`
- 本体パッケージ（`src/`）：`defaults.py`、`core/`、`inference/`、`training/`、`evaluation/`、`cg/`
- データ・重み・素材：`assets/1194M_3/`、`assets/1666MA/`、`data/`、`models/`、`references/`、`outputs/`
- 運用規則：基本は固定1seed（`20260907`）運用とし、3seed学習・評価は明示指定時のみ実行する。エージェントの判断で一時的な検証・可視化に使用したスクリプトやキャッシュ（ワークスペース内・スクラッチ領域を含む）は使用後に必ず片付け、コード・作業領域を肥大化させない。

## 未検証事項と制約

- **未検証事項**：
  - 新しい「アダプタ部＋固定円筒部」データセット（`data/diversified_paf_cylinder/`）の全756形状生成および1seed学習・評価（特に1194M仰角20°の外周誤認率・通常成功率・全13条件成功率への効果）。
  - `data/1666ma/base/` および今後追加予定の他機体PAF CADモデル（さらに2モデル予定）に対する統一評価、ならびに実画像・任意背景での汎化評価。
  - 単体推論で確認したいぶき実画像に対する正解内周定義と複数実画像での定量評価。
- **制約・範囲**：
  - 現行の評価範囲は黒背景・モノクロ256px・単一開口CGであり、実画像一般・任意背景・未知機体一般への汎化は未検証。
  - GPU学習は決定的アルゴリズムを強制しておらず、再学習重みのビット単位一致は保証しない。
  - Zhangの正規出典は Zhang, Pan, Ma, *Sensors* 2019, DOI `10.3390/s19235243`、実装は必ず「Zhang 2019型（再現実装）」と表記する。

## 次の具体的作業

1. `outputs/paf_cylinder_preview/cross_sections.png` と `cg_samples_grid.png` の形状・範囲（特に $H/T$ 下限 `0.20` や $T/B \in [0.40, 1.00]$）をユーザーと確認し、必要があればパラメータ範囲を調整する。
2. 形状仕様確定後、`data/diversified_paf_cylinder/` の全756形状・マルチビューCGを生成し、固定1seed（`20260907`）で学習・評価を実施して単一円錐台版（`B6_mixed`）と比較する。
3. 1666MA（構築済み）および追加予定の他機体PAF CADモデル2種を `scripts/render_cg.py cad` でデータセット化し、統一評価と発表資料・卒論への反映を進める。
