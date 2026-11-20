# PAF再構成：現在の仕様と次の作業

更新日：2026-11-21（JST）

## 方針と実装

ブランチは`codex/paf-rebuild`。独立した`paf_rebuild/`に実行入口`scripts/`、設定`configs/`、共通部品`src/core/`、推論`src/inference/`、評価`src/evaluation/`、学習部品`src/training/`を置く。旧`research/`・`paf_ring_detection/`は実行時にimportしない。

実行用データは`data/`、両学習元×3seedのモデルは`models/`、固定推定は`references/`。設定とmanifestの参照はこのフォルダ内の相対パスで完結する。`data/inventory.json`が全ファイルの内容hashを保持する。検証記録内の過去のパス・実装hashは出典情報であり、実行時の依存先ではない。

必要資産6,396ファイルのhashと参照先の自己完結を確認済み。`outputs/standalone_training_check/verification.json`の全5,446例は既存教師の配列hashと一致し、既定設定の単体推論も保存された確率・楕円・PNGと一致する。旧構成のコード・出力は作業ツリーに残っている。ルートの`Remove-LegacyFiles.ps1`は必要資産と入力照合の検査後、明示した旧構成だけを削除する手動実行用スクリプト。`-WhatIf`で対象を確認できる。Git履歴・仮想環境・研究関連書類・保護された設定フォルダは削除対象に含めない。

画像単体の推論はTiny U-Net16の境界確率→中心線→確率重み付きRANSAC。入力・出力・幾何設定・実行コマンドは[README](../README.md)、学習元・分割・拡張・教師・損失は[学習仕様](TRAINING_DATA.md)を正とする。

学習時の矩形遮蔽例は原画像から直接生成し、固定遮蔽画像は作らない。同じ投影・外観を参照するスロットはキャッシュと対応確認を共有する。JSON読込とファイルhashは`src/core/files.py`、正解投影点からの楕円計算は`src/core/ellipse.py`に共通化する。

## 確認済みの範囲と根拠

|対象|確認結果|根拠|
|---|---|---|
|保存済み6モデルの楕円推論|基本20,748＋強化白飛び6,384＝27,132件の楕円・検出有無・元正解IoU・成功判定が一致|[全件報告](../outputs/reference_verification/REPORT.md)と同フォルダの各seed検証JSON|
|6モデル×2通常画像の確率|12件で確率差0、初期化とstate keyが一致|[確率照合](probability_verification.json)|
|単体CLI|通常1枚の確率・楕円・重ね描き・PNG変換が一致。CPUも同画像で一致。別途代表26条件を照合|[単体推論照合](single_image_verification.json)|
|学習入力・教師|seed20260907、manifest順の全5,446例で旧処理と一致。抽選後の乱数状態も一致|[学習処理照合](training_data_verification.json)|
|学習損失・キャッシュ|代表56例の損失・logit勾配、192例の8bitキャッシュ初回・再読込が一致|同上|
|データ整合性|train・validation・testの群が非重複。全ビューの入力hashと全5,446例の配列列hashを確認|[入力検証](../outputs/training_data_check/verification.json)|
|データ読込と単体出力|キャッシュあり/なしの計512回で入力・教師・乱数状態が一致。遮蔽例は原画像を1回だけ読み、固定劣化は生成しない。単体推論例の確率配列・PNGは保存値と一致|入力検証の`component_checks`|

基本テスト18件合格。確認環境は既存venv、PyTorch 2.13.0+cu130、OpenCV 4.13.0、NumPy 2.2.6、GPUはRTX 4070 Ti。学習処理照合はCPU。保存された検証記録の実装hashは、その検証で使った実装を特定する。

入力検証の空境界教師は1194M train35/validation0、多様形状train26/validation83例。これらは除外しない。trainの件数は指定した抽選列に依存する。

確認結果は実装の数値的一致を示す。精度改善や新たな汎化性能を示すものではない。

## 未実装・未検証

学習ループ・重み保存・checkpoint選択、CG生成、CannyとZhang 2019型（再現実装）の移植、任意データ向けの評価・作図入口は未実装。

shuffle順・ネットワーク勾配・パラメータ更新・再学習の一致、CPU全test、異なる実行環境での完全一致、独立venv、実画像一般への性能は未検証。全testの境界確率配列は照合していない。評価に使うCG testは研究で参照済みで、新規盲検ではない。

## 次の具体的作業

旧構成の削除スクリプトを実行した後、残存ファイルと新構成の動作を確認する。承認要求が無効になる実行環境では、自動削除を完了したと扱わない。

`src/training/`に共通学習ループ、`scripts/train.py`に入口を実装する。条件はランダム初期化、AdamW、一定学習率0.001、weight decay0.0001、batch16、60epoch、元学習元validation loss最小のcheckpoint選択。まずseed20260907の短い学習でshuffle順・損失・ネットワーク勾配・保存重みの再読込を確認する。
