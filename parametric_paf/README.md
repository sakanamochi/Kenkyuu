# PAFのパラメトリック生成

現行のランダムPAF仕様と実行方法は [random_paf/README.md](random_paf/README.md) を参照する。

上端外径Dを基準とし、下端外径1D〜3D、全高0.5D〜3D、穴径0.4D〜0.95Dを独立に抽選する。肉厚は0.02D固定。生成入口は`random_dataset.py`、形状定義は`paf_model.py`、設定は`random_paf/config.json`。

[生成済み1,024画像の一覧](output/paf_random_outer_d_v2/dataset_preview.png) と [manifest](output/paf_random_outer_d_v2/manifest.json) を確認できる。

同じデータでの60epoch学習と穴領域教師の条件付き比較は、[random_paf/CONVERGENCE.md](random_paf/CONVERGENCE.md)を参照する。

既存の標準検出器・学習器はルートREADMEを参照する。保存済みの別設定の画像・重み・評価値を、現行データセットの性能結果として扱わない。
