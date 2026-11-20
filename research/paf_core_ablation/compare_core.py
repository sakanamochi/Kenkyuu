"""mainと最終候補の共通部分・差分を、保存したソースと採用設定から確認する。"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.diagnose import sha


def definitions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def main():
    out = ROOT/'output/paf_core_ablation_v2'
    ref = ROOT/'output/paf_core_ablation_v1/main_reference'
    inventory = read_json(ref/'inventory.json')
    for relative, digest in inventory['files'].items():
        assert sha(ref/relative) == digest
    a, b = definitions(ref/'paflab/model.py'), definitions(ROOT/'paf_ring_detection/methods/cnn.py')
    assert all(a[name] == b[name] for name in ('DoubleConv', 'TinyUNet'))
    names = ('_point_distances', '_angular_coverage', '_ellipse_perimeter', '_ransac_hypothesis_score', '_is_valid_ellipse', 'fit_ellipse_ransac')
    a, b = definitions(ref/'analysis/ellipse_ransac.py'), definitions(ROOT/'paf_ring_detection/methods/ransac.py')
    assert all(a[name] == b[name] for name in names)
    main_config = read_json(ref/'config/research_experiment.json')
    final_path = ROOT/'output/paf_generalization_v2/runs/boundary_projection_relative/20260907/provenance.json'
    final = read_json(final_path)['config']
    differences = {key: {'main': main_config['cnn_ransac'][key], 'final': value} for key, value in final['ransac'].items()
                   if main_config['cnn_ransac'][key] != value}
    assert differences == {'min_inliers': {'main': 45, 'final': 12}}
    params = sum(p.numel() for p in TinyUNet(16).parameters())
    assert params == 482737
    files = [ref/'inventory.json', ref/'paflab/model.py', ref/'paflab/labels.py', ref/'paflab/evaluate_cnn.py',
             ref/'analysis/ellipse_ransac.py', ROOT/'paf_ring_detection/methods/cnn.py',
             ROOT/'paf_ring_detection/methods/ransac.py', ROOT/'research/paf_generalization/boundary.py',
             ROOT/'research/paf_generalization/boundary_detector.py', ROOT/'research/paf_generalization_v2/detector.py', final_path]
    write_json(out/'core_comparison.json', {'main_commit': inventory['main_commit'],
        'cnn_classes_ast_identical': True, 'parameters': params, 'ransac_functions_ast_identical': list(names),
        'ransac_parameter_differences': differences,
        'files': {str(p.relative_to(ROOT)): sha(p) for p in files}, 'implementation_sha256': sha(Path(__file__)),
        'conclusion': 'CNN構造と重み付きRANSAC本体は同じ。教師・点集合・閾値・最小支持点数・学習分布が異なる。最終候補を共通の基礎として寄与比較する。'})
    (out/'CORE_COMPARISON.md').write_text('''# mainと最終候補の基本部分

CNN＋重み付きRANSACという基本構成は同じ。教師と点群処理を含む検出器全体は同一ではない。

|項目|main|hole-region-comparisonの最終候補|
|---|---|---|
|CNN|Tiny U-Net16、482,737パラメータ|同じクラス定義・パラメータ数|
|RANSAC本体|確率によるサンプリング・支持量・楕円再当てはめ|主要6関数のASTが一致|
|教師|幅3px・Gaussian平滑化した完全リング。遮蔽内も含む|穴領域から作る可視境界。既知の矩形遮蔽内は除く|
|推定へ渡す点|確率0.35以上の帯全体。最大6,000点|確率0.5以上の境界を中心線化した点|
|最小支持点数|45点|12点|
|RANSACの他の主設定|450仮説・距離2px・再当てはめ3回など|同じ値|
|一様画像|入力による明示的な保留なし|画像単体の入口で未検出を返す|
|学習画像|1194M|多様形状＋投影・表面模様・開口幅遮蔽|

mainの学習設定には12epochとcosine学習率があり、最終候補は60epoch・一定学習率。画像劣化・分割・入力のモノクロ化も研究段階で異なる。過去集計の差をそのまま特定の変更の効果とは解釈しない。

正しいmainリング教師を推定器へ渡す診断では、1194M通常112例は100%、多様形状通常154例は58.44%。同じ閾値0.35・最小45点で中心線化すると、多様形状は98.05%。最小12点も使うと99.35%。帯のまま最小12点へ変えても58.44%で、点数閾値だけが主因ではない。これはGT教師による処理の診断で、学習器の性能や厳密な上限ではない。

したがって、最終候補の可視境界教師・中心線RANSACを比較の共通基礎にする。mainへの一致を目的にこの差を戻さず、投影・外観の2要因と開口幅遮蔽の寄与を調べる。CNNと幾何の併用を今回の新規変更として扱わない。

根拠：core_comparison.json、output/paf_core_ablation_v1/teacher_diagnosis.json、point_teacher_diagnosis.json。
''', encoding='utf-8')
    print('main/final CNN and six RANSAC functions identical; representation differences recorded', flush=True)


if __name__ == '__main__':
    main()
