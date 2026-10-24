"""固定候補の主比較・追加確認を照合し、卒論用の要約と比較図を作る。"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from research.paf_generalization.diagnose import BASE, sha
from research.paf_generalization_v2 import confirmation, family_confirmation

OUT = ROOT/'output/paf_generalization_v2'
DEST = OUT/'final'
SEEDS = (20260907, 20260908, 20260909)
EFFECTS = [('all', None, '全条件'), ('clean', 0., '通常'),
           ('black_rectangle', .5, '半画面遮蔽'), ('sensor_whiteout', .5, '白飛び0.5'),
           ('sensor_black_crush', .5, '黒つぶれ0.5')]
LABELS = {'cnn_diversified': '穴領域CNN（多様形状学習）', 'cnn_1194m': '穴領域CNN（1194M学習）',
          'canny': 'Canny＋輪郭別RANSAC', 'zhang2019': 'Zhang 2019型（再現実装）',
          'boundary_surface': '前段：境界＋表面模様',
          'boundary_projection_relative': '採用：投影＋開口幅遮蔽'}


def collect_main():
    """同じtest画像の全13条件を保存し、図示する条件だけで集計し直さない。"""
    fixed = confirmation.verify()
    selection = read_json(OUT/'candidate_selection.json')
    assert selection['variant'] == fixed['variant'] == 'boundary_projection_relative'
    for name, digest in selection['files'].items():
        assert sha(ROOT/name) == digest, name
    files = {str((OUT/'candidate_selection.json').relative_to(ROOT)): sha(OUT/'candidate_selection.json')}
    records = []
    methods = ['cnn_diversified', 'cnn_1194m', 'canny', 'zhang2019', fixed['variant']]
    for dataset in ('1194m', 'diversified'):
        ids = {s['sample_id'] for s in load_samples(BASE/'datasets'/dataset, 'test')}
        for method in methods:
            seeds = SEEDS if method not in ('canny', 'zhang2019') else (None,)
            for seed in seeds:
                path = (OUT/f'runs/{method}/{seed}/test_{dataset}.json' if method == fixed['variant'] else
                        BASE/f"results/{dataset}/{method}_{seed if seed is not None else 'fixed'}.json")
                data = read_json(path)
                assert len(data['rows']) == len(ids) and {r['sample_id'] for r in data['rows']} == ids
                files[str(path.relative_to(ROOT))] = sha(path)
                records.extend({**r, 'dataset': dataset, 'method': method, 'seed': seed} for r in data['rows'])
    summaries = []
    for dataset in ('1194m', 'diversified'):
        for method in methods:
            rows = [r for r in records if r['dataset'] == dataset and r['method'] == method]
            for effect, severity in [('all', None)]+sorted({(r['degradation'], r['severity']) for r in rows}):
                part = [r for r in rows if effect == 'all' or (r['degradation'], r['severity']) == (effect, severity)]
                summaries.append({'dataset': dataset, 'method': method, 'degradation': effect, 'severity': severity,
                    'n_images': len({r['sample_id'] for r in part}), 'success_rate': float(np.mean([r['success'] for r in part])),
                    'per_seed': {str(seed): float(np.mean([r['success'] for r in part if r['seed'] == seed]))
                                 for seed in dict.fromkeys(r['seed'] for r in part)}})
    return fixed, summaries, files


def figure():
    """平均とseed範囲を区別し、信頼区間やモデル統合として表示しない。"""
    fixed, rows, files = collect_main()
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'path', 'axes.unicode_minus': False})
    methods = ['cnn_diversified', 'cnn_1194m', 'canny', 'zhang2019', fixed['variant']]
    colors = ['#9aa7b8', '#476a9e', '#b7b7b7', '#7c858b', '#17866a']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, (effect, severity, title) in zip(axes.flat, EFFECTS[1:]):
        for index, method in enumerate(methods):
            row = next(r for r in rows if (r['dataset'], r['method'], r['degradation'], r['severity']) ==
                       ('1194m', method, effect, severity))
            value = 100*row['success_rate']
            repeats = [100*x for x in row['per_seed'].values()]
            ax.barh(index, value, color=colors[index], height=.64, zorder=2)
            if len(repeats) == 3:
                ax.errorbar(value, index, xerr=[[value-min(repeats)], [max(repeats)-value]],
                            fmt='none', ecolor='#222222', capsize=3, linewidth=1, zorder=3)
            ax.text(max(repeats)+1.5, index, f'{value:.1f}', va='center', fontsize=10)
        ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 113)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('検出成功率 [%]')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=.25, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('1194M：同じtest画像での内周検出（IoU ≥ 0.8）', fontsize=15)
    fig.supxlabel('各条件112画像。CNNは3seed平均、線はseed最小〜最大（信頼区間ではない）。幾何は1回評価。', fontsize=10)
    DEST.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(DEST/f'comparison_1194m.{suffix}', dpi=180)
    plt.close(fig)
    write_json(DEST/'main_comparison.json', {'rows': rows, 'files': files,
        'frozen_candidate_sha256': sha(OUT/'confirmation/frozen_candidate.json')})
    print('Saved primary comparison figure and all-condition source data', flush=True)


def table(rows, methods, source=None, view=None):
    """主比較13条件と追加確認4条件の全条件平均を区別して表にする。"""
    lines = ['|方式|全条件|通常|半画面遮蔽|白飛び0.5|黒つぶれ0.5|', '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        selected = [r for r in rows if r.get('method', r.get('variant')) == method and
                    (source is None or r['dataset'] == source) and
                    (view is None or (r['view'] == view and r['family'] == 'all'))]
        values = []
        for effect, severity, _ in EFFECTS:
            matches = [r for r in selected if r['degradation'] == effect and
                       ('severity' not in r or r['severity'] == severity)]
            assert len(matches) == 1, (method, effect, source, view)
            values.append(f"{100*matches[0]['success_rate']:.2f}%")
        lines.append('|'+LABELS[method]+'|'+'|'.join(values)+'|')
    return lines


def report():
    """両追加評価が全8設定揃い、元ラベル照合済みの場合だけ結論資料を生成する。"""
    fixed, main, files = collect_main()
    chart = read_json(DEST/'main_comparison.json')
    assert chart['rows'] == main and chart['files'] == files
    assert chart['frozen_candidate_sha256'] == sha(OUT/'confirmation/frozen_candidate.json')
    family_confirmation.verify()
    checked = {}
    for name, count in [('confirmation', 19264), ('family_confirmation', 6144)]:
        folder = OUT/name
        data = read_json(folder/'summary.json')
        assert data['prediction_checks'] == count
        assert data['frozen_candidate_sha256'] == sha(folder/'frozen_candidate.json')
        for path, digest in data['files'].items():
            assert sha(ROOT/path) == digest, path
        checked[name] = count
        files[str((folder/'summary.json').relative_to(ROOT))] = sha(folder/'summary.json')
    family = read_json(OUT/'family_confirmation/family_summary.json')
    assert family['audited_summary_sha256'] == sha(OUT/'family_confirmation/summary.json')
    assert family['frozen_candidate_sha256'] == sha(OUT/'family_confirmation/frozen_candidate.json')
    extra = read_json(OUT/'confirmation/summary.json')
    audit = read_json(OUT/'audit.json')
    assert len(audit['checkpoints']) == 27 and audit['prediction_checks'] == 93366 and audit['baseline_unchanged']
    reference = read_json(OUT/f"target_reference/{fixed['variant']}.json")
    assert reference['prediction_checks'] == 8736 and reference['seeds'] == list(SEEDS)
    candidate = fixed['variant']
    methods = ['cnn_diversified', 'cnn_1194m', 'canny', 'zhang2019', candidate]
    comparisons = [candidate, 'boundary_surface', 'canny', 'zhang2019']
    lines = ['# 合成PAF学習による未知機体の内周検出', '',
        '**研究の問い：特定機体の詳細CADを使わず、合成PAFでの学習から未知機体の内周を欠損に強く検出できるか。**', '',
        '採用構成は、観測された内周境界を単一CNNで選び、固定RANSACで完全楕円を推定するもの。Tiny U-Net16は482,737パラメータで、モデル統合・対象別閾値・方向投票の追加はない。1194MのCAD・画像・重みはtrain、validation、初期化に使用しない。', '',
        '学習では、合成形状の表面模様・内周の投影サイズ/位置/扁平率を変え、開口幅を基準とする遮蔽を経験させる。元構図と投影拡張、元材質と模様の4ビューを各25%で使う。教師は既知の矩形遮蔽を除いた可視境界。1,988例×60epoch、元の多様形状validation損失最小で重みを選ぶ。', '',
        '## 1194M学習CNNへの近さ：既存test', '',
        '成功条件はIoU≥0.8。全13条件1,456画像を分母に含める。CNNは固定3seedの平均であり、3モデルを統合した推論ではない。既存testを参照した探索結果で、未使用の最終評価ではない。', '']
    lines += table(main, methods, source='1194m')
    lines += ['', '通常は1194M学習CNNとの差0.89pp、白飛び0.5は5.36pp、黒つぶれ0.5は5.06pp。半画面遮蔽では参照CNNの0%に対して72.92%。ユーザー指定の「情報が残る条件で1194M学習CNNに近い性能」をこの差として示す。統計的同等性を検定・証明した結果ではない。', '',
        '![1194M比較](comparison_1194m.png)', '', '## 学習元と同じ合成形状族の未知寸法', '', '全13条件2,002画像。1194M側の改善だけでなく、学習元の性能も確認する。', '']
    lines += table(main, methods, source='diversified')
    lines += ['', '## 候補固定後：1194Mの別カメラ群', '',
        '既存testの16カメラ群と重ならない86群・602原画像に、通常と3劣化0.5を適用した2,408条件。候補・全3seed・推定器を結果参照前に固定。1194M学習CNNは当該原画像を学習/検証に使っているため比較に含めない。同じ機体の別撮影条件であり、未知機体全般の盲検ではない。', '']
    lines += table(extra['rows'], comparisons)
    lines += ['', '## 候補固定後：既存別形状族の再評価', '',
        '角形胴・開放支持・リブ付き胴の各32形状と対応する元構造。各ビュー96形状×4条件、768条件。研究の前段で参照済みの形状集合であり、新規盲検ではない。全8設定の推定を元ラベルから照合した。', '', '### 新構造', '']
    lines += table(family['rows'], comparisons, view='new_family')
    lines += ['', '### 対応する元構造', '']
    lines += table(family['rows'], comparisons, view='control')
    delta = next(r for r in family['paired_shape_differences'] if r['method'] == candidate)
    lines += ['', f"採用方式の新構造−元構造の差は{100*delta['delta']:+.2f}pp、形状族内の対応bootstrap95%区間[{100*delta['ci95'][0]:.2f}, {100*delta['ci95'][1]:.2f}]pp。形状族全般や学習seed母集団への区間ではない。", '',
        '## 研究から示せること', '',
        '形状数を増やすだけでは、撮影規則と穴サイズの相関や、欠損時に残る内周量の偏りを解消できない。投影条件を揃えた3seed対照では、開口幅遮蔽によって1194Mの半画面遮蔽が65.18%→72.92%、多様形状では54.76%→68.40%になった。', '',
        '学習する対象を「完全な穴の塗りつぶし」から「目的の内周を示す観測境界」へ移し、欠けた楕円の整合を幾何に分担する方針を支持する。ただし教師と推論も異なる比較で、境界教師だけの単独因果効果ではない。合成PAFでの外観・投影・欠損の経験と、この分担を対照実験で示すことを卒論の中心に置く。', '',
        '解析境界＋単一BCEは3seedで多様形状側を改善したが、1194Mへの近さを改善しなかったため採用しない。完全形状教師、直接回帰、方向補助、参照範囲、正規化などの条件と負の結果は研究READMEと全推定記録に残す。', '',
        '## 先行研究との位置づけ', '',
        '[Le Gentil et al. 2024](https://arxiv.org/abs/2409.15581)にも、CNNで反射支援構造を強調し幾何推定する構成がある。対象はLM-MAPであり、併用自体を本研究の新規性としない。', '',
        '[Tobin et al. 2017](https://arxiv.org/abs/1703.06907)はシミュレータの描画変動を使うdomain randomizationを扱う。本研究ではPAF内周と撮影・欠損の分布に焦点を当て、同論文の実画像転移の結果をPAFへの保証とは扱わない。', '',
        '[Zhang, Pan, Ma 2019, Sensors](https://doi.org/10.3390/s19235243)は幾何方式の正規出典。比較器は論文構成を参考にしたZhang 2019型（再現実装）で、著者コードの移植や論文性能そのものではない。', '',
        '## 限界と再現', '',
        '黒背景・モノクロ256・単一開口の合成画像で検証した。任意背景、実画像、対象なし、複数開口、異なる上端環定義、一般的な対象非依存性は未検証。比較は固定した幾何2方式に限り、幾何方式一般や最新方式への優位性を示すものではない。矩形外の投影内周が最大約14%しか残らない1194M遮蔽0.75では成功0%。一様入力は未検出にするが、非一様画像の不確実性較正は未実施。', '',
        '全画像分母を保ち、情報消失例を成功率から取り除かない。元の4方式比較、27モデルの主比較、追加確認の全推定とhashを保持する。主比較と追加確認は画像・条件数が異なるため、全条件平均を直接同一母集団の差として比較しない。', '',
        '画像単体の入口：`research/paf_generalization_v2/detector.py`。採用モデルの既定seed07は最良test seedの選択ではない。バックアップの復元入口：`output/paf_generalization_v2/backup/INDEX.md`。原画像はmanifest参照、venvとBlender本体は別途必要。', '']
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    for p in (OUT/'family_confirmation/family_summary.json', OUT/'audit.json', OUT/f"target_reference/{candidate}.json",
              OUT/'confirmation/frozen_candidate.json', OUT/'family_confirmation/frozen_candidate.json', Path(__file__),
              DEST/'main_comparison.json', DEST/'comparison_1194m.png', DEST/'comparison_1194m.svg'):
        files[str(p.relative_to(ROOT))] = sha(p)
    write_json(DEST/'conclusion_audit.json', {'candidate': candidate, 'seeds': list(SEEDS),
        'primary_prediction_checks': 93366, 'additional_prediction_checks': checked,
        'reference_prediction_checks': 8736, 'files': files, 'report_sha256': sha(DEST/'REPORT.md'),
        'scope': '照合済み推定・集計・固定情報の整合を確認。CNNを再学習・全画像再推論した監査ではない。'})
    print('Saved final research report with both completed confirmations', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['figure', 'report'])
    args = parser.parse_args()
    {'figure': figure, 'report': report}[args.stage]()
