"""1194Mの外観変更なし対照を元比較と対応付け、全強度の差・順位・保全情報を保存する。"""

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_fair_comparison import no_surface as control
from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.report import paired_interval

OUT, BASELINE = control.OUT, control.BASELINE
sha = control.common.sha


def report():
    """共通の全件監査を再利用し、外観変更あり/なしの同じseed・同じ画像を比較する。"""
    control.configure()
    from research.paf_fair_comparison import report as core
    core.OUT, core.CONFIG, core.prepare = OUT, control.CONFIG, control.prepare
    checks = read_json(OUT/'implementation_checks.json')['no_surface_checks']
    assert [r['seed'] for r in checks] == read_json(control.CONFIG)['seeds']
    assert all(r['checked_samples'] == 128 and not r['surface_views_used']
               and r['projection_and_occlusion_rng_preserved'] and r['slots_seen'] == [0, 1, 2, 3] for r in checks)
    core.main()
    summary = read_json(OUT/'summary.json')
    old_summary = read_json(BASELINE/'summary.json')
    # 材質の有無で教師や劣化設定が変わっていないことを全train行で照合する。
    roots = [control.common.BASE/'1194m', BASELINE/'augmentation_1194m/original_surface',
             BASELINE/'augmentation_1194m/projection_plain', BASELINE/'augmentation_1194m/projection_surface']
    manifests = [[r for r in read_json(p/'manifest.json')['samples'] if r['split'] == 'train'] for p in roots]
    teacher_pairs = 0
    for a, b in ((0, 1), (2, 3)):
        assert len(manifests[a]) == len(manifests[b]) == 1988
        for left, right in zip(manifests[a], manifests[b]):
            assert left['sample_id'] == right['sample_id']
            assert left['label_sha256'] == right['label_sha256']
            assert left.get('augmentation') == right.get('augmentation')
            teacher_pairs += 1
    deltas, checked = [], 0
    for dataset in ('1194m', 'diversified'):
        old_rows, new_rows = [], []
        for seed in read_json(control.CONFIG)['seeds']:
            old_path = BASELINE/f'results/1194m/{seed}/test_{dataset}.json'
            new_path = OUT/f'results/1194m/{seed}/test_{dataset}.json'
            previous = read_json(old_path)['rows']
            current = read_json(new_path)['rows']
            lookup = {r['sample_id']: r for r in current}
            assert len(previous) == len(current) and set(lookup) == {r['sample_id'] for r in previous}
            for row in previous:
                target = lookup[row['sample_id']]['truth_ellipse']
                iou = evaluate_ellipses(row['ellipse'], target, (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
                assert abs(iou-row['ellipse_iou']) < 1e-10 and row['success'] == (iou >= .8)
                checked += 1
            old_rows += [{**r, 'seed': seed} for r in previous]
            new_rows += [{**r, 'seed': seed} for r in current]
        for effect, severity in core.CONDITIONS:
            a, b = core.select(new_rows, effect, severity), core.select(old_rows, effect, severity)
            deltas.append({'dataset': dataset, 'degradation': effect, 'severity': severity,
                'no_surface_success': float(np.mean([r['success'] for r in a])),
                'with_surface_success': float(np.mean([r['success'] for r in b])),
                'no_surface_mean_iou': float(np.mean([r['ellipse_iou'] for r in a])),
                'with_surface_mean_iou': float(np.mean([r['ellipse_iou'] for r in b])),
                **paired_interval(a, b)})
    summary['surface_removal'] = deltas
    summary['comparison_scope'] = '1194Mだけ外観変更を除去。教師・中心線・投影・遮蔽・損失・最適化・seedは維持。'
    write_json(OUT/'summary.json', summary)
    audit = read_json(OUT/'audit.json')
    assert checked == 10374 and teacher_pairs == 3976
    audit.update(surface_reference_prediction_checks=checked,
                 total_including_surface_reference=audit['prediction_checks']+checked,
                 entry_sha256=sha(Path(__file__)),
                 reference_audit_sha256=sha(BASELINE/'audit.json'),
                 no_surface_data_checks=checks, matched_label_and_degradation_pairs=teacher_pairs)
    write_json(OUT/'audit.json', audit)
    # 本対照の仕様に合わせ、共通監査から生成した報告の説明を明示する。
    path = OUT/'REPORT.md'
    text = path.read_text(encoding='utf-8')
    text = text.replace('# 学習元だけを変更した共通境界CNNの比較', '# 1194Mの外観変更を外した比較')
    text = text.replace('両CNNは可視境界教師、中心線RANSAC、投影・外観4ビューと開口幅遮蔽、学習条件を共通化した。',
        '両CNNの可視境界教師・中心線RANSAC・投影拡張・開口幅遮蔽・損失・最適化を揃え、1194M側だけ外観変更を外した。1194Mは元材質の元投影/拡張投影を各50%、多様形状は従来の4ビュー各25%を使用する。')
    text = text.replace('ここで変えたのは学習データ源一式である。',
        '両CNNの比較では学習データ源に加えて外観拡張の有無が異なる。1194Mの外観変更あり/なし対照では外観拡張だけを変更し、他の条件は固定した。')
    text = text.replace('①はmainのGaussianリング＋帯点群モデルではなく、②と同じ基礎で1194Mから学習したモデル。',
        '①はmainのGaussianリング＋帯点群モデルではなく、共通境界CNNを1194Mの元材質で学習したモデル。')
    text += '\n## 1194M学習モデル：外観変更あり/なしの対応比較\n\n差は「外観変更なし−外観変更あり」、単位は百分率点。各3seed平均。\n\n'
    for dataset in ('1194m', 'diversified'):
        text += f'### {dataset} test\n\n|条件|外観変更あり|外観変更なし|差|\n|---|---:|---:|---:|\n'
        for row in deltas:
            if row['dataset'] != dataset:
                continue
            name = core.EFFECTS[row['degradation']]+(f" {row['severity']:g}" if row['severity'] else '')
            text += f"|{name}|{row['with_surface_success']*100:.2f}%|{row['no_surface_success']*100:.2f}%|{row['delta']*100:+.2f} pp|\n"
        text += '\n'
    text += '既存の外観変更あり1194Mモデルの10,374推定も元GTに照合し、合計38,038推定を確認した。外観変更なしの判断をtestで最良seedを選ぶことに置き換えない。\n'
    path.write_text(text, encoding='utf-8')
    figures(summary, old_summary)


def figures(summary, previous):
    """外観変更の除去の差と多様形状・固定幾何を同じ目盛りに表示する。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 11, 'svg.fonttype': 'path'})
    series = [('1194m', summary, '①1194M学習・外観変更なし', '#355caf', '-'),
              ('1194m', previous, '1194M学習・外観変更あり', '#7c8aa6', ':'),
              ('diversified', summary, '②多様形状学習', '#147d64', '-'),
              ('canny', summary, '③Canny＋輪郭別RANSAC', '#d48427', '--'),
              ('zhang2019', summary, '③Zhang 2019型（再現実装）', '#9b639b', '--')]
    effects = {'black_rectangle': '矩形遮蔽', 'sensor_whiteout': '白飛び', 'sensor_black_crush': '黒つぶれ'}
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    evidence = []
    x = [0, .25, .5, .75, 1.]
    for ri, dataset in enumerate(('1194m', 'diversified')):
        for ci, (effect, name) in enumerate(effects.items()):
            ax = axes[ri, ci]
            for method, table, title, color, style in series:
                rows = [next(r for r in table['rows'] if r['dataset'] == dataset and r['method'] == method
                    and r['degradation'] == e and r['severity'] == s)
                    for e, s in [('clean', 0.)]+[(effect, v) for v in x[1:]]]
                ax.plot(x, [r['success_rate']*100 for r in rows], label=title, color=color,
                        linestyle=style, linewidth=1.8, marker='o', markersize=4,
                        zorder=1 if table is previous else 3)
                if method in ('1194m', 'diversified'):
                    ax.fill_between(x, [min(r['per_seed'].values())*100 for r in rows],
                                    [max(r['per_seed'].values())*100 for r in rows], color=color, alpha=.10)
                evidence += [{'series': title, **r} for r in rows]
            ax.set_ylim(-2, 104)
            ax.set_xticks(x)
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if ri == 0:
                ax.set_title(name, fontsize=16)
            else:
                ax.set_xlabel('劣化強度（0は通常画像）')
            if ci == 0:
                ax.set_ylabel(('1194M test' if ri == 0 else '多様形状 test')+'\n成功率（IoU ≥ 0.8）%')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .05), ncol=3, frameon=False)
    figure.suptitle('1194Mの外観変更を除去：投影拡張・他の学習条件・推論は維持', fontsize=18, y=.98)
    figure.text(.5, .015, 'CNNは固定3seed平均、帯は最小〜最大。多様形状学習側の外観変更・投影拡張は維持。全画像を分母に含む。', ha='center')
    figure.subplots_adjust(left=.075, right=.985, top=.91, bottom=.20, hspace=.12, wspace=.09)
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    paths = []
    for suffix in ('png', 'svg'):
        path = folder/f'surface_removal.{suffix}'
        figure.savefig(path, dpi=160, facecolor='white')
        paths.append(path)
    plt.close(figure)
    assert len(evidence) == 150
    write_json(folder/'surface_removal.json', {'rows': evidence,
        'summary_sha256': sha(OUT/'summary.json'), 'reference_summary_sha256': sha(BASELINE/'summary.json'),
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in paths}})


def snapshot():
    """外観変更ありの確定セットに依存を結び付け、外観変更なしの完成成果物を保全する。"""
    previous = read_json(BASELINE/'backup/latest.json')
    assert sha(ROOT/previous['path']) == previous['sha256']
    assert read_json(OUT/'audit.json')['incomplete_runs'] == []
    backup = OUT/'backup'
    destination = backup/'verified_no_surface_20261025'
    assert not destination.exists()
    paths = list((ROOT/'research/paf_fair_comparison').rglob('*'))
    paths += [ROOT/'.codex/RESEARCH_HANDOFF.md', ROOT/'.codex/PROJECT_CONTEXT.md', ROOT/'README.md']
    paths += [p for p in OUT.rglob('*') if backup not in p.parents]
    files = []
    for p in sorted(set(p for p in paths if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.log'))):
        relative = p.relative_to(ROOT)
        copied = destination/relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(p)
        shutil.copy2(p, copied)
        assert sha(copied) == sha(p) == digest
        files.append({'path': str(relative), 'sha256': digest})
    inventory = {'created_utc': datetime.now(timezone.utc).isoformat(), 'required_previous_snapshot': previous,
                 'files': files, 'incomplete_runs': [],
                 'restore': '外観変更ありsnapshotの依存を復元し、このセットのpathをプロジェクト直下に重ねる。'}
    write_json(destination/'inventory.json', inventory)
    write_json(backup/'latest.json', {'path': str((destination/'inventory.json').relative_to(ROOT)),
        'sha256': sha(destination/'inventory.json'), 'verified_files': len(files)})
    (backup/'INDEX.md').write_text('# 1194M外観変更なし比較のバックアップ\n\n'
        f'確定セット：`{destination.name}`、{len(files)}ファイルのコピーとSHA256を照合済み。\n\n'
        f'依存する外観変更ありセット：`{previous["path"]}`。\n', encoding='utf-8')
    print('SNAPSHOT', len(files), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['report', 'snapshot'], default='report', nargs='?')
    args = parser.parse_args()
    report() if args.stage == 'report' else snapshot()
