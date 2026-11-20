"""固定1seedの原因診断を全GT・学習履歴と照合し、対応表と曲線を保存する。"""

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_training_diagnosis import run as study
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import sha, image_and_truth

OUT, SEED = study.OUT, study.SEED
EFFECTS = {'all': '全13条件', 'clean': '通常', 'black_rectangle': '遮蔽',
           'sensor_whiteout': '白飛び', 'sensor_black_crush': '黒つぶれ'}
NAMES = {'target_plain_60': '現行学習・投影なし', 'target_projection_60': '現行学習・投影あり',
    'target_surface_60': '現行学習・投影＋外観', 'main_band_45': '旧CNN・帯45点',
    'main_band_12': '旧CNN・帯12点', 'main_centerline_45': '旧CNN・中心線45点',
    'main_centerline_12': '旧CNN・中心線12点', 'main_current': '旧CNN・現行推論',
    'source_base': '拡張なし', 'source_projection': '投影のみ', 'source_both': '投影＋外観'}


def summarize(rows, name, dataset):
    """同じ画像集合の全条件と劣化強度別を集計する。"""
    conditions = [('all', None), ('clean', 0.)]
    conditions += [(e, s) for e in list(EFFECTS)[2:] for s in (.25, .5, .75, 1.)]
    result = []
    for effect, severity in conditions:
        group = [r for r in rows if effect == 'all' or
                 (r['degradation'] == effect and r['severity'] == severity)]
        assert group
        result.append({'variant': name, 'dataset': dataset, 'degradation': effect,
            'severity': severity, 'n': len(group), 'successes': sum(r['success'] for r in group),
            'success_rate': float(np.mean([r['success'] for r in group])),
            'mean_iou': float(np.mean([r['ellipse_iou'] for r in group]))})
    return result


def report():
    """保存推定を元GTから再計算し、学習予算・固定モデルの違いを整理する。"""
    plan = read_json(OUT/'plan.json')
    for path, digest in plan['files'].items():
        assert sha(ROOT/path) == digest, path
    tables, files, checks = [], {}, 0
    all_predictions = {}
    for dataset in ('1194m', 'diversified'):
        truth = {s['sample_id']: (s, image_and_truth(study.BASE/dataset, s)[1])
                 for s in load_samples(study.BASE/dataset, 'test')}
        paths = [(p.parent.name, p) for p in sorted((OUT/'results').glob(f'*/test_{dataset}.json'))]
        paths += [('target_projection_60', ROOT/f'output/paf_1194m_no_surface_v1/results/1194m/{SEED}/test_{dataset}.json'),
                  ('target_surface_60', ROOT/f'output/paf_fair_comparison_v1/results/1194m/{SEED}/test_{dataset}.json')]
        old = OUT/f'old_model/test_{dataset}.json'
        assert read_json(old)['checkpoint_sha256'] == sha(study.REFERENCES['main']/'best.pt')
        files[str(old.relative_to(ROOT))] = sha(old)
        groups = [(name, read_json(path)['rows']) for name, path in paths]
        groups += [('main_'+mode, [r for r in read_json(old)['rows'] if r['mode'] == mode])
                   for mode in list(study.MODES)+['current']]
        for name, path in paths:
            result = read_json(path)
            if path.is_relative_to(OUT/'results'):
                variant, budget = name.rsplit('_', 1)
                checkpoint = OUT/f'runs/{variant}/{SEED}/budget_{budget}/best.pt'
                assert result['plan_sha256'] == sha(OUT/'plan.json')
            elif name == 'target_projection_60':
                checkpoint = study.REFERENCES['target_projection']/'best.pt'
            else:
                checkpoint = ROOT/f'output/paf_fair_comparison_v1/runs/1194m/{SEED}/best.pt'
            assert result['checkpoint_sha256'] == sha(checkpoint)
            files[str(path.relative_to(ROOT))] = sha(path)
        assert len(groups) == 14
        for name, rows in groups:
            assert len(rows) == len(truth) and {r['sample_id'] for r in rows} == set(truth)
            for row in rows:
                sample, gt = truth[row['sample_id']]
                assert np.allclose(np.array(row['truth_ellipse'][0]), gt[0])
                assert np.allclose(np.array(row['truth_ellipse'][1]), gt[1])
                assert abs(row['truth_ellipse'][2]-gt[2]) < 1e-8
                assert row['degradation'] == sample['conditions']['degradation']
                assert row['severity'] == sample['conditions']['severity']
                iou = evaluate_ellipses(row['ellipse'], gt, (256, 256))['ellipse_iou'] if row['ellipse'] else 0.
                assert abs(iou-row['ellipse_iou']) < 1e-10 and row['success'] == (iou >= .8)
                checks += 1
            all_predictions[name, dataset] = rows
            tables += summarize(rows, name, dataset)
    models, replay = [], []
    for name in study.VARIANTS:
        folder = OUT/f'runs/{name}/{SEED}'
        history = read_json(folder/'training_history.json')
        assert len(history) == (60 if name == 'target_plain' else 120)
        assert [r['epoch'] for r in history] == list(range(1, len(history)+1))
        assert all(np.isfinite(r[k]) for r in history for k in ('train_loss', 'validation_loss'))
        for budget in ([60] if name == 'target_plain' else [60, 120]):
            checkpoint = folder/f'budget_{budget}/best.pt'
            value = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert value['provenance'] == read_json(folder/'provenance.json')
            assert value['provenance']['plan_sha256'] == sha(OUT/'plan.json')
            best = min(history[:budget], key=lambda r: r['validation_loss'])
            assert value['epoch'] == best['epoch'] and value['validation_loss'] == best['validation_loss']
            assert all(torch.isfinite(x).all() for x in value['model_state'].values())
            assert read_json(folder/f'budget_{budget}/complete.json')['checkpoint_sha256'] == sha(checkpoint)
            files[str(checkpoint.relative_to(ROOT))] = sha(checkpoint)
            models.append({'variant': name, 'budget': budget, 'selected_epoch': value['epoch'],
                'validation_loss': value['validation_loss'], 'checkpoint_sha256': sha(checkpoint)})
        files[str((folder/'training_history.json').relative_to(ROOT))] = sha(folder/'training_history.json')
        if name.startswith('source_'):
            old = study.REFERENCES[name]
            previous = read_json(old/'training_history.json')
            diffs = {k: max(abs(a[k]-b[k]) for a, b in zip(history[:60], previous))
                     for k in ('train_loss', 'validation_loss')}
            a = torch.load(folder/'budget_60/best.pt', map_location='cpu', weights_only=False)['model_state']
            b = torch.load(old/'best.pt', map_location='cpu', weights_only=False)['model_state']
            parameter_names = set(dict(study.TinyUNet(16).named_parameters()))
            # BNの更新回数は学習重みではないため、パラメータ・統計・回数を分ける。
            differences = {k: float((a[k]-b[k]).abs().max()) for k in a}
            replay.append({'variant': name, 'first_60_loss_max_abs_difference': diffs,
                'selected_parameters_max_abs_difference': max(differences[k] for k in parameter_names),
                'selected_float_buffers_max_abs_difference': max(differences[k] for k in a
                    if k not in parameter_names and torch.is_floating_point(a[k])),
                'selected_batch_counter_abs_difference': max(differences[k] for k in a
                    if k.endswith('num_batches_tracked'))})
    differences = []
    comparisons = [('main_band_45', 'main_centerline_45'), ('main_centerline_45', 'main_centerline_12'),
        ('main_centerline_12', 'main_current'), ('main_current', 'target_plain_60'),
        ('target_plain_60', 'target_projection_60'), ('target_projection_60', 'target_surface_60')]
    comparisons += [(f'{n}_60', f'{n}_120') for n in ('source_base', 'source_projection', 'source_both')]
    comparisons += [(f'source_base_{b}', f'source_projection_{b}') for b in (60, 120)]
    for dataset in ('1194m', 'diversified'):
        for first, second in comparisons:
            a = {r['sample_id']: r for r in all_predictions[first, dataset]}
            b = {r['sample_id']: r for r in all_predictions[second, dataset]}
            gained = sum(not a[k]['success'] and b[k]['success'] for k in a)
            lost = sum(a[k]['success'] and not b[k]['success'] for k in a)
            differences.append({'dataset': dataset, 'first': first, 'second': second, 'n': len(a),
                'gained_success': gained, 'lost_success': lost, 'delta_success_rate': (gained-lost)/len(a),
                'delta_mean_iou': float(np.mean([b[k]['ellipse_iou']-a[k]['ellipse_iou'] for k in a]))})
    size_groups = []
    for name in ('main_band_45', 'main_centerline_45', 'main_centerline_12', 'main_current',
                 'target_plain_60', 'target_projection_60'):
        for lower, upper in ((0, 24), (24, 48), (48, 256)):
            selected = [r for r in all_predictions[name, 'diversified'] if r['degradation'] == 'clean'
                        and lower <= max(r['truth_ellipse'][1]) < upper]
            size_groups.append({'variant': name, 'major_min': lower, 'major_max': upper,
                'n': len(selected), 'successes': sum(r['success'] for r in selected),
                'success_rate': float(np.mean([r['success'] for r in selected]))})
    additional = read_json(OUT/'additional_validation.json')
    for path, digest in additional['files'].items():
        assert sha(ROOT/path) == digest
    projection_root = ROOT/'output/paf_generalization_v2/projection_coverage'
    plain = load_samples(projection_root/'native_plain', 'validation')
    surface = load_samples(projection_root/'native_surface', 'validation')
    assert len(plain) == len(surface) == 735
    assert all(a['sample_id'] == b['sample_id'] and a['label_sha256'] == b['label_sha256']
               and a['augmentation'] == b['augmentation'] for a, b in zip(plain, surface))
    files[str((OUT/'additional_validation.json').relative_to(ROOT))] = sha(OUT/'additional_validation.json')
    numerical = read_json(OUT/'numerical_repeat_check.json')
    files[str((OUT/'numerical_repeat_check.json').relative_to(ROOT))] = sha(OUT/'numerical_repeat_check.json')
    summary = {'seed': SEED, 'rows': tables, 'models': models, 'replay': replay,
               'paired_differences': differences, 'clean_diversified_size_groups': size_groups,
               'additional_validation': additional['rows'],
               'numerical_repeat': numerical,
               'paired_surface_validation_labels_and_degradations': len(plain),
               'scope': '固定1seedの探索的対応比較。学習seed母集団の不確実性は評価していない。'}
    write_json(OUT/'summary.json', summary)
    with (OUT/'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    write_json(OUT/'audit.json', {'prediction_checks': checks, 'files': files, 'models': models,
        'first_60_epoch_replay': replay, 'plan_sha256': sha(OUT/'plan.json'),
        'paired_surface_validation_labels_and_degradations': len(plain),
        'implementation_sha256': sha(Path(__file__)), 'incomplete_runs': []})
    markdown(summary)
    figures(summary)
    print('AUDITED', checks, 'predictions;', len(models), 'budget checkpoints', flush=True)
    print('REPLAY', replay, flush=True)


def value(summary, name, dataset, effect='all', severity=None):
    return next(r for r in summary['rows'] if r['variant'] == name and r['dataset'] == dataset
                and r['degradation'] == effect and r['severity'] == severity)


def markdown(summary):
    """結果と分離できる要因の範囲を明示する表を作る。"""
    lines = ['# 投影・学習時間・点群処理の対応診断', '',
        '固定seed20260907。主指標はIoU≥0.8、全画像を分母に含む。通常＋3劣化×4強度の全13条件。'
        'testは既に研究で参照した集合であり、盲検ではない。最良test epoch・seedを選ばない。', '',
        '## 1194M学習モデル：同じ確率での点群処理と、投影除去対照', '',
        '|構成|1194M 全条件|多様形状 全条件|多様形状 通常|', '|---|---:|---:|---:|']
    for name in ('main_band_45', 'main_centerline_45', 'main_band_12', 'main_centerline_12',
                 'main_current', 'target_plain_60', 'target_projection_60', 'target_surface_60'):
        vals = [value(summary, name, d, e, s)['success_rate']*100 for d, e, s in
                [('1194m', 'all', None), ('diversified', 'all', None), ('diversified', 'clean', 0.)]]
        lines.append(f"|{NAMES[name]}|{vals[0]:.2f}%|{vals[1]:.2f}%|{vals[2]:.2f}%|")
    lines += ['', '旧CNNの帯/中心線×最小45/12点は同じ重み・確率・閾値0.35・乱数による対応診断。'
        '「旧CNN・現行推論」は閾値0.5・中心線・最小12点・現行乱数と一様画像棄却も含む。'
        '旧CNN＋現行推論から現行学習・投影なしへの差には、教師・学習率推移・遮蔽再抽選など複数要因が残る。'
        '現行学習の投影なし/ありは、元材質・教師・遮蔽・最適化を共通にした投影だけの対照。'
        '各差はこの比較順に条件付くため、独立な寄与として足し合わせない。', '',
        '## 多様形状学習：同じ学習予算での比較', '',
        '|拡張|予算|採用epoch|元validation loss|1194M 全条件|多様形状 全条件|',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('source_base', 'source_projection', 'source_both'):
        for budget in (60, 120):
            m = next(r for r in summary['models'] if r['variant'] == name and r['budget'] == budget)
            a, b = [value(summary, f'{name}_{budget}', d)['success_rate']*100 for d in ('1194m', 'diversified')]
            lines.append(f"|{NAMES[name]}|{budget}|{m['selected_epoch']}|{m['validation_loss']:.6f}|{a:.2f}%|{b:.2f}%|")
    lines += ['', '同じランダム初期化から120epochを連続学習し、60epochでoptimizerをリセットしない。'
        '各予算内の元多様形状validation loss最小の重みを使う。開口幅遮蔽はこの3構成すべてで使わず、'
        '以前の投影のみの比較と条件を揃える。学習損失は入力分布が違うため、構成間の絶対値だけで学習不足とは判定しない。', '',
        '## 追加投影validation：同じ未学習形状への境界loss', '',
        '|構成|予算|元画像|投影・元材質|投影・外観変更|', '|---|---:|---:|---:|---:|']
    for name in ('source_base', 'source_projection', 'source_both'):
        for budget in (60, 120):
            vals = [next(r['loss'] for r in summary['additional_validation'] if r['variant'] == name
                    and r['budget'] == budget and r['view'] == view)
                    for view in ('original', 'projection_plain', 'projection_surface')]
            lines.append(f'|{NAMES[name]}|{budget}|'+ '|'.join(f'{v:.4f}' for v in vals)+'|')
    lines += ['', '追加投影validationは事後診断のみで、重みの選択には使わない。元画像の値は保存された選択時lossと照合する。'
        '投影・元材質と投影・外観変更の735組は、同じsample ID・ラベルhash・欠損条件であることを照合した。', '']
    for dataset in ('1194m', 'diversified'):
        lines += [f'## {dataset}：全条件別成功率', '',
            '|条件|拡張なし60|拡張なし120|投影60|投影120|投影＋外観60|投影＋外観120|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in summary['rows'] if r['variant'] == 'source_base_60' and r['dataset'] == dataset]:
            label = EFFECTS[r['degradation']]+(f" {r['severity']:g}" if r['severity'] else '')
            vals = [value(summary, f'{name}_{budget}', dataset, r['degradation'], r['severity'])['success_rate']*100
                    for name in ('source_base', 'source_projection', 'source_both') for budget in (60, 120)]
            lines.append('|'+label+'|'+'|'.join(f'{v:.2f}%' for v in vals)+'|')
        lines += ['']
    lines += ['## 照合と範囲', '', '全推定の元GT再計算と、予算内validation最良重み・有限重みをaudit.jsonで確認。'
        '過去60epochとの損失履歴・選択重みの再現誤差はsummary.jsonのreplayに保存する。'
        'GPU学習は決定的アルゴリズムを指定しておらず、同じ初期重み・同じbatchの短い反復でも数値差を確認した。'
        'したがって学習時間の主対照は今回の連続学習内の60/120epochで、過去の60epoch重みへの厳密な継続ではない。'
        '既存3seed平均は変更せず、今回の1seed値とは区別する。黒背景・単一開口CGに限定し、'
        '120epochでの結果から無限の学習時間や他seedの結果を保証しない。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def figures(summary):
    """学習損失の推移と、60/120epochの両対象への効果を同時に示す。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = 'C:/Windows/Fonts/YuGothR.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'font.size': 11, 'svg.fonttype': 'path'})
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ['#727a84', '#3777b8', '#168060']
    names = ('source_base', 'source_projection', 'source_both')
    curve_rows = []
    for name, color in zip(names, colors):
        history = read_json(OUT/f'runs/{name}/{SEED}/training_history.json')
        curve_rows += [{'variant': name, **r} for r in history]
        for ax, field in zip(axes[0], ('train_loss', 'validation_loss')):
            ax.plot([r['epoch'] for r in history], [r[field] for r in history],
                    color=color, label=NAMES[name], alpha=.85, linewidth=1.2)
            ax.set_yscale('log')
            ax.axvline(60, color='#444', linestyle=':', linewidth=1)
            ax.set_xlabel('epoch')
            ax.grid(alpha=.2)
    axes[0, 0].set_title('学習loss（各構成の学習分布）')
    axes[0, 1].set_title('validation loss（共通の元多様形状）')
    axes[0, 0].legend(frameon=False)
    bar_rows = []
    for ax, dataset in zip(axes[1], ('1194m', 'diversified')):
        x = np.arange(3)
        for i, budget in enumerate((60, 120)):
            values = [value(summary, f'{n}_{budget}', dataset) for n in names]
            bar_rows += values
            bars = ax.bar(x+(i-.5)*.34, [r['success_rate']*100 for r in values], width=.32,
                          color=colors, alpha=.45 if budget == 60 else 1.,
                          hatch='//' if budget == 60 else None, label=f'{budget} epoch')
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)
        ax.set_xticks(x, [NAMES[n] for n in names])
        ax.set_ylim(0, 100)
        ax.set_ylabel('全13条件の成功率（IoU ≥ 0.8）%')
        ax.set_title('1194Mへの転移' if dataset == '1194m' else '多様形状自身のtest')
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=.2)
    figure.suptitle('投影のみの悪化は学習不足か：固定1seed・同じ予算で比較', fontsize=17)
    figure.text(.5, .015, 'seed20260907。各予算内の元validation loss最小を採用。曲線は対数軸、棒は0〜100%。', ha='center')
    figure.tight_layout(rect=(0, .045, 1, .95))
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    artifacts = []
    for suffix in ('png', 'svg'):
        path = folder/f'learning_budget.{suffix}'
        figure.savefig(path, dpi=160, facecolor='white')
        artifacts.append(path)
    plt.close(figure)
    write_json(folder/'learning_budget.json', {'curve_rows': curve_rows, 'bar_rows': bar_rows,
        'summary_sha256': sha(OUT/'summary.json'),
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in artifacts}})


def snapshot():
    """確定済みの外観除去比較を依存に持ち、原因診断を別の復元単位へ保全する。"""
    previous = read_json(ROOT/'output/paf_1194m_no_surface_v1/backup/latest.json')
    assert sha(ROOT/previous['path']) == previous['sha256']
    assert read_json(OUT/'audit.json')['incomplete_runs'] == []
    backup = OUT/'backup'
    destination = backup/'verified_training_diagnosis_20261025'
    assert not destination.exists()
    paths = list((ROOT/'research/paf_training_diagnosis').rglob('*'))
    paths += [ROOT/'.codex/RESEARCH_HANDOFF.md', ROOT/'.codex/PROJECT_CONTEXT.md', ROOT/'README.md',
              ROOT/'research/paf_core_ablation/README.md', ROOT/'research/paf_fair_comparison/PRESENTATION_CHANGES.md']
    paths += [p for p in OUT.rglob('*') if backup not in p.parents]
    files = []
    for path in sorted(set(p for p in paths if p.is_file() and '__pycache__' not in p.parts
                           and p.suffix not in ('.pyc', '.log'))):
        relative = path.relative_to(ROOT)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert sha(path) == sha(target)
        files.append({'path': str(relative), 'sha256': sha(path)})
    inventory = {'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
        'required_previous_snapshot': previous, 'incomplete_runs': [],
        'restore': '依存する確定snapshotと原画像参照を復元し、このセットを同じ相対pathへ重ねる。'}
    write_json(destination/'inventory.json', inventory)
    write_json(backup/'latest.json', {'path': str((destination/'inventory.json').relative_to(ROOT)),
        'sha256': sha(destination/'inventory.json'), 'verified_files': len(files)})
    (backup/'INDEX.md').write_text('# 原因診断の確定バックアップ\n\n'
        f'`{destination.name}`：{len(files)}ファイルのSHA256をコピー元と照合。\n\n'
        f'依存セット：`{previous["path"]}`。既存の重み・入力はその復元連鎖と原画像参照で保持する。\n', encoding='utf-8')
    print('SNAPSHOT', len(files), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['report', 'snapshot'], default='report', nargs='?')
    args = parser.parse_args()
    report() if args.stage == 'report' else snapshot()
