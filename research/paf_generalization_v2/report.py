"""完了した継続研究の全推定を元ラベルと照合し、条件別の成績を保存する。"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha
from research.paf_generalization.report import paired_interval

OUT = ROOT/'output/paf_generalization_v2'
LABELS = {'boundary_surface': '可視境界＋表面模様（基準）',
          'boundary_amodal': '完全境界＋表面模様', 'region_surface': '完全穴領域＋表面模様',
          'direct_surface': '5自由度の楕円直接回帰', 'boundary_native': '投影構図を独立化・再描画',
          'boundary_resampled': '投影構図を独立化・補間', 'boundary_context': '可視境界・参照範囲拡大'}
LABELS.update(boundary_native_mix='元画像＋サイズ・位置の再描画',
              boundary_projection_mix='元画像＋サイズ・位置・扁平率の再描画',
              boundary_interior='可視境界・内部構造の対応学習',
              boundary_normal_aux='可視境界・外向き法線の補助学習',
              boundary_online_mask='可視境界・画像幅基準の動的遮蔽',
              boundary_relative_mask='可視境界・開口幅基準の動的遮蔽',
              boundary_projection_relative='投影範囲拡張・開口幅基準の動的遮蔽',
              boundary_projection_validation='投影＋開口幅遮蔽・4ビュー検証選択',
              boundary_projection_analytic='投影＋開口幅遮蔽・解析境界＋単一BCE',
              boundary_group_norm='可視境界・画像内のGroup Normalization',
              boundary_center_aux='可視境界・楕円中心方向の補助学習')


def planned_comparisons(records, summary, sources, plan_path):
    """事前固定した全3seedが候補・対照の両方で揃った場合だけ対応差を集計する。"""
    comparisons = []
    if not plan_path.exists():
        return comparisons
    plan = read_json(plan_path)
    for key in ('files', 'frozen_files', 'initial_result_files'):
        for path, digest in plan.get(key, {}).items():
            assert sha(ROOT/path) == digest, path
    control, candidate = plan['variants'] if 'variants' in plan else (plan['control'], plan['variant'])
    assert plan['seeds'] == [20260907, 20260908, 20260909]
    for source in sources:
        paired = {variant: [r for r in records if r['dataset'] == source and r['variant'] == variant
                            and r['seed'] in plan['seeds']] for variant in (control, candidate)}
        if not all({r['seed'] for r in part} == set(plan['seeds']) for part in paired.values()):
            continue
        assert {(r['seed'], r['sample_id']) for r in paired[control]} == {(r['seed'], r['sample_id']) for r in paired[candidate]}
        for item in summary:
            if item['dataset'] != source or item['variant'] != candidate:
                continue
            assert item['n_seeds'] == 3
            effect, severity = item['degradation'], item['severity']
            selected = {variant: [r for r in part if effect == 'all' or
                                 (r['degradation'], r['severity']) == (effect, severity)]
                        for variant, part in paired.items()}
            comparisons.append({**item, 'control': control,
                'control_success_rate': float(np.mean([r['success'] for r in selected[control]])),
                'replication_plan_sha256': sha(plan_path),
                **paired_interval(selected[candidate], selected[control])})
    return comparisons


def main():
    torch.set_num_threads(2)
    audit = {'checkpoints': [], 'prediction_checks': 0, 'baseline_unchanged': False}
    inventory = read_json(PREVIOUS/'backup/baseline_20261025/inventory.json')
    for item in inventory['files']:
        if not item['path'].startswith('.codex/'):
            assert sha(ROOT/item['path']) == item['sha256']
    audit['baseline_unchanged'] = True
    lookups, truths, seen = {}, {}, set()
    for source in ('1194m', 'diversified'):
        directory = BASE/'datasets'/source
        for sample in read_json(directory/'manifest.json')['samples']:
            for field in ('image', 'label'):
                path = (directory/sample[field]).resolve()
                if path not in seen:
                    assert sha(path) == sample[field+'_sha256']
                    seen.add(path)
        lookups[source] = {s['sample_id']: s for s in load_samples(directory, 'test')}
        truths[source] = {}
        for sid, sample in lookups[source].items():
            label = read_json(directory/sample['label'])
            truths[source][sid] = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    audit['input_hash_checks'] = len(seen)
    records = []
    for source in lookups:
        for seed in (20260907, 20260908, 20260909):
            saved = read_json(PREVIOUS/f'runs/boundary_surface/{seed}/test_{source}.json')
            records.extend({**r, 'dataset': source, 'variant': 'boundary_surface', 'seed': seed} for r in saved['rows'])
    for complete in sorted((OUT/'runs').glob('*/*/complete.json')):
        folder = complete.parent
        if not all((folder/f'test_{source}.json').exists() for source in lookups):
            continue
        metadata = read_json(folder/'provenance.json')
        history = read_json(folder/'training_history.json')
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        assert checkpoint['provenance'] == metadata
        assert len(history) == metadata['config']['epochs'] == read_json(complete)['epochs']
        assert checkpoint['validation_loss'] == min(r['validation_loss'] for r in history)
        assert history[checkpoint['epoch']-1]['validation_loss'] == checkpoint['validation_loss']
        assert all(torch.isfinite(value).all() for value in checkpoint['model_state'].values())
        assert read_json(complete)['checkpoint_sha256'] == sha(folder/'best.pt')
        for path, digest in metadata['files'].items():
            assert sha(ROOT/path) == digest, path
        extra = {'target_implementation_sha256': 'paf_generalization_v2/learning.py',
                 'direct_implementation_sha256': 'paf_generalization_v2/direct.py',
                 'boundary_implementation_sha256': 'paf_generalization/boundary.py',
                 'boundary_surface_implementation_sha256': 'paf_generalization/boundary_surface.py'}
        for key, relative in extra.items():
            if key in metadata['config']:
                assert sha(ROOT/'research'/relative) == metadata['config'][key]
        for key in ('surface_data_hashes', 'framing_files', 'context_boundary_files', 'projection_files', 'interior_boundary_files', 'oriented_files', 'occlusion_sampling_files', 'center_learning_files', 'normalization_files', 'source_validation_files', 'analytic_teacher_files'):
            for path, digest in metadata['config'].get(key, {}).items():
                assert sha(ROOT/path) == digest
        if 'ransac_implementation_sha256' in metadata['config']:
            assert sha(ROOT/'paf_ring_detection/methods/ransac.py') == metadata['config']['ransac_implementation_sha256']
        audit['checkpoints'].append({'variant': metadata['variant'], 'seed': metadata['seed'],
                                     'epochs': len(history), 'selected_epoch': checkpoint['epoch'],
                                     'validation_loss': checkpoint['validation_loss']})
        for source in lookups:
            saved = read_json(folder/f'test_{source}.json')
            assert saved['checkpoint_sha256'] == sha(folder/'best.pt')
            assert saved['provenance'] == metadata
            assert [r['sample_id'] for r in saved['rows']] == list(lookups[source])
            for row in saved['rows']:
                sample = lookups[source][row['sample_id']]
                truth = truths[source][row['sample_id']]
                assert np.allclose([*truth[0], *truth[1], truth[2]],
                                   [*row['truth_ellipse'][0], *row['truth_ellipse'][1], row['truth_ellipse'][2]])
                for field in ('degradation', 'severity'):
                    assert row[field] == sample['conditions'][field]
                assert row['group_id'] == sample['group_id']
                e = row['ellipse']
                ellipse = (tuple(e[0]), tuple(e[1]), e[2]) if e else None
                iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                assert iou == row['ellipse_iou'] and row['success'] == (iou >= .8)
                records.append({**row, 'dataset': source, 'variant': metadata['variant'], 'seed': metadata['seed']})
                audit['prediction_checks'] += 1
    summary, comparisons = [], []
    for source in lookups:
        for variant in sorted({r['variant'] for r in records}):
            rows = [r for r in records if r['dataset'] == source and r['variant'] == variant]
            if not rows:
                continue
            seeds = sorted({r['seed'] for r in rows})
            conditions = [('all', None)]+sorted({(r['degradation'], r['severity']) for r in rows})
            reference = [r for r in records if r['dataset'] == source and r['variant'] == 'boundary_surface' and r['seed'] in seeds]
            for effect, severity in conditions:
                selected = [r for r in rows if effect == 'all' or (r['degradation'], r['severity']) == (effect, severity)]
                item = {'dataset': source, 'variant': variant, 'degradation': effect, 'severity': severity,
                        'n_seeds': len(seeds), 'n_predictions': len(selected),
                        'n_images': len({r['sample_id'] for r in selected}),
                        'success_rate': float(np.mean([r['success'] for r in selected])),
                        'mean_iou': float(np.mean([r['ellipse_iou'] for r in selected])),
                        'per_seed': {str(seed): float(np.mean([r['success'] for r in selected if r['seed'] == seed])) for seed in seeds}}
                summary.append(item)
                if variant != 'boundary_surface':
                    comparisons.append({**item, **paired_interval(selected, reference)})
    projection_comparisons = planned_comparisons(records, summary, lookups,
        OUT/'projection_occlusion_replication_plan.json')
    analytic_comparisons = planned_comparisons(records, summary, lookups,
        OUT/'analytic_learning_replication_plan.json')
    write_json(OUT/'summary.json', {'summary': summary, 'comparisons_to_matching_seed_boundary': comparisons,
                                  'projection_comparisons_all_three_seeds': projection_comparisons,
                                  'analytic_comparisons_all_three_seeds': analytic_comparisons})
    write_json(OUT/'audit.json', audit)
    with (OUT/'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['dataset', 'variant', 'degradation', 'severity', 'n_seeds', 'n_images', 'n_predictions', 'success_rate', 'mean_iou']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(summary)
    lines = ['# 単純なPAF検出器の継続研究', '',
             '目的：PAFに共通する内周構造を合成データで学ばせ、特定機体の詳細CADに頼らず未知PAFの内周を欠損下でも検出できるか検証する。', '',
             '成功はIoU≥0.8。全画像を分母へ含める。1194Mは学習・validation・初期化に使わない。既存testを参照した探索であり、未使用の最終評価とは扱わない。基準は3seed平均、新方式は表のseed数を確認する。同一seedでの対応差と群bootstrap区間はsummary.jsonに保存する。', '']
    for source in lookups:
        lines += [f'## {source}', '', '|構成|seed数|全条件|通常|半画面遮蔽|白飛び0.5|黒つぶれ0.5|', '|---|---:|---:|---:|---:|---:|---:|']
        for variant in LABELS:
            selected = [r for r in summary if r['dataset'] == source and r['variant'] == variant]
            if not selected:
                continue
            values = {(r['degradation'], r['severity']): r for r in selected}
            keys = [('all', None), ('clean', 0.), ('black_rectangle', .5), ('sensor_whiteout', .5), ('sensor_black_crush', .5)]
            scores = '|'.join(f"{values[key]['success_rate']*100:.2f}%" for key in keys)
            lines.append(f"|{LABELS[variant]}|{selected[0]['n_seeds']}|{scores}|")
        lines.append('')
    lines += ['## 照合', '', f"完了した{len(audit['checkpoints'])}モデル、{audit['prediction_checks']:,}推定を元ラベルから照合した。固定比較と{audit['input_hash_checks']:,}入力ファイルのhashは一致した。", '',
              '全黒画像から未知楕円の一意な復元はできない。半画面遮蔽の強度は画像幅の割合であり、内周の欠損率と同じではない。直接回帰は常に楕円を返すため、対象なし・情報消失時の保留能力を備えていない。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'models': len(audit['checkpoints']), 'prediction_checks': audit['prediction_checks']})


if __name__ == '__main__':
    main()
