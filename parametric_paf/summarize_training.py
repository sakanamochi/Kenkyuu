"""小規模学習12本と共通validationの完了を検証し、研究用集計を保存する。"""

import ast
import csv
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parametric_paf.study import PLAN, OUTPUT, read, write, sha, verify_freeze
from paf_ring_detection.methods.cnn import TinyUNet


def main():
    """全seedの完了・入力対応を確認してから、形状を等重みで比較する。"""
    verify_freeze()
    protocol = read(PLAN / 'protocol.json')
    readiness = read(OUTPUT / 'readiness.json')
    assert readiness['status'] == 'passed' and readiness['training_started']
    assert readiness['shape_count'] == 54
    assert readiness['train_images_per_arm'] == 1024 and readiness['validation_images_per_arm'] == 192
    manifest_path = OUTPUT / 'validation/manifest.json'
    samples = read(manifest_path)['samples']
    expected = {s['sample_id']: s for s in samples}
    assert len(expected) == 192 and all(s['split'] == 'validation' for s in samples)
    shape_ids = sorted({s['shape_id'] for s in samples})
    assert len(shape_ids) == 8
    rows, shape_rows, evidence = [], [], []
    geometry_runs = []
    geometry_by_sample = {}
    for p in [OUTPUT / 'readiness.json', OUTPUT / 'render_data_provenance.json', manifest_path]:
        evidence.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)})
    reference = ROOT / 'output/models/cnn_best.pt'
    reference_hash = sha(reference)
    assert reference_hash == 'd9238666ab08572aeb26005b5bbc3bc83d87ae574442c8e89fb7eb4b878fc4f6'

    for arm in protocol['arms']:
        for seed in protocol['training_seeds']:
            model_dir = OUTPUT / 'models' / arm / str(seed)
            checkpoint_path = model_dir / 'cnn_best.pt'
            history_path = model_dir / 'training_history.json'
            history = read(history_path)
            assert [r['epoch'] for r in history] == list(range(1, 13))
            assert all(math.isfinite(r[k]) for r in history for k in ['train_loss', 'validation_loss'])
            log_path = OUTPUT / 'training_logs' / f'{arm}_{seed}.log'
            logged = [ast.literal_eval(line) for line in log_path.read_text(encoding='utf-8-sig').splitlines()
                      if line.startswith("{'epoch':")]
            assert logged == history
            best = min(history, key=lambda r: r['validation_loss'])
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['epoch'] == best['epoch']
            assert checkpoint['validation_loss'] == best['validation_loss']
            assert checkpoint['base_channels'] == 16 and checkpoint['input_size'] == 256
            model = TinyUNet(16)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            assert all(torch.isfinite(t).all().item() for t in checkpoint['model_state'].values())
            assert sha(checkpoint_path) != reference_hash

            evaluation_dir = OUTPUT / 'evaluation' / arm / str(seed) / 'validation'
            provenance = read(evaluation_dir / 'provenance.json')
            assert provenance['checkpoint_sha256'] == sha(checkpoint_path)
            assert provenance['manifest_sha256'] == sha(manifest_path)
            assert provenance['protocol_sha256'] == sha(PLAN / 'protocol.json')
            assert provenance['selected_config_sha256'] == sha(OUTPUT / 'geometry_validation/selected_config.json')
            assert provenance['code_sha256'] == sha(ROOT / 'parametric_paf/evaluate_study.py')
            assert provenance['sample_count'] == 192 and not provenance['limited_preflight']
            with (evaluation_dir / 'results.csv').open(encoding='utf-8-sig', newline='') as f:
                predictions = list(csv.DictReader(f))
            assert len(predictions) == 576
            assert {(r['sample_id'], r['method']) for r in predictions} == {
                (sid, method) for sid in expected for method in ['cnn', 'canny', 'zhang']}
            for r in predictions:
                sample = expected[r['sample_id']]
                assert r['shape_id'] == sample['shape_id'] and r['split'] == 'validation'
                assert r['condition_id'] == sample['condition']['condition_id']
                detected = r['detected'] == 'True'
                iou_success = detected and float(r['ellipse_iou']) >= 0.8
                strict = iou_success and float(r['contour_p95_px']) <= 2
                assert (r['iou_success'] == 'True') == iou_success
                assert (r['strict_success'] == 'True') == strict
                assert (r['missed'] == 'True') == (not detected)
                assert (r['wrong_output'] == 'True') == (detected and not strict)
            cnn = [r for r in predictions if r['method'] == 'cnn']
            geometry = []
            for method in ['canny', 'zhang']:
                group = [r for r in predictions if r['method'] == method]
                for r in group:
                    geometry_by_sample.setdefault((method, r['sample_id']), []).append({
                        'arm': arm, 'seed': seed, 'ransac_seed': int(r['ransac_seed']),
                        'strict_success': r['strict_success'] == 'True',
                        'contour_p95_px': float(r['contour_p95_px']) if r['contour_p95_px'] else None})
                geometry.append({'arm': arm, 'seed': seed, 'method': method, 'n': len(group),
                                 'strict_successes': sum(r['strict_success'] == 'True' for r in group),
                                 'iou_successes': sum(r['iou_success'] == 'True' for r in group),
                                 'missed': sum(r['missed'] == 'True' for r in group),
                                 'wrong_output': sum(r['wrong_output'] == 'True' for r in group)})
            geometry_runs.extend(geometry)
            for sid in shape_ids:
                group = [r for r in cnn if r['shape_id'] == sid]
                assert len(group) == 24
                shape_rows.append({'arm': arm, 'seed': seed, 'shape_id': sid, 'n': 24,
                                   'strict_successes': sum(r['strict_success'] == 'True' for r in group)})
            rows.append({'arm': arm, 'seed': seed, 'n': len(cnn), 'best_epoch': best['epoch'],
                         'best_validation_loss': best['validation_loss'],
                         'final_train_loss': history[-1]['train_loss'],
                         'final_validation_loss': history[-1]['validation_loss'],
                         'strict_successes': sum(r['strict_success'] == 'True' for r in cnn),
                         'iou_successes': sum(r['iou_success'] == 'True' for r in cnn),
                         'missed': sum(r['missed'] == 'True' for r in cnn),
                         'wrong_output': sum(r['wrong_output'] == 'True' for r in cnn),
                         'mean_full_contour_coverage': sum(float(r['probability_full_contour_coverage']) for r in cnn) / len(cnn)})
            files = [checkpoint_path, history_path, log_path, evaluation_dir / 'results.csv',
                     evaluation_dir / 'summary.json', evaluation_dir / 'provenance.json',
                     PLAN / 'configs' / f'{arm}_{seed}.json']
            evidence.extend({'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in files)

    # 同じ形状・撮像条件・学習seedを対応させ、画像数を形状数と混同しない。
    lookup = {(r['arm'], r['seed'], r['shape_id']): r for r in shape_rows}
    contrasts = []
    for seed in protocol['training_seeds']:
        for positive, negative in [('D', 'B'), ('C', 'A')]:
            differences = [(lookup[positive, seed, sid]['strict_successes'] -
                            lookup[negative, seed, sid]['strict_successes']) / 24 * 100 for sid in shape_ids]
            contrasts.append({'contrast': f'{positive}-{negative}', 'seed': seed,
                              'mean_shape_difference_pp': sum(differences) / len(differences),
                              'shape_differences_pp': dict(zip(shape_ids, differences))})
    primary = [r['mean_shape_difference_pp'] for r in contrasts if r['contrast'] == 'D-B']
    mean_difference = sum(primary) / len(primary)
    passed = all(x > 0 for x in primary) and mean_difference >= 10
    variations = [{'method': method, 'sample_id': sid, 'evaluations': group}
                  for (method, sid), group in geometry_by_sample.items()
                  if len({r['strict_success'] for r in group}) > 1]
    result = {'experiment_id': protocol['experiment_id'], 'training_runs_completed': 12,
              'epochs_per_run': 12, 'optimizer_updates_per_run': 768,
              'data_preparation': {'reused_training_images': 4096, 'reused_validation_images': 192,
                                   'newly_rendered_images_this_stage': 0},
              'validation_images_per_run': 192, 'final_test_used': False,
              'runs': rows, 'shape_results': shape_rows, 'contrasts': contrasts,
              'geometry_results_by_evaluation': geometry_runs,
              'geometry_success_variations': variations,
              'mean_D_minus_B_pp': mean_difference, 'continuation_gate_passed': passed,
              'reference_checkpoint_sha256': reference_hash,
              'freeze_sha256': sha(PLAN / 'freeze.json'),
              'summary_code_sha256': sha(Path(__file__)), 'evidence': evidence}
    report_dir = OUTPUT / 'training_summary'
    write(report_dir / 'summary.json', result)
    for name, records in [('runs.csv', rows), ('shape_results.csv', shape_rows), ('geometry_results.csv', geometry_runs)]:
        with (report_dir / name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = ['# 一段階目の多様化学習結果', '', '結果日：2026-09-08（JST）。', '',
             '実験：`paf_shape_appearance_v1`。1194Mの画像・重みを学習に使わず、合成データから12本を学習。',
             'データ準備は保存済みの合成学習4,096枚と共通validation192枚を再検査して利用。新規描画は0枚。',
             '学習は各1,024枚・12epoch。共通validationは未学習8形状×24条件＝192枚。独立テストは未使用。', '',
             '主成功：IoU≥0.80かつ双方向輪郭距離p95≤2px。全画像を分母に集計。', '',
             '|条件|seed|採用epoch|主成功/192|主成功率|IoU成功/192|未検出|誤楕円|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['arm']}|{r['seed']}|{r['best_epoch']}|{r['strict_successes']}|"
                     f"{r['strict_successes']/192:.1%}|{r['iou_successes']}|{r['missed']}|{r['wrong_output']}|")
    lines += ['', '|比較|seed|形状平均差（ポイント）|', '|---|---:|---:|']
    for r in contrasts:
        lines.append(f"|{r['contrast']}|{r['seed']}|{r['mean_shape_difference_pp']:+.2f}|")
    lines += ['', '|幾何比較|主成功/192|主成功率|IoU成功/192|未検出|誤楕円|',
              '|---|---:|---:|---:|---:|---:|']
    names = {'canny': 'Canny＋輪郭別RANSAC', 'zhang': 'Zhang 2019型（再現実装）'}
    for method in ['canny', 'zhang']:
        group = [r for r in geometry_runs if r['method'] == method]
        def span(key):
            values = [r[key] for r in group]
            return str(min(values)) if min(values) == max(values) else f'{min(values)}～{max(values)}'
        rates = [r['strict_successes'] / 192 for r in group]
        rate_text = f'{min(rates):.1%}' if min(rates) == max(rates) else f'{min(rates):.1%}～{max(rates):.1%}'
        lines.append(f"|{names[method]}|{span('strict_successes')}|{rate_text}|"
                     f"{span('iou_successes')}|{span('missed')}|{span('wrong_output')}|")
    lines += ['', '幾何方式はこのvalidationで閾値選択済み。独立テストの性能比較ではない。']
    for item in variations:
        distances = [r['contour_p95_px'] for r in item['evaluations'] if r['contour_p95_px'] is not None]
        seeds = sorted({r['ransac_seed'] for r in item['evaluations']})
        lines.append(f"{names[item['method']]}の{item['sample_id']}に実行間の主成功判定差がある。"
                     f"RANSAC seed={seeds}、輪郭p95は約{min(distances):.2f}～{max(distances):.2f}px。")
    if variations:
        lines += ['原因は未特定で、幾何成績は範囲と実行別表を保存する。',
                  '固定したNumPy側のRANSAC seedだけで全推定の完全再現性を保証しない。独立比較前に再現性の確認が必要。']
    lines += ['', f'D−Bの3 seed平均差：{mean_difference:+.2f}ポイント。小規模継続基準：' + ('合格。' if passed else '未達。'),
              '基準は3 seed全てで正、平均10ポイント以上。validationによる判断であり、独立テストでの汎化結果ではない。', '',
              'A/Bは単一形状、C/Dは16形状。A/Cは基本外観、B/Dは拡張外観。',
              '形状別結果はshape_results.csv、lossとseed別結果はruns.csv、入力・重み・評価のhashはsummary.jsonを参照。',
              '各モデルはvalidation loss最小epochを採用。12epochで十分に収束したとは断定しない。',
              '16形状の必要十分性、撮像条件、対象なし誤検出、実画像への汎化は未検証。', '',
              '学習例：各D形状の最初のcondition_idを固定抽出。緑は教師内周。視点・照明は形状間で異なるため、形状だけの対応比較ではない。',
              '![学習16形状と教師内周](training_examples.png)', '']
    (report_dir / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'completed': 12, 'mean_D_minus_B_pp': mean_difference,
                      'continuation_gate_passed': passed, 'report': str(report_dir / 'REPORT.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
