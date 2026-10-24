"""同じtest画像による4方式の成績を条件別CSVとJSONへまとめる。"""

import csv
from collections import defaultdict
from pathlib import Path

from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.evaluate import METHODS, result_path
from paf_ring_detection.prepare import digest

METHOD_LABELS = {'cnn_1194m': '穴領域CNN（1194M学習）',
    'cnn_diversified': '穴領域CNN（多様形状学習）', 'canny': 'Canny＋輪郭別RANSAC',
    'zhang2019': 'Zhang 2019型（再現実装）'}


def aggregate_scores(scores):
    """CNNはseed平均、幾何は一回の評価として4方式を並べる。"""
    groups = defaultdict(list)
    for row in scores:
        groups[(row['dataset'], row['method'], row['degradation'], row['severity'])].append(row)
    result = []
    for (dataset, method, effect, severity), rows in groups.items():
        count = rows[0]['sample_count']
        if any(row['sample_count'] != count for row in rows):
            raise ValueError('seed間で評価枚数が異なります')
        rates = [row['success_rate'] for row in rows]
        result.append({'dataset': dataset, 'method': method, 'degradation': effect, 'severity': severity,
            'seed_count': len(rows) if rows[0]['seed'] is not None else 0, 'sample_count_per_model': count,
            'success_rate': sum(rates) / len(rows), 'seed_min': min(rates), 'seed_max': max(rates),
            'missed_rate': sum(row['missed_count'] for row in rows) / (len(rows) * count)})
    return result


def write_comparison(root, config, aggregate, complete, limited):
    """二つの比較と条件別成績を、再計算可能な原記録に結び付ける。"""
    lines = ['# 穴領域CNNと幾何2方式の比較', '',
        f'全設定の評価完了：{complete}。少数画像による動作確認：{limited}。', '',
        f"成功はIoU≥{config['success_iou']}、全方式モノクロ{config['input_size']}入力。未検出も分母に含む。",
        f"CNNは同じ条件の{len(config['training_seeds'])}seed平均を用い、幾何は各画像一回の評価。seed別の値と劣化強度別の値はscores.csvを参照する。", '',
        f"学習には通常・遮蔽・白飛び・黒つぶれを含め、両学習元の枚数を揃える。validation loss最小の重みを選択し、各seedを{config['cnn']['epochs']}epoch学習する。", '']
    for number, (dataset, title) in enumerate((('1194m', '1194M'), ('diversified', '多様形状')), start=1):
        rows = [row for row in aggregate if row['dataset'] == dataset]
        lines += [f'## 比較{number}：{title}データセット', '',
            '|方式|評価枚数／モデル|成功率|seed別成功率の範囲|未検出率|',
            '|---|---:|---:|---:|---:|']
        for method in METHOD_LABELS.values():
            overall = next((r for r in rows if r['method'] == method and r['degradation'] == 'all'), None)
            if overall:
                r = overall
                seed_range = f"{r['seed_min']:.2%}〜{r['seed_max']:.2%}" if r['seed_count'] else '—'
                lines.append(f"|{method}|{r['sample_count_per_model']}|{r['success_rate']:.2%}|{seed_range}|{r['missed_rate']:.2%}|")
        lines += ['', '|方式|通常|遮蔽|白飛び|黒つぶれ|', '|---|---:|---:|---:|---:|']
        for method in METHOD_LABELS.values():
            values = []
            for effect in ('clean', 'black_rectangle', 'sensor_whiteout', 'sensor_black_crush'):
                r = next((r for r in rows if r['method'] == method and r['degradation'] == effect and r['severity'] is None), None)
                values.append(f"{r['success_rate']:.2%}" if r else '未評価')
            lines.append('|' + '|'.join([method, *values]) + '|')
        lines += ['']
    lines += ['## 解釈の範囲', '',
        '1194Mのtestは1194M学習CNNにとって同一機体の撮影条件分割。多様形状のtestは学習寸法と非重複だが、同じ合成形状族・分布内の評価である。',
        '両原データの撮影・材質・投影サイズ分布は異なるため、この比較だけで形状多様化の単独因果効果や実画像への汎化を断定しない。',
        '既存画像は方式検討で参照済みのため、この比較は参照評価。矩形遮蔽は画像端からの幅、白飛びは露光増加・飽和・ブルーム、黒つぶれは低露光・ノイズ・黒レベル・量子化の近似。通常画像と強度別の値を併せて読む。', '',
        '正本は各評価集合の推定JSON、summary.json、scores.csv。幾何の閾値は固定し、testで調整していない。',
        'Zhang 2019型（再現実装）：Zhang, Pan, Ma, Sensors 2019, DOI 10.3390/s19235243。著者コードの移植・論文性能の完全再現ではない。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def audit_comparison(config):
    """全重み・全epoch・全推定を元ラベルと照合し、完了の根拠を保存する。"""
    import math
    import numpy as np
    import torch
    from paf_ring_detection.data import label_ellipse, scaled_ellipse
    from paf_ring_detection.evaluate import checkpoint_path
    from paf_ring_detection.geometry import evaluate_ellipses
    from paf_ring_detection.prepare import verify_prepared

    verify_prepared(config)
    root = Path(config['output'])
    provenance = read_json(root / 'provenance.json')
    for name, expected in provenance['sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'固定した学習・検出実装が変わっています: {name}')
    manifests, checkpoint_hashes, training = {}, {}, []
    for source in config['sources']:
        folder = root / 'datasets' / source
        manifest = read_json(folder / 'manifest.json')
        groups = defaultdict(set)
        for sample in manifest['samples']:
            groups[sample['group_id']].add(sample['split'])
        if any(len(splits) != 1 for splits in groups.values()):
            raise ValueError('元画像のグループが複数splitに混在しています')
        manifests[source] = [s for s in manifest['samples'] if s['split'] == 'test']
        for seed in config['training_seeds']:
            checkpoint = checkpoint_path(config, source, seed)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            history = read_json(checkpoint.parent / 'training_history.json')
            if [r['epoch'] for r in history] != list(range(1, config['cnn']['epochs'] + 1)):
                raise ValueError(f'全epochが完了していません: {checkpoint}')
            if not all(math.isfinite(r[key]) for r in history for key in ('train_loss', 'validation_loss')):
                raise ValueError(f'学習履歴に非有限値があります: {checkpoint}')
            best = min(history, key=lambda r: r['validation_loss'])
            if (saved['target'] != 'hole_region' or saved['training_source'] != source or saved['seed'] != seed
                    or saved['input_size'] != config['input_size'] or saved['epoch'] != best['epoch']
                    or saved['validation_loss'] != best['validation_loss']
                    or saved['config']['dataset_manifest_sha256'] != digest(folder / 'manifest.json')
                    or any(saved['config'].get(k) != v for k, v in config.items())
                    or not all(torch.isfinite(t).all().item() for t in saved['model_state'].values())):
                raise ValueError(f'最小validation lossの重みと履歴が一致しません: {checkpoint}')
            checkpoint_hashes[(source, seed)] = digest(checkpoint)
            training.append({'source': source, 'seed': seed, 'epochs': len(history),
                             'selected_epoch': saved['epoch'], 'validation_loss': saved['validation_loss']})
    recomputed, hashes = 0, {}
    for source, samples in manifests.items():
        folder = root / 'datasets' / source
        truths = {}
        for sample in samples:
            label = read_json(folder / sample['label'])
            truths[sample['sample_id']] = scaled_ellipse(label_ellipse(label), config['input_size'] / label['image_width'])
        for method in METHODS:
            for seed in (config['training_seeds'] if method.startswith('cnn_') else [None]):
                path = root / 'results' / source / f'{method}_{seed if seed is not None else "fixed"}.json'
                record = read_json(path)
                rows = record['rows']
                expected_checkpoint = checkpoint_hashes[(method[4:], seed)] if seed is not None else None
                if (record['config'] != config or record['limited'] or record['input_size'] != config['input_size']
                        or record['method'] != method or record['seed'] != seed or record['dataset'] != source
                        or record['manifest_sha256'] != digest(folder / 'manifest.json')
                        or record['checkpoint_sha256'] != expected_checkpoint
                        or record['evaluator_sha256'] != digest(Path('paf_ring_detection/evaluate.py'))
                        or [r['sample_id'] for r in rows] != [s['sample_id'] for s in samples]):
                    raise ValueError(f'評価設定・画像・重みが一致しません: {path}')
                for sample, row in zip(samples, rows):
                    truth = truths[sample['sample_id']]
                    if not np.allclose([*truth[0], *truth[1], truth[2]],
                                       [*row['truth_ellipse'][0], *row['truth_ellipse'][1], row['truth_ellipse'][2]], atol=1e-8, rtol=0):
                        raise ValueError(f'正解楕円が元ラベルと異なります: {path}')
                    if any(row[key] != sample['conditions'][key] for key in ('base_sample_id', 'degradation', 'severity')):
                        raise ValueError(f'劣化条件が異なります: {path}')
                    ellipse = row['ellipse']
                    if ellipse is not None:
                        ellipse = (tuple(ellipse[0]), tuple(ellipse[1]), ellipse[2])
                    metrics = evaluate_ellipses(ellipse, truth, (config['input_size'], config['input_size'])) if ellipse else {'ellipse_iou': 0.0}
                    for key, value in metrics.items():
                        if value is None:
                            if row[key] is not None:
                                raise ValueError(f'未検出の指標が不一致です: {path}')
                        elif not math.isclose(row[key], value, rel_tol=1e-10, abs_tol=1e-10):
                            raise ValueError(f'指標の再計算が不一致です: {path} {key}')
                    success = bool(ellipse and metrics['ellipse_iou'] >= config['success_iou'])
                    if row['detected'] != bool(ellipse) or row['success'] != success:
                        raise ValueError(f'成功・未検出の判定が不一致です: {path}')
                    recomputed += 1
                hashes[path.relative_to(root).as_posix()] = digest(path)
                print(f'再計算照合: {source} {method} seed={seed} {len(rows)}件', flush=True)
    audit = {'status': 'passed', 'training': training, 'recomputed_predictions': recomputed,
             'result_sha256': hashes, 'reporter_sha256': digest(Path(__file__))}
    write_json(root / 'results/audit.json', audit)
    return audit


def build_report(config, limit=None):
    """seed別・劣化条件別に集計し、比較に不足する結果も明示する。"""
    root = result_path(config, limit)
    summary, missing = [], []
    for dataset in config['sources']:
        for method in METHODS:
            for seed in (config['training_seeds'] if method.startswith('cnn_') else [None]):
                path = root / dataset / f'{method}_{seed if seed is not None else "fixed"}.json'
                if not path.exists():
                    missing.append(str(path))
                    continue
                record = read_json(path)
                manifest = Path(config['output']) / 'datasets' / dataset / 'manifest.json'
                if record['manifest_sha256'] != digest(manifest) or record['config'] != config:
                    raise ValueError(f'現在のデータ・設定と結果が不一致です: {path}')
                groups = defaultdict(list)
                for row in record['rows']:
                    groups[('all', None)].append(row)
                    groups[(row['degradation'], None)].append(row)
                    groups[(row['degradation'], row['severity'])].append(row)
                for (effect, severity), rows in groups.items():
                    count = len(rows)
                    summary.append({'dataset': dataset, 'method': METHOD_LABELS[method], 'seed': seed,
                        'degradation': effect, 'severity': severity, 'sample_count': count,
                        'success_count': sum(r['success'] for r in rows),
                        'success_rate': sum(r['success'] for r in rows) / count,
                        'missed_count': sum(not r['detected'] for r in rows),
                        'wrong_output_count': sum(r['detected'] and not r['success'] for r in rows)})
    if not summary:
        raise ValueError('評価結果がありません。evaluateを先に実行してください')
    if not missing and limit is None:
        audit_comparison(config)
    aggregate = aggregate_scores(summary)
    write_json(root / 'summary.json', {'complete': not missing, 'limited': limit is not None,
                                      'missing': missing, 'scores': summary, 'aggregate': aggregate})
    write_comparison(root, config, aggregate, not missing, limit is not None)
    with (root / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'集計: {root}（未実施の評価: {len(missing)}件）', flush=True)
