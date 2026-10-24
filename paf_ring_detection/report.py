"""同じtest画像による4方式の成績を条件別CSVとJSONへまとめる。"""

import csv
from collections import defaultdict
from pathlib import Path

from paf_ring_detection.data import read_json, write_json
from paf_ring_detection.evaluate import METHODS, result_path
from paf_ring_detection.prepare import digest

METHOD_LABELS = {'cnn_m1194m': '穴領域CNN（M1194M学習）',
    'cnn_diversified': '穴領域CNN（汎化形状学習）', 'canny': 'Canny＋輪郭別RANSAC',
    'zhang2019': 'Zhang 2019型（再現実装）'}


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
                    groups[(row['degradation'], row['severity'])].append(row)
                for (effect, severity), rows in groups.items():
                    count = len(rows)
                    summary.append({'dataset': dataset, 'method': METHOD_LABELS[method], 'seed': seed,
                        'degradation': effect, 'severity': severity, 'sample_count': count,
                        'strict_success_count': sum(r['strict_success'] for r in rows),
                        'strict_success_rate': sum(r['strict_success'] for r in rows) / count,
                        'iou_success_rate': sum(r['iou_success'] for r in rows) / count,
                        'missed_count': sum(not r['detected'] for r in rows),
                        'wrong_output_count': sum(r['detected'] and not r['strict_success'] for r in rows)})
    if not summary:
        raise ValueError('評価結果がありません。evaluateを先に実行してください')
    write_json(root / 'summary.json', {'complete': not missing, 'limited': limit is not None,
                                      'missing': missing, 'scores': summary})
    with (root / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'集計: {root}（未実施の評価: {len(missing)}件）', flush=True)
