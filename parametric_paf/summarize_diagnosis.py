"""保存済み診断から、操作的な失敗分類と評価閾値の根拠を集計する。"""

import csv
import json
from pathlib import Path
import sys
from collections import Counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parametric_paf.diagnosis_metrics import contour_metrics, nearest_distances
from parametric_paf.diagnose import save_json, digest


def main():
    root = ROOT/'parametric_paf/output'
    output = root/'diagnosis_v1'
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((output/'records').rglob('*.json'))]
    geometry = json.loads((root/'diagnostic_geometry_v1/geometry.json').read_text(encoding='utf-8'))
    lookup = {(g['model'],g['camera_tilt'],g['camera_azimuth']):g for g in geometry}
    classifications, boundary_calibration = [], []
    for g in geometry:
        truth = cv2.fitEllipse(np.float32(g['image_points'])*256/480)
        for name, points in g['boundaries'].items():
            values = contour_metrics(cv2.fitEllipse(np.float32(points)*256/480), truth)
            boundary_calibration.append({'model':g['model'], 'camera_tilt':g['camera_tilt'],
                                         'camera_azimuth':g['camera_azimuth'], 'boundary':name, **values})
    for r in records:
        if r['model']=='reference_1194M':
            continue
        g = lookup[r['model'], r['camera_tilt'], r['camera_azimuth']]
        prediction = r['runs'][0]
        points = np.load(output/'arrays'/r['model']/f"{r['condition_id']}.npz")['threshold_points']
        visible = np.float32(g['image_points'])[np.asarray(g['visible'])]*256/480
        support = nearest_distances(visible, points)
        boundaries = {name: contour_metrics(prediction['result']['ellipse'], cv2.fitEllipse(np.float32(p)*256/480))['contour_p95_px']
                      for name,p in g['boundaries'].items()} if prediction['result'] else {}
        nearest = min(boundaries, key=boundaries.get) if boundaries else None
        # 分類は単一原因の断定ではなく、明示した観測条件による一次分類。
        if prediction['iou_success']:
            category = 'iou_success'
        elif len(visible)/len(g['visible']) < 0.1:
            category = 'geometrically_invisible'
        elif nearest and boundaries[nearest] <= 2.0:
            category = 'other_boundary_match'
        elif r['oracle_truth_band']['iou_success']:
            category = 'fit_selection_with_competing_response'
        else:
            category = 'insufficient_or_biased_extraction'
        classifications.append({'model':r['model'], 'condition_id':r['condition_id'],
            'category':category, 'visible_fraction':len(visible)/len(g['visible']),
            'visible_point_coverage':float((support<=2).mean()) if len(visible) else None,
            'nearest_other_boundary':nearest, 'nearest_other_p95_px':boundaries.get(nearest),
            'seed_successes':sum(x['iou_success'] for x in r['runs']),
            'iou':prediction['ellipse_iou'], 'contour_p95_px':prediction['contour_p95_px'],
            'strict_success':bool(prediction['iou_success'] and prediction['contour_p95_px']<=2),
            'oracle_band_iou':r['oracle_truth_band']['ellipse_iou'], **r['support']})
    for filename, rows in [('classification.csv',classifications),('boundary_calibration.csv',boundary_calibration)]:
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    soft = [json.loads(p.read_text(encoding='utf-8')) for p in (root/'diagnosis_soft_v1/records').rglob('*.json')]
    result = {}
    for name in ('baseline','low_taper','thick_wall'):
        rows = [r for r in records if r['model']==name]
        selected = [r for r in classifications if r['model']==name]
        result[name] = {
            'n':len(rows), 'classification':dict(Counter(r['category'] for r in selected)),
            'seed_successes':[sum(r['runs'][i]['iou_success'] for r in rows) for i in range(5)],
            'seed_unstable_images':sum(0<r['seed_successes']<5 for r in selected),
            'strict_successes':sum(r['strict_success'] for r in selected),
            'full_contour_coverage_mean':float(np.mean([r['full_contour_coverage'] for r in selected])),
            'original_fixed_light_successes':sum(r['runs'][0]['iou_success'] for r in rows if r['light_tilt']==60 and r['light_azimuth']==0),
            'soft_fixed_light_successes':sum(r['runs'][0]['iou_success'] for r in soft if r['model']==name),
        }
    save_json(output/'diagnostic_findings.json', {'models':result,
        'threshold_calibration':{'contour_p95_px':2.0,
            'nearest_tested_wrong_boundary_p95_px':min(r['contour_p95_px'] for r in boundary_calibration),
            'note':'256入力、教師リング幅3px、RANSAC許容2px。開発形状の既知別境界を棄却できる固定閾値。運用上の安全許容値ではない。'},
        'source_hashes':{'geometry':digest(root/'diagnostic_geometry_v1/geometry.json'),
                         'diagnosis_provenance':digest(output/'provenance.json'),
                         'soft_provenance':digest(root/'diagnosis_soft_v1/provenance.json')},
        'classification_caveat':'oracle成功は全点群からのフィット/選択への感度を示す。oracle失敗は抽出不足だけを証明せず、短い弧の不安定性も含む。',
        'visual_review_scope':'baseline/low_taper/thick_wallの全144画像を視点別24シートで確認。'})
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
