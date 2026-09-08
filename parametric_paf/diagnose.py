"""既存画像の抽出・楕円推定を分解し、再現可能な診断結果を保存する。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, read_image, write_image, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import load_model
from paf_ring_detection.methods.cnn_ransac import detect_from_probability
from paf_ring_detection.methods.ransac import fit_ellipse_ransac
from paf_ring_detection.methods.canny_contour_ransac import detect_canny_contour_ransac
from paf_ring_detection.methods.zhang2019 import detect_zhang_arc_candidates
from paf_ring_detection.methods.zhang2019_paf import score_zhang_candidates_for_paf, select_zhang_inner_boundary
from parametric_paf.diagnosis_metrics import contour_metrics, probability_support


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(result):
    """巨大なインライアマスクを除き、候補の数値情報を保持する。"""
    if result is None:
        return None
    return {k: v for k, v in result.items() if k != 'inlier_mask'}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')


def measure(result, truth, shape):
    ellipse = result['ellipse'] if result else None
    values = evaluate_ellipses(ellipse, truth, shape) if ellipse else {'ellipse_iou': 0.0}
    return {'detected': ellipse is not None, 'iou_success': values['ellipse_iou'] >= 0.8,
            **values, **contour_metrics(ellipse, truth)}


def panel(image, truth, ellipse, caption):
    image = image.copy()
    cv2.ellipse(image, truth, (0, 255, 0), 1, cv2.LINE_AA)
    if ellipse is not None:
        cv2.ellipse(image, ellipse, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(image, caption, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'parametric_paf/output/comparison')
    parser.add_argument('--output', type=Path, default=ROOT/'parametric_paf/output/diagnosis_v1')
    parser.add_argument('--models', nargs='+', default=['baseline', 'low_taper', 'thick_wall', 'reference_1194M'])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    manifest = read_json(source/'manifest.json')
    config = read_json(ROOT/'config/experiment.json')
    checkpoint = ROOT/config['paths']['checkpoint']
    samples = [s for s in manifest['samples'] if s['model'] in args.models]
    if not samples:
        raise ValueError('対象画像がありません')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    model = load_model(checkpoint, device)
    provenance = {
        'source': str(source), 'manifest_sha256': digest(source/'manifest.json'),
        'checkpoint_sha256': digest(checkpoint), 'config': config, 'models': args.models,
        'seed_offsets': [0, 1, 2, 3, 4], 'oracle_band_px': 2.0,
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                        [Path(__file__), ROOT/'parametric_paf/diagnosis_metrics.py']
                        + sorted((ROOT/'paf_ring_detection').rglob('*.py'))},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'torch': torch.__version__, 'opencv': cv2.__version__, 'device': str(device),
    }
    if (output/'provenance.json').exists() and read_json(output/'provenance.json') != provenance:
        raise ValueError('既存診断と設定またはコードが異なります。別の出力先を指定してください')
    save_json(output/'provenance.json', provenance)
    records, overview = [], []
    size = config['input_size']
    for index, sample in enumerate(samples):
        key = sample['model']+'/'+sample['condition_id']
        record_path = output/'records'/f'{key}.json'
        if record_path.exists():
            records.append(read_json(record_path))
            continue
        image_original = read_image(source/sample['image'])
        image = cv2.resize(image_original, (size, size), interpolation=cv2.INTER_AREA)
        truth = scaled_ellipse(label_ellipse(sample), size/image_original.shape[1])
        seed = (config['seed'] + zlib.crc32(sample['condition_id'].encode())) % 2**32
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        with torch.inference_mode():
            probability = torch.sigmoid(model(torch.from_numpy(rgb).float().unsqueeze(0).to(device)/127.5-1))[0, 0].cpu().numpy()
        points, distances, support = probability_support(probability, truth)
        predictions = [detect_from_probability(probability, config['ransac'], (seed+i) % 2**32) for i in range(5)]
        runs = [{'seed': (seed+i) % 2**32, 'result': compact(r), **measure(r, truth, image.shape)}
                for i, r in enumerate(predictions)]
        # 正解近傍の抽出点のみを残すoracle。通常の検出結果には含めない。
        selected = points[distances <= 2.0]
        weights = probability[selected[:, 1].astype(int), selected[:, 0].astype(int)]
        oracle = fit_ellipse_ransac(selected, image.shape, config['ransac'], weights=weights, random_seed=seed)
        # 投影点の理想入力で、ラベルと共通RANSACの幾何学的整合を確認する。
        ideal = fit_ellipse_ransac(np.asarray(sample['image_points'], np.float32)*size/image_original.shape[1],
                                   image.shape, config['ransac'], random_seed=seed)
        canny, cs = detect_canny_contour_ransac(image, config['canny_contour_ransac'], config['ransac'], seed)
        zs = config['zhang2019']
        zc, stages = detect_zhang_arc_candidates(image, zs['preprocess'], zs['detector'])
        zc = score_zhang_candidates_for_paf(zc, stages, zs['paf_postprocess']['candidate_validation'])
        zhang = select_zhang_inner_boundary(zc, zs['paf_postprocess']['selector'])
        record = {'model': sample['model'], 'condition_id': sample['condition_id'],
                  'camera_tilt': sample['camera_tilt'], 'camera_azimuth': sample['camera_azimuth'],
                  'light_tilt': sample['light_tilt'], 'light_azimuth': sample['light_azimuth'],
                  'image_sha256': digest(source/sample['image']), 'truth': truth,
                  'support': support, 'runs': runs,
                  'oracle_truth_band': {'result': compact(oracle), **measure(oracle, truth, image.shape)},
                  'oracle_ideal_points': {'result': compact(ideal), **measure(ideal, truth, image.shape)},
                  'canny': {'result': compact(canny), 'candidates': [compact(c) for c in cs['candidates']], **measure(canny, truth, image.shape)},
                  'zhang': {'result': compact(zhang), 'candidates': [compact(c) for c in zc], **measure(zhang, truth, image.shape)}}
        array_path = output/'arrays'/f'{key}.npz'
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(array_path, probability=probability, threshold_points=points,
                            truth_band_points=selected, canny_edges=cs['edges'])
        prediction = predictions[0]['ellipse'] if predictions[0] else None
        heatmap = cv2.applyColorMap(np.uint8(np.clip(probability, 0, 1)*255), cv2.COLORMAP_INFERNO)
        edge_image = cv2.cvtColor(cs['edges'], cv2.COLOR_GRAY2BGR)
        tile = np.vstack((np.hstack((panel(image, truth, prediction, sample['condition_id']+' CNN'),
                                    panel(heatmap, truth, None, f"prob coverage={support['full_contour_coverage']:.2f}"))),
                          np.hstack((panel(edge_image, truth, canny['ellipse'] if canny else None, 'Canny'),
                                    panel(image, truth, oracle['ellipse'] if oracle else None, f"oracle IoU={record['oracle_truth_band']['ellipse_iou']:.2f}")))))
        write_image(output/'panels'/f'{key}.png', tile)
        save_json(record_path, record)
        records.append(record)
        print(f'{index+1}/{len(samples)} {key} IoU={runs[0]["ellipse_iou"]:.3f}', flush=True)
    # 成否で選別せず、視点ごとに全照明条件の画像を並べる。
    for name in args.models:
        keys = sorted({(r['camera_tilt'], r['camera_azimuth']) for r in records if r['model'] == name})
        for tilt, azimuth in keys:
            subset = [r for r in records if (r['model'], r['camera_tilt'], r['camera_azimuth']) == (name, tilt, azimuth)]
            tiles = [read_image(output/'panels'/name/f"{r['condition_id']}.png") for r in subset]
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            write_image(output/'sheets'/f'{name}_{tilt:02d}_{azimuth:03d}.png',
                        np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]))
    for r in records:
        overview.append({k:r[k] for k in ('model', 'condition_id', 'camera_tilt', 'camera_azimuth', 'light_tilt', 'light_azimuth')} |
                        r['support'] | {'seed0_iou': r['runs'][0]['ellipse_iou'],
                        'seed_successes': sum(x['iou_success'] for x in r['runs']),
                        'oracle_band_iou': r['oracle_truth_band']['ellipse_iou'],
                        'ideal_iou': r['oracle_ideal_points']['ellipse_iou']})
    with (output/'overview.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(overview[0]))
        writer.writeheader()
        writer.writerows(overview)
    save_json(output/'summary.json', {name: {'n':sum(r['model']==name for r in records),
              'seed_successes':[sum(r['runs'][i]['iou_success'] for r in records if r['model']==name) for i in range(5)],
              'oracle_band_successes':sum(r['oracle_truth_band']['iou_success'] for r in records if r['model']==name),
              'ideal_successes':sum(r['oracle_ideal_points']['iou_success'] for r in records if r['model']==name)} for name in args.models})


if __name__ == '__main__':
    main()
