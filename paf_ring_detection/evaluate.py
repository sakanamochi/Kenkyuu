"""二つの穴領域CNNと幾何2方式を同じ256入力で評価する。"""

import zlib
from pathlib import Path

import cv2
import torch

from paf_ring_detection.data import (label_ellipse, load_samples, read_json,
    sample_image, scaled_ellipse, write_json)
from paf_ring_detection.geometry import evaluate_ellipses, contour_metrics
from paf_ring_detection.methods.canny_contour_ransac import detect_canny_contour_ransac
from paf_ring_detection.methods.cnn import load_model
from paf_ring_detection.methods.cnn_region import region_ellipse
from paf_ring_detection.methods.zhang2019 import detect_zhang_arc_candidates
from paf_ring_detection.methods.zhang2019_paf import score_zhang_candidates_for_paf, select_zhang_inner_boundary
from paf_ring_detection.prepare import digest

METHODS = ('cnn_m1194m', 'cnn_diversified', 'canny', 'zhang2019')


def checkpoint_path(config, source, seed):
    return Path(config['output']) / 'models' / source / str(seed) / 'best.pt'


def result_path(config, limit=None):
    # 少数画像の動作確認を正式な比較結果へ混ぜない。
    return Path(config['output']) / ('results' if limit is None else f'checks/limit_{limit}')


def _geometry(image, method, config, seed):
    if method == 'canny':
        selected, _ = detect_canny_contour_ransac(image, config['canny_contour_ransac'], config['ransac'], seed)
    else:
        settings = config['zhang2019']
        candidates, stages = detect_zhang_arc_candidates(image, settings['preprocess'], settings['detector'])
        candidates = score_zhang_candidates_for_paf(candidates, stages, settings['paf_postprocess']['candidate_validation'])
        selected = select_zhang_inner_boundary(candidates, settings['paf_postprocess']['selector'])
    return selected['ellipse'] if selected else None


def evaluate(config, methods=None, seeds=None, limit=None):
    """全方式の入力を揃え、未検出を含む推定原記録を保存する。"""
    methods = list(METHODS if methods is None else methods)
    seeds = config['training_seeds'] if seeds is None else seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 必要な重みを先に確認し、不足時は結果を書き出さない。
    for method in methods:
        if method.startswith('cnn_'):
            for seed in seeds:
                path = checkpoint_path(config, method[4:], seed)
                if not path.exists():
                    raise FileNotFoundError(f'穴領域CNNの学習が必要です: {path}')
    for method in methods:
        for seed in (seeds if method.startswith('cnn_') else [None]):
            model, checkpoint = None, None
            if seed is not None:
                checkpoint = checkpoint_path(config, method[4:], seed)
                metadata = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if (metadata.get('target') != 'hole_region' or metadata['training_source'] != method[4:]
                        or metadata['input_size'] != config['input_size'] or metadata['seed'] != seed):
                    raise ValueError(f'穴領域CNNの仕様が不一致です: {checkpoint}')
                manifest = Path(config['output']) / 'datasets' / method[4:] / 'manifest.json'
                if metadata['config']['dataset_manifest_sha256'] != digest(manifest):
                    raise ValueError(f'学習後にデータ分割が変わっています: {manifest}')
                if any(metadata['config'].get(key) != value for key, value in config.items()):
                    raise ValueError(f'学習時と現在の比較設定が異なります: {checkpoint}')
                model = load_model(checkpoint, device)
            for dataset_name in config['sources']:
                folder = Path(config['output']) / 'datasets' / dataset_name
                samples = load_samples(folder, 'test')
                if limit is not None:
                    samples = samples[:limit]
                if not samples:
                    raise ValueError(f'test画像がありません: {folder}')
                rows = []
                for sample in samples:
                    image = cv2.resize(sample_image(folder, sample), (config['input_size'], config['input_size']),
                                       interpolation=cv2.INTER_AREA)
                    label = read_json(folder / sample['label'])
                    truth = scaled_ellipse(label_ellipse(label), config['input_size'] / label['image_width'])
                    random_seed = (config['evaluation_seed'] + zlib.crc32(sample['sample_id'].encode('utf-8'))) % 2**32
                    if model is not None:
                        tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())
                        with torch.inference_mode():
                            probability = torch.sigmoid(model((tensor.float()[None].to(device) / 127.5) - 1))[0, 0].cpu().numpy()
                        ellipse = region_ellipse(probability, config['region_probability_threshold'])
                    else:
                        ellipse = _geometry(image, method, config, random_seed)
                    metrics = evaluate_ellipses(ellipse, truth, image.shape) if ellipse is not None else {'ellipse_iou': 0.0}
                    distances = contour_metrics(ellipse, truth)
                    iou_success = metrics['ellipse_iou'] >= config['success_iou']
                    success = bool(ellipse is not None and iou_success and distances['contour_p95_px'] <= config['contour_p95_px'])
                    rows.append({'sample_id': sample['sample_id'], 'base_sample_id': sample['conditions']['base_sample_id'],
                        'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                        'ellipse': ellipse, 'truth_ellipse': truth, 'detected': ellipse is not None,
                        'iou_success': iou_success, 'strict_success': success, **metrics, **distances})
                destination = result_path(config, limit) / dataset_name / f'{method}_{seed if seed is not None else "fixed"}.json'
                write_json(destination, {'method': method, 'seed': seed, 'dataset': dataset_name,
                    'split': 'test', 'limited': limit is not None, 'input_size': config['input_size'],
                    'manifest_sha256': digest(folder / 'manifest.json'), 'config': config,
                    'checkpoint_sha256': digest(checkpoint) if checkpoint else None,
                    'evaluator_sha256': digest(Path(__file__)), 'rows': rows})
                print(dataset_name, method, seed, sum(r['strict_success'] for r in rows), '/', len(rows), flush=True)
