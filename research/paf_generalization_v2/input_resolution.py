"""重みを固定し、入力解像度だけを変える比較をsource validationで選択する。"""

import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, sample_image, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.geometry import evaluate_ellipses
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha
from research.paf_generalization_v2.oriented import OrientedUNet

OUT = ROOT/'output/paf_generalization_v2/input_resolution'


def input_and_truth(root, sample, size):
    """480で既存劣化を適用し、指定解像度へ一度だけリサイズする。"""
    image = cv2.resize(sample_image(root, sample), (size, size), interpolation=cv2.INTER_AREA)
    label = read_json(root/sample['label'])
    truth = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    return image, truth


def estimate(probability, settings):
    """距離・軸長・点数の閾値を倍率で換算し、楕円を共通256座標へ戻す。"""
    size = probability.shape[0]
    scale = size/256
    options = copy.deepcopy(settings)
    options['distance_threshold_px'] *= scale
    options['min_axis_px'] *= scale
    options['min_inliers'] = round(options['min_inliers']*scale)
    cv2.setRNGSeed(20261025)
    ellipse = boundary_ellipse(probability, options)
    return scaled_ellipse(ellipse, 1/scale) if ellipse else None


def evaluate(model, protocol, root, split, name, sizes, destination):
    """入力と出力の解像度以外は共通とし、成功判定は256座標で行う。"""
    path = destination/f'{name}.json'
    if path.exists():
        saved = read_json(path)
        assert saved['protocol'] == protocol and saved['sizes'] == sizes
        assert saved['manifest_sha256'] == sha(root/'manifest.json')
        return saved['rates']
    rows = []
    for sample in load_samples(root, split):
        for size in sizes:
            image, truth = input_and_truth(root, sample, size)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            ellipse = estimate(probability, protocol['ransac'])
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                         'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                         'input_size': size, 'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8})
    rates = {str(size): float(np.mean([r['success'] for r in rows if r['input_size'] == size])) for size in sizes}
    write_json(path, {'protocol': protocol, 'manifest_sha256': sha(root/'manifest.json'),
                      'sizes': sizes, 'rates': rates, 'rows': rows})
    print(name, rates, flush=True)
    return rates


def main(stage):
    folder = ROOT/'output/paf_generalization_v2/runs/boundary_normal_aux/20260907'
    digest = sha(folder/'best.pt')
    assert read_json(folder/'complete.json')['checkpoint_sha256'] == digest
    checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    model = OrientedUNet(16).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])
    protocol = {'implementation_sha256': sha(Path(__file__)), 'checkpoint_sha256': digest,
                'oriented_sha256': sha(Path(__file__).with_name('oriented.py')),
                'boundary_sha256': sha(ROOT/'research/paf_generalization/boundary.py'),
                'ransac_sha256': sha(ROOT/'paf_ring_detection/methods/ransac.py'),
                'sizes': [256, 384, 512], 'training_size': 256,
                'ransac': checkpoint['provenance']['config']['ransac'],
                'opencv_seed_per_image': 20261025, 'inference_batch_size': 1,
                'minimum_source_gain': .01,
                'selection': 'mean original/surface source validation success; require 1pp gain over256; ties prefer lower resolution',
                'coordinate_convention': 'proportional ellipse coordinates as in the fixed256 evaluation',
                'interpretation': 'fixed256-trained weights; higher inference resolution also changes apparent scale and relative receptive field, not pure subpixel precision'}
    path = OUT/'protocol.json'
    if path.exists():
        assert read_json(path) == protocol
    else:
        write_json(path, protocol)
    if stage == 'validation':
        names = ('original_validation', 'surface_validation')
        scores = [evaluate(model, protocol, root, 'validation', name, protocol['sizes'], OUT) for name, root in
                  zip(names, (BASE/'datasets/diversified', PREVIOUS/'surface_counterfactual'))]
        rates = {str(size): float(np.mean([s[str(size)] for s in scores])) for size in protocol['sizes']}
        best = max(protocol['sizes'], key=lambda size: rates[str(size)])
        selected = best if rates[str(best)]-rates['256'] >= protocol['minimum_source_gain'] else 256
        write_json(OUT/'selection.json', {'rates': rates, 'selected_size': selected,
                   'input_hashes': [sha(OUT/f'{name}.json') for name in names]})
    else:
        selected = read_json(OUT/'selection.json')['selected_size']
        if selected == 256:
            print('Source validation retains256 input.', flush=True)
            return
        for source in ('1194m', 'diversified'):
            evaluate(model, protocol, BASE/'datasets'/source, 'test', source, [256, selected], OUT)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validation', 'test'])
    main(parser.parse_args().stage)
