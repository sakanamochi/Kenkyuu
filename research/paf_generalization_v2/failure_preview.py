"""角度ごとの半画面遮蔽失敗例で、基準と候補の境界確率を可視化する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, write_image
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.context import ContextUNet
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha, image_and_truth

OUT = ROOT/'output/paf_generalization_v2'


def draw_ellipse(image, ellipse, color):
    if ellipse:
        cv2.ellipse(image, (tuple(ellipse[0]), tuple(ellipse[1]), ellipse[2]), color, 1, cv2.LINE_AA)


def main(variant, seed):
    directory = OUT/f'runs/{variant}/{seed}'
    reference = PREVIOUS/f'runs/boundary_surface/{seed}'
    candidate_rows = read_json(directory/'test_1194m.json')['rows']
    reference_rows = {r['sample_id']: r for r in read_json(reference/'test_1194m.json')['rows']}
    clean = {r['base_sample_id']: r for r in candidate_rows if r['degradation'] == 'clean'}
    samples = {s['sample_id']: s for s in load_samples(BASE/'datasets/1194m', 'test')}
    models = []
    for folder, cls in [(reference, TinyUNet), (directory, ContextUNet if variant == 'boundary_context' else TinyUNet)]:
        model = cls(16).cuda().eval()
        model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=False)['model_state'])
        models.append(model)
    canvas = np.zeros((4*286, 4*256, 3), np.uint8)
    selected = []
    for index, tilt in enumerate((20, 40, 60, 75)):
        failed = sorted([r for r in candidate_rows if r['sample_id'].startswith(f'camera_t{tilt:03}')
                         and r['degradation'] == 'black_rectangle' and r['severity'] == .5 and not r['success']],
                        key=lambda r: r['sample_id'])
        if not failed:
            continue
        paired = [r for r in failed if clean[r['base_sample_id']]['success']]
        row = (paired or failed)[0]
        image, truth = image_and_truth(BASE/'datasets/1194m', samples[row['sample_id']])
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
        with torch.inference_mode():
            maps = [torch.sigmoid(model(x))[0, 0].cpu().numpy() for model in models]
        tiles = [image.copy()]
        tiles += [cv2.cvtColor(np.uint8(np.clip(p*255, 0, 255)), cv2.COLOR_GRAY2BGR) for p in maps]
        tiles += [image.copy()]
        draw_ellipse(tiles[0], truth, (0, 255, 0))
        draw_ellipse(tiles[-1], truth, (0, 255, 0))
        draw_ellipse(tiles[-1], row['ellipse'], (255, 0, 255))
        labels = ['input / truth', 'reference probability', 'candidate probability', f'candidate IoU={row["ellipse_iou"]:.3f}']
        for column, (tile, label) in enumerate(zip(tiles, labels)):
            canvas[index*286:index*286+256, column*256:(column+1)*256] = tile
            cv2.putText(canvas, label, (column*256+4, index*286+274), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.putText(canvas, f'tilt {tilt}', (4, index*286+18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        selected.append({'tilt': tilt, 'sample_id': row['sample_id'], 'candidate': row,
                         'reference': reference_rows[row['sample_id']], 'candidate_clean': clean[row['base_sample_id']]})
    destination = OUT/f'previews/{variant}_{seed}_half'
    write_image(destination.with_suffix('.png'), canvas)
    write_json(destination.with_suffix('.json'), {'selection': 'first_id_failure_per_tilt_prefer_clean_success',
               'implementation_sha256': sha(Path(__file__)), 'candidate_checkpoint_sha256': sha(directory/'best.pt'),
               'reference_checkpoint_sha256': sha(reference/'best.pt'), 'rows': selected})


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant')
    parser.add_argument('--seed', type=int, default=20260907)
    args = parser.parse_args()
    main(args.variant, args.seed)
