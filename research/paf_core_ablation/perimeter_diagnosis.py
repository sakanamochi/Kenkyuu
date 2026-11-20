"""固定CNNで支持総量と周長当たり支持密度を比較し、source validationで判定する。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_generalization.diagnose import sha, image_and_truth

OUT = ROOT/'output/paf_core_ablation_v2/perimeter_diagnosis'
MODEL = ROOT/'output/paf_generalization_v2/runs/boundary_projection_relative/20260907'
SOURCES = {'original_validation': ROOT/'output/hole_region_comparison_v1/datasets/diversified',
           'surface_validation': ROOT/'output/paf_generalization_v1/surface_counterfactual'}


def evaluate(model, protocol, name, root):
    destination = OUT/f'{name}.json'
    if destination.exists():
        saved = read_json(destination)
        assert saved['protocol_sha256'] == sha(OUT/'protocol.json')
        return saved['summary']
    rows = []
    samples = load_samples(root, 'validation')
    for index, sample in enumerate(samples):
        for field in ('image', 'label'):
            assert sha(root/sample[field]) == sample[field+'_sha256']
        image, truth = image_and_truth(root, sample)
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None]/127.5-1
        with torch.inference_mode():
            probability = torch.sigmoid(model(x))[0, 0].numpy()
        for power in (0., 1.):
            settings = {**protocol['ransac'], 'perimeter_power': power}
            # 仮説抽選は同一にし、周長正規化の有無だけを変える。
            cv2.setRNGSeed(protocol['opencv_seed_per_image'])
            ellipse = boundary_ellipse(probability, settings)
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                         'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                         'perimeter_power': power, 'truth_ellipse': truth, 'ellipse': ellipse,
                         'ellipse_iou': iou, 'success': iou >= .8})
        if (index+1) % 210 == 0:
            print(name, index+1, len(samples), flush=True)
    summary = []
    conditions = [('all', None)]+sorted({(r['degradation'], r['severity']) for r in rows})
    for power in (0., 1.):
        for effect, severity in conditions:
            selected = [r for r in rows if r['perimeter_power'] == power and
                        (effect == 'all' or (r['degradation'] == effect and r['severity'] == severity))]
            assert selected
            summary.append({'perimeter_power': power, 'degradation': effect, 'severity': severity,
                            'n_images': len(selected), 'success_rate': float(np.mean([r['success'] for r in selected])),
                            'mean_iou': float(np.mean([r['ellipse_iou'] for r in selected]))})
    write_json(destination, {'protocol_sha256': sha(OUT/'protocol.json'), 'summary': summary, 'rows': rows})
    print(name, summary, flush=True)
    return summary


def main():
    torch.set_num_threads(2)
    metadata = read_json(MODEL/'provenance.json')
    checkpoint = torch.load(MODEL/'best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['provenance'] == metadata
    assert read_json(MODEL/'complete.json')['checkpoint_sha256'] == sha(MODEL/'best.pt')
    assert metadata['target_data_used_for_training'] is False
    protocol = {'seed': 20260907, 'checkpoint_sha256': sha(MODEL/'best.pt'),
                'implementation_sha256': sha(Path(__file__)), 'ransac': metadata['config']['ransac'],
                'implementation_files': {str(p.relative_to(ROOT)): sha(p) for p in
                    (ROOT/'research/paf_generalization/boundary.py', ROOT/'paf_ring_detection/methods/ransac.py',
                     ROOT/'research/paf_generalization/diagnose.py', ROOT/'paf_ring_detection/geometry.py')},
                'manifests': {name: sha(root/'manifest.json') for name, root in SOURCES.items()},
                'device': 'cpu', 'opencv_seed_per_image': 20261025, 'powers': [0., 1.],
                'hypothesis': '支持総量による長い誤輪郭の優遇を、既存の周長正規化で減らせるか。',
                'gate': '両source validationで全体および各遮蔽強度の成功率が非低下、かつ少なくとも片方の全体が改善すれば次の反復へ進む。',
                'scope': 'source validationだけを使う1seed診断。通過しても採用ではなく、反復とtest確認が必要。'}
    assert protocol['ransac']['perimeter_power'] == 0.
    if (OUT/'protocol.json').exists():
        assert read_json(OUT/'protocol.json') == protocol
    else:
        write_json(OUT/'protocol.json', protocol)
    model = TinyUNet(16).cpu().eval()
    model.load_state_dict(checkpoint['model_state'])
    summaries = {name: evaluate(model, protocol, name, root) for name, root in SOURCES.items()}
    differences = {}
    for name, summary in summaries.items():
        differences[name] = {}
        conditions = {(r['degradation'], r['severity']) for r in summary if r['degradation'] in ('all', 'black_rectangle')}
        for effect, severity in sorted(conditions):
            values = {r['perimeter_power']: r['success_rate'] for r in summary
                      if r['degradation'] == effect and r['severity'] == severity}
            differences[name]['all' if effect == 'all' else f'{effect}:{severity}'] = values[1.]-values[0.]
    can_continue = all(v >= 0 for d in differences.values() for v in d.values()) and any(d['all'] > 0 for d in differences.values())
    write_json(OUT/'decision.json', {'can_continue_to_replication': can_continue, 'differences': differences,
        'protocol_sha256': sha(OUT/'protocol.json'), 'validation_hashes': {name: sha(OUT/f'{name}.json') for name in SOURCES},
        'selected_for_deployment': False})
    print('perimeter normalization source-validation gate', can_continue, differences, flush=True)


if __name__ == '__main__':
    main()
