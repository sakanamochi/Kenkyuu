"""候補と3seedの重みを固定してから、別カメラ群の1194Mで追加確認する。"""

import argparse
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.evaluate import _geometry
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.boundary import boundary_ellipse
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha, image_and_truth
from research.paf_generalization_v2.oriented import OrientedUNet
from research.paf_generalization_v2.group_normalization import GroupUNet

OUT = ROOT/'output/paf_generalization_v2/confirmation'
DATA = BASE/'datasets/1194m'
SEEDS = (20260907, 20260908, 20260909)
ARCHITECTURES = {name: TinyUNet for name in ('boundary_amodal', 'boundary_native', 'boundary_resampled',
    'boundary_native_mix', 'boundary_projection_mix', 'boundary_interior', 'boundary_online_mask', 'boundary_relative_mask',
    'boundary_surface', 'boundary_projection_relative', 'boundary_projection_validation', 'boundary_projection_analytic')}
ARCHITECTURES.update(boundary_normal_aux=OrientedUNet, boundary_center_aux=OrientedUNet)
ARCHITECTURES['boundary_group_norm'] = GroupUNet


def cases():
    """既定602画像と4条件を展開する。学習側の分割や画像は変更しない。"""
    plan = read_json(OUT/'plan.json')
    assert plan['source_manifest_sha256'] == sha(DATA/'manifest.json')
    original = {s['sample_id']: s for s in read_json(DATA/'manifest.json')['samples']}
    test_groups = {s['group_id'] for s in load_samples(DATA, 'test')}
    result = []
    for reference in plan['samples']:
        sample = original[reference['sample_id']]
        assert sample['split'] in plan['source_splits'] and sample['group_id'] not in test_groups
        assert sample['group_id'] == reference['group_id'] and sample['conditions']['base_sample_id'] == reference['base_sample_id']
        for field in ('image', 'label'):
            assert sample[field] == reference[field]
            assert sha(DATA/sample[field]) == reference[field+'_sha256'] == sample[field+'_sha256']
        for effect, severity in plan['effects']:
            identity = f"{reference['base_sample_id']}:{effect}:{severity}:confirmation_v2:20261025"
            seed = zlib.crc32(identity.encode('utf-8'))
            result.append({**sample, 'sample_id': identity, 'base_sample_id': reference['base_sample_id'],
                'split': 'confirmation', 'conditions': {**sample['conditions'], 'degradation': effect, 'severity': severity},
                'augmentation': None if effect == 'clean' else {'name': effect, 'severity': severity, 'seed': seed},
                'effect_seed': seed})
    assert len(result) == 2408 and len({s['sample_id'] for s in result}) == 2408
    assert len({s['group_id'] for s in result}) == 86
    return result


def freeze(variant):
    """単一の境界CNN候補と基準3seedを固定し、結果による重み選択を防ぐ。"""
    if variant not in ARCHITECTURES or variant == 'boundary_surface':
        raise ValueError('通常のTiny U-Netまたは3出力Tiny U-Netの境界候補を指定してください。')
    destination = OUT/'frozen_candidate.json'
    if destination.exists():
        fixed = verify()
        assert fixed['variant'] == variant
        return fixed
    cases()
    jobs = []
    for name, root in [(variant, ROOT/'output/paf_generalization_v2'), ('boundary_surface', PREVIOUS)]:
        common_config = None
        for seed in SEEDS:
            folder = root/f'runs/{name}/{seed}'
            checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
            digest = sha(folder/'best.pt')
            assert read_json(folder/'complete.json')['checkpoint_sha256'] == digest
            history = read_json(folder/'training_history.json')
            assert len(history) == 60 and checkpoint['validation_loss'] == min(r['validation_loss'] for r in history)
            metadata = read_json(folder/'provenance.json')
            assert checkpoint['provenance'] == metadata
            assert metadata['training_source'] == 'diversified' and metadata['target_data_used_for_training'] is False
            assert all(torch.isfinite(value).all() for value in checkpoint['model_state'].values())
            assert checkpoint['provenance']['variant'] == name and checkpoint['provenance']['seed'] == seed
            config = checkpoint['provenance']['config']
            assert config['source'] == 'diversified'
            # 全seedで同じ学習・選択・推論仕様を用い、対象側で都合のよい組合せを作らない。
            if common_config is None:
                common_config = config
            else:
                assert config == common_config
            # 学習時の構造・入力・設定を結果と一緒に保存する。
            jobs.append({'method': name, 'seed': seed, 'checkpoint': str((folder/'best.pt').relative_to(ROOT)),
                'checkpoint_sha256': digest, 'config': config,
                'provenance_sha256': sha(folder/'provenance.json')})
    jobs.extend({'method': name, 'seed': None} for name in ('canny', 'zhang2019'))
    files = list((ROOT/'paf_ring_detection').rglob('*.py'))
    files += [ROOT/'config/experiment.json', Path(__file__),
              ROOT/'research/paf_generalization/boundary.py', ROOT/'research/paf_generalization/diagnose.py',
              ROOT/'research/paf_generalization_v2/oriented.py', ROOT/'research/paf_generalization_v2/group_normalization.py']
    fixed = {'created_utc': datetime.now(timezone.utc).isoformat(), 'variant': variant,
        'plan_sha256': sha(OUT/'plan.json'), 'jobs': jobs, 'success_iou': .8,
        'frozen_files': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'estimator': 'standard_weighted_ransac', 'opencv_seed_per_image': 20261025,
        'uniform_policy': '一様画像は未検出として全画像分母に含める。',
        'scope': '同一1194Mの別カメラ群。未知機体全般の盲検ではない。追加確認後に候補・設定・seedを選び直さない。'}
    write_json(destination, fixed)
    return fixed


def verify():
    fixed = read_json(OUT/'frozen_candidate.json')
    assert fixed['plan_sha256'] == sha(OUT/'plan.json')
    for path, digest in fixed['frozen_files'].items():
        assert sha(ROOT/path) == digest, path
    for job in fixed['jobs']:
        if 'checkpoint' in job:
            path = ROOT/job['checkpoint']
            assert sha(path) == job['checkpoint_sha256']
            assert sha(path.with_name('provenance.json')) == job['provenance_sha256']
    return fixed


def evaluate():
    fixed = verify()
    samples = cases()
    config = read_json(ROOT/'config/experiment.json')
    frozen_hash = sha(OUT/'frozen_candidate.json')
    for job in fixed['jobs']:
        method, seed = job['method'], job['seed']
        destination = OUT/f"results/{method}_{seed if seed is not None else 'fixed'}.json"
        if destination.exists():
            assert read_json(destination)['frozen_candidate_sha256'] == frozen_hash
            continue
        model = None
        if seed is not None:
            state = torch.load(ROOT/job['checkpoint'], map_location='cpu', weights_only=False)['model_state']
            # 最終層のチャネル数を厳密ロードで確認し、方向出力は本評価の推定に使わない。
            cls = ARCHITECTURES[method]
            model = cls(16).cuda().eval()
            model.load_state_dict(state, strict=True)
        rows = []
        for index, sample in enumerate(samples):
            image, truth = image_and_truth(DATA, sample)
            cv2.setRNGSeed(fixed['opencv_seed_per_image'])
            if np.ptp(image) == 0:
                ellipse = None
            elif model is not None:
                tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).cuda().float()[None]/127.5-1
                with torch.inference_mode():
                    probability = torch.sigmoid(model(tensor)[0, 0]).cpu().numpy()
                ellipse = boundary_ellipse(probability, job['config']['ransac'])
            else:
                geometry_seed = (config['evaluation_seed']+zlib.crc32(sample['sample_id'].encode('utf-8'))) % (2**32)
                ellipse = _geometry(image, method, config, geometry_seed)
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'base_sample_id': sample['base_sample_id'],
                'group_id': sample['group_id'], 'degradation': sample['conditions']['degradation'],
                'severity': sample['conditions']['severity'], 'effect_seed': sample['effect_seed'],
                'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= fixed['success_iou']})
            if (index+1) % 602 == 0:
                print('CONFIRMATION', method, seed, index+1, len(samples), flush=True)
        write_json(destination, {'frozen_candidate_sha256': frozen_hash, 'job': job, 'rows': rows})


def report():
    """全推定を元ラベルで照合し、最良seedではなく全3seedと平均を保存する。"""
    fixed = verify()
    samples = cases()
    truth = {}
    for sample in samples:
        label = read_json(DATA/sample['label'])
        truth[sample['sample_id']] = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    records, files = [], {}
    for job in fixed['jobs']:
        seed = job['seed']
        path = OUT/f"results/{job['method']}_{seed if seed is not None else 'fixed'}.json"
        saved = read_json(path)
        assert saved['frozen_candidate_sha256'] == sha(OUT/'frozen_candidate.json') and saved['job'] == job
        assert [r['sample_id'] for r in saved['rows']] == [s['sample_id'] for s in samples]
        for sample, row in zip(samples, saved['rows']):
            assert row['group_id'] == sample['group_id'] and row['effect_seed'] == sample['effect_seed']
            assert (row['degradation'], row['severity']) == (sample['conditions']['degradation'], sample['conditions']['severity'])
            e = row['ellipse']
            ellipse = (tuple(e[0]), tuple(e[1]), e[2]) if e else None
            iou = evaluate_ellipses(ellipse, truth[row['sample_id']], (256, 256))['ellipse_iou'] if ellipse else 0.
            assert row['ellipse_iou'] == iou and row['success'] == (iou >= fixed['success_iou'])
            records.append({**row, 'method': job['method'], 'seed': seed})
        files[str(path.relative_to(ROOT))] = sha(path)
    summaries = []
    for method in dict.fromkeys(job['method'] for job in fixed['jobs']):
        for effect in ('all', 'clean', 'black_rectangle', 'sensor_whiteout', 'sensor_black_crush'):
            rows = [r for r in records if r['method'] == method and (effect == 'all' or r['degradation'] == effect)]
            summaries.append({'method': method, 'degradation': effect, 'n_images': len({r['sample_id'] for r in rows}),
                'success_rate': float(np.mean([r['success'] for r in rows])),
                'per_seed': {str(s): float(np.mean([r['success'] for r in rows if r['seed'] == s]))
                             for s in dict.fromkeys(r['seed'] for r in rows)}})
    write_json(OUT/'summary.json', {'rows': summaries, 'prediction_checks': len(records), 'files': files,
        'frozen_candidate_sha256': sha(OUT/'frozen_candidate.json'), 'scope': fixed['scope']})
    print('confirmation predictions checked', len(records), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify_data', 'freeze', 'evaluate', 'report'])
    parser.add_argument('--variant')
    args = parser.parse_args()
    if args.stage == 'verify_data':
        print('disjoint fixed cases', len(cases()))
    elif args.stage == 'freeze':
        freeze(args.variant)
    else:
        {'evaluate': evaluate, 'report': report}[args.stage]()
