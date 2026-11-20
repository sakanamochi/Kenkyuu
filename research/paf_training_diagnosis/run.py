"""固定1seedで、投影の寄与・学習時間・旧CNNの点群処理を分離する。"""

import argparse
import copy
import shutil
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization import experiment as trainer, boundary
from research.paf_generalization.diagnose import sha, image_and_truth
from research.paf_generalization_v2 import projection
from research.paf_core_ablation import boundary_core
from research.paf_core_ablation.point_representation import predict, MODES
from research.paf_fair_comparison import experiment as fair

OUT = ROOT/'output/paf_training_diagnosis_v1'
BASE = fair.BASE
SEED = 20260907
VARIANTS = ('target_plain', 'source_projection', 'source_base', 'source_both')
REFERENCES = {
    'source_base': ROOT/f'output/paf_core_ablation_v2/runs/core_base/{SEED}',
    'source_projection': ROOT/f'output/paf_core_ablation_v2/runs/core_projection/{SEED}',
    'source_both': ROOT/f'output/paf_generalization_v2/runs/boundary_projection_mix/{SEED}',
    'target_projection': ROOT/f'output/paf_1194m_no_surface_v1/runs/1194m/{SEED}',
    'main': ROOT/f'output/paf_core_ablation_v1/runs/main_1194m/{SEED}',
}
WRITE = trainer.write_json


class PlainTargetDataset(fair.FairDataset):
    """4回分の抽選枠・開口幅遮蔽を保ち、元材質・元投影だけを返す。"""

    def __init__(self, root, split, size, cache=False):
        # 共有入口の保存先に依存せず、既存4ビューを読み元画像へ対応させる。
        super().__init__(BASE/'1194m', split, size, cache)
        if split == 'train':
            self.views = [self.views[0]]*4


def prepare():
    """結果を見る前に比較条件・参照モデル・共通実装と入力を固定する。"""
    common = read_json(ROOT/'research/paf_core_ablation/boundary_core.json')
    target = read_json(ROOT/'research/paf_fair_comparison/no_surface.json')
    configs = {}
    for name in VARIANTS:
        config = copy.deepcopy(target if name == 'target_plain' else common)
        config.update(experiment_id='paf_training_diagnosis_v1', seeds=[SEED],
                      epochs=60 if name == 'target_plain' else 120)
        config['variants'] = {name: {'spatial': False, 'geometry': False,
                                    'new_view': 'projection_coverage'}}
        config['diagnostic_mapping'] = ([0, 0, 2, 2] if name == 'source_projection'
            else [0, 1, 2, 3] if name == 'source_both' else [0, 0, 0, 0])
        config['diagnostic_source'] = '1194m' if name == 'target_plain' else 'diversified'
        path = OUT/f'configs/{name}.json'
        if path.exists():
            assert read_json(path) == config
        else:
            write_json(path, config)
        configs[name] = config
    # 完了済み比較の入力・共有実装を全件照合し、新しい保存先へ依存を記録する。
    previous = ROOT/'output/paf_1194m_no_surface_v1/plan.json'
    files = dict(read_json(previous)['files'])
    files[str(previous.relative_to(ROOT))] = sha(previous)
    for folder in REFERENCES.values():
        metadata = read_json(folder/'provenance.json')
        for path, digest in metadata.get('files', {}).items():
            assert sha(ROOT/path) == digest
            files[path] = digest
        for name in ('best.pt', 'provenance.json', 'training_history.json', 'complete.json'):
            path = folder/name
            files[str(path.relative_to(ROOT))] = sha(path)
        assert read_json(folder/'complete.json')['checkpoint_sha256'] == sha(folder/'best.pt')
    for path in (Path(__file__), ROOT/'research/paf_core_ablation/boundary_core.py',
                 ROOT/'research/paf_core_ablation/point_representation.py',
                 ROOT/'research/paf_core_ablation/main_compat.py'):
        files[str(path.relative_to(ROOT))] = sha(path)
    for path, digest in files.items():
        assert sha(ROOT/path) == digest, path
    plan = {'seed': SEED, 'configs': configs, 'files': files,
            'budgets': [60, 120], 'selection': 'minimum original training-source validation loss within each budget',
            'source_training': 'from the same random initialization; no optimizer restart at epoch 60',
            'target_control': 'remove projection only from the no-surface, relative-occlusion 1194M model',
            'test_scope': 'existing inspected test; no target test checkpoint selection',
            'torch': torch.__version__, 'device': torch.cuda.get_device_name()}
    path = OUT/'plan.json'
    if path.exists():
        assert read_json(path) == plan
    else:
        write_json(path, plan)
    return plan


def freeze(folder, config, variant, seed):
    """別保存先で、学習予算と学習元を参照計画に結び付ける。"""
    value = {'config': config, 'variant': variant, 'seed': seed,
             'plan_sha256': sha(OUT/'plan.json'), 'training_source': config['diagnostic_source']}
    path = folder/'provenance.json'
    if path.exists():
        assert read_json(path) == value
    else:
        write_json(path, value)
    return value


def save_history(path, value):
    """共有学習ループは変えず、予算60/120時点のvalidation最良重みを保存する。"""
    WRITE(path, value)
    path = Path(path)
    if path.name == 'training_history.json' and len(value) in (60, 120):
        destination = path.parent/f'budget_{len(value)}'
        destination.mkdir(exist_ok=True)
        shutil.copy2(path.parent/'best.pt', destination/'best.pt')
        WRITE(destination/'training_history.json', value)
        WRITE(destination/'complete.json', {'epochs': len(value),
              'checkpoint_sha256': sha(destination/'best.pt'),
              'best_validation_loss': min(r['validation_loss'] for r in value)})


def train(name):
    """元と同じ初期化・データ抽選・AdamWを使い、学習予算だけを延ばす。"""
    trainer.CONFIG = OUT/f'configs/{name}.json'
    trainer.OUT = OUT
    trainer.objective = boundary.boundary_objective
    trainer.freeze_run = freeze
    trainer.write_json = save_history
    if name == 'target_plain':
        trainer.RingDataset = PlainTargetDataset
    else:
        projection.NEW_VIEW = 'projection_coverage'
        boundary_core.RETAIN = name != 'source_base', name == 'source_both'
        trainer.RingDataset = boundary_core.FactorDataset
    trainer.train_variant(name, SEED)


def evaluate(checkpoint, name):
    """固定の現行推論で、両testの通常・全欠損強度を同じ規則で採点する。"""
    settings = read_json(ROOT/'research/paf_fair_comparison/experiment.json')['ransac']
    model = TinyUNet(16).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state'])
    for dataset in ('1194m', 'diversified'):
        path = OUT/f'results/{name}/test_{dataset}.json'
        if path.exists():
            assert read_json(path)['checkpoint_sha256'] == sha(checkpoint)
            continue
        rows = []
        for sample in load_samples(BASE/dataset, 'test'):
            image, truth = image_and_truth(BASE/dataset, sample)
            ellipse = None
            if np.ptp(image):
                x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
                with torch.inference_mode():
                    probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                cv2.setRNGSeed(20261025)
                ellipse = boundary.boundary_ellipse(probability, settings)
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            rows.append({'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                'ellipse': ellipse, 'truth_ellipse': truth, 'ellipse_iou': iou, 'success': iou >= .8})
        write_json(path, {'checkpoint_sha256': sha(checkpoint), 'plan_sha256': sha(OUT/'plan.json'),
                         'dataset': dataset, 'variant': name, 'seed': SEED, 'rows': rows})
        print('EVAL', name, dataset, sum(r['success'] for r in rows), len(rows), flush=True)


def old_model():
    """旧1194Mの同じ確率画像に、帯/中心線・45/12点と現行推論を適用する。"""
    folder = REFERENCES['main']
    settings = read_json(ROOT/'research/paf_core_ablation/experiment.json')['cnn_ransac']
    current = read_json(ROOT/'research/paf_fair_comparison/experiment.json')['ransac']
    model = TinyUNet(16).cuda().eval()
    model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=False)['model_state'])
    for dataset in ('1194m', 'diversified'):
        path = OUT/f'old_model/test_{dataset}.json'
        if path.exists():
            assert read_json(path)['checkpoint_sha256'] == sha(folder/'best.pt')
            continue
        saved = {r['sample_id']: r for r in read_json(folder/f'test_{dataset}.json')['rows']}
        rows = []
        for sample in load_samples(BASE/dataset, 'test'):
            image, truth = image_and_truth(BASE/dataset, sample)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
            with torch.inference_mode():
                probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            seed = (settings['random_seed']+zlib.crc32(sample['sample_id'].encode('utf-8'))) % 2**32
            for mode in list(MODES)+['current']:
                if mode == 'current':
                    cv2.setRNGSeed(20261025)
                    ellipse = boundary.boundary_ellipse(probability, current) if np.ptp(image) else None
                else:
                    ellipse = predict(probability, settings, seed, mode)
                iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
                if mode == 'band_45':
                    assert abs(iou-saved[sample['sample_id']]['ellipse_iou']) < 1e-10
                rows.append({'mode': mode, 'sample_id': sample['sample_id'], 'group_id': sample['group_id'],
                    'degradation': sample['conditions']['degradation'], 'severity': sample['conditions']['severity'],
                    'ellipse': ellipse, 'truth_ellipse': truth, 'major_axis_px': max(truth[1]),
                    'ellipse_iou': iou, 'success': iou >= .8})
        write_json(path, {'checkpoint_sha256': sha(folder/'best.pt'), 'plan_sha256': sha(OUT/'plan.json'),
                         'original_replay_max_error': 0., 'dataset': dataset, 'rows': rows})
        print('OLD MODEL', dataset, len(rows), flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'old', 'run'])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    prepare()
    if args.stage in ('old', 'run'):
        old_model()
    if args.stage in ('train', 'evaluate', 'run'):
        for name in args.variants:
            if args.stage in ('train', 'run'):
                train(name)
            if args.stage in ('evaluate', 'run'):
                for budget in ([60] if name == 'target_plain' else [60, 120]):
                    evaluate(OUT/f'runs/{name}/{SEED}/budget_{budget}/best.pt', f'{name}_{budget}')


if __name__ == '__main__':
    main()
