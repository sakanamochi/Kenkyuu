"""最終候補の可視境界CNN・中心線RANSACを固定した、投影・外観の除去対照。"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from research.paf_generalization import experiment, boundary
from research.paf_generalization_v2 import projection

OUT = ROOT/'output/paf_core_ablation_v2'
CONFIG = Path(__file__).with_suffix('.json')
RETAIN = True, True


class FactorDataset(projection.ProjectionDataset):
    """4ビューの抽選列を保ち、除く要因だけを対応する元ビューへ置き換える。"""

    def __init__(self, dataset_dir, split, input_size, cache=False):
        super().__init__(dataset_dir, split, input_size, cache)
        if len(self.views) == 4:
            keep_projection, keep_surface = RETAIN
            # indexは投影ビット×2＋表面ビット。抽選数と他方の要因を変えない。
            mapping = [(i//2*2 if keep_projection else 0)+(i % 2 if keep_surface else 0) for i in range(4)]
            self.views = [self.views[i] for i in mapping]


def freeze(folder, config, variant, seed):
    """共通実装の保全に加え、要因の対応写像と既存の両要因あり結果を固定する。"""
    config = copy.deepcopy(config)
    config['factor_mapping_implementation_sha256'] = experiment.sha(Path(__file__))
    config['shared_reference_config_sha256'] = experiment.sha(ROOT/'research/paf_generalization_v2/projection.json')
    return projection.freeze(folder, config, variant, seed)


def train(variant, seed):
    global RETAIN
    settings = read_json(CONFIG)['variants'][variant]
    RETAIN = settings['retain_projection'], settings['retain_surface']
    return projection.train(variant, seed)


def verify_references():
    """既存の両要因あり・開口幅遮蔽ありの全3seedを上書きせず参照する。"""
    references = {}
    config = read_json(CONFIG)
    shared_keys = ('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'positive_weight', 'ransac')
    for name, variant in [('projection_surface', 'boundary_projection_mix'),
                          ('projection_surface_relative', 'boundary_projection_relative')]:
        for seed in config['seeds']:
            folder = projection.DATA_ROOT/'runs'/variant/str(seed)
            metadata = read_json(folder/'provenance.json')
            assert metadata['training_source'] == 'diversified' and metadata['target_data_used_for_training'] is False
            assert all(metadata['config'][key] == config[key] for key in shared_keys)
            assert read_json(folder/'complete.json')['checkpoint_sha256'] == experiment.sha(folder/'best.pt')
            for path, digest in metadata['files'].items():
                assert experiment.sha(ROOT/path) == digest, path
            for file in ('best.pt', 'provenance.json', 'complete.json', 'training_history.json', 'test_1194m.json', 'test_diversified.json'):
                path = folder/file
                references[str(path.relative_to(ROOT))] = experiment.sha(path)
    record = {'reference_files': references, 'config_sha256': experiment.sha(CONFIG),
              'implementation_sha256': experiment.sha(Path(__file__)),
              'scope': '最終候補の基礎を固定し、mainへの一致を要求しない。既存の両要因ありと遮蔽追加の全3seedを再利用。'}
    path = OUT/'references.json'
    if path.exists():
        assert read_json(path) == record
    else:
        write_json(path, record)


if __name__ == '__main__':
    verify_references()
    experiment.CONFIG = CONFIG
    experiment.OUT = OUT
    experiment.RingDataset = FactorDataset
    experiment.objective = boundary.boundary_objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.train_variant = train
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
