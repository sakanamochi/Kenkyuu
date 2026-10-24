"""投影範囲を広げたsource検証で、角度被覆スコアを省く単純化を評価する。"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from research.paf_generalization.diagnose import BASE, OUT as PREVIOUS, sha
from research.paf_generalization_v2 import consensus

OUT = ROOT/'output/paf_generalization_v2/consensus_projection'


def main(stage):
    model, baseline = consensus.load()
    views = {'original_validation': BASE/'datasets/diversified', 'surface_validation': PREVIOUS/'surface_counterfactual',
             'projection_plain_validation': OUT.parent/'projection_coverage/native_plain',
             'projection_surface_validation': OUT.parent/'projection_coverage/native_surface'}
    protocol = {'implementation_sha256': sha(Path(__file__)), 'scorer_sha256': sha(Path(consensus.__file__)),
        'checkpoint_sha256': baseline['checkpoint_sha256'], 'boundary_sha256': baseline['boundary_sha256'],
        'ransac_sha256': baseline['ransac_sha256'], 'ransac': baseline['ransac'], 'methods': ['coverage', 'support'],
        'inference_batch_size': 1, 'opencv_seed_per_image': 20261025, 'minimum_gain': .01,
        'validation_views': {name: {'root': str(root), 'manifest_sha256': sha(root/'manifest.json')} for name, root in views.items()},
        'selection': 'mean success on four paired source views; require1pp gain; no target criterion',
        'scope': '元2ビューの結論を保持し、投影範囲を加えた固定source検証で再検討する。方式・入力testは既存探索と関連し独立な最終評価ではない。4ビューは同じ105形状。処理は角度被覆による乗算の省略だけ。'}
    path = OUT/'protocol.json'
    if path.exists():
        assert read_json(path) == protocol
    else:
        write_json(path, protocol)
    consensus.OUT = OUT

    def evaluate(root, split, name):
        path = OUT/f'{name}.json'
        if path.exists():
            saved = read_json(path)
            assert saved['protocol'] == protocol and saved['manifest_sha256'] == sha(root/'manifest.json')
            return saved['rates']
        previous = OUT.parent/f'consensus/{name}.json'
        if name in ('original_validation', 'surface_validation'):
            # 同じ重み・乱数・推定式で照合済みの2ビューを、出典hashを付けて再利用する。
            saved = read_json(previous)
            for field in ('checkpoint_sha256', 'boundary_sha256', 'ransac_sha256', 'ransac', 'methods', 'inference_batch_size', 'opencv_seed_per_image'):
                assert saved['protocol'][field] == protocol[field]
            assert saved['manifest_sha256'] == sha(root/'manifest.json')
            write_json(path, {**saved, 'protocol': protocol, 'reused_file': str(previous.relative_to(ROOT)),
                              'reused_sha256': sha(previous)})
            return saved['rates']
        return consensus.evaluate(model, protocol, root, split, name)

    if stage == 'validation':
        scores = [evaluate(root, 'validation', name) for name, root in views.items()]
        rates = {m: float(np.mean([row[m] for row in scores])) for m in protocol['methods']}
        selected = 'support' if rates['support']-rates['coverage'] >= protocol['minimum_gain'] else 'coverage'
        write_json(OUT/'selection.json', {'rates': rates, 'selected': selected,
            'input_hashes': [sha(OUT/f'{name}.json') for name in views]})
        print('four-view selection', rates, selected, flush=True)
    else:
        if read_json(OUT/'selection.json')['selected'] != 'support':
            print('Four source views retain angular coverage.', flush=True)
            return
        for source in ('1194m', 'diversified'):
            evaluate(BASE/'datasets'/source, 'test', source)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validation', 'test'])
    main(parser.parse_args().stage)
