"""同じパラメータ数のまま深部の参照範囲を広げる対照実験。"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_generalization import experiment
from paf_ring_detection.methods.cnn import TinyUNet


class ContextUNet(TinyUNet):
    """ボトルネックの2畳み込みだけをdilation=4へ変更する。"""

    def __init__(self, base_channels=16):
        super().__init__(base_channels)
        for layer in (self.bottleneck[0], self.bottleneck[3]):
            layer.dilation = (4, 4)
            layer.padding = (4, 4)


original_freeze = experiment.freeze_run


def freeze_context(folder, config, variant, seed):
    """共通学習器に加えて、参照範囲を定める実装のhashも保存する。"""
    config = copy.deepcopy(config)
    config['context_implementation_sha256'] = experiment.sha(Path(__file__))
    return original_freeze(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_name('context.json')
    experiment.TinyUNet = ContextUNet
    experiment.freeze_run = freeze_context
    experiment.main()
