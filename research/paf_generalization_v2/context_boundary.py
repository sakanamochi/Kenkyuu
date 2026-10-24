"""同じ可視境界教師で、深部の参照範囲だけを広げる。"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_generalization import experiment, boundary, boundary_surface, context

BASE_FREEZE = boundary_surface.freeze_boundary_surface


def freeze(folder, config, variant, seed):
    """畳み込みのdilationと境界・模様の学習仕様を一緒に固定する。"""
    config = copy.deepcopy(config)
    config['context_boundary_files'] = {str(p.relative_to(ROOT)): experiment.sha(p)
                                        for p in (Path(__file__), Path(context.__file__))}
    return BASE_FREEZE(folder, config, variant, seed)


if __name__ == '__main__':
    experiment.CONFIG = Path(__file__).with_suffix('.json')
    experiment.OUT = ROOT/'output/paf_generalization_v2'
    experiment.RingDataset = boundary_surface.SurfaceBoundaryDataset
    experiment.TinyUNet = context.ContextUNet
    experiment.objective = boundary.boundary_objective
    experiment.freeze_run = freeze
    boundary.freeze_boundary = freeze
    experiment.evaluate_variant = boundary.evaluate_boundary
    experiment.main()
