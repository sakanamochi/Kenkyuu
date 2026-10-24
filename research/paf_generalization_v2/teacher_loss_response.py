"""正例重み付きBCEの画素独立最適値で、滑らかな教師の推定結果を診断する。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from research.paf_generalization import boundary
from research.paf_generalization.diagnose import BASE, sha
from research.paf_generalization_v2 import analytic_teacher as teacher


def bce_response(target, weight=12.):
    """BCE単独を画素ごとに最小化したp=w*t/(1+(w-1)*t)。Diceとの最適解ではない。"""
    return weight*target/(1+(weight-1)*target)


def main(audit=False):
    out = ROOT/'output/paf_generalization_v2/teacher_loss_response'
    spec = teacher.protocol()
    spec['files'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    spec['positive_weight'] = 12.
    assert read_json(ROOT/'research/paf_generalization_v2/projection_occlusion.json')['positive_weight'] == 12.
    spec['selection'] = '事後の損失応答診断。方式や重みの選択には使用しない。'
    spec['scope'] = 'GT教師を正例重み12のBCE単独の画素独立最適値へ変換して推定器へ渡す。Dice項・CNN制約・最適化過程を含まない。実学習の出力や性能上限ではない。'
    original_out, original_fit = teacher.OUT, boundary.boundary_ellipse

    def estimate(probability, settings):
        return original_fit(bce_response(probability), settings)

    try:
        teacher.OUT = out
        boundary.boundary_ellipse = estimate
        views = [('original_validation', BASE/'datasets/diversified', 'validation'),
                 ('projection_validation', ROOT/'output/paf_generalization_v2/projection_coverage/native_plain', 'validation'),
                 ('1194m', BASE/'datasets/1194m', 'test'), ('diversified', BASE/'datasets/diversified', 'test')]
        for name, root, split in views:
            teacher.evaluate(name, root, split, spec, audit=audit)
        if audit:
            write_json(out/'audit.json', {'recomputed': {name: sha(out/f'{name}.json') for name, _, _ in views},
                'fit_checks': 2324, 'scope': 'GT教師・BCE応答・細線化・RANSAC・元GTのIoU・集計を再計算して完全一致。'})
    finally:
        teacher.OUT, boundary.boundary_ellipse = original_out, original_fit


if __name__ == '__main__':
    teacher.torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    main(parser.parse_args().audit)
