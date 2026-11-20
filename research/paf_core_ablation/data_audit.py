"""対照画像の対応と、実際の投影内周の分布を原ラベルから確認する。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from research.paf_generalization.diagnose import sha
from research.paf_core_ablation.experiment import OUT, BASE, SURFACE, PROJECTION


def projected(root, sample):
    """継承されたconditionsのカメラ値ではなく、描画済み頂点の投影を使う。"""
    label = read_json(root/sample['label'])
    center, axes, angle = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
    return {'cx': center[0], 'cy': center[1], 'major': max(axes),
            'axis_ratio': min(axes)/max(axes), 'angle': angle}


def main():
    roots = {'original_plain': BASE/'diversified', 'original_surface': SURFACE,
             'projection_plain': PROJECTION/'native_plain', 'projection_surface': PROJECTION/'native_surface'}
    paired = 0
    for split in ('train', 'validation'):
        views = {name: load_samples(root, split) for name, root in roots.items()}
        ids = [s['sample_id'] for s in views['original_plain']]
        assert all([s['sample_id'] for s in samples] == ids for samples in views.values())
        for index in range(len(ids)):
            samples = {name: values[index] for name, values in views.items()}
            original = samples['original_plain']
            for sample in samples.values():
                for field in ('sample_id', 'group_id', 'parameters', 'augmentation', 'split'):
                    assert sample[field] == original[field], (ids[index], field)
            for first, second in [('original_plain', 'original_surface'), ('projection_plain', 'projection_surface')]:
                a = projected(roots[first], samples[first])
                b = projected(roots[second], samples[second])
                assert np.allclose(list(a.values()), list(b.values()), rtol=0, atol=1e-5)
                paired += 1
    summaries = []
    for name, root, split in [('source_original', BASE/'diversified', 'train'),
                              ('source_projection', PROJECTION/'native_plain', 'train'),
                              ('source_test', BASE/'diversified', 'test'),
                              ('target_test', BASE/'1194m', 'test')]:
        samples = [s for s in load_samples(root, split) if s['conditions']['degradation'] == 'clean']
        projections = [projected(root, s) for s in samples]
        summary = {'dataset': name, 'n_images': len(samples), 'statistics': {}}
        for field in ('cx', 'cy', 'major', 'axis_ratio'):
            values = np.array([p[field] for p in projections])
            summary['statistics'][field] = dict(zip(('min', 'q25', 'median', 'q75', 'max'), np.quantile(values, [0, .25, .5, .75, 1]).tolist()))
        if name.startswith('source_'):
            summary['pearson_major_to_dimensions'] = {field: float(np.corrcoef(
                [p['major'] for p in projections], [s['parameters'][field] for s in samples])[0, 1])
                for field in ('height', 'bottom_outer_diameter', 'hole_diameter')}
        summaries.append(summary)
    write_json(OUT/'data_audit.json', {'plan_sha256': sha(OUT/'plan.json'),
        'implementation_sha256': sha(Path(__file__)), 'paired_surface_projection_checks': paired,
        'statistics': summaries, 'scope': '投影分布と対応画像の教師/形状/劣化の一致を測定。汎化性能や因果の証明ではない。target testは診断のみで学習に使用しない。'})
    print({'paired_checks': paired, 'datasets': [(s['dataset'], s['n_images'], s['statistics']['major']) for s in summaries]}, flush=True)


if __name__ == '__main__':
    main()
