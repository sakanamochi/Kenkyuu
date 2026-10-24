"""学習回数の効果と、条件付きの穴領域教師を同じPAF画像で検証する。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parametric_paf import train_random_paf as base
from parametric_paf.random_dataset import read, write, digest

SPEC = ROOT / 'parametric_paf/random_paf/convergence.json'
OUT = ROOT / 'parametric_paf/output/paf_convergence_v1'
OLD = ROOT / 'parametric_paf/output/paf_outer_d_training_v2'


def configure(mode):
    """元実験を保存したまま、独立した出力先に既存の学習器を適用する。"""
    base.OUTPUT = OUT / mode
    base.SPEC = base.OUTPUT / 'spec.json'
    base.FROZEN = base.OUTPUT / 'freeze.json'


def prepare(mode):
    configure(mode)
    spec = read(SPEC)
    spec['experiment_id'] += '_' + mode
    write(base.SPEC, spec)
    if base.FROZEN.exists():
        base.verify_freeze()
        return
    base.prepare()
    frozen = read(base.FROZEN)
    for path in (Path(__file__), SPEC):
        frozen['sha256'][path.relative_to(ROOT).as_posix()] = digest(path)
    write(base.FROZEN, frozen)
    original = read(OLD / 'dataset/manifest.json')
    current = read(base.OUTPUT / 'dataset/manifest.json')
    # 保存先の階層が異なるため、相対パスを実際の参照先へ直して照合する。
    for manifest, folder in ((original, OLD / 'dataset'), (current, base.OUTPUT / 'dataset')):
        for sample in manifest['samples']:
            for field in ('image', 'label'):
                sample[field] = str((folder / sample[field]).resolve())
    if original != current:
        raise ValueError('元実験と学習・検証・参照評価の分割が異なります')


def train(mode, seed):
    configure(mode)
    if mode == 'region':
        import cv2
        import numpy as np
        import torch
        from paf_ring_detection import train as trainer
        from paf_ring_detection.data import RingDataset, label_ellipse, scaled_ellipse

        class RegionDataset(RingDataset):
            """同じ画像に対し、完全な穴領域だけを教師として返す。"""

            def __getitem__(self, index):
                image, _, item = super().__getitem__(index)
                label = read(self.root / self.samples[index]['label'])
                ellipse = scaled_ellipse(label_ellipse(label), self.input_size / label['image_width'])
                mask = np.zeros((self.input_size, self.input_size), np.uint8)
                cv2.ellipse(mask, ellipse, 255, -1, cv2.LINE_AA)
                return image, torch.from_numpy(mask[None]).float() / 255, item

        # この学習プロセス内だけで教師を切り替え、既存学習器は変更しない。
        trainer.RingDataset = RegionDataset
    base.train(seed)
    history = read(base.checkpoint(seed).parent / 'training_history.json')
    if mode == 'ring':
        original = read(OLD / f'models/{seed}/training_history.json')
        # CUDAの非決定的演算による差も記録し、同一seedを完全一致と扱わない。
        differences = {key: max(abs(a[key] - b[key]) for a, b in zip(history[:12], original))
                       for key in ('train_loss', 'validation_loss')}
        write(base.checkpoint(seed).parent / 'prefix_verification.json',
              {'compared_epochs': 12, 'bitwise_match': history[:12] == original,
               'max_absolute_loss_differences': differences, 'original_history_sha256':
               digest(OLD / f'models/{seed}/training_history.json')})


def region_ellipse(probability):
    """0.5以上の最大連結領域の外周に楕円を当てはめる。正解は参照しない。"""
    import cv2
    import numpy as np
    mask = (probability >= read(SPEC)['region_probability_threshold']).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = [c for c in contours if len(c) >= 5 and cv2.contourArea(c) > 0]
    if not candidates:
        return None
    contour = max(candidates, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(contour)
    if not np.isfinite(np.asarray((*ellipse[0], *ellipse[1], ellipse[2]))).all():
        return None
    return ellipse


def context(mode, seed):
    import cv2
    import torch
    from paf_ring_detection.methods.cnn import load_model
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    config = read(OUT / ('ring' if mode in ('baseline', 'canny', 'zhang') else mode) / f'configs/{seed}.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = OLD / f'models/{seed}/cnn_best.pt' if mode == 'baseline' else OUT / mode / f'models/{seed}/cnn_best.pt'
    model = None if mode in ('canny', 'zhang') else load_model(checkpoint, device)
    return model, device, config, checkpoint


def predict(sample, folder, mode, model, device, config):
    """推論には画像だけを渡し、楕円推定後に正解と比較する。"""
    import cv2
    import numpy as np
    import torch
    from paf_ring_detection.data import read_image, label_ellipse, scaled_ellipse
    from paf_ring_detection.methods.cnn_ransac import detect_from_probability
    from parametric_paf.tune_geometry import run_geometry
    from parametric_paf.evaluate_study import evaluate_prediction
    spec = read(SPEC)
    image = cv2.resize(read_image(folder / sample['image']), (256, 256), interpolation=cv2.INTER_AREA)
    rng = (spec['evaluation_seed'] + zlib.crc32(sample['sample_id'].encode())) % 2**32
    cv2.setRNGSeed(rng & 0x7fffffff)
    phash = None
    if model is None:
        result = run_geometry(image, mode, config, rng)
        ellipse = result['ellipse'] if result else None
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        with torch.inference_mode():
            probability = torch.sigmoid(model(torch.from_numpy(rgb).float().unsqueeze(0).to(device) / 127.5 - 1))[0, 0].cpu().numpy()
        phash = hashlib.sha256(probability.tobytes()).hexdigest()
        if mode == 'region':
            ellipse = region_ellipse(probability)
        else:
            result = detect_from_probability(probability, config['ransac'], rng)
            ellipse = result['ellipse'] if result else None
    label = read(folder / sample['label'])
    truth = scaled_ellipse(label_ellipse(label), 256 / label['image_width'])
    return {'sample_id': sample['sample_id'], 'method': mode, 'seed': config['seed'],
            'ransac_seed': rng, 'estimated_ellipse': ellipse, 'truth_ellipse': truth,
            'probability_sha256': phash, **evaluate_prediction(ellipse, truth, image.shape, spec)}


def selection(split):
    folder = OUT / 'fresh' if split == 'fresh' else OUT / 'ring/dataset'
    samples = read(folder / 'manifest.json')['samples']
    if split != 'fresh':
        samples = [s for s in samples if s['split'] == ('validation' if split == 'validation' else 'test')]
    return folder, samples


def evaluate(mode, seed, split):
    """固定checkpointと固定画像の全推定を再開可能な原記録に保存する。"""
    folder, samples = selection(split)
    if split == 'validation':
        samples = [samples[i * 9] for i in range(17)]
    model, device, config, checkpoint = context(mode, seed)
    if split == 'validation':
        print(json.dumps([predict(s, folder, mode, model, device, config) for s in samples]))
        return
    destination = OUT / f'evaluation/{split}/{mode}/{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {'code_sha256': digest(Path(__file__)), 'spec_sha256': digest(SPEC),
                  'manifest_sha256': digest(folder / 'manifest.json'), 'n': len(samples),
                  'checkpoint_sha256': digest(checkpoint) if model else None}
    ppath = destination / 'provenance.json'
    if ppath.exists() and read(ppath) != provenance:
        raise ValueError('評価の固定条件が変わっています')
    write(ppath, provenance)
    path = destination / 'records.jsonl'
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if [r['sample_id'] for r in rows] != [s['sample_id'] for s in samples[:len(rows)]]:
        raise ValueError('原記録と画像順が異なります')
    with path.open('a', encoding='utf-8') as stream:
        for sample in samples[len(rows):]:
            row = predict(sample, folder, mode, model, device, config)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            rows.append(row)
    score = {'mode': mode, 'seed': seed, 'n': len(rows),
             'strict_successes': sum(r['strict_success'] for r in rows),
             'iou_successes': sum(r['iou_success'] for r in rows),
             'missed': sum(r['missed'] for r in rows), 'records_sha256': digest(path)}
    write(destination / 'completion.json', {**provenance, **score, 'status': 'complete'})
    print(json.dumps(score), flush=True)


def reproduce(mode, seed):
    """独立した2プロセスで検証画像17枚の確率と推定を照合する。"""
    results = []
    for repeat in range(2):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__)),
                                'evaluate', '--mode', mode, '--seed', str(seed), '--split', 'validation'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=True)
        rows = json.loads(result.stdout)
        write(OUT / f'reproducibility/{mode}_{seed}_{repeat}.json', rows)
        results.append(rows)
    if results[0] != results[1]:
        raise ValueError('独立プロセスの推論が一致しません')
    write(OUT / f'reproducibility/{mode}_{seed}.json', {'status': 'passed', 'n': 17})


def fresh(stage):
    """学習や条件分岐に使用しない、同じ分布の新規154画像を作る。"""
    from parametric_paf import random_dataset as generator
    generator.CONFIG = OUT / 'fresh_config.json'
    generator.OUTPUT = OUT / 'fresh'
    if stage == 'render':
        generator.render()
    else:
        generator.verify()
        original = read(OLD / 'dataset/manifest.json')['samples']
        generated = read(generator.OUTPUT / 'manifest.json')['samples']
        keys = ('top_outer_diameter', 'bottom_outer_diameter', 'height', 'hole_diameter', 'wall_thickness')
        prior = {tuple(s['parameters'][k] for k in keys) for s in original}
        current = {tuple(s['parameters'][k] for k in keys) for s in generated}
        if prior & current:
            raise ValueError('新規評価画像と元データの寸法が重複しています')
        write(OUT / 'fresh/disjoint.json', {'status': 'passed', 'unique_new_dimensions': len(current),
              'original_manifest_sha256': digest(OLD / 'dataset/manifest.json')})


def scores(split, modes):
    seeds = read(SPEC)['training_seeds']
    return [read(OUT / f'evaluation/{split}/{m}/{s}/completion.json')
            for m in modes for s in (seeds[:1] if m in ('canny', 'zhang') else seeds)]


def audit(modes):
    """全原記録を元ラベルから再評価し、画像・重み・履歴・分割を照合する。"""
    import cv2
    import numpy as np
    from paf_ring_detection.data import label_ellipse, scaled_ellipse, read_image
    from parametric_paf.evaluate_study import evaluate_prediction
    total = 0
    for mode in modes:
        if mode not in ('baseline', 'canny', 'zhang'):
            configure(mode)
            base.verify_freeze()
            for seed in read(SPEC)['training_seeds']:
                base.training_evidence(seed)
                if read(OUT / f'reproducibility/{mode}_{seed}_0.json') != read(OUT / f'reproducibility/{mode}_{seed}_1.json'):
                    raise ValueError('推論再現性の原記録が不一致')
    for split in ('reference', 'fresh'):
        folder, samples = selection(split)
        for s in samples:
            for field in ('image', 'label'):
                if digest(folder / s[field]) != s[field + '_sha256']:
                    raise ValueError('評価元データhash不一致')
        active = [m for m in modes if split == 'fresh' or m in ('ring', 'region')]
        for score in scores(split, active):
            path = OUT / f'evaluation/{split}/{score["mode"]}/{score["seed"]}/records.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(rows) != len(samples) or digest(path) != score['records_sha256']:
                raise ValueError('原記録の件数・hash不一致')
            expected_path = OLD if score['mode'] == 'baseline' else OUT / score['mode']
            if score['mode'] not in ('canny', 'zhang') and score['checkpoint_sha256'] != digest(expected_path / f'models/{score["seed"]}/cnn_best.pt'):
                raise ValueError('評価checkpoint不一致')
            for s, row in zip(samples, rows):
                label = read(folder / s['label'])
                truth = scaled_ellipse(label_ellipse(label), 256 / label['image_width'])
                image = cv2.resize(read_image(folder / s['image']), (256, 256), interpolation=cv2.INTER_AREA)
                metrics = evaluate_prediction(row['estimated_ellipse'], truth, image.shape, read(SPEC))
                if s['sample_id'] != row['sample_id'] or not np.allclose(np.asarray((*truth[0], *truth[1], truth[2])),
                        np.asarray((*row['truth_ellipse'][0], *row['truth_ellipse'][1], row['truth_ellipse'][2]))):
                    raise ValueError('原記録の正解対応不一致')
                if any(row[k] != v for k, v in metrics.items()):
                    raise ValueError('元ラベルからの指標再計算不一致')
            if score['strict_successes'] != sum(r['strict_success'] for r in rows):
                raise ValueError('成功数の集計不一致')
            total += len(rows)
    write(OUT / 'report/audit.json', {'status': 'passed', 'recomputed_predictions': total,
          'modes': modes, 'code_sha256': digest(Path(__file__)), 'spec_sha256': digest(SPEC)})


def run():
    """60epochを検証し、平均が最良幾何手法を超えなければ穴領域教師も実行する。"""
    base.verify_freeze()
    spec = read(SPEC)
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)

    def step(name, args, executable=None):
        print('Start: ' + name, flush=True)
        with (logs / f'{name}.log').open('a', encoding='utf-8') as stream:
            command = executable or [sys.executable, '-u', '-X', 'utf8', '-B', str(Path(__file__))]
            subprocess.run(command + args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print('Complete: ' + name, flush=True)

    def experiment(mode):
        step('prepare_' + mode, ['prepare', '--mode', mode])
        for seed in spec['training_seeds']:
            step(f'train_{mode}_{seed}', ['train', '--mode', mode, '--seed', str(seed)])
            step(f'repro_{mode}_{seed}', ['reproduce', '--mode', mode, '--seed', str(seed)])
            step(f'reference_{mode}_{seed}', ['evaluate', '--mode', mode, '--seed', str(seed)])

    experiment('ring')
    reference = scores('reference', ['ring'])
    geometry = read(OLD / 'report/summary.json')['scores']
    strongest = max(r['strict_successes'] for r in geometry if r['method'] != 'cnn')
    mean = sum(r['strict_successes'] for r in reference) / len(reference)
    run_region = mean <= strongest
    write(OUT / 'decision.json', {'reference_mean_successes': mean, 'strongest_geometry_successes': strongest,
          'n': 154, 'region_required': run_region, 'rule': 'three_seed_mean_strictly_exceeds_strongest_geometry',
          'fresh_test_used_for_decision': False})
    modes = ['baseline', 'ring', 'canny', 'zhang']
    if run_region:
        experiment('region')
        modes.insert(2, 'region')
    config = read(ROOT / 'parametric_paf/random_paf/config.json')
    config.update(dataset_id='paf_convergence_fresh', seed=spec['fresh_test_seed'], image_count=spec['fresh_test_count'])
    write(OUT / 'fresh_config.json', config)
    step('render_fresh', ['--', 'fresh-render'],
         ['C:/Program Files/Blender Foundation/Blender 5.2/blender.exe', '--background', '--python-exit-code', '1', '--python', str(Path(__file__))])
    step('verify_fresh', ['fresh-verify'])
    for mode in modes:
        for seed in (spec['training_seeds'][:1] if mode in ('canny', 'zhang') else spec['training_seeds']):
            step(f'fresh_{mode}_{seed}', ['evaluate', '--mode', mode, '--seed', str(seed), '--split', 'fresh'])
    audit(modes)
    write(OUT / 'report/summary.json', {'status': 'complete', 'decision': read(OUT / 'decision.json'),
          'reference': scores('reference', [m for m in modes if m in ('ring', 'region')]),
          'fresh': scores('fresh', modes), 'audit': read(OUT / 'report/audit.json')})
    print('All training and evaluations complete.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'reproduce', 'fresh-render', 'fresh-verify', 'run'])
    parser.add_argument('--mode', choices=['ring', 'region', 'baseline', 'canny', 'zhang'], default='ring')
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--split', choices=['reference', 'validation', 'fresh'], default='reference')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    if args.stage == 'run':
        run()
    elif args.stage.startswith('fresh-'):
        fresh(args.stage.split('-')[1])
    elif args.stage == 'prepare':
        prepare(args.mode)
    elif args.stage == 'train':
        train(args.mode, args.seed)
    elif args.stage == 'reproduce':
        reproduce(args.mode, args.seed)
    else:
        evaluate(args.mode, args.seed, args.split)


if __name__ == '__main__':
    main()
