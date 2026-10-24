"""穴領域CNN二系統と幾何二方式の最小実行入口。"""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

from paf_ring_detection.data import read_json
from paf_ring_detection.evaluate import METHODS, checkpoint_path, evaluate
from paf_ring_detection.prepare import digest, prepare_training_data, verify_prepared
from paf_ring_detection.report import build_report
from paf_ring_detection.train import train

ROOT = Path(__file__).resolve().parent


def render(config, sources):
    """1194Mと指定範囲の多様形状を既存の生成器で描画する。"""
    for source in sources:
        if source == '1194m':
            script, arguments = 'blender/generate_dataset.py', ['--config', 'config/render_base.json']
        else:
            script, arguments = 'parametric_paf/random_dataset.py', ['render']
        subprocess.run([config['blender'], '--background', '--python-exit-code', '1',
                        '--python', str(ROOT / script), '--', *arguments], cwd=ROOT, check=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['render', 'prepare', 'train', 'evaluate', 'report'])
    parser.add_argument('--config', default='config/experiment.json')
    parser.add_argument('--source', nargs='+', choices=['1194m', 'diversified'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--limit', type=int, help='evaluate/reportの動作確認用画像数。正式結果と分離する')
    args = parser.parse_args()
    config = read_json(Path(args.config))
    sources = args.source or list(config['sources'])
    seeds = args.seeds or config['training_seeds']
    if args.limit is not None and (args.limit < 1 or args.stage not in ('evaluate', 'report')):
        parser.error('--limitはevaluate/reportで正の値を指定してください')
    if not set(seeds).issubset(config['training_seeds']):
        parser.error('--seedsは設定ファイルのtraining_seedsから指定してください')
    if args.source and args.stage not in ('render', 'train'):
        parser.error('--sourceはrender/trainで指定してください。評価は両test集合に共通です')
    if args.methods and args.stage != 'evaluate':
        parser.error('--methodsはevaluateで指定してください')
    if args.stage == 'render':
        render(config, sources)
    elif args.stage == 'prepare':
        prepare_training_data(config)
    elif args.stage == 'train':
        verify_prepared(config)
        # 指定した学習元・seedの重みと履歴を同じ保存先で更新する。
        for source in sources:
            for seed in seeds:
                checkpoint = checkpoint_path(config, source, seed)
                checkpoint.unlink(missing_ok=True)
                (checkpoint.parent / 'training_history.json').unlink(missing_ok=True)
        for source in sources:
            for seed in seeds:
                current = copy.deepcopy(config)
                folder = Path(config['output']) / 'datasets' / source
                current.update(seed=seed, training_source=source, dataset_manifest_sha256=digest(folder / 'manifest.json'))
                current['paths'] = {'training_dataset': str(folder), 'checkpoint': str(checkpoint_path(config, source, seed))}
                train(current)
    elif args.stage == 'evaluate':
        verify_prepared(config)
        evaluate(config, args.methods, seeds, args.limit)
    else:
        build_report(config, args.limit)


if __name__ == '__main__':
    main()
