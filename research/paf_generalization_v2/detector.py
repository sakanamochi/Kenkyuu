"""投影・遮蔽を学習した単一境界CNNで、画像ファイルからPAF内周を推定する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.boundary_detector import BoundaryPAFDetector
from research.paf_generalization.detector import predict_file
from research.paf_generalization.diagnose import sha

OUT = ROOT/'output/paf_generalization_v2'
VARIANTS = ('boundary_projection_relative', 'boundary_projection_analytic')
SEEDS = (20260907, 20260908, 20260909)


class ProjectionPAFDetector(BoundaryPAFDetector):
    """元の前処理とRANSACを共有し、指定した1モデルだけを読み込む。"""

    def __init__(self, variant='boundary_projection_relative', seed=20260907):
        if variant not in VARIANTS or seed not in SEEDS:
            raise ValueError('対応する投影境界モデルと固定3seedの一つを指定してください。')
        folder = OUT/f'runs/{variant}/{seed}'
        path = folder/'best.pt'
        checkpoint_hash = sha(path)
        assert read_json(folder/'complete.json')['checkpoint_sha256'] == checkpoint_hash
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        metadata = read_json(folder/'provenance.json')
        assert checkpoint['provenance'] == metadata
        assert metadata['variant'] == variant and metadata['seed'] == seed
        assert metadata['training_source'] == 'diversified' and metadata['target_data_used_for_training'] is False
        for relative, digest in metadata['files'].items():
            assert sha(ROOT/relative) == digest
        config = metadata['config']
        assert config['boundary_implementation_sha256'] == sha(ROOT/'research/paf_generalization/boundary.py')
        assert config['ransac_implementation_sha256'] == sha(ROOT/'paf_ring_detection/methods/ransac.py')
        self.settings = config['ransac']
        self.model = TinyUNet(16).cuda().eval()
        self.model.load_state_dict(checkpoint['model_state'], strict=True)
        self.provenance = {'variant': variant, 'seed': seed, 'checkpoint_sha256': checkpoint_hash,
            'selected_epoch': checkpoint['epoch'], 'settings': self.settings,
            'implementation_sha256': sha(Path(__file__)),
            'shared_predict_sha256': sha(ROOT/'research/paf_generalization/boundary_detector.py'),
            'boundary_implementation_sha256': config['boundary_implementation_sha256'],
            'ransac_implementation_sha256': config['ransac_implementation_sha256'],
            'reject_uniform_images': True, 'input': 'square_grayscale_256',
            'opencv_seed_per_image': 20261025, 'ensemble': False}

    def predict(self, image):
        """正方形uint8画像を受け、追加確認と同じOpenCV乱数で推定する。"""
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError('入力は正方形のuint8グレー画像またはBGR画像にしてください。')
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError('カラー入力は3チャンネルBGR画像にしてください。')
        if min(image.shape[:2]) == 0:
            raise ValueError('空の画像は推論できません。')
        cv2.setRNGSeed(self.provenance['opencv_seed_per_image'])
        return super().predict(image)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default=VARIANTS[0])
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    predict_file(ProjectionPAFDetector(args.variant, args.seed), args.image, args.output)
