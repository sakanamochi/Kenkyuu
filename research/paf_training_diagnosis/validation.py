"""選択に使わない追加投影validationで、拡張分布への適合を事後確認する。"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paf_training_diagnosis import run as study
from research.paf_generalization_v2.projection import PackedBoundaryDataset
from research.paf_generalization.boundary import boundary_objective
from paf_ring_detection.data import read_json, write_json, load_samples, label_ellipse, scaled_ellipse
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.diagnose import sha


def distribution():
    """画像上の実際の楕円ラベルで、元/追加投影と評価対象の範囲を測る。"""
    specs = [('1194m_train', study.BASE/'1194m', 'train'),
             ('1194m_projection_train', ROOT/'output/paf_fair_comparison_v1/augmentation_1194m/projection_plain', 'train'),
             ('diversified_train', study.BASE/'diversified', 'train'),
             ('diversified_projection_train', ROOT/'output/paf_generalization_v2/projection_coverage/native_plain', 'train'),
             ('1194m_test', study.BASE/'1194m', 'test'), ('diversified_test', study.BASE/'diversified', 'test')]
    result, files = [], {}
    for name, root, split in specs:
        items = []
        files[str((root/'manifest.json').relative_to(ROOT))] = sha(root/'manifest.json')
        for s in load_samples(root, split):
            if s['conditions']['degradation'] != 'clean':
                continue
            path = root/s['label']
            assert sha(path) == s['label_sha256']
            label = read_json(path)
            center, axes, _ = scaled_ellipse(label_ellipse(label), 256/label['image_width'])
            items.append({'sample_id': s['sample_id'], 'major': max(axes), 'minor': min(axes),
                          'axis_ratio': min(axes)/max(axes), 'cx': center[0], 'cy': center[1]})
        stats = {k: dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                 np.quantile([r[k] for r in items], [0, .25, .5, .75, 1]).tolist()))
                 for k in ('major', 'minor', 'axis_ratio', 'cx', 'cy')}
        result.append({'name': name, 'n': len(items), 'statistics': stats, 'rows': items})
    write_json(study.OUT/'projection_distribution.json', {'datasets': result, 'files': files,
        'implementation_sha256': sha(Path(__file__)), 'scope': 'GTは事後統計のみ。モデル入力・推論には使わない。'})


def validation():
    """同じ未学習形状の元/投影/投影外観でlossを測り、checkpoint選択は変えない。"""
    torch.set_num_threads(4)
    config = read_json(study.OUT/'configs/source_projection.json')
    roots = {'original': study.BASE/'diversified',
             'projection_plain': ROOT/'output/paf_generalization_v2/projection_coverage/native_plain',
             'projection_surface': ROOT/'output/paf_generalization_v2/projection_coverage/native_surface'}
    datasets = {k: PackedBoundaryDataset(p, 'validation', 256, True) for k, p in roots.items()}
    original_ids = [s['sample_id'] for s in datasets['original'].samples]
    assert all([s['sample_id'] for s in data.samples] == original_ids for data in datasets.values())
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.], device='cuda'))
    rows, files = [], {}
    for name in ('source_base', 'source_projection', 'source_both'):
        for budget in (60, 120):
            path = study.OUT/f'runs/{name}/{study.SEED}/budget_{budget}/best.pt'
            model = TinyUNet(16).cuda().eval()
            saved = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model_state'])
            files[str(path.relative_to(ROOT))] = sha(path)
            for view, dataset in datasets.items():
                loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
                total = 0.
                with torch.inference_mode():
                    for images, masks, _ in loader:
                        loss = boundary_objective(model(images.cuda()), masks.cuda(), {}, config, bce)
                        total += float(loss)*len(images)
                mean = total/len(dataset)
                if view == 'original':
                    assert abs(mean-saved['validation_loss']) < 1e-6
                rows.append({'variant': name, 'budget': budget, 'view': view,
                             'n': len(dataset), 'selected_epoch': saved['epoch'], 'loss': mean})
            print('VALIDATION', name, budget, flush=True)
    for root in roots.values():
        files[str((root/'manifest.json').relative_to(ROOT))] = sha(root/'manifest.json')
        for sample in load_samples(root, 'validation'):
            for field in ('image', 'label'):
                path = root/sample[field]
                assert sha(path) == sample[field+'_sha256']
                files[str(path.relative_to(ROOT))] = sha(path)
    write_json(study.OUT/'additional_validation.json', {'rows': rows, 'files': files,
        'implementation_sha256': sha(Path(__file__)),
        'scope': '追加投影validationは事後診断のみ。重み・方式の選択には使用しない。'})


def numerical_repeat():
    """同一初期状態・同一batchの短い2反復で、既存GPU設定の数値差を測る。"""
    torch.set_num_threads(4)
    torch.manual_seed(study.SEED)
    study.projection.NEW_VIEW = 'projection_coverage'
    study.boundary_core.RETAIN = True, False
    data = study.boundary_core.FactorDataset(study.BASE/'diversified', 'train', 256, True)
    samples = [data[i] for i in range(16)]
    images = torch.stack([r[0] for r in samples]).cuda()
    masks = torch.stack([r[1] for r in samples]).cuda()
    torch.manual_seed(study.SEED)
    model = TinyUNet(16).cuda()
    initial = copy.deepcopy(model.state_dict())
    parameters = set(dict(model.named_parameters()))
    states, losses = [], []
    for repeat in range(2):
        model = TinyUNet(16).cuda().train()
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.], device='cuda'))
        track = []
        for step in range(12):
            optimizer.zero_grad()
            loss = boundary_objective(model(images), masks, {}, {}, bce)
            loss.backward()
            optimizer.step()
            track.append(float(loss.detach()))
        states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        losses.append(track)
    difference = {k: float((states[0][k]-states[1][k]).abs().max()) for k in states[0]}
    write_json(study.OUT/'numerical_repeat_check.json', {
        'seed': study.SEED, 'same_initial_weights_and_fixed_batch': True, 'steps': 12,
        'sample_ids': [s['sample_id'] for s in data.samples[:16]], 'losses': losses,
        'max_loss_abs_difference': max(abs(a-b) for a, b in zip(*losses)),
        'max_parameter_abs_difference': max(difference[k] for k in parameters),
        'max_state_abs_difference': max(difference.values()),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'implementation_sha256': sha(Path(__file__)),
        'scope': '同一重み・入力・optimizerの短い反復診断。主比較の学習設定は変更しない。'})


if __name__ == '__main__':
    if '--numerical-repeat-only' in sys.argv:
        numerical_repeat()
    else:
        distribution()
        if '--distribution-only' not in sys.argv:
            validation()
