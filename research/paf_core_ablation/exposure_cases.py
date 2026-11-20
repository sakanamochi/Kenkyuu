"""強い黒つぶれで専用学習と多様形状学習に差が出る画像を固定規則で表示する。"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json, load_samples
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.diagnose import sha, image_and_truth
from research.paf_core_ablation.boundary_report import OUT, BASE
from research.paf_core_ablation.failure_cases import draw_ellipse


def main():
    folders = [ROOT/'output/paf_core_ablation_v1/runs/main_1194m/20260907',
               ROOT/'output/paf_generalization_v2/runs/boundary_projection_relative/20260907']
    saved = [read_json(p/'test_1194m.json') for p in folders]
    lookups = [{r['sample_id']: r for r in result['rows']} for result in saved]
    selected = [next(r for r in saved[0]['rows'] if r['degradation'] == 'sensor_black_crush'
                     and r['severity'] == severity and r['success']
                     and not lookups[1][r['sample_id']]['success']) for severity in (.75, 1.)]
    models = []
    for folder, result in zip(folders, saved):
        assert sha(folder/'best.pt') == result['checkpoint_sha256']
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        assert checkpoint['provenance'] == result['provenance']
        model = TinyUNet(16).cpu().eval()
        model.load_state_dict(checkpoint['model_state'])
        models.append(model)
    samples = {s['sample_id']: s for s in load_samples(BASE/'1194m', 'test')}
    font = Path('C:/Windows/Fonts/YuGothR.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 12, 'svg.fonttype': 'path'})
    figure, axes = plt.subplots(2, 3, figsize=(12, 8.7))
    evidence = []
    for index, reference in enumerate(selected):
        sample = samples[reference['sample_id']]
        for field in ('image', 'label'):
            assert sha(BASE/'1194m'/sample[field]) == sample[field+'_sha256']
        image, truth = image_and_truth(BASE/'1194m', sample)
        target = reference['truth_ellipse']
        assert np.allclose([*truth[0], *truth[1], truth[2]], [*target[0], *target[1], target[2]])
        # 原入力に線を重ねず、低輝度の内周情報がどこに残るかを確認できるようにする。
        axes[index, 0].imshow(image[..., 0], cmap='gray', vmin=0, vmax=255)
        axes[index, 0].set_ylabel(f"黒つぶれ {reference['severity']}", fontsize=13)
        axes[index, 0].set_xlabel(f"最大画素値 {image.max()} / 非零画素 {np.mean(image[..., 0] > 0)*100:.1f}%")
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None]/127.5-1
        predictions = []
        for column, (model, lookup) in enumerate(zip(models, lookups), 1):
            with torch.inference_mode():
                probability = torch.sigmoid(model(x))[0, 0].numpy()
            row = lookup[sample['sample_id']]
            axes[index, column].imshow(probability, cmap='gray', vmin=0, vmax=1)
            draw_ellipse(axes[index, column], truth, '#53df8c')
            draw_ellipse(axes[index, column], row['ellipse'], '#ff67ce')
            axes[index, column].set_xlabel(f"保存済み推定のIoU {row['ellipse_iou']:.3f}")
            predictions.append({'variant': saved[column-1]['variant'], 'ellipse': row['ellipse'],
                                'ellipse_iou': row['ellipse_iou']})
        for ax in axes[index]:
            ax.set_xticks([])
            ax.set_yticks([])
        evidence.append({'sample_id': sample['sample_id'], 'severity': reference['severity'],
                         'truth_ellipse': truth, 'predictions': predictions,
                         'image_sha256': sample['image_sha256'], 'label_sha256': sample['label_sha256'],
                         'max_pixel_value': int(image.max()), 'nonzero_pixel_fraction': float(np.mean(image[..., 0] > 0))})
    for ax, title in zip(axes[0], ('入力（明るさ変更なし）', 'main参考：1194M学習', '投影＋外観＋開口幅遮蔽')):
        ax.set_title(title, pad=12, fontsize=13)
    figure.suptitle('強い黒つぶれで残る専用学習との差', fontsize=18, y=.98)
    figure.text(.5, .065, '緑：正解内周　桃：保存済み推定楕円　白：CNNの高い出力確率', ha='center')
    figure.text(.5, .038, '各強度で「専用学習が成功・多様形状学習が失敗」する固定test順の先頭、seed07。', ha='center', fontsize=10)
    figure.text(.5, .016, '教師・学習率が異なるため単一要因の対照ではない。非零画素率は内周の情報量を表す指標ではない。', ha='center', fontsize=10)
    figure.subplots_adjust(left=.06, right=.98, top=.88, bottom=.14, wspace=.15, hspace=.28)
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    paths = [folder/f'strong_black_crush_cases.{suffix}' for suffix in ('png', 'svg')]
    for path in paths:
        figure.savefig(path, dpi=150, facecolor='white')
    plt.close(figure)
    write_json(folder/'strong_black_crush_cases.json', {'seed': 20260907, 'cases': evidence,
        'selection': '強度0.75/1.0それぞれ専用学習成功・多様形状学習失敗の固定順先頭。全体頻度の推定には用いない。',
        'scope': '入力・表示の露光補正なし。図のCNNはCPU、楕円は保存済み。教師・学習率の異なる参照比較。',
        'implementation_sha256': sha(Path(__file__)),
        'inputs': {str((p/name).relative_to(ROOT)): sha(p/name) for p in folders for name in ('best.pt', 'test_1194m.json')},
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in paths}})
    print('strong exposure comparison saved', flush=True)


if __name__ == '__main__':
    main()
