"""通常画像の小差を、固定CNNの帯点群と中心線の対応診断で調べる。"""

import sys
import zlib
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
from paf_ring_detection.geometry import evaluate_ellipses
from paf_ring_detection.methods.cnn import TinyUNet
from research.paf_generalization.diagnose import image_and_truth, sha
from research.paf_core_ablation.point_representation import predict
from research.paf_core_ablation.failure_cases import draw_ellipse

OUT = ROOT/'output/paf_core_ablation_v2/clean_gap_diagnosis'
MODEL = ROOT/'output/paf_core_ablation_v1/runs/main_1194m/20260907'
DATA = ROOT/'output/hole_region_comparison_v1/datasets/1194m'


def main():
    torch.set_num_threads(4)
    config_path = ROOT/'research/paf_core_ablation/experiment.json'
    settings = read_json(config_path)['cnn_ransac']
    saved = read_json(MODEL/'test_1194m.json')
    assert saved['checkpoint_sha256'] == sha(MODEL/'best.pt')
    checkpoint = torch.load(MODEL/'best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['provenance'] == saved['provenance']
    model = TinyUNet(16).cuda().eval()
    model.load_state_dict(checkpoint['model_state'])
    main_rows = {r['sample_id']: r for r in saved['rows'] if r['degradation'] == 'clean'}
    reference_paths = [ROOT/f'output/paf_generalization_v2/runs/boundary_projection_relative/{seed}/test_1194m.json'
                       for seed in (20260907, 20260908, 20260909)]
    references = [{r['sample_id']: r for r in read_json(p)['rows'] if r['degradation'] == 'clean'}
                  for p in reference_paths]
    paths = [Path(__file__), config_path, MODEL/'best.pt', MODEL/'test_1194m.json',
             ROOT/'research/paf_core_ablation/point_representation.py',
             ROOT/'research/paf_core_ablation/main_compat.py',
             ROOT/'research/paf_generalization/boundary.py',
             ROOT/'paf_ring_detection/methods/ransac.py']+reference_paths
    rows, examples = [], []
    for sample in load_samples(DATA, 'test'):
        if sample['conditions']['degradation'] != 'clean':
            continue
        for field in ('image', 'label'):
            assert sha(DATA/sample[field]) == sample[field+'_sha256']
        image, truth = image_and_truth(DATA, sample)
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().cuda()[None]/127.5-1
        with torch.inference_mode():
            probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        seed = (settings['random_seed']+zlib.crc32(sample['sample_id'].encode('utf-8'))) % 2**32
        row = {'sample_id': sample['sample_id'], 'truth_ellipse': truth,
               'saved_main': main_rows[sample['sample_id']],
               'diversified': [r[sample['sample_id']] for r in references], 'diagnosis': {}}
        for mode in ('band_45', 'centerline_45'):
            # CNN・閾値0.35・最小45点・乱数seedを固定し、渡す点集合だけを変える。
            ellipse = predict(probability, settings, seed, mode)
            iou = evaluate_ellipses(ellipse, truth, (256, 256))['ellipse_iou'] if ellipse else 0.
            row['diagnosis'][mode] = {'ellipse': ellipse, 'ellipse_iou': iou, 'success': iou >= .8}
        rows.append(row)
        if not row['saved_main']['success']:
            examples.append((image, probability, row))
    assert len(rows) == 112
    replay_errors = [abs(r['diagnosis']['band_45']['ellipse_iou']-r['saved_main']['ellipse_iou']) for r in rows]
    assert max(replay_errors) < 1e-10, '元推定の再現を確認してから中心線との差を解釈する'
    summary = {}
    for mode in ('band_45', 'centerline_45'):
        values = [r['diagnosis'][mode] for r in rows]
        summary[mode] = {'n': len(rows), 'successes': sum(r['success'] for r in values),
                         'mean_iou': float(np.mean([r['ellipse_iou'] for r in values]))}
    font_manager.fontManager.addfont('C:/Windows/Fonts/YuGothR.ttc')
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname='C:/Windows/Fonts/YuGothR.ttc').get_name(),
                         'font.size': 12, 'svg.fonttype': 'path'})
    figure, axes = plt.subplots(len(examples), 3, figsize=(12.5, 7.8), squeeze=False)
    for index, (image, probability, row) in enumerate(examples):
        truth = row['truth_ellipse']
        for col in range(3):
            ax = axes[index, col]
            ax.imshow(image[..., 0] if col == 0 else probability, cmap='gray',
                      vmin=0, vmax=255 if col == 0 else 1, interpolation='nearest')
            draw_ellipse(ax, truth, '#53df8c')
            if col:
                result = row['diagnosis']['band_45' if col == 1 else 'centerline_45']
                draw_ellipse(ax, result['ellipse'], '#ff67ce')
                ax.set_xlabel(f"IoU {result['ellipse_iou']:.4f}")
            # 全例で同じ範囲を拡大表示し、扁平な内周の短径方向の誤差を見せる。
            ax.set_xlim(48, 208)
            ax.set_ylim(116, 55)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[index, 0].set_ylabel(f'通常失敗例 {index+1}')
    for ax, label in zip(axes[0], ('入力と正解内周', '①のCNN確率＋帯の推定', '同じCNN確率＋中心線の推定')):
        ax.set_title(label, pad=12)
    figure.suptitle('①の通常画像の失敗3枚：CNNを固定した点群処理の診断', fontsize=17, y=.98)
    figure.text(.5, .05, '緑：正解　桃：推定　確率画像は同一。閾値・最小点数・CNN重みは変更なし。', ha='center', fontsize=11)
    figure.text(.5, .02, '失敗した全3枚を表示。通常112枚で診断し、元の評価結果・運用構成は上書きしない。', ha='center', fontsize=10)
    figure.subplots_adjust(left=.06, right=.985, top=.88, bottom=.12, hspace=.38, wspace=.15)
    OUT.mkdir(exist_ok=True)
    artifacts = []
    for suffix in ('png', 'svg'):
        path = OUT/f'normal_failure_cases.{suffix}'
        figure.savefig(path, dpi=160, facecolor='white')
        artifacts.append(path)
    plt.close(figure)
    write_json(OUT/'results.json', {'summary': summary, 'rows': rows,
        'max_original_replay_iou_error': max(replay_errors),
        'files': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'artifacts': {str(p.relative_to(ROOT)): sha(p) for p in artifacts},
        'scope': '①の固定CNN・通常112例の事後診断。帯/中心線以外は共通。②との全差や学習元の因果効果を分離する実験ではない。',
        'deployment_changed': False})
    print(summary, 'original replay maximum error', max(replay_errors), flush=True)


if __name__ == '__main__':
    main()
