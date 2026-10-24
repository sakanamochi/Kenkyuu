"""収束検証の原記録を照合し、学習曲線・比較表・同じ画像の推定例を保存する。"""

import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parametric_paf import convergence_study as study
from parametric_paf.random_dataset import read, write, digest


def report():
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw
    from paf_ring_detection.data import read_image

    summary = read(study.OUT / 'report/summary.json')
    if summary['status'] != 'complete':
        raise ValueError('全学習と評価が未完了')
    modes = summary['audit']['modes']
    spec = read(study.SPEC)
    seeds = spec['training_seeds']
    destination = study.OUT / 'report'
    study.audit(modes)
    old_summary = read(study.OLD / 'report/summary.json')
    for run in old_summary['runs']:
        folder = study.OLD / f'models/{run["seed"]}'
        if digest(folder / 'cnn_best.pt') != run['checkpoint_sha256'] or digest(folder / 'training_history.json') != run['history_sha256']:
            raise ValueError('元の12epoch重み・履歴が変わっています')
    for mode in (m for m in modes if m in ('ring', 'region')):
        for seed in seeds:
            previous = read(study.OLD / f'configs/{seed}.json')
            current = read(study.OUT / mode / f'configs/{seed}.json')
            if previous['cnn'].pop('epochs') != 12 or current['cnn'].pop('epochs') != spec['epochs']:
                raise ValueError('指定した学習回数と不一致')
            if any(previous[k] != current[k] for k in
                   ('cnn', 'ransac', 'canny_contour_ransac', 'zhang2019', 'seed', 'input_size')):
                raise ValueError('学習回数以外の既存設定が異なります')
    original = read(study.OLD / 'dataset/manifest.json')['samples']
    for sample in original:
        for field in ('image', 'label'):
            if digest(study.OLD / 'dataset' / sample[field]) != sample[field + '_sha256']:
                raise ValueError('元の学習・検証・参照画像のhash不一致')

    learned = [m for m in modes if m in ('ring', 'region')]
    fig, axes = plt.subplots(len(learned), 3, figsize=(14, 4 * len(learned)), squeeze=False)
    histories = []
    for i, mode in enumerate(learned):
        for j, seed in enumerate(seeds):
            folder = study.OUT / mode / f'models/{seed}'
            history = read(folder / 'training_history.json')
            logged = [ast.literal_eval(line) for line in
                      (study.OUT / f'logs/train_{mode}_{seed}.log').read_text(encoding='utf-8').splitlines()
                      if line.startswith("{'epoch':")]
            if history != logged:
                raise ValueError('全epochの履歴と実行ログが一致しません')
            best = min(history, key=lambda r: r['validation_loss'])
            row = {'mode': mode, 'seed': seed, 'epochs': len(history), 'best_epoch': best['epoch'],
                   'best_validation_loss': best['validation_loss'],
                   'epoch12_train_loss': history[11]['train_loss'],
                   'final_train_loss': history[-1]['train_loss'],
                   'last10_best_validation_loss': min(r['validation_loss'] for r in history[-10:]),
                   'previous10_best_validation_loss': min(r['validation_loss'] for r in history[-20:-10])}
            histories.append(row)
            ax = axes[i, j]
            ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Train')
            ax.plot([r['epoch'] for r in history], [r['validation_loss'] for r in history], alpha=.8, label='Validation')
            ax.axvline(12, color='gray', linestyle=':', label='12 epochs')
            ax.scatter([best['epoch']], [best['validation_loss']], color='red', label='Selected checkpoint')
            ax.set(title=f'{mode}, seed {seed}', xlabel='Epoch', ylabel='Loss', yscale='log')
            ax.grid(alpha=.2)
            if i == j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / 'learning_curves.png', dpi=160)
    plt.close(fig)
    write(destination / 'convergence.json', histories)

    # 各方式の先頭seedと、新規評価の先頭3画像を成否で選ばず表示する。
    folder, samples = study.selection('fresh')
    canvas = Image.new('RGB', (256 * len(modes), 292 * 3), 'white')
    rows_by_mode = {}
    for mode in modes:
        path = study.OUT / f'evaluation/fresh/{mode}/{seeds[0]}/records.jsonl'
        rows_by_mode[mode] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    for i, sample in enumerate(samples[:3]):
        for j, mode in enumerate(modes):
            row = rows_by_mode[mode][i]
            image = cv2.resize(read_image(folder / sample['image']), (256, 256), interpolation=cv2.INTER_AREA)
            for field, color in (('truth_ellipse', (0, 220, 0)), ('estimated_ellipse', (0, 0, 255))):
                if row[field] is not None:
                    center, axes, angle = row[field]
                    cv2.ellipse(image, (tuple(center), tuple(axes), angle), color, 1, cv2.LINE_AA)
            canvas.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (256*j, 292*i))
            draw = ImageDraw.Draw(canvas)
            draw.text((256*j+4, 292*i+257), f'{mode} | {sample["sample_id"]}', fill='black')
            draw.text((256*j+4, 292*i+273), f'success={row["strict_success"]}; IoU={row["ellipse_iou"]:.3f}', fill='black')
    canvas.save(destination / 'predictions.png')

    names = {'baseline': 'CNN・12epoch', 'ring': 'CNN・内周線・60epoch',
             'region': 'CNN・穴領域・60epoch', 'canny': 'Canny＋輪郭別RANSAC', 'zhang': 'Zhang 2019型（再現実装）'}
    lines = ['# PAF内周検出：学習回数と穴領域教師の検証', '',
             '同じ716枚で学習し、154枚のvalidationで最小損失のcheckpointを採用する。'
             '形状・撮影条件の範囲、CNN構造、学習率、損失設定は共通。全seedを報告する。', '',
             '内周線方式は既存の重み付きRANSAC。穴領域方式は確率0.5以上の最大連結領域の外周への楕円当てはめ。'
             '教師変更に伴う後処理変更を含む比較であり、教師だけの因果効果とはしない。', '',
             '元のtest154枚は改善比較と、穴領域方式を実行する条件分岐のための参照評価である。'
             '方式確定後、同じ分布から新規154画像を生成し、元12epochモデルを含む全方式を共通評価する。'
             '新規画像はcheckpoint選択や閾値調整には使用していない。', '',
             '主成功：IoU≥0.8かつ双方向輪郭距離p95≤2px。入力256。未検出も分母に含める。', '',
             '## 新規154画像での評価', '', '|方式|学習seed|主成功/154|主成功率|IoU成功|未検出|',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary['fresh']:
        seed = 'なし' if row['mode'] in ('canny', 'zhang') else row['seed']
        lines.append(f'|{names[row["mode"]]}|{seed}|{row["strict_successes"]}|'
                     f'{row["strict_successes"]/row["n"]*100:.2f}%|{row["iou_successes"]}|{row["missed"]}|')
    lines += ['', '## 3seed平均と参照評価', '', '|方式|新規画像の平均主成功率|元画像の平均主成功率|', '|---|---:|---:|']
    old = old_summary['scores']
    aggregates = []
    for mode in modes:
        fresh = [r for r in summary['fresh'] if r['mode'] == mode]
        if mode in ('ring', 'region'):
            reference = [r for r in summary['reference'] if r['mode'] == mode]
        else:
            reference = [r for r in old if r['method'] == ('cnn' if mode == 'baseline' else mode)]
        fmean = np.mean([r['strict_successes']/r['n']*100 for r in fresh])
        rmean = np.mean([r['strict_successes']/r['n']*100 for r in reference])
        lines.append(f'|{names[mode]}|{fmean:.2f}%|{rmean:.2f}%|')
        aggregates.append({'mode': mode, 'fresh_mean_percent': float(fmean), 'reference_mean_percent': float(rmean)})
    lines += ['', '## 学習履歴', '', '|方式|seed|採用epoch|最小validation loss|12epoch train loss|60epoch train loss|',
              '|---|---:|---:|---:|---:|---:|']
    for row in histories:
        lines.append(f'|{row["mode"]}|{row["seed"]}|{row["best_epoch"]}|{row["best_validation_loss"]:.5f}|'
                     f'{row["epoch12_train_loss"]:.5f}|{row["final_train_loss"]:.5f}|')
    lines += ['', '![学習曲線](learning_curves.png)', '',
              '学習損失の低下と検出成功率は別の指標である。教師が異なる内周線方式と穴領域方式の損失値を直接比較しない。'
              '60epochの終了だけで収束を断定しない。'
              'GPUの非決定的演算により、同じseedでも元実験との最初の12epochは完全一致しない。'
              '差は各モデルのprefix_verification.jsonに保存する。', '',
              '## 推定例', '', '新規評価の固定順の先頭3画像。CNNは先頭seed20260907。緑が正解、赤が推定。'
              '例を成否で選んでおらず、全seedの平均性能の代用にはしない。', '', '![推定例](predictions.png)', '',
              '## 検証範囲', '',
              '単一PAFモデルの指定寸法・描画分布内の合成画像評価である。実画像・実在PAF全般への汎化、'
              '単一寸法学習との因果比較、対象なし画像の検出は未検証。', '',
              'Zhang 2019型（再現実装）は著者コードの完全再現ではない。'
              '原出典はZhang, Pan, Ma, Sensors 2019, DOI 10.3390/s19235243。', '',
              '原記録・全指標の再計算・固定コードと入力・全epoch履歴・checkpoint・独立2プロセス推論を照合した。'
              'audit.json、summary.json、convergence.jsonと各evaluationフォルダを根拠とする。', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write(destination / 'aggregate.json', aggregates)
    write(destination / 'report_verification.json', {'status': 'passed', 'original_images_verified': len(original),
          'original_checkpoints_unchanged': True, 'shared_training_and_detection_settings_verified': True,
          'histories_verified': len(histories), 'recomputed_predictions': read(destination / 'audit.json')['recomputed_predictions'],
          'report_code_sha256': digest(Path(__file__)),
          'artifact_sha256': {name: digest(destination / name) for name in
                             ('REPORT.md', 'learning_curves.png', 'predictions.png', 'convergence.json', 'aggregate.json')}})
    print(json.dumps(aggregates, ensure_ascii=False))


if __name__ == '__main__':
    report()
