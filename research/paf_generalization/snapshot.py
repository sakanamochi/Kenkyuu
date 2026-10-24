"""研究コード・重み・生成画像・集計をSHA256付きの復元用コピーへ保存する。"""

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from diagnose import ROOT, OUT, sha
from paf_ring_detection.data import write_json


def snapshot(name):
    """既存コピーを上書きせず、バックアップ自身を含めない一覧を先に固定する。"""
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name):
        raise ValueError('名前には英小文字・数字・ハイフン・下線だけを使用してください。')
    backup_root = (OUT / 'backup').resolve()
    destination = (backup_root / name).resolve()
    if destination.parent != backup_root or destination.exists():
        raise ValueError('保存先はbackup直下の未使用の名前にしてください。')
    sources = list((ROOT / 'research/paf_generalization').rglob('*'))
    sources += list(OUT.rglob('*'))
    sources += [ROOT / '.codex/PROJECT_CONTEXT.md', ROOT / '.codex/RESEARCH_HANDOFF.md']
    # 元の描画・検出コードも含め、研究実装が参照する依存関係を保全する。
    sources += list((ROOT / 'paf_ring_detection').rglob('*.py'))
    sources += [p for p in (ROOT / 'parametric_paf').rglob('*.py') if 'output' not in p.parts]
    sources += list((ROOT / 'blender').rglob('*.py'))
    sources += list((ROOT / 'parametric_paf/random_paf').rglob('*.json'))
    sources += [ROOT / 'config/experiment.json', ROOT / 'requirements.txt', ROOT / 'requirements-cuda.txt']
    files = sorted({p.resolve() for p in sources if p.is_file()
                    and backup_root not in p.resolve().parents
                    and '__pycache__' not in p.parts and p.suffix not in ('.log', '.pyc')})
    inventory = []
    for source in files:
        relative = source.relative_to(ROOT)
        target = destination / relative
        digest = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != digest or sha(source) != digest:
            raise RuntimeError(f'コピー中に内容が変化しました: {relative}')
        inventory.append({'path': relative.as_posix(), 'sha256': digest,
                          'bytes': source.stat().st_size})
    write_json(destination / 'inventory.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'files': inventory,
        'restore': '各pathをプロジェクト直下からの相対パスとして復元する。既存ファイルとの衝突は内容を比較する。',
        'original_inputs': '既存原画像と固定比較はbaseline_20261025および各manifestのhash付き参照で保持する。',
        'environment': '.venvおよびBlender本体は含めない。manifestにある絶対パスは元のworkspaceを前提とする。'})
    print(f'{destination}: {len(inventory)} files, {sum(r["bytes"] for r in inventory)} bytes')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    snapshot(parser.parse_args().name)
