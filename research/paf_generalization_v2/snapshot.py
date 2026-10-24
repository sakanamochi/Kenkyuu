"""継続研究のコード・入力・完了結果を、識別可能な名前で保全する。"""

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from research.paf_generalization.diagnose import sha

OUT = ROOT/'output/paf_generalization_v2'


def snapshot(name):
    """未完了実験を除き、旧研究の保全コピーへの参照と一緒に保存する。"""
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name):
        raise ValueError('名前には英小文字・数字・ハイフン・下線を使用してください。')
    backup = (OUT/'backup').resolve()
    destination = (backup/name).resolve()
    if destination.parent != backup or destination.exists():
        raise ValueError('backup直下の新しい名前を指定してください。')
    previous = ROOT/'output/paf_generalization_v1/backup/verified_candidates_20261025/inventory.json'
    read_json(previous)
    files = list((ROOT/'research/paf_generalization_v2').rglob('*'))
    files += [ROOT/'.codex/RESEARCH_HANDOFF.md', ROOT/'.codex/PROJECT_CONTEXT.md']
    incomplete = [p for p in (OUT/'runs').glob('*/*') if p.is_dir()
                  and not all((p/f).exists() for f in ('complete.json', 'test_1194m.json', 'test_diversified.json'))]
    files += [p for p in OUT.rglob('*') if backup not in p.resolve().parents
              and not any(folder in p.parents for folder in incomplete)]
    sources = sorted({p.resolve() for p in files if p.is_file() and '__pycache__' not in p.parts
                      and p.suffix not in ('.log', '.pyc')})
    inventory = []
    for source in sources:
        relative = source.relative_to(ROOT)
        target = destination/relative
        digest = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == sha(source) == digest, relative
        inventory.append({'path': relative.as_posix(), 'sha256': digest, 'bytes': source.stat().st_size})
    write_json(destination/'inventory.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
               'files': inventory, 'excluded_incomplete_runs': [str(p.relative_to(ROOT)) for p in incomplete],
               'required_previous_snapshot': {'path': str(previous.relative_to(ROOT)), 'sha256': sha(previous)},
               'restore': '先に参照先v1 snapshotの依存コード・画像を復元し、本一覧のpathをプロジェクト直下へ復元する。既存ファイルとの衝突は内容を確認する。',
               'environment': '.venvとBlender本体は含めない。manifestの絶対パスは元workspaceを前提とする。'})
    print({'name': name, 'n_files': len(inventory), 'bytes': sum(r['bytes'] for r in inventory)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    snapshot(parser.parse_args().name)
