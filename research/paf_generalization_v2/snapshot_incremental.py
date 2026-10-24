"""固定した完全バックアップに対する差分を保存し、全ファイルの復元元を明示する。"""

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
BASE_NAME = 'normal_and_occlusion_20261025'


def snapshot(name):
    """復元は固定した完全版＋この差分の2段階。差分同士を連鎖させない。"""
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name):
        raise ValueError('名前には英小文字・数字・ハイフン・下線を使用してください。')
    backup = (OUT/'backup').resolve()
    destination = (backup/name).resolve()
    if destination.parent != backup or destination.exists():
        raise ValueError('backup直下の新しい名前を指定してください。')
    base = backup/BASE_NAME
    base_inventory = base/'inventory.json'
    original = read_json(base_inventory)
    assert 'files' in original and 'required_previous_snapshot' in original
    lookup = {r['path']: r for r in original['files']}
    incomplete = [p for p in (OUT/'runs').glob('*/*') if p.is_dir()
                  and not all((p/f).exists() for f in ('complete.json', 'test_1194m.json', 'test_diversified.json'))]
    paths = list((ROOT/'research/paf_generalization_v2').rglob('*'))
    paths += [ROOT/'.codex/RESEARCH_HANDOFF.md', ROOT/'.codex/PROJECT_CONTEXT.md']
    paths += [p for p in OUT.rglob('*') if backup not in p.resolve().parents
              and not any(folder in p.parents for folder in incomplete)]
    sources = sorted({p.resolve() for p in paths if p.is_file() and '__pycache__' not in p.parts
                      and p.suffix not in ('.pyc', '.log')})
    inventory, copied = [], []
    for source in sources:
        relative = source.relative_to(ROOT).as_posix()
        digest = sha(source)
        same = relative in lookup and lookup[relative]['sha256'] == digest
        if same:
            restored = base/relative
            assert sha(restored) == digest, relative
            storage = 'base'
        else:
            restored = destination/relative
            restored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, restored)
            assert sha(restored) == sha(source) == digest, relative
            storage = 'delta'
            copied.append(relative)
        inventory.append({'path': relative, 'sha256': digest, 'bytes': source.stat().st_size, 'storage': storage})
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'inventory.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'type': 'delta_against_fixed_full_snapshot',
        'required_base_snapshot': {'path': str(base_inventory.relative_to(ROOT)), 'sha256': sha(base_inventory)},
        'files': inventory, 'copied_paths': copied,
        'excluded_incomplete_runs': [str(p.relative_to(ROOT)) for p in incomplete],
        'absent_from_current': sorted(set(lookup)-{r['path'] for r in inventory}),
        'restore': 'まず完全版のinventoryに従ってv1依存と固定v2完全版を復元し、この差分のstorage=deltaを同じpathへ重ねる。各復元元のhashを確認する。absent_from_currentは自動削除せず内容を確認する。差分同士の適用は不要。',
        'environment': '.venvとBlender本体は含めない。manifestの絶対パスは元workspaceを前提とする。'})
    print({'name': name, 'n_files': len(inventory), 'copied_files': len(copied),
           'copied_bytes': sum(r['bytes'] for r in inventory if r['storage'] == 'delta'),
           'total_bytes': sum(r['bytes'] for r in inventory)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    snapshot(parser.parse_args().name)
