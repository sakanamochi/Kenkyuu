"""公平比較の完成結果を前段の確定バックアップと結び付けて保全する。"""

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from research.paf_fair_comparison.experiment import OUT, CONFIG, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    assert re.fullmatch('[a-z0-9][a-z0-9_-]*', args.name)
    previous_pointer = ROOT/'output/paf_core_ablation_v2/backup/latest.json'
    previous = read_json(previous_pointer)
    previous_path = ROOT/previous['path']
    assert sha(previous_path) == previous['sha256']
    audit = read_json(OUT/'audit.json')
    assert audit['incomplete_runs'] == [] and len(audit['models']) == 6
    assert audit['plan_sha256'] == sha(OUT/'plan.json')
    backup = (OUT/'backup').resolve()
    destination = backup/args.name
    assert destination.parent == backup and not destination.exists()
    paths = list((ROOT/'research/paf_fair_comparison').rglob('*'))
    paths += [ROOT/'.codex/RESEARCH_HANDOFF.md', ROOT/'.codex/PROJECT_CONTEXT.md', ROOT/'README.md']
    paths += [p for p in OUT.rglob('*') if backup not in p.resolve().parents]
    # 前段に含まれない1194M拡張画像と今回の学習重みも全て同じセットに保存する。
    files = []
    for path in sorted(set(p.resolve() for p in paths if p.is_file() and '__pycache__' not in p.parts
                           and p.suffix not in ('.pyc', '.log'))):
        relative = path.relative_to(ROOT)
        copied = destination/relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(path)
        shutil.copy2(path, copied)
        assert sha(copied) == sha(path) == digest
        files.append({'path': relative.as_posix(), 'sha256': digest, 'bytes': path.stat().st_size})
    inventory = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'required_previous_snapshot': {'path': str(previous_path.relative_to(ROOT)), 'sha256': sha(previous_path)},
        'files': files, 'incomplete_runs': [],
        'restore': '前段snapshotの依存を復元し、本セットのpathをプロジェクト直下へ重ねる。原画像・参照重みはplan.jsonのhashと照合する。venv・Blenderは別途必要。'}
    write_json(destination/'inventory.json', inventory)
    write_json(backup/'latest.json', {'path': str((destination/'inventory.json').relative_to(ROOT)),
        'sha256': sha(destination/'inventory.json'), 'verified_files': len(files)})
    (backup/'INDEX.md').write_text('# 公平比較バックアップ\n\n'
        f'確定セット：`{args.name}`、全{len(files)}ファイルのコピーとSHA256を照合済み。\n\n'
        f'前段固定セット：`{previous_path.relative_to(ROOT)}`。依存を復元してから本セットを重ねる。\n', encoding='utf-8')
    print('SNAPSHOT', args.name, len(files), flush=True)


if __name__ == '__main__':
    main()
