"""前段の固定バックアップを参照し、寄与比較の完了成果物を保全する。"""

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

OUT = ROOT/'output/paf_core_ablation_v1'
PREVIOUS = ROOT/'output/paf_generalization_v2/backup/verified_final_detector_20261025/inventory.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--study', choices=['main', 'boundary'], default='main')
    args = parser.parse_args()
    name = args.name
    output = OUT if args.study == 'main' else ROOT/'output/paf_core_ablation_v2'
    previous_path = PREVIOUS
    if args.study == 'boundary':
        pointer = read_json(OUT/'backup/latest.json')
        previous_path = ROOT/pointer['path']
        assert sha(previous_path) == pointer['sha256']
    assert re.fullmatch('[a-z0-9][a-z0-9_-]*', name)
    backup = (output/'backup').resolve()
    destination = (backup/name).resolve()
    assert destination.parent == backup and not destination.exists()
    previous = read_json(previous_path)
    if args.study == 'main':
        assert previous['excluded_incomplete_runs'] == []
    incomplete = [p for p in (output/'runs').glob('*/*') if p.is_dir() and not all(
        (p/f).exists() for f in ('complete.json', 'test_1194m.json', 'test_diversified.json'))]
    paths = list((ROOT/'research/paf_core_ablation').rglob('*'))
    paths += [ROOT/'.codex/RESEARCH_HANDOFF.md']
    paths += [p for p in output.rglob('*') if backup not in p.resolve().parents
              and not any(folder in p.parents for folder in incomplete)]
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
                 'files': files, 'excluded_incomplete_runs': [str(p.relative_to(ROOT)) for p in incomplete],
                 'restore': '前段v2固定セットの依存順に復元し、このセットのpathをプロジェクト直下へ対応させる。原画像・venv・Blenderは別途必要。'}
    write_json(destination/'inventory.json', inventory)
    write_json(backup/'latest.json', {'path': str((destination/'inventory.json').relative_to(ROOT)),
                                    'sha256': sha(destination/'inventory.json'), 'verified_files': len(files)})
    (backup/'INDEX.md').write_text(
        '# PAF寄与比較のバックアップ\n\n'
        f'最新セット：[{name}]({(destination/"inventory.json").as_posix()})。全{len(files)}ファイルのhashを照合済み。\n\n'
        f'前段固定セット：[{previous_path.parent.name}]({previous_path.as_posix()})。その依存順を復元後、最新セットを重ねる。\n\n'
        '未完了runはinventoryの除外一覧に記載する。原画像は固定計画のpathとhashで参照し、venv・Blenderは含めない。\n', encoding='utf-8')
    print({'snapshot': name, 'verified_files': len(files), 'excluded_incomplete_runs': len(incomplete)}, flush=True)


if __name__ == '__main__':
    main()
