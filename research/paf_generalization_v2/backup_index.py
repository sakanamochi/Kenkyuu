"""最新の保全セットと復元順を明示し、既存バックアップを変更せず索引を作る。"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paf_ring_detection.data import read_json, write_json
from research.paf_generalization.diagnose import sha

BACKUP = ROOT/'output/paf_generalization_v2/backup'


def dependencies(path, seen=None):
    """inventoryの依存先を先に並べ、循環とhash不一致を拒否する。"""
    seen = set() if seen is None else seen
    path = path.resolve()
    assert path.is_relative_to(ROOT) and path not in seen
    seen.add(path)
    data = read_json(path)
    chain = []
    dependency = data.get('required_base_snapshot', data.get('required_previous_snapshot'))
    if dependency:
        previous = ROOT/dependency['path']
        assert sha(previous) == dependency['sha256']
        chain.extend(dependencies(previous, seen))
    return chain+[path]


def main():
    inventories = [(p, read_json(p)) for p in BACKUP.glob('*/inventory.json')]
    assert inventories
    inventories.sort(key=lambda item: datetime.fromisoformat(item[1]['created_utc']))
    latest, data = inventories[-1]
    chain = dependencies(latest)
    baseline = ROOT/'output/paf_generalization_v1/backup/baseline_20261025/inventory.json'
    assert baseline.exists()
    if baseline not in chain:
        chain.insert(0, baseline)
    base = (ROOT/data['required_base_snapshot']['path']).parent if 'required_base_snapshot' in data else latest.parent
    for item in data['files']:
        folder = base if item.get('storage') == 'base' else latest.parent
        source = (folder/item['path']).resolve()
        assert source.is_relative_to(folder.resolve()) and sha(source) == item['sha256']
    pointer = {'latest_inventory': str(latest.relative_to(ROOT)), 'latest_inventory_sha256': sha(latest),
        'snapshot_created_utc': data['created_utc'], 'restore_order': [
            {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in chain],
        'verified_latest_restore_files': len(data['files']),
        'scope': '最新セットの全復元元ファイルと依存inventoryのhashを照合。原画像は既存manifest参照で別途必要。venvとBlender本体は含まない。'}
    write_json(BACKUP/'latest.json', pointer)
    lines = ['# PAF継続研究のバックアップ', '',
        f'最新セット：[{latest.parent.name}]({latest.as_posix()})', '',
        '以下の順に各inventoryのpathをプロジェクト直下へ対応させる。差分セット同士の連鎖適用は不要。既存ファイルへの上書きは内容を比較して判断する。', '']
    for index, path in enumerate(chain, 1):
        lines.append(f'{index}. [{path.parent.name}]({path.as_posix()})')
    lines += ['', '最新の差分はstorage=deltaのファイルだけを重ね、storage=baseは指定の完全版を参照する。未完了runは各inventoryのexcluded_incomplete_runsで確認できる。', '',
        '元の1194M/多様形状の原画像と教師はmanifestのパスとhashで参照しており、元データを別途保持する必要がある。venv・Blenderは同梱しない。', '',
        f'最新セットの復元元{len(data["files"]):,}ファイルと依存inventoryのhashを照合済み。過去セットのファイル実体を毎回再検査したことを意味しない。', '',
        '|保全セット|作成日時 JST|形式|復元対象ファイル|このセットの保存量 MiB|', '|---|---|---|---:|---:|']
    for path, item in reversed(inventories):
        delta = item.get('type') == 'delta_against_fixed_full_snapshot'
        size = sum(r['bytes'] for r in item['files'] if not delta or r['storage'] == 'delta')/2**20
        created = datetime.fromisoformat(item['created_utc']).astimezone(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')
        lines.append(f'|[{path.parent.name}]({path.as_posix()})|{created}|{"差分" if delta else "完全版"}|{len(item["files"]):,}|{size:.1f}|')
    (BACKUP/'INDEX.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print({'latest': latest.parent.name, 'restore_steps': len(chain), 'verified_files': len(data['files'])})


if __name__ == '__main__':
    main()
