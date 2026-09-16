"""確認済みSQLite DBを、完成済みの検査結果へ置き換える。"""

from hashlib import sha256
from pathlib import Path
from shutil import copy2
from uuid import uuid4


def database_snapshot(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("上書き先には通常のSQLite DBファイルを指定してください")
    for suffix in ('-wal', '-shm', '-journal'):
        if Path(str(path) + suffix).exists():
            raise ValueError("保存先DBが使用中の可能性があります / 使用中のアプリを終了してから再実行してください")
    with path.open('rb') as source:
        if source.read(16) != b'SQLite format 3\x00':
            raise ValueError("保存先がSQLite DBではないため上書きできません")
    return str(path.resolve()), sha256(path.read_bytes()).hexdigest()


def publish_replacement(completed: Path, destination: Path, snapshot: tuple[str, str]) -> Path:
    if database_snapshot(destination) != snapshot:
        raise ValueError("確認後に保存先DBが変更されました / 内容を確認して再実行してください")
    backup = destination.with_name(f'{destination.stem}.backup-{uuid4().hex}.db')
    copy2(destination, backup)
    if database_snapshot(destination) != snapshot:
        raise ValueError("保存先DBが変更されたため置き換えを中止しました")
    completed.replace(destination)
    return backup
