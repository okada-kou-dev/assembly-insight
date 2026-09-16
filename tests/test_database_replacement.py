from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from src.storage.database import save_inspection
from src.storage.replacement import database_snapshot, publish_replacement


class DatabaseReplacementTest(unittest.TestCase):
    def test_changed_target_and_journal_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, completed = root / 'target.db', root / 'completed.db'
            for path in (target, completed):
                save_inspection(path, 'a.jpg', datetime(2032, 1, 1), 'OK', None, .9)
            snapshot = database_snapshot(target)
            save_inspection(target, 'b.jpg', datetime(2032, 1, 1), 'OK', None, .9)
            before = target.read_bytes()
            with self.assertRaisesRegex(ValueError, '確認後'):
                publish_replacement(completed, target, snapshot)
            self.assertEqual(target.read_bytes(), before)
            self.assertTrue(completed.exists())
            Path(str(target) + '-wal').write_bytes(b'active')
            with self.assertRaisesRegex(ValueError, '使用中'):
                database_snapshot(target)

    def test_non_sqlite_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'model.pt'
            target.write_bytes(b'not a database')
            with self.assertRaisesRegex(ValueError, 'SQLite DBではない'):
                database_snapshot(target)
            self.assertEqual(target.read_bytes(), b'not a database')
