import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path

from src.storage.database import (
    initialize_database,
    list_inspections,
    save_inspection,
)


class TestDatabase(unittest.TestCase):
    def test_initialize_database_creates_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "inspection.db"

            initialize_database(db_path)

            with closing(sqlite3.connect(db_path)) as connection:
                row = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    AND name = 'inspections'
                    """
                ).fetchone()

            self.assertIsNotNone(row)

    def test_save_and_read_ng_inspection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "inspection.db"

            image_id = save_inspection(
                db_path=db_path,
                image_path="20260901_080000_001.jpg",
                captured_at=datetime(2026, 9, 1, 8, 0, 0),
                inspection_result="NG",
                defect_type="washer_missing",
                confidence=0.936,
            )

            rows = list_inspections(db_path)

            self.assertEqual(image_id, 1)
            self.assertEqual(len(rows), 1)

            row = rows[0]

            self.assertEqual(
                row["image_path"],
                "20260901_080000_001.jpg",
            )
            self.assertEqual(
                row["captured_at"],
                "2026-09-01T08:00:00",
            )
            self.assertEqual(
                row["inspection_result"],
                "NG",
            )
            self.assertEqual(
                row["defect_type"],
                "washer_missing",
            )
            self.assertAlmostEqual(
                row["confidence"],
                0.936,
            )

    def test_ok_inspection_allows_no_defect_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "inspection.db"

            save_inspection(
                db_path=db_path,
                image_path="20260901_090000_002.jpg",
                captured_at=datetime(2026, 9, 1, 9, 0, 0),
                inspection_result="OK",
                defect_type=None,
                confidence=0.941,
            )

            rows = list_inspections(db_path)

            self.assertEqual(
                rows[0]["inspection_result"],
                "OK",
            )
            self.assertIsNone(rows[0]["defect_type"])

    def test_list_inspections_orders_by_captured_at(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "inspection.db"

            save_inspection(
                db_path=db_path,
                image_path="later.jpg",
                captured_at=datetime(2026, 9, 5, 13, 0, 0),
                inspection_result="NG",
                defect_type="cap_nut_missing",
                confidence=0.90,
            )

            save_inspection(
                db_path=db_path,
                image_path="earlier.jpg",
                captured_at=datetime(2026, 9, 1, 8, 0, 0),
                inspection_result="OK",
                defect_type=None,
                confidence=0.95,
            )

            rows = list_inspections(db_path)

            self.assertEqual(
                rows[0]["image_path"],
                "earlier.jpg",
            )
            self.assertEqual(
                rows[1]["image_path"],
                "later.jpg",
            )


if __name__ == "__main__":
    unittest.main()
