import unittest
from datetime import datetime
from pathlib import Path

from src.inspection.metadata import parse_captured_at


class TestParseCapturedAt(unittest.TestCase):
    def test_parse_valid_filename(self):
        result = parse_captured_at("20260901_080000_001.jpg")

        self.assertEqual(
            result,
            datetime(2026, 9, 1, 8, 0, 0),
        )

    def test_parse_path_object(self):
        result = parse_captured_at(Path("data/final/20260905_130000_004.jpg"))

        self.assertEqual(
            result,
            datetime(2026, 9, 5, 13, 0, 0),
        )

    def test_invalid_filename_format_raises_error(self):
        with self.assertRaises(ValueError):
            parse_captured_at("normal_test_001.jpg")

    def test_missing_sequence_raises_error(self):
        with self.assertRaises(ValueError):
            parse_captured_at("20260901_080000.jpg")

    def test_invalid_date_raises_error(self):
        with self.assertRaises(ValueError):
            parse_captured_at("20260230_080000_001.jpg")

    def test_invalid_time_raises_error(self):
        with self.assertRaises(ValueError):
            parse_captured_at("20260901_250000_001.jpg")


if __name__ == "__main__":
    unittest.main()
