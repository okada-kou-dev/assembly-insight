import unittest

from quality_fixtures import (
    build_dummy_records,
)
from src.analysis.quality import (
    calculate_defect_occurrence_summary,
)


class DefectOccurrenceTest(unittest.TestCase):
    def test_dummy_scenario_defect_occurrence(self) -> None:
        records = build_dummy_records()

        summaries = calculate_defect_occurrence_summary(records)

        by_defect = {row["defect_type"]: row for row in summaries}

        washer = by_defect["washer_missing"]

        self.assertEqual(
            washer["total_count"],
            14,
        )
        self.assertEqual(
            washer["first_date"],
            "2026-09-01",
        )
        self.assertEqual(
            washer["last_date"],
            "2026-09-05",
        )
        self.assertTrue(washer["present_on_latest_date"])
        self.assertEqual(
            washer["days_since_last_occurrence"],
            0,
        )

        cap_nut = by_defect["cap_nut_missing"]

        self.assertEqual(
            cap_nut["total_count"],
            6,
        )
        self.assertEqual(
            cap_nut["first_date"],
            "2026-09-01",
        )
        self.assertEqual(
            cap_nut["last_date"],
            "2026-09-05",
        )
        self.assertTrue(cap_nut["present_on_latest_date"])

        spacer = by_defect["spacer_missing"]

        self.assertEqual(
            spacer["total_count"],
            2,
        )
        self.assertEqual(
            spacer["first_date"],
            "2026-09-01",
        )
        self.assertEqual(
            spacer["last_date"],
            "2026-09-02",
        )
        self.assertFalse(spacer["present_on_latest_date"])
        self.assertEqual(
            spacer["days_since_last_occurrence"],
            3,
        )

    def test_no_defects_returns_empty_list(self) -> None:
        inspections = [
            {
                "inspection_result": "OK",
                "defect_type": None,
                "captured_at": "2026-09-01T08:00:00",
            }
        ]

        self.assertEqual(
            calculate_defect_occurrence_summary(inspections),
            [],
        )

    def test_invalid_captured_at_raises_value_error(self) -> None:
        inspections = [
            {
                "inspection_result": "NG",
                "defect_type": "washer_missing",
                "captured_at": "invalid-date",
            }
        ]

        with self.assertRaises(ValueError):
            calculate_defect_occurrence_summary(inspections)


if __name__ == "__main__":
    unittest.main()
