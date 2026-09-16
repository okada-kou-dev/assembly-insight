import unittest

from quality_fixtures import (
    build_dummy_records,
)
from src.analysis.quality import (
    calculate_daily_defect_counts,
)


class DailyDefectCountsTest(unittest.TestCase):
    def test_dummy_scenario_daily_defect_counts(self) -> None:
        records = build_dummy_records()

        daily = calculate_daily_defect_counts(records)

        self.assertEqual(
            daily,
            [
                {
                    "date": "2026-09-01",
                    "total_inspections": 10,
                    "ng_count": 3,
                    "defect_counts": {
                        "cap_nut_missing": 1,
                        "spacer_missing": 1,
                        "washer_missing": 1,
                    },
                },
                {
                    "date": "2026-09-02",
                    "total_inspections": 10,
                    "ng_count": 4,
                    "defect_counts": {
                        "cap_nut_missing": 1,
                        "spacer_missing": 1,
                        "washer_missing": 2,
                    },
                },
                {
                    "date": "2026-09-03",
                    "total_inspections": 10,
                    "ng_count": 5,
                    "defect_counts": {
                        "cap_nut_missing": 1,
                        "washer_missing": 4,
                    },
                },
                {
                    "date": "2026-09-04",
                    "total_inspections": 10,
                    "ng_count": 7,
                    "defect_counts": {
                        "cap_nut_missing": 2,
                        "washer_missing": 5,
                    },
                },
                {
                    "date": "2026-09-05",
                    "total_inspections": 10,
                    "ng_count": 3,
                    "defect_counts": {
                        "cap_nut_missing": 1,
                        "washer_missing": 2,
                    },
                },
            ],
        )

    def test_empty_data_returns_empty_list(self) -> None:
        self.assertEqual(
            calculate_daily_defect_counts([]),
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
            calculate_daily_defect_counts(inspections)


if __name__ == "__main__":
    unittest.main()
