import unittest

from quality_fixtures import (
    build_dummy_records,
)
from src.analysis.quality import (
    calculate_time_band_quality_metrics,
    classify_time_band,
)


class TimeBandQualityTest(unittest.TestCase):
    def test_all_hours_and_exact_boundaries_cover_four_equal_bands(self):
        expected = ["00_06"] * 6 + ["06_12"] * 6 + ["12_18"] * 6 + ["18_24"] * 6
        self.assertEqual([classify_time_band(hour) for hour in range(24)], expected)
        timestamps = ["00:00:00", "05:59:59", "06:00:00", "11:59:59", "12:00:00", "17:59:59", "18:00:00", "23:59:59"]
        records = [{"captured_at": f"2032-03-01T{at}", "inspection_result": "OK", "defect_type": None} for at in timestamps]
        bands = calculate_time_band_quality_metrics(records)
        self.assertEqual([row["total_inspections"] for row in bands], [2, 2, 2, 2])

    def test_classify_time_band(self) -> None:
        self.assertEqual(
            classify_time_band(8),
            "06_12",
        )
        self.assertEqual(
            classify_time_band(13),
            "12_18",
        )
        self.assertEqual(
            classify_time_band(19),
            "18_24",
        )
        self.assertEqual(
            classify_time_band(23),
            "18_24",
        )
        self.assertEqual(
            classify_time_band(2),
            "00_06",
        )

    def test_invalid_hour_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            classify_time_band(24)

    def test_dummy_scenario_time_band_metrics(self) -> None:
        records = build_dummy_records()

        metrics = calculate_time_band_quality_metrics(records)

        by_band = {row["time_band"]: row for row in metrics}

        morning = by_band["06_12"]

        self.assertEqual(
            morning["total_inspections"],
            20,
        )
        self.assertEqual(
            morning["ok_count"],
            15,
        )
        self.assertEqual(
            morning["ng_count"],
            5,
        )
        self.assertAlmostEqual(
            morning["defect_rate"],
            0.25,
        )
        self.assertEqual(
            morning["defect_counts"],
            {
                "cap_nut_missing": 2,
                "spacer_missing": 2,
                "washer_missing": 1,
            },
        )

        afternoon = by_band["12_18"]

        self.assertEqual(
            afternoon["total_inspections"],
            20,
        )
        self.assertEqual(
            afternoon["ok_count"],
            11,
        )
        self.assertEqual(
            afternoon["ng_count"],
            9,
        )
        self.assertAlmostEqual(
            afternoon["defect_rate"],
            0.45,
        )
        self.assertEqual(
            afternoon["defect_counts"],
            {
                "cap_nut_missing": 4,
                "washer_missing": 5,
            },
        )

        evening = by_band["18_24"]

        self.assertEqual(
            evening["total_inspections"],
            10,
        )
        self.assertEqual(
            evening["ok_count"],
            2,
        )
        self.assertEqual(
            evening["ng_count"],
            8,
        )
        self.assertAlmostEqual(
            evening["defect_rate"],
            0.8,
        )
        self.assertEqual(
            evening["defect_counts"],
            {
                "washer_missing": 8,
            },
        )

        night = by_band["00_06"]

        self.assertEqual(
            night["total_inspections"],
            0,
        )
        self.assertEqual(
            night["ng_count"],
            0,
        )
        self.assertEqual(
            night["defect_rate"],
            0.0,
        )
        self.assertEqual(
            night["defect_counts"],
            {},
        )

    def test_invalid_captured_at_raises_value_error(self) -> None:
        inspections = [
            {
                "inspection_result": "OK",
                "defect_type": None,
                "captured_at": "invalid-date",
            }
        ]

        with self.assertRaises(ValueError):
            calculate_time_band_quality_metrics(inspections)


if __name__ == "__main__":
    unittest.main()
