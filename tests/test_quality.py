import unittest

from src.analysis.quality import (
    build_quality_trend_data,
    calculate_daily_quality_metrics,
    calculate_quality_metrics,
)


class TestCalculateQualityMetrics(unittest.TestCase):
    def test_calculates_basic_quality_metrics(self):
        inspections = [
            {
                "inspection_result": "OK",
                "defect_type": None,
            },
            {
                "inspection_result": "NG",
                "defect_type": "washer_missing",
            },
            {
                "inspection_result": "NG",
                "defect_type": "washer_missing",
            },
            {
                "inspection_result": "NG",
                "defect_type": "spacer_missing",
            },
            {
                "inspection_result": "OK",
                "defect_type": None,
            },
        ]

        result = calculate_quality_metrics(inspections)

        self.assertEqual(result["total_inspections"], 5)
        self.assertEqual(result["ok_count"], 2)
        self.assertEqual(result["ng_count"], 3)
        self.assertAlmostEqual(result["defect_rate"], 0.6)
        self.assertEqual(
            result["defect_counts"],
            {
                "spacer_missing": 1,
                "washer_missing": 2,
            },
        )

    def test_empty_inspections_returns_zero_metrics(self):
        result = calculate_quality_metrics([])

        self.assertEqual(result["total_inspections"], 0)
        self.assertEqual(result["ok_count"], 0)
        self.assertEqual(result["ng_count"], 0)
        self.assertEqual(result["defect_rate"], 0.0)
        self.assertEqual(result["defect_counts"], {})

    def test_unexpected_inspection_result_raises_value_error(self):
        inspections = [
            {
                "inspection_result": "UNKNOWN",
                "defect_type": None,
            },
        ]

        with self.assertRaises(ValueError):
            calculate_quality_metrics(inspections)


class TestCalculateDailyQualityMetrics(unittest.TestCase):
    def test_calculates_daily_quality_metrics_in_date_order(self):
        inspections = [
            {
                "captured_at": "2026-09-02T10:00:00",
                "inspection_result": "NG",
                "defect_type": "spacer_missing",
            },
            {
                "captured_at": "2026-09-01T09:00:00",
                "inspection_result": "OK",
                "defect_type": None,
            },
            {
                "captured_at": "2026-09-02T11:00:00",
                "inspection_result": "OK",
                "defect_type": None,
            },
            {
                "captured_at": "2026-09-01T10:00:00",
                "inspection_result": "NG",
                "defect_type": "washer_missing",
            },
        ]

        result = calculate_daily_quality_metrics(inspections)

        self.assertEqual(
            result,
            [
                {
                    "date": "2026-09-01",
                    "total_inspections": 2,
                    "ok_count": 1,
                    "ng_count": 1,
                    "defect_rate": 0.5,
                },
                {
                    "date": "2026-09-02",
                    "total_inspections": 2,
                    "ok_count": 1,
                    "ng_count": 1,
                    "defect_rate": 0.5,
                },
            ],
        )

    def test_empty_inspections_returns_empty_daily_metrics(self):
        result = calculate_daily_quality_metrics([])

        self.assertEqual(result, [])

    def test_invalid_captured_at_raises_value_error(self):
        inspections = [
            {
                "captured_at": "invalid-date",
                "inspection_result": "OK",
                "defect_type": None,
            },
        ]

        with self.assertRaises(ValueError):
            calculate_daily_quality_metrics(inspections)


class TestBuildQualityTrendData(unittest.TestCase):
    def test_builds_chart_ready_daily_data(self):
        inspections = [
            {
                "captured_at": "2026-09-02T09:00:00",
                "inspection_result": "NG",
                "defect_type": "washer_missing",
            },
            {
                "captured_at": "2026-09-01T09:00:00",
                "inspection_result": "OK",
                "defect_type": None,
            },
            {
                "captured_at": "2026-09-02T10:00:00",
                "inspection_result": "OK",
                "defect_type": None,
            },
            {
                "captured_at": "2026-09-01T10:00:00",
                "inspection_result": "NG",
                "defect_type": "spacer_missing",
            },
        ]

        result = build_quality_trend_data(inspections)

        self.assertEqual(
            result,
            [
                {
                    "date": "2026-09-01",
                    "total_inspections": 2,
                    "ok_count": 1,
                    "ng_count": 1,
                    "defect_rate": 0.5,
                    "defect_rate_percent": 50.0,
                },
                {
                    "date": "2026-09-02",
                    "total_inspections": 2,
                    "ok_count": 1,
                    "ng_count": 1,
                    "defect_rate": 0.5,
                    "defect_rate_percent": 50.0,
                },
            ],
        )

    def test_empty_inspections_returns_empty_trend_data(self):
        result = build_quality_trend_data([])

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
