import unittest

from quality_fixtures import (
    build_dummy_records,
)
from src.analysis.quality import (
    calculate_quality_trend_patterns,
)


class QualityTrendPatternsTest(unittest.TestCase):
    def test_dummy_scenario_trend_patterns(self) -> None:
        records = build_dummy_records()

        patterns = calculate_quality_trend_patterns(records)

        self.assertEqual(
            patterns["peak"]["date"],
            "2026-09-04",
        )
        self.assertAlmostEqual(
            patterns["peak"]["defect_rate"],
            0.7,
        )
        self.assertEqual(
            patterns["peak"]["ng_count"],
            7,
        )

        post_peak = patterns["post_peak"]

        self.assertEqual(
            post_peak["peak_date"],
            "2026-09-04",
        )
        self.assertEqual(
            post_peak["latest_date"],
            "2026-09-05",
        )
        self.assertAlmostEqual(
            post_peak["peak_defect_rate"],
            0.7,
        )
        self.assertAlmostEqual(
            post_peak["latest_defect_rate"],
            0.3,
        )
        self.assertAlmostEqual(
            post_peak["defect_rate_change"],
            -0.4,
        )
        self.assertTrue(post_peak["recovered"])

        increase = patterns["longest_increase_streak"]

        self.assertEqual(
            increase["start_date"],
            "2026-09-01",
        )
        self.assertEqual(
            increase["end_date"],
            "2026-09-04",
        )
        self.assertEqual(
            increase["consecutive_increases"],
            3,
        )
        self.assertAlmostEqual(
            increase["start_defect_rate"],
            0.3,
        )
        self.assertAlmostEqual(
            increase["end_defect_rate"],
            0.7,
        )

        decrease = patterns["longest_decrease_streak"]

        self.assertEqual(
            decrease["start_date"],
            "2026-09-04",
        )
        self.assertEqual(
            decrease["end_date"],
            "2026-09-05",
        )
        self.assertEqual(
            decrease["consecutive_decreases"],
            1,
        )

    def test_empty_data(self) -> None:
        patterns = calculate_quality_trend_patterns([])

        self.assertIsNone(patterns["peak"])
        self.assertIsNone(patterns["post_peak"])
        self.assertIsNone(patterns["longest_increase_streak"])
        self.assertIsNone(patterns["longest_decrease_streak"])


if __name__ == "__main__":
    unittest.main()
