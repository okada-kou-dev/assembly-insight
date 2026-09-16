import unittest

from inspection_rules import judge_part_counts


class TestJudgePartCounts(unittest.TestCase):
    def test_normal_is_ok(self):
        result = judge_part_counts(
            {
                "bolt": 1,
                "washer": 2,
                "spacer": 1,
                "cap_nut": 1,
            }
        )

        self.assertEqual(result.inspection_result, "OK")
        self.assertIsNone(result.defect_type)

    def test_washer_missing_is_ng(self):
        result = judge_part_counts(
            {
                "bolt": 1,
                "washer": 1,
                "spacer": 1,
                "cap_nut": 1,
            }
        )

        self.assertEqual(result.inspection_result, "NG")
        self.assertEqual(result.defect_type, "washer_missing")

    def test_spacer_missing_is_ng(self):
        result = judge_part_counts(
            {
                "bolt": 1,
                "washer": 2,
                "spacer": 0,
                "cap_nut": 1,
            }
        )

        self.assertEqual(result.inspection_result, "NG")
        self.assertEqual(result.defect_type, "spacer_missing")

    def test_cap_nut_missing_is_ng(self):
        result = judge_part_counts(
            {
                "bolt": 1,
                "washer": 2,
                "spacer": 1,
                "cap_nut": 0,
            }
        )

        self.assertEqual(result.inspection_result, "NG")
        self.assertEqual(result.defect_type, "cap_nut_missing")

    def test_bolt_missing_is_ng(self):
        result = judge_part_counts(
            {
                "bolt": 0,
                "washer": 2,
                "spacer": 1,
                "cap_nut": 1,
            }
        )

        self.assertEqual(result.inspection_result, "NG")
        self.assertEqual(result.defect_type, "bolt_missing")

    def test_multiple_missing_is_ng(self):
        result = judge_part_counts(
            {
                "bolt": 1,
                "washer": 1,
                "spacer": 0,
                "cap_nut": 1,
            }
        )

        self.assertEqual(result.inspection_result, "NG")
        self.assertEqual(result.defect_type, "multiple_missing")

    def test_excess_count_is_quantity_mismatch(self):
        result = judge_part_counts(
            {
                "bolt": 1,
                "washer": 3,
                "spacer": 1,
                "cap_nut": 1,
            }
        )

        self.assertEqual(result.inspection_result, "NG")
        self.assertEqual(result.defect_type, "quantity_mismatch")

    def test_missing_key_is_treated_as_zero(self):
        result = judge_part_counts(
            {
                "bolt": 1,
                "washer": 2,
                "spacer": 1,
            }
        )

        self.assertEqual(result.inspection_result, "NG")
        self.assertEqual(result.defect_type, "cap_nut_missing")
        self.assertEqual(result.counts["cap_nut"], 0)

    def test_negative_count_raises_value_error(self):
        with self.assertRaises(ValueError):
            judge_part_counts(
                {
                    "bolt": 1,
                    "washer": -1,
                    "spacer": 1,
                    "cap_nut": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
