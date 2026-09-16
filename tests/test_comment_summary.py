import copy
import csv
import io
import unittest

from src.analysis.comment_summary import build_quality_summary


def record(at, result="OK", defect=None):
    return {"captured_at": at, "inspection_result": result, "defect_type": defect}


def table(summary, title):
    content = summary.split(title + "(CSV)\n", 1)[1].split("\n\n", 1)[0]
    return list(csv.DictReader(io.StringIO(content)))


class TestQualitySummary(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("2031-03-01T08:00:00", "NG", "spacer_missing"),
            record("2031-03-01T12:00:00"),
            record("2031-03-01T17:00:00", "NG", "washer_missing"),
            record("2031-03-03T05:59:00", "NG", "washer_missing"),
            record("2031-03-03T06:00:00"),
            record("2031-03-06T22:00:00"),
        ]

    def test_distinguishes_inspection_and_ng_denominators(self):
        summary = build_quality_summary(self.records, data_type="test_data")
        rows = table(summary, "全体・日別・時間帯別集計")
        overall = rows[0]
        self.assertEqual(overall["検査件数"], "6")
        self.assertEqual(overall["OK件数"], "3")
        self.assertEqual(overall["不良率(NG/検査)"], "3/6=50.0%")
        self.assertEqual(overall["ワッシャ欠品（位置未確定）/検査"], "2/6=33.3%")
        self.assertEqual(overall["ワッシャ欠品（位置未確定）/NG"], "2/3=66.7%")
        day = next(row for row in rows if row["区分"] == "2031-03-01")
        self.assertEqual(day["ワッシャ欠品（位置未確定）/NG"], "1/2=50.0%")

    def test_uses_observed_dates_only_and_correct_time_boundaries(self):
        summary = build_quality_summary(self.records, data_type="test_data")
        self.assertIn("記録のある日数：3日、検査6件", summary)
        self.assertNotIn("2031-03-04", summary)
        rows = table(summary, "欠品の観測記録")
        washer = next(row for row in rows if row["欠品種別"] == "ワッシャ欠品（位置未確定）")
        self.assertEqual(washer["初回記録"], "2031-03-01T17:00:00")
        self.assertEqual(washer["最終記録"], "2031-03-03T05:59:00")
        self.assertEqual(washer["最終発生日後の観測日(発生0件)"], "2031-03-06")
        self.assertEqual(washer["後続観測日数"], "1")
        cross = table(summary, "日付×時間帯集計")
        self.assertEqual(len(cross), 12)
        by_label = {row["日付・時間帯"]: row for row in cross}
        self.assertEqual(by_label["2031-03-03 00:00–06:00"]["NG件数"], "1")
        self.assertEqual(by_label["2031-03-03 06:00–12:00"]["NG件数"], "0")
        shares = table(summary, "欠品種別ごとの時間帯構成")
        washer_night = next(row for row in shares if
                            row["時間帯"] == "00:00–06:00" and row["欠品種別"] == "ワッシャ欠品（位置未確定）")
        self.assertEqual(washer_night["当該時間帯の件数/全期間の同種欠品件数"], "1/2=50.0%")

    def test_all_ok_and_unobserved_bands_are_not_equivalent(self):
        summary = build_quality_summary([record("2032-01-02T08:00:00")], data_type="inspection_data")
        rows = table(summary, "全体・日別・時間帯別集計")
        self.assertEqual(rows[0]["不良率(NG/検査)"], "0/1=0.0%")
        night = next(row for row in rows if row["区分"] == "00:00–06:00")
        self.assertEqual(night["不良率(NG/検査)"], "算出不可(分母0件)")
        self.assertEqual(table(summary, "不良の観測記録"), [])
        self.assertNotIn("ダミー", summary)
        self.assertNotIn("NG1件につき欠品種別1つ", summary)

    def test_zero_ng_denominator_is_not_zero_percent(self):
        summary = build_quality_summary(self.records, data_type="test_data")
        day = next(row for row in table(summary, "全体・日別・時間帯別集計") if row["区分"] == "2031-03-06")
        self.assertEqual(day["ワッシャ欠品（位置未確定）/検査"], "0/1=0.0%")
        self.assertEqual(day["ワッシャ欠品（位置未確定）/NG"], "算出不可(分母0件)")

    def test_includes_other_defect_types_and_unclassified_ng(self):
        defects = ["bolt_missing", "multiple_missing", "quantity_mismatch", "new_defect", None]
        records = [record("2032-01-02T08:00:00", "NG", kind) for kind in defects]
        summary = build_quality_summary(records, data_type="test_data")
        overall = table(summary, "全体・日別・時間帯別集計")[0]
        self.assertEqual(overall["NG件数"], "5")
        for name in ["ボルト欠品", "複数部品欠品", "数量異常", "new_defect", "不良種別未記録"]:
            self.assertEqual(overall[name + "件数"], "1")
            self.assertEqual(overall[name + "/NG"], "1/5=20.0%")
        self.assertNotIn("NG1件につき欠品種別1つ", summary)

    def test_input_order_and_unused_fields_do_not_change_summary(self):
        records = copy.deepcopy(self.records)
        records[0]["image_path"] = "not_for_the_model.jpg"
        original = copy.deepcopy(records)
        summary = build_quality_summary(iter(records), data_type="test_data")
        self.assertEqual(records, original)
        self.assertEqual(summary, build_quality_summary(reversed(records), data_type="test_data"))
        self.assertNotIn("not_for_the_model.jpg", summary)

    def test_washer_sides_and_legacy_missing_remain_separate_record_categories(self):
        records = [record("2032-01-02T08:00:00", "NG", kind) for kind in (
            "washer_missing", "washer_before_spacer_missing", "washer_after_spacer_missing",
        )]
        summary = build_quality_summary(records, data_type="inspection_data")
        overall = table(summary, "全体・日別・時間帯別集計")[0]
        self.assertEqual(overall["NG件数"], "3")
        for label in ("ワッシャ欠品（位置未確定）", "ワッシャ欠品（ボルト側）", "ワッシャ欠品（化粧ナット側）"):
            self.assertEqual(overall[label + "件数"], "1")
            self.assertEqual(overall[label + "/NG"], "1/3=33.3%")
        self.assertEqual(len(table(summary, "欠品の観測記録")), 3)

    def test_empty_input_and_invalid_input(self):
        self.assertEqual(build_quality_summary([], data_type="test_data"), "分析対象の検査記録はありません。")
        with self.assertRaises(ValueError):
            build_quality_summary(self.records, data_type="")
        with self.assertRaisesRegex(ValueError, "Invalid captured_at"):
            build_quality_summary([record("not-a-date")], data_type="test_data")
        with self.assertRaisesRegex(ValueError, "Unexpected inspection_result"):
            build_quality_summary([record("2032-01-02", "ERROR")], data_type="test_data")


if __name__ == "__main__":
    unittest.main()
