import unittest
from unittest.mock import patch

from src.analysis.service import (
    analyze_quality,
    build_llm_context,
)


class TestBuildLlmContext(unittest.TestCase):
    def test_builds_flags_without_quality_numbers(self):
        result = build_llm_context(
            total_inspections=30,
            data_type="test_data",
        )

        self.assertIsNotNone(result)

        self.assertEqual(
            result["data_type"],
            "test_data",
        )
        self.assertNotIn("increase_candidate_exists", result)
        self.assertTrue(
            result["sample_size_limited"],
        )

        self.assertNotIn(
            "total_inspections",
            result,
        )
        self.assertNotIn(
            "defect_rate",
            result,
        )


class TestAnalyzeQuality(unittest.TestCase):
    @patch("src.analysis.service.generate_free_quality_comment")
    @patch("src.analysis.service.list_inspections")
    def test_default_generator_receives_summary_and_returns_free_text(self, rows, generator):
        rows.return_value = [
            {"captured_at": "2031-03-01T08:00:00", "inspection_result": "OK", "defect_type": None},
            {"captured_at": "2031-03-01T09:00:00", "inspection_result": "NG", "defect_type": "washer_missing"},
        ]
        generator.return_value = "自由な品質考察。"
        result = analyze_quality("dummy.db")
        rows.assert_called_once_with("dummy.db")
        generator.assert_called_once_with(result["llm_context"])
        self.assertIn("1/2=50.0%", result["llm_context"]["quality_summary"])
        self.assertIn("evidence", result["llm_context"])
        self.assertEqual(result["llm_comment"], "自由な品質考察。")
        self.assertEqual(result["overall"]["total_inspections"], 2)
        self.assertEqual(result["source_db_path"], "dummy.db")
        self.assertEqual([row["total_inspections"] for row in result["time_bands"]], [0, 2, 0, 0])
        self.assertEqual(result["time_bands"][1]["ng_count"], 1)
        self.assertEqual(result["time_bands"][1]["defect_rate"], 0.5)

    @patch("src.analysis.service.generate_free_quality_comment")
    @patch("src.analysis.service.list_inspections", return_value=[])
    def test_empty_database_does_not_call_default_generator(self, rows, generator):
        result = analyze_quality("empty.db")
        generator.assert_not_called()
        self.assertIsNone(result["llm_comment"])
        self.assertEqual([row["total_inspections"] for row in result["time_bands"]], [0, 0, 0, 0])

    @patch("src.analysis.service.generate_free_quality_comment")
    @patch("src.analysis.service.list_inspections")
    def test_generator_error_is_propagated_without_retry(self, rows, generator):
        rows.return_value = [
            {"captured_at": "2031-03-01T08:00:00", "inspection_result": "OK", "defect_type": None},
        ]
        error = RuntimeError("communication failed")
        generator.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            analyze_quality("dummy.db")
        self.assertIs(raised.exception, error)
        generator.assert_called_once()

    @patch("src.analysis.service.list_inspections")
    def test_integrates_quality_analysis_and_llm(
        self,
        mock_list_inspections,
    ):
        mock_list_inspections.return_value = [
            {
                "captured_at": "2026-09-01T09:00:00",
                "inspection_result": "OK",
                "defect_type": None,
            },
            {
                "captured_at": "2026-09-01T10:00:00",
                "inspection_result": "OK",
                "defect_type": None,
            },
            {
                "captured_at": "2026-09-02T09:00:00",
                "inspection_result": "NG",
                "defect_type": "washer_missing",
            },
            {
                "captured_at": "2026-09-02T10:00:00",
                "inspection_result": "NG",
                "defect_type": "spacer_missing",
            },
        ]

        received_context = None

        def fake_comment_generator(context):
            nonlocal received_context
            received_context = context

            return "時間帯別の品質集計です。"

        result = analyze_quality(
            "dummy.db",
            data_type="test_data",
            comment_generator=fake_comment_generator,
        )

        # DB取得はサービス層で1回だけ行う。
        mock_list_inspections.assert_called_once_with(
            "dummy.db",
        )

        # 全体品質指標
        self.assertEqual(
            result["overall"]["total_inspections"],
            4,
        )
        self.assertEqual(
            result["overall"]["ok_count"],
            2,
        )
        self.assertEqual(
            result["overall"]["ng_count"],
            2,
        )
        self.assertEqual(
            result["overall"]["defect_rate"],
            0.5,
        )

        self.assertNotIn("increase_candidate", result)
        self.assertNotIn("period_comparison", result)

        # LLMへ渡されたコンテキスト
        self.assertIsNotNone(
            received_context,
        )

        self.assertEqual(
            received_context["data_type"],
            "test_data",
        )
        self.assertNotIn("increase_candidate_exists", received_context)
        self.assertTrue(
            received_context["sample_size_limited"],
        )

        # Python分析で生成したevidenceがLLMへ渡される。
        self.assertIn(
            "evidence",
            received_context,
        )
        self.assertIsInstance(
            received_context["evidence"],
            list,
        )
        self.assertGreater(
            len(received_context["evidence"]),
            0,
        )

        # evidenceの最低限の構造を確認する。
        first_evidence = received_context["evidence"][0]

        self.assertIn(
            "evidence_id",
            first_evidence,
        )
        self.assertIn(
            "type",
            first_evidence,
        )
        self.assertIn(
            "summary",
            first_evidence,
        )
        self.assertIn(
            "facts",
            first_evidence,
        )

        # LLM生成結果がサービス戻り値へ統合される。
        self.assertEqual(result["llm_comment"], "時間帯別の品質集計です。")


    @patch("src.analysis.service.list_inspections")
    def test_empty_database_skips_llm(
        self,
        mock_list_inspections,
    ):
        mock_list_inspections.return_value = []

        def should_not_be_called(context):
            self.fail("LLM should not be called for empty data.")

        result = analyze_quality(
            "empty.db",
            comment_generator=should_not_be_called,
        )

        mock_list_inspections.assert_called_once_with(
            "empty.db",
        )

        self.assertEqual(
            result["overall"]["total_inspections"],
            0,
        )
        self.assertEqual(
            result["daily"],
            [],
        )
        self.assertEqual(
            result["trend"],
            [],
        )
        self.assertNotIn("increase_candidate", result)
        self.assertNotIn("period_comparison", result)
        self.assertIsNone(
            result["llm_comment"],
        )


if __name__ == "__main__":
    unittest.main()
