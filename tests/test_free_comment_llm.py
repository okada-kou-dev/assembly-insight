import io
import json
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from src.llm.client import (
    FREE_COMMENT_FINAL_REMINDER,
    FREE_COMMENT_INSTRUCTION,
    generate_free_quality_comment,
)


def stream(*chunks):
    return io.BytesIO(b"\n".join(json.dumps(chunk, ensure_ascii=False).encode("utf-8") for chunk in chunks))


def chunk(content, *, done=False, reason="stop"):
    result = {"message": {"content": content}, "done": done}
    if done:
        result["done_reason"] = reason
    return result


class TestFreeQualityComment(unittest.TestCase):
    @patch("src.llm.client.urlopen")
    def test_uses_accepted_conditions_and_returns_unmodified_text(self, urlopen):
        urlopen.return_value = stream(chunk(" 文章\n"), chunk("続き。", done=True))
        summary = "全体,7,3,4,4/7=57.1%\n"
        result = generate_free_quality_comment({"quality_summary": summary, "evidence": [{"summary": "送信しない指定傾向"}]})
        self.assertEqual(result, " 文章\n続き。")
        urlopen.assert_called_once()
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 900.0})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        body = json.loads(request.data)
        self.assertEqual(body, {
            "model": "gemma4:26b",
            "messages": [{"role": "user", "content": (
                FREE_COMMENT_INSTRUCTION + "\n\n" + summary + "\n\n" + FREE_COMMENT_FINAL_REMINDER
            )}],
            "stream": True,
            "options": {"temperature": 0, "num_predict": 1536, "num_ctx": 8192},
            "keep_alive": 0,
            "think": False,
        })
        self.assertEqual(FREE_COMMENT_INSTRUCTION,
                         'Python集計の数値を使い、重要な品質傾向と考察を日本語で3点、全体1000字程度にまとめてください。前置きと総括は不要です。着眼点・構成・表現は自由です。割合を書くときは、同じ表の同じ行にある対象期間、分母の名称と総数、分子の名称と件数、割合を一組のまま使い、別の行の数値と組み替えないでください。日付は「以降・以前」がその日を含み、「より後・より前」はその日を含みません。最終発生日後の0件期間に最終発生日を含めず、後続の観測日がなければ0件とは書かないでください。')
        self.assertEqual(FREE_COMMENT_FINAL_REMINDER,
                         '【最後に守ること】割合は、表にある対象期間・分母・分子・割合の組を崩さず、分母と分子の名称と件数を明記してください。日付境界では「以降・以前」はその日を含み、「より後・より前」は含みません。最終発生日後の0件期間に最終発生日を含めず、後続の観測日がなければ0件とは書かないでください。')

    @patch("src.llm.client.urlopen")
    def test_portfolio_demo_sends_summary_only_with_the_same_simple_instruction(self, urlopen):
        urlopen.return_value = stream(chunk('本文', done=True))
        generate_free_quality_comment({'quality_summary':'実測集計', 'data_type':'portfolio_demo',
                                      'records':[{'image_path':'private_photo.jpg','detections':[1,2,3]}],
                                      'evidence':[{'summary':'送信しない指定傾向'}]})
        messages = json.loads(urlopen.call_args.args[0].data)['messages']
        self.assertEqual(messages, [{'role':'user', 'content':(
            FREE_COMMENT_INSTRUCTION + '\n\n実測集計\n\n' + FREE_COMMENT_FINAL_REMINDER
        )}])

    @patch("src.llm.client.urlopen")
    def test_explicit_options_and_blank_stream_lines(self, urlopen):
        urlopen.return_value = io.BytesIO(b"\n" + stream(chunk("本文", done=True)).getvalue())
        self.assertEqual(generate_free_quality_comment({"quality_summary": "集計"}, model="test-model", timeout=10), "本文")
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 10})
        self.assertEqual(json.loads(urlopen.call_args.args[0].data)["model"], "test-model")
        self.assertNotIn("think", json.loads(urlopen.call_args.args[0].data))

    @patch("src.llm.client.urlopen")
    def test_missing_summary_does_not_call_api(self, urlopen):
        for summary in [None, "", " \n", [], 1]:
            with self.subTest(summary=summary), self.assertRaises(ValueError):
                generate_free_quality_comment({"quality_summary": summary})
        urlopen.assert_not_called()

    @patch("src.llm.client.urlopen")
    def test_rejects_empty_or_incomplete_responses(self, urlopen):
        cases = [
            ([], "完了通知なし"),
            ([chunk("途中")], "完了通知なし"),
            ([chunk(" \n", done=True)], "空の品質コメント"),
            ([chunk("途中", done=True, reason="length")], "出力上限"),
            ([chunk("本文", done=True, reason=None)], "自然終了"),
            ([chunk("本文", done=True, reason="unload")], "自然終了"),
        ]
        for chunks, message in cases:
            with self.subTest(message=message):
                urlopen.reset_mock()
                urlopen.return_value = stream(*chunks)
                with self.assertRaisesRegex(ValueError, message):
                    generate_free_quality_comment({"quality_summary": "集計"})
                urlopen.assert_called_once()

    @patch("src.llm.client.urlopen")
    def test_truncated_comment_is_preserved_only_on_the_error(self, urlopen):
        ending = {**chunk('文章', done=True, reason='length'), 'prompt_eval_count': 5100, 'eval_count': 1280}
        urlopen.return_value = stream(chunk('途中の'), ending)
        stats = {}
        with self.assertRaisesRegex(ValueError, '出力上限') as raised:
            generate_free_quality_comment({'quality_summary':'集計'}, response_stats=stats)
        self.assertEqual(raised.exception.partial_comment, '途中の文章')
        self.assertEqual(stats, {'done_reason': 'length', 'prompt_eval_count': 5100, 'eval_count': 1280})

    @patch("src.llm.client.urlopen")
    def test_records_natural_completion_and_usage_without_modifying_comment(self, urlopen):
        urlopen.return_value = stream({**chunk('日本語の本文。', done=True),
                                      'prompt_eval_count': 5200, 'eval_count': 550, 'total_duration': 123})
        stats = {}
        self.assertEqual(generate_free_quality_comment({'quality_summary': '集計'}, response_stats=stats), '日本語の本文。')
        self.assertEqual(stats, {'done_reason': 'stop', 'prompt_eval_count': 5200,
                                'eval_count': 550, 'total_duration': 123})

    @patch("src.llm.client.urlopen")
    def test_rejects_malformed_stream(self, urlopen):
        for payload in [b"not-json", b"\xff", b"[]", b"null", b"{}",
                        b'{"message":{"content":3},"done":true}',
                        b'{"message":{"content":"x"},"done":"false"}']:
            with self.subTest(payload=payload):
                urlopen.return_value = io.BytesIO(payload)
                with self.assertRaises(ValueError):
                    generate_free_quality_comment({"quality_summary": "集計"})

    @patch("src.llm.client.urlopen")
    def test_stream_error_is_preserved(self, urlopen):
        urlopen.return_value = stream(chunk("途中"), {"error": "model runner stopped"})
        with self.assertRaisesRegex(RuntimeError, "model runner stopped"):
            generate_free_quality_comment({"quality_summary": "集計"})
        urlopen.assert_called_once()

    @patch("src.llm.client.urlopen")
    def test_transport_errors_preserve_cause_without_retry(self, urlopen):
        for error in [URLError("offline"), TimeoutError("timeout"), ConnectionResetError("reset")]:
            with self.subTest(error=error):
                urlopen.reset_mock()
                urlopen.side_effect = error
                with self.assertRaises(RuntimeError) as raised:
                    generate_free_quality_comment({"quality_summary": "集計"})
                self.assertIs(raised.exception.__cause__, error)
                urlopen.assert_called_once()

    @patch("src.llm.client.urlopen")
    def test_timeout_during_stream_is_not_a_success(self, urlopen):
        class InterruptedStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                yield stream(chunk("途中")).getvalue()
                raise TimeoutError("read timeout")

        urlopen.return_value = InterruptedStream()
        with self.assertRaises(RuntimeError) as raised:
            generate_free_quality_comment({"quality_summary": "集計"})
        self.assertIsInstance(raised.exception.__cause__, TimeoutError)
        urlopen.assert_called_once()

    @patch("src.llm.client.urlopen")
    def test_http_error_retains_status_and_body(self, urlopen):
        error_body = io.BytesIO(b"model unavailable")
        urlopen.side_effect = HTTPError("http://localhost", 503, "unavailable", {}, error_body)
        with self.assertRaisesRegex(RuntimeError, "HTTP 503: model unavailable"):
            generate_free_quality_comment({"quality_summary": "集計"})
        urlopen.assert_called_once()
        self.assertTrue(error_body.closed)


if __name__ == "__main__":
    unittest.main()
