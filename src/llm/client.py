import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4:26b"
DEFAULT_TIMEOUT = 900.0
FREE_COMMENT_MAX_TOKENS = 1536
FREE_COMMENT_CONTEXT_TOKENS = 8192
FREE_COMMENT_INSTRUCTION = (
    "Python集計の数値を使い、重要な品質傾向と考察を日本語で3点、"
    "全体1000字程度にまとめてください。前置きと総括は不要です。"
    "着眼点・構成・表現は自由です。"
    "割合を書くときは、同じ表の同じ行にある対象期間、分母の名称と総数、"
    "分子の名称と件数、割合を一組のまま使い、別の行の数値と組み替えないでください。"
    "日付は「以降・以前」がその日を含み、「より後・より前」はその日を含みません。"
    "最終発生日後の0件期間に最終発生日を含めず、後続の観測日がなければ0件とは書かないでください。"
)
FREE_COMMENT_FINAL_REMINDER = (
    "【最後に守ること】割合は、表にある対象期間・分母・分子・割合の組を崩さず、"
    "分母と分子の名称と件数を明記してください。日付境界では「以降・以前」はその日を含み、"
    "「より後・より前」は含みません。最終発生日後の0件期間に最終発生日を含めず、"
    "後続の観測日がなければ0件とは書かないでください。"
)


def generate_free_quality_comment(
    analysis_data: dict[str, Any],
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    response_stats: dict[str, Any] | None = None,
) -> str:
    """Python集計を正式な指示とともに送り、自然終了した自由文を返す。"""
    summary = analysis_data.get("quality_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("自由コメント用のPython集計がありません。")
    content = FREE_COMMENT_INSTRUCTION + '\n\n' + summary + '\n\n' + FREE_COMMENT_FINAL_REMINDER
    messages = [{'role': 'user', 'content': content}]
    request_body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0, "num_predict": FREE_COMMENT_MAX_TOKENS,
                    "num_ctx": FREE_COMMENT_CONTEXT_TOKENS},
        "keep_alive": 0,
    }
    if model == "gemma4:26b":
        request_body["think"] = False
    request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    received_parts: list[str] = []
    try:
        with urlopen(request, timeout=timeout) as response:
            return _read_free_comment_stream(response, parts=received_parts, response_stats=response_stats)
    except HTTPError as exc:
        with exc:
            error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama API returned HTTP {exc.code}: {error_body}"
        ) from exc
    except (URLError, TimeoutError, ConnectionError) as exc:
        error = RuntimeError(
            "Ollamaとの通信に失敗しました（接続切断またはタイムアウトを含む）。"
            "Ollamaの起動状態を確認してください。"
        )
        error.partial_comment = ''.join(received_parts)
        raise error from exc
    except (ValueError, RuntimeError) as exc:
        # 診断用に保持するだけで、不完全な本文を成功結果として返さない。
        exc.partial_comment = ''.join(received_parts)
        raise


def _read_free_comment_stream(response: Any, *, parts: list[str] | None = None,
                              response_stats: dict[str, Any] | None = None) -> str:
    parts = [] if parts is None else parts
    for line in response:
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Ollamaのストリームに不正なJSONがあります。") from exc
        if not isinstance(chunk, dict):
            raise ValueError("OllamaのストリームがJSON objectではありません。")
        if "error" in chunk:
            raise RuntimeError(f"Ollama API error: {chunk['error']}")
        message = chunk.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Ollamaのストリームに有効な本文がありません。")
        if not isinstance(chunk.get("done"), bool):
            raise ValueError("Ollamaのストリームに有効な終了状態がありません。")
        parts.append(message["content"])
        if chunk["done"]:
            if response_stats is not None:
                for key in ('done_reason', 'prompt_eval_count', 'eval_count', 'total_duration',
                            'load_duration', 'prompt_eval_duration', 'eval_duration'):
                    if key in chunk:
                        response_stats[key] = chunk[key]
            reason = chunk.get("done_reason")
            if reason == "length":
                raise ValueError("品質コメントが出力上限で打ち切られました（done_reason=length）。")
            if reason != "stop":
                raise ValueError(f"品質コメントの自然終了を確認できません（done_reason={reason!r}）。")
            comment = "".join(parts)
            if not comment.strip():
                raise ValueError("Ollamaから空の品質コメントが返されました。")
            return comment
    raise ValueError("品質コメントのストリームが完了通知なしで終了しました。")
