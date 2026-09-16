"""表示用の文章だけを整え、保存本文やLLM入力は変更しない。"""

import re


NUMBERED_HEADING = re.compile(r"^(?:[0-9０-９]+[.．、]|[①-⑳])\s*\S")


def display_text(value: str) -> str:
    return value.replace("。", " / ").strip().rstrip(" /")


def display_lines(value: str) -> str:
    """グラフなどの補足文を、文ごとにMarkdown改行して表示する。"""
    return value.replace("。", "  \n").strip()


def comment_markdown(value: str) -> str:
    """LLM原文の番号付きタイトルをMarkdown見出しとして表示する。"""
    rendered: list[str] = []
    for line in value.splitlines():
        if NUMBERED_HEADING.match(line.strip()):
            if rendered and rendered[-1]:
                rendered.append("")
            rendered.extend((f"#### {line.strip()}", ""))
        else:
            rendered.append(line)
    return "\n".join(rendered).strip()
