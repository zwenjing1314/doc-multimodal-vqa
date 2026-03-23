from __future__ import annotations

import re


def normalize_text(text: str | None) -> str:
    """
    Normalize text before comparison.

    Current strategy:
    - handle None
    - strip leading/trailing whitespace
    - collapse multiple spaces into one
    - casefold for case-insensitive matching

    We intentionally do NOT remove punctuation globally,
    because answers like dates or initials may rely on it.
    """
    if text is None:
        return ""

    text = text.casefold()  # 比 lower() 更激进的方法，将字符串转换为适合不区分大小写比较的形式
    text = text.strip()  # 去除首尾空白字符
    text = re.sub(r"\s+", " ", text)  #  将连续空白字符替换为单个空格
    return text


def normalize_answers(answers: list[str] | None) -> list[str]:
    """
    Normalize a list of answers and drop empty ones.
    """
    if answers is None:
        return []

    normalized: list[str] = []
    for answer in answers:
        norm = normalize_text(answer)
        if norm:
            normalized.append(norm)
    return normalized
