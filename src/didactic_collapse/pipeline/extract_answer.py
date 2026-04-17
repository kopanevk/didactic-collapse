from __future__ import annotations

import re


FINAL_ANSWER_REGEX = re.compile(r"(?:final answer|answer)\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def extract_final_answer(text: str) -> str | None:
    """Best-effort parser for final numeric/text answer from free-form reasoning."""
    m = FINAL_ANSWER_REGEX.search(text)
    if m:
        return m.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]
