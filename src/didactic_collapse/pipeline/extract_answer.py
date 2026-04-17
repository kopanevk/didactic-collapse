from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd

_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+)?")
_EXPLICIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "final_answer_marker",
        re.compile(r"(?:^|\n)\s*final\s+answer\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
    ),
    (
        "answer_marker",
        re.compile(r"(?:^|\n)\s*answer\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
    ),
    (
        "the_answer_is",
        re.compile(r"(?:^|\n)\s*the\s+answer\s+is\s+(.+?)(?:\n|$)", re.IGNORECASE),
    ),
)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")


@dataclass(frozen=True)
class ExtractedAnswer:
    """Typed result of final-answer extraction and normalization."""

    extracted_answer: str | None
    normalized_answer: str | None
    parse_success: bool
    parse_strategy: str
    parse_failure_reason: str | None


@dataclass(frozen=True)
class _Candidate:
    raw_value: str
    normalized_value: str
    strategy: str


def _clean_fragment(fragment: str) -> str:
    cleaned = fragment.strip()
    cleaned = cleaned.strip("`'\" ")
    cleaned = cleaned.strip(".,;:!?)]}>")
    cleaned = cleaned.strip("([<{")
    return cleaned.strip()


def _extract_numeric_token(fragment: str) -> str | None:
    """Extract first numeric token from fragment in safe formats."""
    cleaned = _clean_fragment(fragment)
    if not cleaned:
        return None

    match = _NUMERIC_TOKEN_RE.search(cleaned)
    if not match:
        return None
    token = match.group(0)
    token = re.sub(r"\s+", "", token)
    token = token.replace(",", "")
    return token


def normalize_extracted_answer(raw_answer: object) -> str | None:
    """Normalize extracted raw answer to numeric token when possible."""
    if raw_answer is None or pd.isna(raw_answer):
        return None
    return _extract_numeric_token(str(raw_answer))


def _collect_candidates(text: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []

    for match in _BOXED_RE.finditer(text):
        raw_value = _clean_fragment(match.group(1))
        normalized = _extract_numeric_token(raw_value)
        if normalized is not None:
            candidates.append(_Candidate(raw_value=raw_value, normalized_value=normalized, strategy="boxed"))

    for strategy_name, pattern in _EXPLICIT_PATTERNS:
        for match in pattern.finditer(text):
            raw_value = _clean_fragment(match.group(1))
            normalized = _extract_numeric_token(raw_value)
            if normalized is not None:
                candidates.append(
                    _Candidate(
                        raw_value=raw_value,
                        normalized_value=normalized,
                        strategy=strategy_name,
                    )
                )

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        normalized = _extract_numeric_token(tail)
        if normalized is not None:
            candidates.append(_Candidate(raw_value=_clean_fragment(tail), normalized_value=normalized, strategy="trailing_numeric"))

    return candidates


def _resolve_candidates(candidates: Iterable[_Candidate]) -> ExtractedAnswer:
    items = list(candidates)
    if not items:
        return ExtractedAnswer(
            extracted_answer=None,
            normalized_answer=None,
            parse_success=False,
            parse_strategy="none",
            parse_failure_reason="no_numeric_candidate",
        )

    normalized_values = {item.normalized_value for item in items}
    if len(normalized_values) > 1:
        return ExtractedAnswer(
            extracted_answer=None,
            normalized_answer=None,
            parse_success=False,
            parse_strategy="ambiguous",
            parse_failure_reason="multiple_candidate_answers",
        )

    best = items[-1]
    return ExtractedAnswer(
        extracted_answer=best.raw_value,
        normalized_answer=best.normalized_value,
        parse_success=True,
        parse_strategy=best.strategy,
        parse_failure_reason=None,
    )


def extract_final_answer_result(text: str) -> ExtractedAnswer:
    """Extract final numeric answer from model output with ambiguity handling."""
    if not text or not text.strip():
        return ExtractedAnswer(
            extracted_answer=None,
            normalized_answer=None,
            parse_success=False,
            parse_strategy="none",
            parse_failure_reason="empty_output",
        )

    candidates = _collect_candidates(text)
    return _resolve_candidates(candidates)


def extract_final_answer(text: str) -> str | None:
    """Backward-compatible helper for generation pipeline.

    Returns raw extracted answer string or None on parse failure.
    """
    result = extract_final_answer_result(text)
    return result.extracted_answer if result.parse_success else None
