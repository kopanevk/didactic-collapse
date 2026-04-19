from __future__ import annotations

import pytest

from didactic_collapse.clients.judge_client import (
    JudgeResponseValidationError,
    parse_and_validate_judge_response,
)


def _valid_payload() -> str:
    return (
        '{'
        '"clarity": 2,'
        '"structure": 1,'
        '"terminology": 2,'
        '"reasoning_soundness": 1,'
        '"overall_pedagogical_score": 6,'
        '"is_silent_error": false,'
        '"comment": "solid explanation"'
        '}'
    )


def test_parse_valid_json() -> None:
    result = parse_and_validate_judge_response(_valid_payload())
    assert result.score.overall_pedagogical_score == 6
    assert result.repair_applied is False


def test_parse_fenced_json() -> None:
    text = f"```json\n{_valid_payload()}\n```"
    result = parse_and_validate_judge_response(text)
    assert result.score.clarity == 2
    assert result.repair_applied is True


def test_parse_prose_plus_json() -> None:
    text = f"Here is my evaluation:\n{_valid_payload()}\nThank you."
    result = parse_and_validate_judge_response(text)
    assert result.score.terminology == 2
    assert result.repair_applied is True


def test_missing_field_fails() -> None:
    text = (
        '{'
        '"clarity": 2,'
        '"structure": 1,'
        '"terminology": 2,'
        '"overall_pedagogical_score": 5,'
        '"is_silent_error": false,'
        '"comment": "missing reasoning"'
        '}'
    )
    with pytest.raises(JudgeResponseValidationError, match="Missing required fields"):
        parse_and_validate_judge_response(text)


def test_invalid_score_range_fails() -> None:
    text = (
        '{'
        '"clarity": 3,'
        '"structure": 1,'
        '"terminology": 2,'
        '"reasoning_soundness": 1,'
        '"overall_pedagogical_score": 7,'
        '"is_silent_error": false,'
        '"comment": "out of range"'
        '}'
    )
    with pytest.raises(JudgeResponseValidationError, match="schema validation failed"):
        parse_and_validate_judge_response(text)


def test_wrong_type_fails() -> None:
    text = (
        '{'
        '"clarity": "two",'
        '"structure": 1,'
        '"terminology": 2,'
        '"reasoning_soundness": 1,'
        '"overall_pedagogical_score": 6,'
        '"is_silent_error": false,'
        '"comment": "wrong type"'
        '}'
    )
    with pytest.raises(JudgeResponseValidationError, match="clarity must be integer or int-like value"):
        parse_and_validate_judge_response(text)


def test_inconsistent_overall_score_fails() -> None:
    text = (
        '{'
        '"clarity": 2,'
        '"structure": 1,'
        '"terminology": 2,'
        '"reasoning_soundness": 1,'
        '"overall_pedagogical_score": 5,'
        '"is_silent_error": false,'
        '"comment": "inconsistent total"'
        '}'
    )
    result = parse_and_validate_judge_response(text)
    assert result.score.overall_pedagogical_score == 6
    assert result.repair_applied is True
    assert any(action.startswith("recomputed_overall_from_subscores:") for action in result.repair_actions)


def test_inconsistent_overall_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    text = (
        '{'
        '"clarity": 2,'
        '"structure": 2,'
        '"terminology": 0,'
        '"reasoning_soundness": 0,'
        '"overall_pedagogical_score": 6,'
        '"is_silent_error": false,'
        '"comment": "inconsistent overall from provider"'
        '}'
    )
    result = parse_and_validate_judge_response(text)
    assert result.score.overall_pedagogical_score == 4
    assert "provider_overall_inconsistent" in caplog.text


def test_empty_response_fails() -> None:
    with pytest.raises(JudgeResponseValidationError, match="empty response"):
        parse_and_validate_judge_response("   ")


def test_truncated_json_like_with_all_keys_is_safely_recovered() -> None:
    text = (
        '{'
        '"clarity": 2,'
        '"structure": 1,'
        '"terminology": 2,'
        '"reasoning_soundness": 1,'
        '"overall_pedagogical_score": 6,'
        '"is_silent_error": false,'
        '"comment": "short comment"'
    )
    result = parse_and_validate_judge_response(text)
    assert result.score.overall_pedagogical_score == 6
    assert "safe_truncated_json_recovery" in result.repair_actions


def test_irreparable_non_json_still_fails_explicitly() -> None:
    text = '{"clarity": 2, "structure": 1, "comment": "missing keys and broken"'
    with pytest.raises(JudgeResponseValidationError, match="Could not extract JSON object"):
        parse_and_validate_judge_response(text)
