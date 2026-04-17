from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from didactic_collapse.clients.base import JudgeClient

logger = logging.getLogger(__name__)


class JudgeResponseValidationError(ValueError):
    """Raised when judge response cannot be safely parsed/validated/repaired."""


class JudgeRubricScore(BaseModel):
    """Strict schema for judge rubric output."""

    model_config = ConfigDict(extra="forbid", strict=True)

    clarity: int
    structure: int
    terminology: int
    reasoning_soundness: int
    overall_pedagogical_score: int
    is_silent_error: bool
    comment: str

    @model_validator(mode="after")
    def validate_ranges_and_consistency(self) -> "JudgeRubricScore":
        per_criterion = [
            ("clarity", self.clarity),
            ("structure", self.structure),
            ("terminology", self.terminology),
            ("reasoning_soundness", self.reasoning_soundness),
        ]
        for name, value in per_criterion:
            if not (0 <= value <= 2):
                raise ValueError(f"{name} must be in [0, 2], got {value}")

        if not (0 <= self.overall_pedagogical_score <= 8):
            raise ValueError(
                "overall_pedagogical_score must be in [0, 8], "
                f"got {self.overall_pedagogical_score}"
            )

        expected_overall = self.clarity + self.structure + self.terminology + self.reasoning_soundness
        if self.overall_pedagogical_score != expected_overall:
            raise ValueError(
                "overall_pedagogical_score must equal sum of sub-scores: "
                f"expected {expected_overall}, got {self.overall_pedagogical_score}"
            )
        return self


@dataclass(frozen=True)
class JudgeParseResult:
    """Typed parse result with repair metadata for observability."""

    score: JudgeRubricScore
    repair_applied: bool
    repair_actions: tuple[str, ...]


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_INT_LIKE_RE = re.compile(r"^[+-]?\d+$")


def _extract_first_json_object(raw_text: str) -> tuple[str, bool, str]:
    """Extract first JSON object from text, supporting fences and prose wrappers."""
    stripped = raw_text.strip()
    if not stripped:
        raise JudgeResponseValidationError("Judge returned empty response")

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, False, "direct_json"

    fenced_match = _FENCED_JSON_RE.search(stripped)
    if fenced_match:
        return fenced_match.group(1).strip(), True, "extracted_fenced_json"

    # Balanced-brace extraction for prose+JSON responses.
    in_string = False
    escaped = False
    depth = 0
    start: int | None = None

    for idx, ch in enumerate(stripped):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return stripped[start : idx + 1], True, "extracted_json_from_prose"

    raise JudgeResponseValidationError("Could not extract JSON object from judge response")


def _parse_json_dict(json_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise JudgeResponseValidationError(f"Invalid JSON from judge: {exc}") from exc

    if not isinstance(parsed, dict):
        raise JudgeResponseValidationError("Judge JSON must be an object")
    return parsed


def _safe_bool(value: Any) -> tuple[bool, bool]:
    if isinstance(value, bool):
        return value, False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True, True
        if lowered == "false":
            return False, True
    raise JudgeResponseValidationError("is_silent_error must be boolean")


def _safe_int(value: Any, field_name: str) -> tuple[int, bool]:
    if isinstance(value, bool):
        raise JudgeResponseValidationError(f"{field_name} must be integer, got boolean")
    if isinstance(value, int):
        return value, False
    if isinstance(value, float) and value.is_integer():
        return int(value), True
    if isinstance(value, str) and _INT_LIKE_RE.match(value.strip()):
        return int(value.strip()), True
    raise JudgeResponseValidationError(f"{field_name} must be integer or int-like value")


def _repair_payload(raw_payload: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Apply bounded safe repairs only (no guessing, no score invention)."""
    required_fields = {
        "clarity",
        "structure",
        "terminology",
        "reasoning_soundness",
        "overall_pedagogical_score",
        "is_silent_error",
        "comment",
    }
    payload_fields = set(raw_payload.keys())

    missing = required_fields.difference(payload_fields)
    extra = payload_fields.difference(required_fields)
    if missing:
        raise JudgeResponseValidationError(f"Missing required fields: {sorted(missing)}")
    if extra:
        raise JudgeResponseValidationError(f"Unexpected extra fields: {sorted(extra)}")

    repaired: dict[str, Any] = dict(raw_payload)
    actions: list[str] = []

    for field_name in (
        "clarity",
        "structure",
        "terminology",
        "reasoning_soundness",
        "overall_pedagogical_score",
    ):
        as_int, changed = _safe_int(repaired[field_name], field_name)
        repaired[field_name] = as_int
        if changed:
            actions.append(f"coerced_{field_name}_to_int")

    as_bool, bool_changed = _safe_bool(repaired["is_silent_error"])
    repaired["is_silent_error"] = as_bool
    if bool_changed:
        actions.append("coerced_is_silent_error_to_bool")

    if not isinstance(repaired["comment"], str):
        raise JudgeResponseValidationError("comment must be a string")

    return repaired, tuple(actions)


def parse_and_validate_judge_response(raw_text: str) -> JudgeParseResult:
    """Parse, repair (bounded), and validate judge output into typed rubric score."""
    json_text, extracted, extraction_action = _extract_first_json_object(raw_text)
    raw_payload = _parse_json_dict(json_text)
    repaired_payload, repair_actions = _repair_payload(raw_payload)

    actions: list[str] = []
    if extracted:
        actions.append(extraction_action)
    actions.extend(repair_actions)

    try:
        score = JudgeRubricScore.model_validate(repaired_payload)
    except ValidationError as exc:
        raise JudgeResponseValidationError(f"Judge schema validation failed: {exc}") from exc

    return JudgeParseResult(
        score=score,
        repair_applied=bool(actions),
        repair_actions=tuple(actions),
    )


class OpenAICompatibleJudgeClient(JudgeClient):
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key_env: str,
        timeout_sec: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env: {api_key_env}")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = httpx.Client(timeout=timeout_sec, headers=self._headers)

    def _extract_content_from_api_response(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeResponseValidationError("Malformed API response: missing choices[0].message.content") from exc
        if content is None:
            return ""
        if not isinstance(content, str):
            raise JudgeResponseValidationError("Judge content must be string")
        return content

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def score_typed(
        self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str
    ) -> JudgeParseResult:
        user_prompt = (
            "Evaluate educational quality and silent error risk. "
            "Return strict JSON only.\n\n"
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Model output: {model_output}\n"
        )
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": rubric_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        resp = self._http.post(f"{self._base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        raw_content = self._extract_content_from_api_response(data)

        try:
            parsed = parse_and_validate_judge_response(raw_content)
        except JudgeResponseValidationError as exc:
            if "schema validation failed" in str(exc):
                logger.error("judge_schema_violation error=%s", exc)
            else:
                logger.error("judge_parse_failure error=%s", exc)
            raise

        if parsed.repair_applied:
            logger.warning(
                "judge_repair_applied actions=%s",
                ",".join(parsed.repair_actions),
            )
        return parsed

    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        """Backward-compatible dict API used by existing pipeline."""
        parsed = self.score_typed(
            question=question,
            gold_answer=gold_answer,
            model_output=model_output,
            rubric_prompt=rubric_prompt,
        )
        return parsed.score.model_dump()


class MockJudgeClient(JudgeClient):
    """Deterministic non-scientific judge for pilot smoke tests only."""

    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        payload = f"{question}\n{gold_answer}\n{model_output}"
        h = int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)
        clarity = int(h % 3)
        structure = int((h >> 2) % 3)
        terminology = int((h >> 4) % 3)
        reasoning_soundness = int((h >> 6) % 3)
        overall = clarity + structure + terminology + reasoning_soundness
        is_silent_error = (h % 7) == 0
        return {
            "clarity": clarity,
            "structure": structure,
            "terminology": terminology,
            "reasoning_soundness": reasoning_soundness,
            "overall_pedagogical_score": overall,
            "is_silent_error": bool(is_silent_error),
            "comment": "MOCK_JUDGE_OUTPUT_NON_SCIENTIFIC",
        }
