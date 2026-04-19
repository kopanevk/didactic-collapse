from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
import random
import re
import socket
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from didactic_collapse.clients.base import JudgeClient

logger = logging.getLogger(__name__)


class JudgeResponseValidationError(ValueError):
    """Raised when judge response cannot be safely parsed/validated/repaired."""


class GeminiAuthConfigurationError(RuntimeError):
    """Raised when Gemini auth configuration is missing/ambiguous/unsupported."""


class CerebrasAuthConfigurationError(RuntimeError):
    """Raised when Cerebras auth configuration is missing/invalid."""


class GeminiJudgeRequestError(RuntimeError):
    """Raised when Gemini SDK request fails with categorized diagnostics."""

    def __init__(self, *, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


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


@dataclass(frozen=True)
class GeminiAuthPreflightResult:
    """Safe auth diagnostics for Gemini credential setup."""

    selected_mode: str
    selected_source: str
    selected_key_fingerprint: str
    present_sources: tuple[str, ...]


@dataclass(frozen=True)
class CerebrasAuthPreflightResult:
    """Safe auth diagnostics for Cerebras credential setup."""

    selected_source: str
    selected_key_fingerprint: str


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_INT_LIKE_RE = re.compile(r"^[+-]?\d+$")
_PREVIEW_LEN = 320

_CEREBRAS_JSON_CONTRACT = (
    "You are an evaluation engine. Output must be exactly one JSON object and nothing else.\n"
    "No markdown fences. No prefix/suffix text. No explanations.\n"
    "Do not output any keys other than the required rubric keys.\n"
    "Required fields and types:\n"
    "- clarity: integer in [0,2]\n"
    "- structure: integer in [0,2]\n"
    "- terminology: integer in [0,2]\n"
    "- reasoning_soundness: integer in [0,2]\n"
    "- overall_pedagogical_score: integer in [0,8], must equal clarity+structure+terminology+reasoning_soundness\n"
    "- is_silent_error: boolean\n"
    "- comment: one short sentence string, <=120 chars, no markdown, no line breaks, avoid quotes\n"
    "Return only the JSON object."
)

_CEREBRAS_FORMAT_REPAIR_PROMPT = (
    "Your previous response was not parseable JSON.\n"
    "Re-emit the same evaluation as exactly one minified valid JSON object.\n"
    "No markdown fences. No prose. No prefix/suffix.\n"
    "Allowed keys only: clarity, structure, terminology, reasoning_soundness, "
    "overall_pedagogical_score, is_silent_error, comment.\n"
    "Comment must be <=80 chars, one short sentence, no line breaks.\n"
)

_RUBRIC_REQUIRED_FIELDS: tuple[str, ...] = (
    "clarity",
    "structure",
    "terminology",
    "reasoning_soundness",
    "overall_pedagogical_score",
    "is_silent_error",
    "comment",
)


def _mask_secret(secret: str) -> str:
    if len(secret) <= 6:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-2:]}"


def _trim_preview(text: str, limit: int = _PREVIEW_LEN) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "...<trimmed>"


def _should_retry_openai_compatible(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or (500 <= code <= 599)
    return False


def _should_retry_gemini(exc: Exception) -> bool:
    if isinstance(exc, GeminiJudgeRequestError):
        return exc.category in {"network_timeout", "network_transport", "quota_or_rate_limit"}
    return False


def _parse_retry_after_seconds(retry_after_value: str | None) -> float | None:
    if retry_after_value is None:
        return None
    raw = retry_after_value.strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


def _classify_openai_compatible_exception(exc: Exception) -> tuple[str, bool, float | None]:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, socket.timeout)):
        return ("timeout", True, None)
    if isinstance(exc, httpx.TransportError):
        return ("transport_error", True, None)
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            retry_after = _parse_retry_after_seconds(exc.response.headers.get("Retry-After"))
            return ("http_429_rate_limited", True, retry_after)
        if 500 <= code <= 599:
            return (f"http_{code}_server_error", True, None)
        return (f"http_{code}_non_retryable", False, None)
    return (exc.__class__.__name__.lower(), False, None)


def preflight_validate_gemini_auth(
    *, api_key_env: str = "GEMINI_API_KEY", explicit_api_key: str | None = None
) -> GeminiAuthPreflightResult:
    """Validate that exactly one supported Gemini auth source is configured."""
    explicit = (explicit_api_key or "").strip()
    env_primary = (os.getenv(api_key_env) or "").strip()
    env_gemini = (os.getenv("GEMINI_API_KEY") or "").strip()
    env_google = (os.getenv("GOOGLE_API_KEY") or "").strip()
    adc = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    vertex_mode = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()

    present_sources: list[str] = []
    key_candidates: list[tuple[str, str]] = []

    if explicit:
        present_sources.append("explicit_api_key")
        key_candidates.append(("explicit_api_key", explicit))
    if env_primary:
        present_sources.append(f"env:{api_key_env}")
        key_candidates.append((f"env:{api_key_env}", env_primary))
    if api_key_env != "GEMINI_API_KEY" and env_gemini:
        present_sources.append("env:GEMINI_API_KEY")
        key_candidates.append(("env:GEMINI_API_KEY", env_gemini))
    if env_google:
        present_sources.append("env:GOOGLE_API_KEY")
        key_candidates.append(("env:GOOGLE_API_KEY", env_google))
    if adc:
        present_sources.append("env:GOOGLE_APPLICATION_CREDENTIALS")
    if vertex_mode in {"1", "true", "yes"}:
        present_sources.append("env:GOOGLE_GENAI_USE_VERTEXAI")

    # API-key path only: do not allow mixing with ADC.
    if adc and key_candidates:
        raise GeminiAuthConfigurationError(
            "Multiple auth sources found for Gemini (API key + ADC). "
            "Use exactly one auth path."
        )
    if adc and not key_candidates:
        raise GeminiAuthConfigurationError(
            "Unsupported auth mode for this project: ADC-only configuration detected. "
            "Use API key auth via GEMINI_API_KEY."
        )
    if vertex_mode in {"1", "true", "yes"}:
        raise GeminiAuthConfigurationError(
            "Unsupported auth mode for this project: Vertex AI mode detected "
            "(GOOGLE_GENAI_USE_VERTEXAI=true). Use API key auth path only."
        )
    if not key_candidates:
        raise GeminiAuthConfigurationError(
            f"No API key found for Gemini. Set {api_key_env} or pass explicit api_key."
        )
    if len(key_candidates) > 1:
        raise GeminiAuthConfigurationError(
            "Multiple API key sources found for Gemini. "
            "Keep exactly one source and clear others."
        )

    selected_source, selected_key = key_candidates[0]

    result = GeminiAuthPreflightResult(
        selected_mode="api_key",
        selected_source=selected_source,
        selected_key_fingerprint=_mask_secret(selected_key),
        present_sources=tuple(present_sources),
    )
    logger.info(
        "gemini_auth_preflight mode=%s source=%s key=%s present=%s",
        result.selected_mode,
        result.selected_source,
        result.selected_key_fingerprint,
        ",".join(result.present_sources),
    )
    return result


def preflight_validate_cerebras_auth(*, api_key_env: str = "CEREBRAS_API_KEY") -> CerebrasAuthPreflightResult:
    """Validate Cerebras API key presence for OpenAI-compatible API path."""
    api_key = (os.getenv(api_key_env) or "").strip()
    if not api_key:
        raise CerebrasAuthConfigurationError(
            f"Missing Cerebras API key. Set {api_key_env}."
        )
    result = CerebrasAuthPreflightResult(
        selected_source=f"env:{api_key_env}",
        selected_key_fingerprint=_mask_secret(api_key),
    )
    logger.info(
        "cerebras_auth_preflight source=%s key=%s",
        result.selected_source,
        result.selected_key_fingerprint,
    )
    return result


def _classify_gemini_exception(exc: Exception) -> tuple[str, str]:
    """Classify Gemini SDK failures into actionable categories."""
    raw = str(exc) or exc.__class__.__name__
    msg = raw.lower()

    if "multiple authentication credentials" in msg:
        return (
            "multiple_auth_credentials",
            "Gemini auth failed: multiple authentication credentials detected. "
            "Keep exactly one API key source.",
        )
    if "invalid api key" in msg or "api key not valid" in msg or "permission denied" in msg:
        return (
            "invalid_api_key",
            "Gemini auth failed: API key rejected by provider.",
        )
    if "quota" in msg or "rate limit" in msg or "resource exhausted" in msg:
        return (
            "quota_or_rate_limit",
            "Gemini request failed due to quota/rate limit.",
        )
    if "model" in msg and ("not found" in msg or "unavailable" in msg or "unsupported" in msg):
        return (
            "model_unavailable",
            "Gemini request failed: configured model is unavailable or unsupported.",
        )
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, socket.timeout)):
        return ("network_timeout", "Gemini request timed out.")
    if isinstance(exc, (httpx.TransportError, OSError)):
        return ("network_transport", "Gemini request failed due to network/transport error.")
    return ("sdk_misconfiguration_or_unknown", f"Gemini SDK call failed: {raw}")


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


def _count_unescaped_quotes(text: str) -> int:
    count = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            count += 1
    return count


def _brace_depth_outside_strings(text: str) -> int:
    in_string = False
    escaped = False
    depth = 0
    for ch in text:
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
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
    return depth


def _non_json_diagnostics(raw_text: str) -> tuple[bool, int, bool]:
    stripped = raw_text.strip()
    keys_detected = all(f'"{field}"' in stripped for field in _RUBRIC_REQUIRED_FIELDS)
    quote_unbalanced = (_count_unescaped_quotes(stripped) % 2) == 1
    brace_depth = _brace_depth_outside_strings(stripped)
    looks_truncated = stripped.startswith("{") and (
        not stripped.endswith("}") or quote_unbalanced or brace_depth > 0
    )
    return looks_truncated, len(raw_text), keys_detected


def _attempt_safe_truncated_json_recovery(raw_text: str) -> tuple[str, bool]:
    """Try bounded recovery for truncated JSON-like payloads without inventing fields."""
    stripped = raw_text.strip()
    if not stripped.startswith("{"):
        return raw_text, False
    if not all(f'"{field}"' in stripped for field in _RUBRIC_REQUIRED_FIELDS):
        return raw_text, False

    quote_unbalanced = (_count_unescaped_quotes(stripped) % 2) == 1
    brace_depth = _brace_depth_outside_strings(stripped)
    looks_truncated = not stripped.endswith("}") or quote_unbalanced or brace_depth > 0
    if not looks_truncated:
        return raw_text, False

    candidate = stripped.rstrip("` \n\r\t")
    if quote_unbalanced:
        candidate += '"'
    if brace_depth > 0:
        candidate += "}" * brace_depth

    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*$", "", candidate)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return raw_text, False
    if not isinstance(parsed, dict):
        return raw_text, False
    return candidate, True


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

    repaired: dict[str, Any] = dict(raw_payload)
    actions: list[str] = []

    # Safe bounded repair for known provider quirk:
    # sometimes model returns {"type":"object", ...rubric fields...}
    if extra == {"type"} and repaired.get("type") == "object":
        repaired.pop("type", None)
        actions.append("dropped_type_object_field")
        payload_fields = set(repaired.keys())
        extra = payload_fields.difference(required_fields)

    if extra:
        raise JudgeResponseValidationError(f"Unexpected extra fields: {sorted(extra)}")

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

    # Safe bounded repair: if provider-level overall conflicts with valid
    # criterion subscores, trust subscores and recompute overall.
    expected_overall = (
        int(repaired["clarity"])
        + int(repaired["structure"])
        + int(repaired["terminology"])
        + int(repaired["reasoning_soundness"])
    )
    provider_overall = int(repaired["overall_pedagogical_score"])
    if provider_overall != expected_overall:
        logger.warning(
            "provider_overall_inconsistent provider_overall=%s repaired_overall=%s",
            provider_overall,
            expected_overall,
        )
        repaired["overall_pedagogical_score"] = expected_overall
        actions.append(
            f"recomputed_overall_from_subscores:{provider_overall}->{expected_overall}"
        )

    return repaired, tuple(actions)


def parse_and_validate_judge_response(raw_text: str) -> JudgeParseResult:
    try:
        json_text, extracted, extraction_action = _extract_first_json_object(raw_text)
    except JudgeResponseValidationError as exc:
        if is_non_json_format_drift_error(exc):
            recovered_json, recovered = _attempt_safe_truncated_json_recovery(raw_text)
            if not recovered:
                raise
            json_text = recovered_json
            extracted = True
            extraction_action = "safe_truncated_json_recovery"
        else:
            raise
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


def is_non_json_format_drift_error(exc: JudgeResponseValidationError) -> bool:
    """Return True for non-JSON formatting drift where extractor found no JSON object."""
    msg = str(exc)
    return "Could not extract JSON object from judge response" in msg


class OpenAICompatibleJudgeClient(JudgeClient):
    """Generic OpenAI-compatible judge client (not for Gemini auth path)."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key_env: str,
        timeout_sec: int = 60,
        max_retries: int = 3,
        provider_name: str = "openai_compatible",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._provider_name = provider_name
        self._timeout_sec = int(timeout_sec)
        self._max_retries = int(max_retries)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env: {api_key_env}")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=float(self._timeout_sec), write=20.0, pool=20.0),
            headers=self._headers,
        )

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

    def _build_system_prompt(self, rubric_prompt: str) -> str:
        return rubric_prompt

    def _build_user_prompt(self, *, question: str, gold_answer: str, model_output: str) -> str:
        return (
            "Evaluate educational quality and silent error risk. "
            "Return strict JSON only.\n\n"
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Model output: {model_output}\n"
        )

    def _build_response_format(self) -> dict[str, str] | None:
        return {"type": "json_object"}

    def _build_payload(self, *, rubric_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": self._build_system_prompt(rubric_prompt)},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        response_format = self._build_response_format()
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _build_format_repair_prompt(self, *, previous_raw_content: str) -> str:
        return (
            f"{_CEREBRAS_FORMAT_REPAIR_PROMPT}\n"
            "Previous non-JSON response:\n"
            f"{previous_raw_content}\n"
            "Return JSON only."
        )

    def _request_raw_content(self, *, payload: dict[str, Any]) -> str:
        resp = self._http.post(f"{self._base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_content_from_api_response(data)

    def _compute_retry_sleep(
        self,
        *,
        attempt_number: int,
        retry_after_sec: float | None,
        base_backoff_sec: float = 1.5,
        max_backoff_sec: float = 45.0,
        jitter_max_sec: float = 0.75,
    ) -> float:
        if retry_after_sec is not None:
            return max(0.0, retry_after_sec)
        exp = min(max_backoff_sec, base_backoff_sec * (2 ** max(0, attempt_number - 1)))
        jitter = random.uniform(0.0, jitter_max_sec)
        return exp + jitter

    def score_typed(
        self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str
    ) -> JudgeParseResult:
        user_prompt = self._build_user_prompt(
            question=question,
            gold_answer=gold_answer,
            model_output=model_output,
        )
        payload = self._build_payload(rubric_prompt=rubric_prompt, user_prompt=user_prompt)
        max_attempts = max(1, self._max_retries + 1)
        attempt = 1
        while True:
            try:
                raw_content = self._request_raw_content(payload=payload)
                try:
                    return parse_and_validate_judge_response(raw_content)
                except JudgeResponseValidationError as exc:
                    looks_truncated, raw_len, keys_detected = _non_json_diagnostics(raw_content)
                    logger.warning(
                        "judge_validation_failed provider=%s model=%s category=%s looks_truncated=%s raw_len=%s keys_detected=%s preview=%s",
                        self._provider_name,
                        self._model_name,
                        "format_non_json_response"
                        if is_non_json_format_drift_error(exc)
                        else "schema_or_validation_failure",
                        looks_truncated,
                        raw_len,
                        keys_detected,
                        _trim_preview(raw_content),
                    )
                    if is_non_json_format_drift_error(exc):
                        repair_payload = self._build_payload(
                            rubric_prompt=rubric_prompt,
                            user_prompt=self._build_format_repair_prompt(
                                previous_raw_content=raw_content
                            ),
                        )
                        repair_raw_content = self._request_raw_content(payload=repair_payload)
                        try:
                            repaired = parse_and_validate_judge_response(repair_raw_content)
                            logger.warning(
                                "judge_format_repair_applied provider=%s model=%s preview=%s",
                                self._provider_name,
                                self._model_name,
                                _trim_preview(repair_raw_content),
                            )
                            return repaired
                        except JudgeResponseValidationError:
                            logger.warning(
                                "judge_format_repair_failed provider=%s model=%s category=%s preview=%s",
                                self._provider_name,
                                self._model_name,
                                "format_non_json_response",
                                _trim_preview(repair_raw_content),
                            )
                    raise
            except Exception as exc:  # noqa: BLE001
                category, retryable, retry_after_sec = _classify_openai_compatible_exception(exc)
                if (not retryable) or attempt >= max_attempts:
                    raise
                sleep_sec = self._compute_retry_sleep(
                    attempt_number=attempt,
                    retry_after_sec=retry_after_sec,
                )
                logger.warning(
                    "judge_retry provider=%s model=%s category=%s attempt=%d/%d sleep_sec=%.2f timeout_read_sec=%s",
                    self._provider_name,
                    self._model_name,
                    category,
                    attempt,
                    max_attempts,
                    sleep_sec,
                    self._timeout_sec,
                )
                time.sleep(sleep_sec)
                attempt += 1

    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        parsed = self.score_typed(
            question=question,
            gold_answer=gold_answer,
            model_output=model_output,
            rubric_prompt=rubric_prompt,
        )
        return parsed.score.model_dump()


class CerebrasJudgeClient(OpenAICompatibleJudgeClient):
    """Cerebras judge via OpenAI-compatible endpoint with strict auth preflight."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key_env: str = "CEREBRAS_API_KEY",
        timeout_sec: int = 60,
        max_retries: int = 3,
    ) -> None:
        self._preflight = preflight_validate_cerebras_auth(api_key_env=api_key_env)
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key_env=api_key_env,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            provider_name="cerebras",
        )

    def _build_system_prompt(self, rubric_prompt: str) -> str:
        return f"{_CEREBRAS_JSON_CONTRACT}\n\nRubric:\n{rubric_prompt}"

    def _build_user_prompt(self, *, question: str, gold_answer: str, model_output: str) -> str:
        return (
            "Score the model output using the rubric and return exactly one JSON object.\n"
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Model output: {model_output}\n"
            "Remember: output only JSON object with required fields. "
            "Comment must be one short sentence <=120 chars, no line breaks."
        )

    def _build_payload(self, *, rubric_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = super()._build_payload(rubric_prompt=rubric_prompt, user_prompt=user_prompt)
        payload["temperature"] = 0
        payload["top_p"] = 1
        return payload

    def smoke_check(self) -> str:
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": "Reply with a short text: ok"}],
            "temperature": 0,
            "max_tokens": 8,
        }
        resp = self._http.post(f"{self._base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_content_from_api_response(data)

    def rubric_format_check(self) -> JudgeParseResult:
        return self.score_typed(
            question="What is 1 + 1?",
            gold_answer="2",
            model_output="I think answer might be 2.",
            rubric_prompt=(
                "Evaluate pedagogical quality using the rubric. "
                "Output strict rubric JSON."
            ),
        )


class GeminiJudgeClient(JudgeClient):
    """Gemini judge client using official google-genai SDK with API key auth only."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key_env: str = "GEMINI_API_KEY",
        explicit_api_key: str | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini judge provider") from exc

        preflight = preflight_validate_gemini_auth(
            api_key_env=api_key_env,
            explicit_api_key=explicit_api_key,
        )
        key = (explicit_api_key or os.getenv(api_key_env) or os.getenv("GOOGLE_API_KEY") or "").strip()
        self._model_name = model_name
        self._preflight = preflight
        self._client = genai.Client(api_key=key)

    def _generate_content(self, prompt: str) -> Any:
        try:
            return self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
        except Exception as exc:  # noqa: BLE001
            category, message = _classify_gemini_exception(exc)
            logger.error("gemini_judge_request_failed category=%s detail=%s", category, str(exc))
            raise GeminiJudgeRequestError(category=category, message=message) from exc

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        # Last-resort stringify for diagnostics; validator will still enforce JSON.
        return str(response)

    @retry(
        wait=wait_exponential(min=1, max=20),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_should_retry_gemini),
        reraise=True,
    )
    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        prompt = (
            f"{rubric_prompt}\n\n"
            "Return strict JSON only.\n"
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Model output: {model_output}\n"
        )
        response = self._generate_content(prompt)
        parsed = parse_and_validate_judge_response(self._extract_text(response))
        if parsed.repair_applied:
            logger.warning("judge_repair_applied actions=%s", ",".join(parsed.repair_actions))
        return parsed.score.model_dump()

    def smoke_check(self) -> str:
        response = self._generate_content("Respond with JSON only: {\"ok\": true}")
        return self._extract_text(response)


def build_gemini_judge_client(
    *,
    model_name: str,
    api_key_env: str = "GEMINI_API_KEY",
    explicit_api_key: str | None = None,
) -> GeminiJudgeClient:
    """Factory for Gemini judge client used by both smoke-check and pipeline."""
    return GeminiJudgeClient(
        model_name=model_name,
        api_key_env=api_key_env,
        explicit_api_key=explicit_api_key,
    )


def build_cerebras_judge_client(
    *,
    model_name: str,
    base_url: str,
    api_key_env: str = "CEREBRAS_API_KEY",
    timeout_sec: int = 60,
    max_retries: int = 3,
) -> CerebrasJudgeClient:
    """Factory for Cerebras judge client used by both smoke-check and pipeline."""
    return CerebrasJudgeClient(
        base_url=base_url,
        model_name=model_name,
        api_key_env=api_key_env,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
    )


def gemini_judge_auth_smoke_check(*, model_name: str, api_key_env: str = "GEMINI_API_KEY") -> str:
    """Run minimal Gemini SDK call via the same auth path as judge pipeline."""
    client = build_gemini_judge_client(model_name=model_name, api_key_env=api_key_env)
    return client.smoke_check()


def cerebras_judge_auth_smoke_check(
    *,
    model_name: str,
    base_url: str,
    api_key_env: str = "CEREBRAS_API_KEY",
    timeout_sec: int = 60,
) -> str:
    """Run minimal Cerebras OpenAI-compatible call via the same auth path as pipeline."""
    client = build_cerebras_judge_client(
        model_name=model_name,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_sec=timeout_sec,
    )
    return client.smoke_check()


def cerebras_judge_rubric_format_check(
    *,
    model_name: str,
    base_url: str,
    api_key_env: str = "CEREBRAS_API_KEY",
    timeout_sec: int = 60,
) -> dict[str, Any]:
    """Check real Cerebras rubric-format readiness via the same parser as pipeline."""
    client = build_cerebras_judge_client(
        model_name=model_name,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_sec=timeout_sec,
    )
    parsed = client.rubric_format_check()
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
