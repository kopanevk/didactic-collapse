import httpx
from types import SimpleNamespace

import pytest

from didactic_collapse.clients.judge_client import (
    GeminiJudgeRequestError,
    JudgeResponseValidationError,
    build_cerebras_judge_client,
    _should_retry_gemini,
)
from didactic_collapse.clients.ollama_client import _should_retry_ollama


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError("http error", request=request, response=response)


def test_should_retry_gemini_only_transient_categories() -> None:
    assert _should_retry_gemini(GeminiJudgeRequestError(category="network_timeout", message="timeout")) is True
    assert _should_retry_gemini(GeminiJudgeRequestError(category="network_transport", message="transport")) is True
    assert _should_retry_gemini(GeminiJudgeRequestError(category="quota_or_rate_limit", message="rate")) is True

    assert _should_retry_gemini(GeminiJudgeRequestError(category="invalid_api_key", message="bad key")) is False
    assert _should_retry_gemini(
        GeminiJudgeRequestError(category="multiple_auth_credentials", message="multiple")
    ) is False
    assert _should_retry_gemini(
        GeminiJudgeRequestError(category="sdk_misconfiguration_or_unknown", message="misconfig")
    ) is False


def test_should_retry_ollama_only_transient_transport_or_http() -> None:
    assert _should_retry_ollama(httpx.TimeoutException("timeout")) is True
    assert _should_retry_ollama(httpx.TransportError("transport")) is True
    assert _should_retry_ollama(_http_status_error(429)) is True
    assert _should_retry_ollama(_http_status_error(503)) is True

    assert _should_retry_ollama(_http_status_error(400)) is False
    assert _should_retry_ollama(ValueError("schema")) is False


def test_cerebras_retries_on_429_with_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-retry")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=120,
        max_retries=2,
        cache_enabled=False,
        min_request_interval_sec=0.0,
    )
    calls = {"count": 0}
    slept: list[float] = []

    def _fake_sleep(sec: float) -> None:
        slept.append(sec)

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        if calls["count"] == 1:
            request = httpx.Request("POST", url)
            response = httpx.Response(status_code=429, request=request, headers={"Retry-After": "0"})
            return SimpleNamespace(
                raise_for_status=lambda: (_ for _ in ()).throw(
                    httpx.HTTPStatusError("429", request=request, response=response)
                ),
                json=lambda: {},
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"clarity":1,"structure":1,"terminology":1,"reasoning_soundness":1,'
                                '"overall_pedagogical_score":4,"is_silent_error":false,"comment":"ok"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("didactic_collapse.clients.judge_client.time.sleep", _fake_sleep)
    monkeypatch.setattr(client._http, "post", _fake_post)

    out = client.score(question="q", gold_answer="1", model_output="a", rubric_prompt="rubric")
    assert out["overall_pedagogical_score"] == 4
    assert calls["count"] == 2
    assert slept, "expected retry backoff sleep to be invoked"


def test_cerebras_validation_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-retry")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=120,
        max_retries=3,
        cache_enabled=False,
        min_request_interval_sec=0.0,
    )
    calls = {"count": 0}

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"clarity":3,"structure":1,"terminology":1,"reasoning_soundness":1,'
                                '"overall_pedagogical_score":6,"is_silent_error":false,"comment":"bad range"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    with pytest.raises(JudgeResponseValidationError):
        client.score(question="q", gold_answer="1", model_output="a", rubric_prompt="rubric")
    assert calls["count"] == 1


def test_cerebras_non_json_triggers_single_format_repair_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-format-repair")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=120,
        max_retries=3,
        cache_enabled=False,
        min_request_interval_sec=0.0,
    )
    calls = {"count": 0}

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": "non-json prose response"}}]},
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"clarity":1,"structure":1,"terminology":1,"reasoning_soundness":1,'
                                '"overall_pedagogical_score":4,"is_silent_error":false,"comment":"ok"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    out = client.score(question="q", gold_answer="1", model_output="a", rubric_prompt="rubric")
    assert out["overall_pedagogical_score"] == 4
    assert calls["count"] == 2


def test_cerebras_irreparable_non_json_fails_without_retry_storm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-format-fail")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=120,
        max_retries=3,
        cache_enabled=False,
        min_request_interval_sec=0.0,
    )
    calls = {"count": 0}
    slept: list[float] = []

    def _fake_sleep(sec: float) -> None:
        slept.append(sec)

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "still non-json"}}]},
        )

    monkeypatch.setattr("didactic_collapse.clients.judge_client.time.sleep", _fake_sleep)
    monkeypatch.setattr(client._http, "post", _fake_post)

    with pytest.raises(JudgeResponseValidationError, match="Could not extract JSON object"):
        client.score(question="q", gold_answer="1", model_output="a", rubric_prompt="rubric")
    assert calls["count"] == 2
    assert slept == []


def test_cerebras_timeout_config_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-timeout")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=150,
        max_retries=3,
        cache_enabled=False,
        min_request_interval_sec=0.0,
    )
    assert float(client._http.timeout.read) == 150.0


def test_cerebras_payload_keeps_json_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-format")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=120,
        max_retries=1,
        cache_enabled=False,
        min_request_interval_sec=0.0,
    )
    seen_payloads: list[dict] = []

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        seen_payloads.append(json)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"clarity":1,"structure":1,"terminology":1,"reasoning_soundness":1,'
                                '"overall_pedagogical_score":4,"is_silent_error":false,"comment":"ok"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    out = client.score(question="q", gold_answer="1", model_output="a", rubric_prompt="rubric")
    assert out["overall_pedagogical_score"] == 4
    assert seen_payloads
    assert seen_payloads[0].get("response_format") == {"type": "json_object"}


def test_cerebras_repair_path_handles_malformed_long_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-long-comment")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=120,
        max_retries=1,
        cache_enabled=False,
        min_request_interval_sec=0.0,
    )
    calls = {"count": 0}

    long_comment = "this is a very long comment " * 8

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        if calls["count"] == 1:
            malformed = (
                'Evaluation: '
                '{"clarity":1,"structure":1,"terminology":1,"reasoning_soundness":1,'
                '"overall_pedagogical_score":4,"is_silent_error":false,'
                f'"comment":"{long_comment}'
            )
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": malformed}}]},
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"clarity":1,"structure":1,"terminology":1,"reasoning_soundness":1,'
                                '"overall_pedagogical_score":4,"is_silent_error":false,'
                                '"comment":"short repaired comment"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    out = client.score(question="q", gold_answer="1", model_output="a", rubric_prompt="rubric")
    assert out["overall_pedagogical_score"] == 4
    assert calls["count"] == 2


def test_cerebras_prompt_contract_mentions_comment_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-contract")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=120,
        max_retries=1,
    )
    system_prompt = client._build_system_prompt("rubric")
    repair_prompt = client._build_format_repair_prompt(previous_raw_content="bad")
    assert "<=120 chars" in system_prompt
    assert "<=80 chars" in repair_prompt
