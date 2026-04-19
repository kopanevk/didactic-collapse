from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from didactic_collapse.clients.judge_client import (
    CerebrasAuthConfigurationError,
    CerebrasJudgeClient,
    GeminiAuthConfigurationError,
    build_cerebras_judge_client,
    cerebras_judge_rubric_format_check,
    gemini_judge_auth_smoke_check,
    parse_and_validate_judge_response,
    JudgeResponseValidationError,
    preflight_validate_cerebras_auth,
    preflight_validate_gemini_auth,
)


def _clear_env() -> None:
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "CEREBRAS_API_KEY",
    ):
        if key in os.environ:
            del os.environ[key]


def test_single_gemini_source_accepted() -> None:
    _clear_env()
    os.environ["GEMINI_API_KEY"] = "AQ.SomeAiStudioKeyLikeToken"
    result = preflight_validate_gemini_auth(api_key_env="GEMINI_API_KEY")
    assert result.selected_source == "env:GEMINI_API_KEY"


def test_multiple_gemini_sources_rejected() -> None:
    _clear_env()
    os.environ["GEMINI_API_KEY"] = "key_1"
    os.environ["GOOGLE_API_KEY"] = "key_2"
    with pytest.raises(GeminiAuthConfigurationError, match="Multiple API key sources"):
        preflight_validate_gemini_auth(api_key_env="GEMINI_API_KEY")


def test_cerebras_missing_key_rejected() -> None:
    _clear_env()
    with pytest.raises(CerebrasAuthConfigurationError, match="Missing Cerebras API key"):
        preflight_validate_cerebras_auth(api_key_env="CEREBRAS_API_KEY")


def test_cerebras_key_present_accepted() -> None:
    _clear_env()
    os.environ["CEREBRAS_API_KEY"] = "csk-live-verysecret"
    result = preflight_validate_cerebras_auth(api_key_env="CEREBRAS_API_KEY")
    assert result.selected_source == "env:CEREBRAS_API_KEY"
    assert "verysecret" not in result.selected_key_fingerprint.lower()


def test_cerebras_provider_output_uses_existing_validation_layer() -> None:
    parsed = parse_and_validate_judge_response(
        '{"clarity":2,"structure":2,"terminology":2,'
        '"reasoning_soundness":2,"overall_pedagogical_score":8,'
        '"is_silent_error":false,"comment":"ok"}'
    )
    assert parsed.score.overall_pedagogical_score == 8


def test_cerebras_type_object_extra_field_is_safely_repaired() -> None:
    parsed = parse_and_validate_judge_response(
        '{"type":"object","clarity":1,"structure":0,"terminology":0,'
        '"reasoning_soundness":0,"overall_pedagogical_score":1,'
        '"is_silent_error":false,"comment":"ok"}'
    )
    assert parsed.score.overall_pedagogical_score == 1
    assert parsed.repair_applied is True
    assert "dropped_type_object_field" in parsed.repair_actions


def test_cerebras_valid_rubric_json_passes_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env()
    os.environ["CEREBRAS_API_KEY"] = "csk-live-verysecret"
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=10,
    )

    calls = {"count": 0}

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        assert url.endswith("/chat/completions")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"clarity":2,"structure":2,"terminology":2,'
                                '"reasoning_soundness":2,"overall_pedagogical_score":8,'
                                '"is_silent_error":false,"comment":"ok"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    score = client.score(
        question="q",
        gold_answer="1",
        model_output="a",
        rubric_prompt="rubric",
    )
    assert score["overall_pedagogical_score"] == 8
    assert calls["count"] == 1


def test_cerebras_chatty_output_fails_without_retry_storm(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _clear_env()
    os.environ["CEREBRAS_API_KEY"] = "csk-live-verysecret"
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=10,
    )
    calls = {"count": 0}

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": "Great answer! The student seems correct."}}]
            },
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    caplog.set_level("WARNING")

    with pytest.raises(JudgeResponseValidationError, match="Could not extract JSON"):
        client.score(
            question="q",
            gold_answer="1",
            model_output="a",
            rubric_prompt="rubric",
        )

    assert calls["count"] == 1
    assert "judge_validation_failed" in caplog.text
    assert "preview=Great answer!" in caplog.text


def test_validation_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env()
    os.environ["CEREBRAS_API_KEY"] = "csk-live-verysecret"
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=10,
    )
    calls = {"count": 0}

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '{"foo":"bar"}'}}]},
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    with pytest.raises(JudgeResponseValidationError, match="Missing required fields"):
        client.score(
            question="q",
            gold_answer="1",
            model_output="a",
            rubric_prompt="rubric",
        )
    assert calls["count"] == 1


def test_build_cerebras_client_uses_explicit_provider_class() -> None:
    _clear_env()
    os.environ["CEREBRAS_API_KEY"] = "csk-live-verysecret"
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=10,
    )
    assert isinstance(client, CerebrasJudgeClient)


def test_smoke_and_pipeline_share_gemini_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    class _DummyClient:
        def smoke_check(self) -> str:
            return '{"ok":true}'

    def _fake_build(*, model_name: str, api_key_env: str = "GEMINI_API_KEY", explicit_api_key: str | None = None):
        calls.append((model_name, api_key_env))
        return _DummyClient()

    monkeypatch.setattr("didactic_collapse.clients.judge_client.build_gemini_judge_client", _fake_build)
    _ = gemini_judge_auth_smoke_check(model_name="gemini-2.5-flash", api_key_env="GEMINI_API_KEY")
    assert calls == [("gemini-2.5-flash", "GEMINI_API_KEY")]


def test_rubric_check_uses_same_cerebras_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class _DummyClient:
        def rubric_format_check(self):
            return parse_and_validate_judge_response(
                '{"clarity":0,"structure":0,"terminology":0,'
                '"reasoning_soundness":0,"overall_pedagogical_score":0,'
                '"is_silent_error":false,"comment":"ok"}'
            )

    def _fake_build(
        *,
        model_name: str,
        base_url: str,
        api_key_env: str = "CEREBRAS_API_KEY",
        timeout_sec: int = 60,
    ):
        calls.append((model_name, base_url, api_key_env))
        return _DummyClient()

    monkeypatch.setattr("didactic_collapse.clients.judge_client.build_cerebras_judge_client", _fake_build)
    out = cerebras_judge_rubric_format_check(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=30,
    )
    assert out["overall_pedagogical_score"] == 0
    assert calls == [("llama-3.1-8b", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY")]
