from __future__ import annotations

import os

import pytest

from didactic_collapse.clients.judge_client import (
    GeminiAuthConfigurationError,
    preflight_validate_gemini_auth,
)


def _clear_env() -> None:
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
        if key in os.environ:
            del os.environ[key]


def test_single_auth_source_env_accepted() -> None:
    _clear_env()
    os.environ["GEMINI_API_KEY"] = "AIzaSyDummyKey123456"

    result = preflight_validate_gemini_auth(api_key_env="GEMINI_API_KEY")
    assert result.selected_source == "env:GEMINI_API_KEY"
    assert result.selected_mode == "api_key"


def test_multiple_auth_sources_rejected() -> None:
    _clear_env()
    os.environ["GEMINI_API_KEY"] = "AIzaSyDummyKey123456"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/creds/adc.json"

    with pytest.raises(GeminiAuthConfigurationError, match="Multiple auth sources"):
        preflight_validate_gemini_auth(api_key_env="GEMINI_API_KEY")


def test_no_auth_source_rejected() -> None:
    _clear_env()
    with pytest.raises(GeminiAuthConfigurationError, match="No API key found"):
        preflight_validate_gemini_auth(api_key_env="GEMINI_API_KEY")


def test_explicit_api_key_preferred() -> None:
    _clear_env()
    result = preflight_validate_gemini_auth(
        api_key_env="GEMINI_API_KEY",
        explicit_api_key="AIzaSyExplicit000000",
    )
    assert result.selected_source == "explicit_api_key"


def test_preflight_diagnostic_masks_key() -> None:
    _clear_env()
    os.environ["GEMINI_API_KEY"] = "AIzaSyVerySecretKey1234"

    result = preflight_validate_gemini_auth(api_key_env="GEMINI_API_KEY")
    assert "..." in result.selected_key_fingerprint
    assert "VerySecret" not in result.selected_key_fingerprint


def test_oauth_like_token_rejected() -> None:
    _clear_env()
    os.environ["GEMINI_API_KEY"] = "AQ.SomeOAuthToken"

    with pytest.raises(GeminiAuthConfigurationError, match="OAuth"):
        preflight_validate_gemini_auth(api_key_env="GEMINI_API_KEY")
