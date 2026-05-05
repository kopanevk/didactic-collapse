from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import httpx
import pandas as pd
import pytest

from didactic_collapse.clients.judge_client import build_cerebras_judge_client
from didactic_collapse.pipeline.judge_outputs import run_judging


def _tmp_dir(prefix: str) -> Path:
    base = Path("outputs/.tmp") / f"{prefix}_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _valid_judge_content() -> str:
    return (
        '{"clarity":1,"structure":1,"terminology":1,"reasoning_soundness":1,'
        '"overall_pedagogical_score":4,"is_silent_error":false,"comment":"ok"}'
    )


def test_cerebras_cache_hit_avoids_second_http_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-cache")
    base = _tmp_dir("judge_cache_hit")
    cache_path = base / "judge_cache.sqlite"
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=60,
        max_retries=1,
        cache_enabled=True,
        cache_path=cache_path,
        min_request_interval_sec=0.0,
    )
    calls = {"count": 0}

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": _valid_judge_content()}}]},
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    out1 = client.score(question="q1", gold_answer="1", model_output="a1", rubric_prompt="rubric-v1")
    out2 = client.score(question="q1", gold_answer="1", model_output="a1", rubric_prompt="rubric-v1")

    assert out1["overall_pedagogical_score"] == 4
    assert out2["overall_pedagogical_score"] == 4
    assert calls["count"] == 1
    stats = client.get_runtime_stats_snapshot()
    assert stats["cache_hits"] >= 1


def test_cerebras_cache_key_changes_on_model_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-cache-key")
    base = _tmp_dir("judge_cache_key")
    cache_path = base / "judge_cache.sqlite"
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=60,
        max_retries=1,
        cache_enabled=True,
        cache_path=cache_path,
        min_request_interval_sec=0.0,
    )
    calls = {"count": 0}

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": _valid_judge_content()}}]},
        )

    monkeypatch.setattr(client._http, "post", _fake_post)
    _ = client.score(question="q1", gold_answer="1", model_output="a1", rubric_prompt="rubric-v1")
    _ = client.score(question="q1", gold_answer="1", model_output="a2", rubric_prompt="rubric-v1")
    assert calls["count"] == 2


def test_cerebras_cache_key_changes_on_model_and_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-cache-key2")
    base = _tmp_dir("judge_cache_key_model_rubric")
    cache_path = base / "judge_cache.sqlite"

    client_a = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=60,
        cache_enabled=True,
        cache_path=cache_path,
        min_request_interval_sec=0.0,
    )
    client_b = build_cerebras_judge_client(
        model_name="qwen-3-235b-a22b-instruct-2507",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=60,
        cache_enabled=True,
        cache_path=cache_path,
        min_request_interval_sec=0.0,
    )
    calls = {"count": 0}

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": _valid_judge_content()}}]},
        )

    monkeypatch.setattr(client_a._http, "post", _fake_post)
    monkeypatch.setattr(client_b._http, "post", _fake_post)

    _ = client_a.score(question="q1", gold_answer="1", model_output="a1", rubric_prompt="rubric-v1")
    _ = client_a.score(question="q1", gold_answer="1", model_output="a1", rubric_prompt="rubric-v2")
    _ = client_b.score(question="q1", gold_answer="1", model_output="a1", rubric_prompt="rubric-v1")
    assert calls["count"] == 3


def test_retry_after_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-live-retry-after")
    client = build_cerebras_judge_client(
        model_name="llama-3.1-8b",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        timeout_sec=60,
        max_retries=1,
        max_429_retries=2,
        max_retry_after_sec=5.0,
        min_request_interval_sec=0.0,
        cache_enabled=False,
    )
    calls = {"count": 0}
    sleeps: list[float] = []

    def _fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    def _fake_post(url: str, json: dict) -> SimpleNamespace:  # noqa: A002
        calls["count"] += 1
        if calls["count"] == 1:
            request = httpx.Request("POST", url)
            response = httpx.Response(status_code=429, request=request, headers={"Retry-After": "120"})
            return SimpleNamespace(
                raise_for_status=lambda: (_ for _ in ()).throw(
                    httpx.HTTPStatusError("429", request=request, response=response)
                ),
                json=lambda: {},
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": _valid_judge_content()}}]},
        )

    monkeypatch.setattr("didactic_collapse.clients.judge_client.time.sleep", _fake_sleep)
    monkeypatch.setattr(client._http, "post", _fake_post)

    out = client.score(question="q1", gold_answer="1", model_output="a1", rubric_prompt="rubric-v1")
    assert out["overall_pedagogical_score"] == 4
    assert calls["count"] == 2
    assert any(sec <= 5.0 and sec >= 1.0 for sec in sleeps)


class _CountingJudgeClient:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        self.calls += 1
        return {
            "clarity": 1,
            "structure": 1,
            "terminology": 1,
            "reasoning_soundness": 1,
            "overall_pedagogical_score": 4,
            "is_silent_error": False,
            "comment": "ok",
        }

    def get_runtime_stats_snapshot(self) -> dict[str, float | int]:
        return {
            "cache_hits": 3,
            "cache_misses": 2,
            "api_calls": self.calls,
            "rate_limit_retries": 1,
            "total_sleep_sec_due_to_rate_limit": 2.5,
            "pacing_sleeps": 0,
            "total_sleep_sec_due_to_pacing": 0.0,
        }


def test_run_judging_dedup_and_progress_stats_written() -> None:
    base = _tmp_dir("judge_dedup_stats")
    generations_df = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex1",
                "raw_response": "Final answer: 1",
            },
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex2",
                "raw_response": "Final answer: 1",
            },
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex3",
                "raw_response": "Final answer: 3",
            },
        ]
    )
    questions_df = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex2", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex3", "question": "2+1", "answer_gold": "3"},
        ]
    )
    client = _CountingJudgeClient()
    out = run_judging(
        client=client,
        generations_df=generations_df,
        questions_df=questions_df,
        judge_provider="cerebras",
        judge_model="llama-3.1-8b",
        rubric_prompt="rubric-v1",
        out_path=base / "judge_outputs.parquet",
        metadata_path=base / "judge_progress.json",
        partial_save_every_n=1,
        continue_on_row_error=True,
    )
    assert len(out) == 3
    # ex1/ex2 have identical judge payload, so only two client calls.
    assert client.calls == 2
    progress = json.loads((base / "judge_progress.json").read_text(encoding="utf-8"))
    assert progress["unique_judge_inputs"] == 2
    assert progress["duplicate_count"] == 1
    assert progress["api_calls_saved_by_dedupe"] == 1
    assert progress["cache_hits"] == 3
    assert progress["rate_limit_retries"] == 1
