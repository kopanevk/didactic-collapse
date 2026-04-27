from pathlib import Path
import uuid

import pandas as pd
import pytest

from didactic_collapse.pipeline.judge_outputs import run_judging


def _mk_base_dir(prefix: str) -> Path:
    base_dir = Path("outputs/.tmp") / f"{prefix}_{uuid.uuid4().hex}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


class _DummyJudgeClient:
    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        return {
            "clarity": 1,
            "structure": 1,
            "terminology": 1,
            "reasoning_soundness": 1,
            "overall_pedagogical_score": 4,
            "is_silent_error": False,
            "comment": "ok",
        }


class _FlakyJudgeClient:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        key = str(model_output)
        self.calls[key] = self.calls.get(key, 0) + 1
        if "bad" in key:
            raise RuntimeError("simulated row failure")
        return {
            "clarity": 1,
            "structure": 1,
            "terminology": 1,
            "reasoning_soundness": 1,
            "overall_pedagogical_score": 4,
            "is_silent_error": False,
            "comment": "ok",
        }


def _single_generation_row(example_id: str) -> dict:
    return {
        "run_id": "r1",
        "branch": "pure_recycling",
        "generation": 0,
        "model_name": "qwen2.5:0.5b",
        "example_id": example_id,
        "raw_response": "Final answer: 1",
    }


def test_run_judging_fails_on_duplicate_example_ids() -> None:
    generations = pd.DataFrame([_single_generation_row("ex1"), _single_generation_row("ex1")])
    questions = pd.DataFrame([{"example_id": "ex1", "question": "1+0", "answer_gold": "1"}])

    with pytest.raises(ValueError, match="cardinality violation"):
        run_judging(
            client=_DummyJudgeClient(),
            generations_df=generations,
            questions_df=questions,
            judge_provider="cerebras",
            judge_model="llama-3.1-8b",
            rubric_prompt="rubric",
            out_path=_mk_base_dir("judge_unused") / "judge_outputs.parquet",
        )


def test_run_judging_applies_request_pacing_for_real_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    generations = pd.DataFrame([_single_generation_row("ex1"), _single_generation_row("ex2")])
    questions = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex2", "question": "2-1", "answer_gold": "1"},
        ]
    )
    slept: list[float] = []
    monkeypatch.setattr("didactic_collapse.pipeline.judge_outputs.time.sleep", lambda sec: slept.append(sec))

    base_dir = _mk_base_dir("judge_pacing")
    out_df = run_judging(
        client=_DummyJudgeClient(),
        generations_df=generations,
        questions_df=questions,
        judge_provider="cerebras",
        judge_model="llama-3.1-8b",
        rubric_prompt="rubric",
        out_path=base_dir / "judge_outputs.parquet",
        request_delay_sec=0.2,
    )
    assert len(out_df) == 2
    assert slept == [0.2]


def test_run_judging_resume_from_partial_and_dedup_by_example_id() -> None:
    base_dir = _mk_base_dir("judge_resume_contract")
    out_path = base_dir / "judge_outputs.parquet"
    partial_path = base_dir / "judge_partial.parquet"
    failures_path = base_dir / "judge_failures.parquet"
    metadata_path = base_dir / "judge_progress.json"
    base_dir.mkdir(parents=True, exist_ok=True)

    partial_df = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex1",
                "judge_provider": "cerebras",
                "judge_model": "llama-3.1-8b",
                "clarity": 1,
                "structure": 1,
                "terminology": 1,
                "reasoning_soundness": 1,
                "overall_pedagogical_score": 4,
                "is_silent_error": False,
                "comment": "old",
            },
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex1",
                "judge_provider": "cerebras",
                "judge_model": "llama-3.1-8b",
                "clarity": 1,
                "structure": 1,
                "terminology": 1,
                "reasoning_soundness": 1,
                "overall_pedagogical_score": 4,
                "is_silent_error": False,
                "comment": "latest",
            },
        ]
    )
    partial_df.to_parquet(partial_path, index=False)
    pd.DataFrame(
        columns=[
            "run_id",
            "branch",
            "generation",
            "model_name",
            "example_id",
            "judge_provider",
            "judge_model",
            "error_category",
            "error_message",
        ]
    ).to_parquet(failures_path, index=False)

    generations = pd.DataFrame([_single_generation_row("ex1"), _single_generation_row("ex2")])
    questions = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex2", "question": "2-1", "answer_gold": "1"},
        ]
    )

    out_df = run_judging(
        client=_DummyJudgeClient(),
        generations_df=generations,
        questions_df=questions,
        judge_provider="cerebras",
        judge_model="llama-3.1-8b",
        rubric_prompt="rubric",
        out_path=out_path,
        partial_path=partial_path,
        failures_path=failures_path,
        metadata_path=metadata_path,
        partial_save_every_n=1,
        max_row_failures=1,
        continue_on_row_error=True,
    )
    assert len(out_df) == 2
    assert out_df["example_id"].nunique() == 2
    meta = Path(metadata_path).read_text(encoding="utf-8")
    assert "skipped_from_checkpoint" in meta


def test_run_judging_one_bad_row_does_not_kill_stage_under_threshold() -> None:
    generations = pd.DataFrame(
        [
            {**_single_generation_row("ex1"), "raw_response": "good-1"},
            {**_single_generation_row("ex2"), "raw_response": "bad-row"},
            {**_single_generation_row("ex3"), "raw_response": "good-3"},
        ]
    )
    questions = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex2", "question": "2-1", "answer_gold": "1"},
            {"example_id": "ex3", "question": "3-2", "answer_gold": "1"},
        ]
    )
    base_dir = _mk_base_dir("judge_fault_tolerant")

    out_df = run_judging(
        client=_FlakyJudgeClient(),
        generations_df=generations,
        questions_df=questions,
        judge_provider="cerebras",
        judge_model="llama-3.1-8b",
        rubric_prompt="rubric",
        out_path=base_dir / "judge_outputs.parquet",
        partial_save_every_n=1,
        max_row_failures=1,
        continue_on_row_error=True,
    )
    assert len(out_df) == 2
    failures_df = pd.read_parquet(base_dir / "judge_failures.parquet")
    assert len(failures_df) == 1
    assert failures_df.loc[0, "example_id"] == "ex2"


def test_run_judging_threshold_exceeded_fails_explicitly() -> None:
    generations = pd.DataFrame(
        [
            {**_single_generation_row("ex1"), "raw_response": "bad-row-1"},
            {**_single_generation_row("ex2"), "raw_response": "bad-row-2"},
        ]
    )
    questions = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex2", "question": "2-1", "answer_gold": "1"},
        ]
    )
    base_dir = _mk_base_dir("judge_threshold")

    with pytest.raises(RuntimeError, match="threshold exceeded"):
        run_judging(
            client=_FlakyJudgeClient(),
            generations_df=generations,
            questions_df=questions,
            judge_provider="cerebras",
            judge_model="llama-3.1-8b",
            rubric_prompt="rubric",
            out_path=base_dir / "judge_outputs.parquet",
            partial_save_every_n=1,
            max_row_failures=0,
            continue_on_row_error=True,
        )


def test_run_judging_final_artifact_merges_partial_and_remaining() -> None:
    base_dir = _mk_base_dir("judge_merge_contract")
    partial_path = base_dir / "judge_partial.parquet"
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex1",
                "judge_provider": "cerebras",
                "judge_model": "llama-3.1-8b",
                "clarity": 1,
                "structure": 1,
                "terminology": 1,
                "reasoning_soundness": 1,
                "overall_pedagogical_score": 4,
                "is_silent_error": False,
                "comment": "ok",
            }
        ]
    ).to_parquet(partial_path, index=False)

    generations = pd.DataFrame(
        [
            _single_generation_row("ex1"),
            _single_generation_row("ex2"),
            _single_generation_row("ex3"),
        ]
    )
    questions = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex2", "question": "2-1", "answer_gold": "1"},
            {"example_id": "ex3", "question": "3-2", "answer_gold": "1"},
        ]
    )

    out_df = run_judging(
        client=_DummyJudgeClient(),
        generations_df=generations,
        questions_df=questions,
        judge_provider="cerebras",
        judge_model="llama-3.1-8b",
        rubric_prompt="rubric",
        out_path=base_dir / "judge_outputs.parquet",
        partial_path=partial_path,
        failures_path=base_dir / "judge_failures.parquet",
        metadata_path=base_dir / "judge_progress.json",
        partial_save_every_n=2,
        max_row_failures=2,
        continue_on_row_error=True,
    )
    assert len(out_df) == 3
    assert set(out_df["example_id"].tolist()) == {"ex1", "ex2", "ex3"}


def test_run_judging_resume_retries_previous_failures_and_clears_resolved() -> None:
    base_dir = _mk_base_dir("judge_retry_failed_rows")
    out_path = base_dir / "judge_outputs.parquet"
    partial_path = base_dir / "judge_partial.parquet"
    failures_path = base_dir / "judge_failures.parquet"

    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex1",
                "judge_provider": "cerebras",
                "judge_model": "llama-3.1-8b",
                "clarity": 1,
                "structure": 1,
                "terminology": 1,
                "reasoning_soundness": 1,
                "overall_pedagogical_score": 4,
                "is_silent_error": False,
                "comment": "ok",
            }
        ]
    ).to_parquet(partial_path, index=False)
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex2",
                "judge_provider": "cerebras",
                "judge_model": "llama-3.1-8b",
                "error_category": "ConnectError",
                "error_message": "transient",
            }
        ]
    ).to_parquet(failures_path, index=False)

    generations = pd.DataFrame([_single_generation_row("ex1"), _single_generation_row("ex2")])
    questions = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex2", "question": "2-1", "answer_gold": "1"},
        ]
    )

    out_df = run_judging(
        client=_DummyJudgeClient(),
        generations_df=generations,
        questions_df=questions,
        judge_provider="cerebras",
        judge_model="llama-3.1-8b",
        rubric_prompt="rubric",
        out_path=out_path,
        partial_path=partial_path,
        failures_path=failures_path,
        metadata_path=base_dir / "judge_progress.json",
        partial_save_every_n=1,
        max_row_failures=0,
        continue_on_row_error=True,
    )

    assert len(out_df) == 2
    assert set(out_df["example_id"].tolist()) == {"ex1", "ex2"}
    failures_df = pd.read_parquet(failures_path)
    assert failures_df.empty
