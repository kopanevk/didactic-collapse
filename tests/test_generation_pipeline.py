from pathlib import Path
import uuid

import pandas as pd
import pytest

from didactic_collapse.pipeline.generate_outputs import build_generation_prompt, run_generation


def _mk_base_dir(prefix: str) -> Path:
    base_dir = Path("outputs/.tmp") / f"{prefix}_{uuid.uuid4().hex}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


class _DummyClient:
    def generate(self, *, model_name: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        return "Reasoning here.\nFinal answer: 1"


class _FlakyClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, model_name: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        self.calls += 1
        if "fail_case" in prompt:
            raise RuntimeError("simulated generation failure")
        return "Reasoning here.\nFinal answer: 1"


def test_generation_fails_on_missing_required_columns() -> None:
    examples = pd.DataFrame([{"example_id": "ex1"}])
    with pytest.raises(ValueError, match="missing required columns"):
        run_generation(
            client=_DummyClient(),
            examples_df=examples,
            model_name="qwen2.5:0.5b",
            branch="pure_recycling",
            generation=0,
            run_id="run",
            prompt_version="v1",
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
            out_path=_mk_base_dir("generation_missing_cols") / "model_outputs.parquet",
        )


def test_generation_fails_on_duplicate_example_ids() -> None:
    examples = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+1"},
            {"example_id": "ex1", "question": "2+2"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate example_id"):
        run_generation(
            client=_DummyClient(),
            examples_df=examples,
            model_name="qwen2.5:0.5b",
            branch="pure_recycling",
            generation=0,
            run_id="run",
            prompt_version="v1",
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
            out_path=_mk_base_dir("generation_dup_ids") / "model_outputs.parquet",
        )


def test_build_generation_prompt_v2_enforces_final_answer_line() -> None:
    prompt = build_generation_prompt(question="What is 1+1?", prompt_version="v2_strict_final")
    assert "Final answer: <number>" in prompt
    assert "Do not add any text after the final line." in prompt


def test_generation_v2_prompt_downstream_extract_compatibility() -> None:
    base_dir = _mk_base_dir("generation_v2")
    examples = pd.DataFrame([{"example_id": "ex1", "question": "1+0"}])
    out = run_generation(
        client=_DummyClient(),
        examples_df=examples,
        model_name="qwen2.5:0.5b",
        branch="pure_recycling",
        generation=0,
        run_id="run",
        prompt_version="v2_strict_final",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        out_path=base_dir / "model_outputs.parquet",
    )
    assert out.loc[0, "parsed_final_answer"] is not None
    assert "1" in str(out.loc[0, "parsed_final_answer"])
    assert out.loc[0, "prompt_version"] == "v2_strict_final"
    assert "Final answer: <number>" in out.loc[0, "prompt_text"]


def test_generation_resume_from_partial_skips_completed_example_ids() -> None:
    base_dir = _mk_base_dir("generation_resume")
    partial_path = base_dir / "generation_partial.parquet"
    pd.DataFrame(
        [
            {
                "run_id": "run",
                "branch": "pure_recycling",
                "generation": 0,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex1",
                "prompt_version": "v2_strict_final",
                "prompt_text": "p",
                "raw_response": "Reasoning here.\nFinal answer: 1",
                "parsed_final_answer": "1",
            }
        ]
    ).to_parquet(partial_path, index=False)
    pd.DataFrame(
        columns=[
            "run_id",
            "branch",
            "generation",
            "model_name",
            "example_id",
            "error_category",
            "error_message",
        ]
    ).to_parquet(base_dir / "generation_failures.parquet", index=False)

    examples = pd.DataFrame(
        [{"example_id": "ex1", "question": "1+0"}, {"example_id": "ex2", "question": "2-1"}]
    )
    out = run_generation(
        client=_DummyClient(),
        examples_df=examples,
        model_name="qwen2.5:0.5b",
        branch="pure_recycling",
        generation=0,
        run_id="run",
        prompt_version="v2_strict_final",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        out_path=base_dir / "model_outputs.parquet",
        partial_path=partial_path,
        failures_path=base_dir / "generation_failures.parquet",
        metadata_path=base_dir / "generation_progress.json",
        partial_save_every_n=1,
    )
    assert len(out) == 2
    assert set(out["example_id"].tolist()) == {"ex1", "ex2"}


def test_generation_continues_on_single_row_failure_under_threshold() -> None:
    examples = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "ok_case"},
            {"example_id": "ex2", "question": "fail_case"},
            {"example_id": "ex3", "question": "ok_case_2"},
        ]
    )
    client = _FlakyClient()
    base_dir = _mk_base_dir("generation_fault_tolerant")

    out = run_generation(
        client=client,
        examples_df=examples,
        model_name="qwen2.5:0.5b",
        branch="pure_recycling",
        generation=0,
        run_id="run",
        prompt_version="v2_strict_final",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        out_path=base_dir / "model_outputs.parquet",
        partial_save_every_n=1,
        max_row_failures=1,
        continue_on_row_error=True,
    )
    assert len(out) == 2
    failures_df = pd.read_parquet(base_dir / "generation_failures.parquet")
    assert len(failures_df) == 1
    assert failures_df.loc[0, "example_id"] == "ex2"
