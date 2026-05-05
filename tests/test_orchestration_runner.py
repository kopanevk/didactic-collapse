from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.runner import (
    CONTEXT_STAGES,
    ExperimentRunner,
    OrchestrationError,
    StageContext,
    StageExecutionResult,
)


def _make_cfg(tmp_path: Path) -> AppConfig:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    prompt_dir = tmp_path / "configs" / "prompts"
    split_dir = data_root / "splits"

    split_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "judge_system.txt").write_text("judge prompt", encoding="utf-8")

    base = pd.DataFrame([{"example_id": "b1", "question": "q", "answer_gold": "1"}])
    heldout = pd.DataFrame([{"example_id": "h1", "question": "q", "answer_gold": "1"}])
    anchor = pd.DataFrame([{"example_id": "a1", "question": "q", "answer_gold": "1"}])
    base.to_parquet(split_dir / "base_train.parquet", index=False)
    heldout.to_parquet(split_dir / "heldout_test.parquet", index=False)
    anchor.to_parquet(split_dir / "anchor_pool.parquet", index=False)
    (split_dir / "split_metadata.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    cfg_dict = {
        "project": {"name": "dc", "seed": 42, "run_tag": "test"},
        "paths": {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "prompt_dir": str(prompt_dir),
        },
        "models": {"local_models": [{"name": "qwen2.5:0.5b", "role": "subject"}]},
        "judge": {
            "provider": "gemini_openai_compatible",
            "model_name": "gemini-2.5-flash",
            "base_url": "https://example.org/openai",
            "api_key_env": "DUMMY_KEY",
            "timeout_sec": 60,
            "max_retries": 1,
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16},
        "experiment": {
            "generations": 1,
            "branches": [{"name": "pure_recycling", "anchor_ratio": 0.0}],
        },
        "dataset": {
            "source": "gsm8k",
            "base_train_size": 1,
            "anchor_pool_size": 1,
            "heldout_test_size": 1,
        },
        "runtime": {"force_recompute": False, "save_parquet": True, "save_csv": True},
    }
    return AppConfig.model_validate(cfg_dict)


def _make_cerebras_cfg(tmp_path: Path) -> AppConfig:
    cfg = _make_cfg(tmp_path)
    payload = cfg.model_dump(mode="python")
    payload["judge"]["provider"] = "cerebras"
    payload["judge"]["base_url"] = "https://api.cerebras.ai/v1"
    payload["judge"]["api_key_env"] = "CEREBRAS_API_KEY"
    payload["judge"]["model_name"] = "llama-3.1-8b"
    return AppConfig.model_validate(payload)


def _mk_local_temp_dir(prefix: str) -> Path:
    root = Path.cwd() / "test_workdirs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _executor_factory(counter: dict[str, int]):
    def _exec(ctx: StageContext) -> StageExecutionResult:
        counter[ctx.stage_name] = counter.get(ctx.stage_name, 0) + 1
        if ctx.stage_name == "generation":
            df = pd.DataFrame([
                {
                    "example_id": "h1",
                    "raw_response": "Final answer: 1",
                    "parsed_final_answer": "1",
                }
            ])
            ctx.artifacts["model_outputs"].parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ctx.artifacts["model_outputs"], index=False)
            df.to_parquet(ctx.artifacts["generation_partial"], index=False)
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
            ).to_parquet(ctx.artifacts["generation_failures"], index=False)
            ctx.artifacts["generation_metadata"].write_text(
                json.dumps({"stage": "generation", "total_rows": 1, "completed_rows": 1, "failed_rows": 0}),
                encoding="utf-8",
            )
            return StageExecutionResult(row_count=1)

        if ctx.stage_name == "answer_extraction":
            df = pd.DataFrame([
                {
                    "example_id": "h1",
                    "raw_response": "Final answer: 1",
                    "parsed_final_answer": "1",
                    "normalized_predicted": "1",
                }
            ])
            df.to_parquet(ctx.artifacts["answer_extraction"], index=False)
            return StageExecutionResult(row_count=1)

        if ctx.stage_name == "accuracy":
            df = pd.DataFrame([
                {
                    "example_id": "h1",
                    "is_correct": True,
                }
            ])
            df.to_parquet(ctx.artifacts["accuracy_table"], index=False)
            return StageExecutionResult(row_count=1)

        if ctx.stage_name == "judge":
            judge_df = pd.DataFrame([
                {
                    "example_id": "h1",
                    "overall_pedagogical_score": 8,
                    "is_silent_error": False,
                }
            ])
            eval_df = pd.DataFrame([
                {
                    "example_id": "h1",
                    "is_correct": True,
                    "overall_pedagogical_score": 8,
                    "is_silent_error": False,
                }
            ])
            judge_df.to_parquet(ctx.artifacts["judge_outputs"], index=False)
            judge_df.to_parquet(ctx.artifacts["judge_partial"], index=False)
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
            ).to_parquet(ctx.artifacts["judge_failures"], index=False)
            ctx.artifacts["judge_metadata"].write_text(
                json.dumps({"stage": "judge", "total_rows": 1, "completed_rows": 1, "failed_rows": 0}),
                encoding="utf-8",
            )
            eval_df.to_parquet(ctx.artifacts["eval_merged"], index=False)
            return StageExecutionResult(row_count=1)

        if ctx.stage_name == "synthetic_build":
            df = pd.DataFrame([
                {
                    "example_id": "h1",
                    "question": "q",
                    "answer_for_training": "Final answer: 1",
                    "source": "synthetic",
                }
            ])
            df.to_parquet(ctx.artifacts["synthetic_base"], index=False)
            return StageExecutionResult(row_count=1)

        if ctx.stage_name == "anchoring":
            df = pd.DataFrame([
                {
                    "example_id": "h1",
                    "question": "q",
                    "answer_for_training": "Final answer: 1",
                    "source": "synthetic",
                }
            ])
            df.to_parquet(ctx.artifacts["synthetic_train_next"], index=False)
            ctx.artifacts["anchor_metadata"].write_text(
                json.dumps(
                    {
                        "model_name": ctx.model_name,
                        "branch": ctx.branch,
                        "generation": ctx.generation,
                        "seed": ctx.seed,
                        "anchor_ratio_requested": 0.0,
                        "anchor_ratio_realized": 0.0,
                        "synthetic_count": 1,
                        "anchor_count": 0,
                        "total_count": 1,
                        "remaining_anchor_pool_size": 1,
                        "selected_anchor_ids": [],
                        "reused_anchor_ids": [],
                    }
                ),
                encoding="utf-8",
            )
            ctx.artifacts["used_anchor_ids"].write_text("[]", encoding="utf-8")
            return StageExecutionResult(row_count=1)

        raise AssertionError(f"unexpected stage {ctx.stage_name}")

    return _exec


def test_completed_stage_is_skipped(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    counter: dict[str, int] = {}
    runner = ExperimentRunner(
        cfg,
        run_dir=tmp_path / "run",
        stage_executors={"generation": _executor_factory(counter)},
    )

    runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
    runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)

    assert counter.get("generation") == 1


def test_force_rerun_works(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    counter: dict[str, int] = {}
    runner = ExperimentRunner(
        cfg,
        run_dir=tmp_path / "run",
        stage_executors={"generation": _executor_factory(counter)},
    )

    runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
    runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1, force=True)

    assert counter.get("generation") == 2


def test_resume_continues_from_next_incomplete_stage(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    counter: dict[str, int] = {}
    exec_fn = _executor_factory(counter)
    runner = ExperimentRunner(
        cfg,
        run_dir=tmp_path / "run",
        stage_executors={stage: exec_fn for stage in CONTEXT_STAGES},
    )

    runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
    runner.resume_from_checkpoint(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)

    assert counter.get("generation") == 1
    assert counter.get("answer_extraction") == 1
    assert counter.get("accuracy") == 1
    assert counter.get("judge") == 1
    assert counter.get("synthetic_build") == 1
    assert counter.get("anchoring") == 1


def test_resume_cleans_stale_outputs_for_pending_synthetic_and_anchoring() -> None:
    base = _mk_local_temp_dir("runner_stale_nonrow")
    try:
        cfg = _make_cfg(base)
        counter: dict[str, int] = {}
        exec_fn = _executor_factory(counter)
        runner = ExperimentRunner(
            cfg,
            run_dir=base / "run",
            stage_executors={stage: exec_fn for stage in CONTEXT_STAGES},
        )

        for stage in ("generation", "answer_extraction", "accuracy", "judge", "synthetic_build", "anchoring"):
            runner.run_stage(stage, model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)

        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
        manifest_path = step_dir / "stage_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["stages"]["synthetic_build"]["status"] = "pending"
        payload["stages"]["synthetic_build"]["timestamp_end"] = None
        payload["stages"]["anchoring"]["status"] = "pending"
        payload["stages"]["anchoring"]["timestamp_end"] = None
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        assert (step_dir / "synthetic_base.parquet").exists()
        assert (step_dir / "synthetic_train_next.parquet").exists()

        runner.resume_from_checkpoint(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)

        assert counter.get("synthetic_build") == 2
        assert counter.get("anchoring") == 2
        assert (step_dir / "synthetic_base.parquet").exists()
        assert (step_dir / "synthetic_train_next.parquet").exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_resume_cleans_stale_outputs_for_pending_answer_extraction() -> None:
    base = _mk_local_temp_dir("runner_stale_extract")
    try:
        cfg = _make_cfg(base)
        counter: dict[str, int] = {}
        exec_fn = _executor_factory(counter)
        runner = ExperimentRunner(
            cfg,
            run_dir=base / "run",
            stage_executors={stage: exec_fn for stage in CONTEXT_STAGES},
        )

        for stage in ("generation", "answer_extraction", "accuracy", "judge", "synthetic_build", "anchoring"):
            runner.run_stage(stage, model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)

        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
        manifest_path = step_dir / "stage_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["stages"]["answer_extraction"]["status"] = "pending"
        payload["stages"]["answer_extraction"]["timestamp_end"] = None
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        assert (step_dir / "answer_extraction.parquet").exists()
        runner.resume_from_checkpoint(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)

        assert counter.get("answer_extraction") == 2
        # Downstream stages remain completed and are skipped unless explicitly
        # marked incomplete in manifest.
        assert counter.get("accuracy") == 1
        assert counter.get("judge") == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_corrupted_manifest_causes_explicit_failure(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    runner = ExperimentRunner(cfg, run_dir=tmp_path / "run", stage_executors={})

    step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "stage_manifest.json").write_text("{not_valid_json", encoding="utf-8")

    with pytest.raises(OrchestrationError, match="Corrupted manifest"):
        runner.resume_from_checkpoint(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)


def test_missing_artifact_causes_explicit_failure(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    counter: dict[str, int] = {}
    runner = ExperimentRunner(
        cfg,
        run_dir=tmp_path / "run",
        stage_executors={"generation": _executor_factory(counter)},
    )

    runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
    outputs_path = runner._step_dir(
        model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1
    ) / "model_outputs.parquet"
    outputs_path.unlink()

    with pytest.raises(OrchestrationError, match="Missing artifact file"):
        runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)


def test_lineage_mismatch_causes_explicit_failure(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    counter: dict[str, int] = {}
    runner = ExperimentRunner(
        cfg,
        run_dir=tmp_path / "run",
        stage_executors={"generation": _executor_factory(counter)},
    )

    runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
    manifest_path = runner._step_dir(
        model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1
    ) / "stage_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["stages"]["generation"]["model_name"] = "another_model"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OrchestrationError, match="Lineage mismatch"):
        runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)


def test_provider_selection_for_cerebras_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cerebras_cfg(tmp_path)
    runner = ExperimentRunner(cfg, run_dir=tmp_path / "run", stage_executors={})
    calls: list[tuple[str, str, str]] = []

    class _DummyJudgeClient:
        def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
            return {
                "clarity": 0,
                "structure": 0,
                "terminology": 0,
                "reasoning_soundness": 0,
                "overall_pedagogical_score": 0,
                "is_silent_error": False,
                "comment": "dummy",
            }

    def _fake_build(
        *,
        model_name: str,
        base_url: str,
        api_key_env: str = "CEREBRAS_API_KEY",
        timeout_sec: int = 60,
        max_retries: int = 3,
        max_completion_tokens: int = 180,
        comment_max_chars: int = 120,
        cache_enabled: bool = True,
        cache_path: str | None = None,
        min_request_interval_sec: float = 0.8,
        max_retry_after_sec: float = 60.0,
        max_429_retries: int = 6,
        jitter_sec: float = 0.75,
    ):
        calls.append((model_name, base_url, api_key_env))
        return _DummyJudgeClient()

    monkeypatch.setattr("didactic_collapse.orchestration.runner.build_cerebras_judge_client", _fake_build)
    _ = runner._get_judge_client()
    assert calls == [("llama-3.1-8b", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY")]


def test_synthetic_build_uses_heldout_question_column_contract(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    def _answer_extraction_exec(ctx: StageContext) -> StageExecutionResult:
        df = pd.DataFrame(
            [
                {
                    "example_id": "h1",
                    "raw_response": "Final answer: 1",
                }
            ]
        )
        ctx.artifacts["answer_extraction"].parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ctx.artifacts["answer_extraction"], index=False)
        return StageExecutionResult(row_count=1)

    runner = ExperimentRunner(
        cfg,
        run_dir=tmp_path / "run",
        stage_executors={"answer_extraction": _answer_extraction_exec},
    )

    runner.run_stage("answer_extraction", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
    runner.run_stage("synthetic_build", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)

    out_path = runner._step_dir(
        model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1
    ) / "synthetic_base.parquet"
    out_df = pd.read_parquet(out_path)
    assert out_df.loc[0, "question"] == "q"


def test_judge_stage_row_level_resume_reuses_partial_progress() -> None:
    base = Path("outputs/.tmp") / f"runner_row_resume_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    cfg = _make_cfg(base)
    cfg.runtime.continue_on_row_error = True
    cfg.runtime.partial_save_every_n = 1
    cfg.runtime.max_row_failures = 0

    split_dir = cfg.paths.data_root / "splits"
    heldout = pd.DataFrame(
        [
            {"example_id": "ex1", "question": "1+0", "answer_gold": "1"},
            {"example_id": "ex2", "question": "2-1", "answer_gold": "1"},
            {"example_id": "ex3", "question": "3-2", "answer_gold": "1"},
        ]
    )
    heldout.to_parquet(split_dir / "heldout_test.parquet", index=False)

    runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
    step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
    step_dir.mkdir(parents=True, exist_ok=True)
    outputs = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 1,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex1",
                "raw_response": "good-1",
            },
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 1,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex2",
                "raw_response": "bad-2",
            },
            {
                "run_id": "r1",
                "branch": "pure_recycling",
                "generation": 1,
                "model_name": "qwen2.5:0.5b",
                "example_id": "ex3",
                "raw_response": "good-3",
            },
        ]
    )
    outputs.to_parquet(step_dir / "model_outputs.parquet", index=False)
    pd.DataFrame(
        [
            {"example_id": "ex1", "is_correct": True},
            {"example_id": "ex2", "is_correct": False},
            {"example_id": "ex3", "is_correct": True},
        ]
    ).to_parquet(step_dir / "accuracy_table.parquet", index=False)

    class _FailingJudge:
        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
            self.calls[model_output] = self.calls.get(model_output, 0) + 1
            if "bad-2" in model_output:
                raise RuntimeError("simulated bad row")
            return {
                "clarity": 1,
                "structure": 1,
                "terminology": 1,
                "reasoning_soundness": 1,
                "overall_pedagogical_score": 4,
                "is_silent_error": False,
                "comment": "ok",
            }

    judge = _FailingJudge()
    runner._judge_client = judge
    runner._heldout_df = heldout
    runner._judge_prompt = "rubric"

    try:
        with pytest.raises(RuntimeError, match="threshold exceeded"):
            runner.run_stage("judge", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)

        assert (step_dir / "judge_partial.parquet").exists()
        assert (step_dir / "judge_failures.parquet").exists()
        assert judge.calls == {"good-1": 1, "bad-2": 1}

        cfg.runtime.max_row_failures = 1
        runner.run_stage("judge", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
        assert judge.calls == {"good-1": 1, "bad-2": 2, "good-3": 1}

        merged = pd.read_parquet(step_dir / "eval_merged.parquet")
        assert set(merged["example_id"].tolist()) == {"ex1", "ex3"}
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_anchoring_stage_pvf_branch_writes_filter_artifacts() -> None:
    base = _mk_local_temp_dir("runner_pvf")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["experiment"]["branches"] = [
            {
                "name": "pvf_medium",
                "branch_type": "pvf_medium",
                "anchor_ratio": 0.0,
                "mixing_mode": "append",
                "pvf_threshold_score": 5,
                "pvf_min_keep_ratio": 0.2,
            }
        ]
        cfg = AppConfig.model_validate(payload)

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="pvf_medium", generation=1)
        step_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"example_id": "h1", "question": "q", "answer_for_training": "A", "source": "synthetic"},
                {"example_id": "h2", "question": "q2", "answer_for_training": "B", "source": "synthetic"},
            ]
        ).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "h1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
                {"example_id": "h2", "pred_parse_success": False, "accuracy_label": "wrong", "is_correct": False},
            ]
        ).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "h1", "overall_pedagogical_score": 6, "is_silent_error": False},
                {"example_id": "h2", "overall_pedagogical_score": 2, "is_silent_error": True},
            ]
        ).to_parquet(step_dir / "judge_outputs.parquet", index=False)

        runner.run_stage("anchoring", model_name="qwen2.5:0.5b", branch="pvf_medium", generation=1)

        assert (step_dir / "filtered_training_dataset.parquet").exists()
        assert (step_dir / "rejected_examples.parquet").exists()
        assert (step_dir / "pvf_filter_report.json").exists()
        out_df = pd.read_parquet(step_dir / "synthetic_train_next.parquet")
        assert set(out_df["example_id"].tolist()) == {"h1"}
        meta = json.loads((step_dir / "anchor_selection_manifest.json").read_text(encoding="utf-8"))
        assert meta["method"] == "pvf_medium"
        assert json.loads((step_dir / "used_anchor_ids.json").read_text(encoding="utf-8")) == []
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_anchoring_stage_human_branch_still_works_after_pvf_integration() -> None:
    base = _mk_local_temp_dir("runner_anchor")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["experiment"]["branches"] = [
            {
                "name": "anchor_50_append",
                "branch_type": "human_anchoring",
                "anchor_ratio": 0.5,
                "mixing_mode": "append",
            }
        ]
        cfg = AppConfig.model_validate(payload)

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="anchor_50_append", generation=1)
        step_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"example_id": "s1", "question": "q1", "answer_for_training": "A1", "source": "synthetic"},
                {"example_id": "s2", "question": "q2", "answer_for_training": "A2", "source": "synthetic"},
            ]
        ).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "s1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
                {"example_id": "s2", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            ]
        ).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "s1", "overall_pedagogical_score": 6, "is_silent_error": False},
                {"example_id": "s2", "overall_pedagogical_score": 6, "is_silent_error": False},
            ]
        ).to_parquet(step_dir / "judge_outputs.parquet", index=False)

        runner.run_stage("anchoring", model_name="qwen2.5:0.5b", branch="anchor_50_append", generation=1)

        out_df = pd.read_parquet(step_dir / "synthetic_train_next.parquet")
        assert len(out_df) == 3  # 2 synthetic + 1 anchor (append, ratio 0.5 on n=2)
        assert int((out_df["source"] == "human_anchor").sum()) == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_anchoring_stage_pvf_tolerates_missing_judge_rows_with_continue_on_error() -> None:
    base = _mk_local_temp_dir("runner_pvf_partial")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["runtime"]["continue_on_row_error"] = True
        payload["experiment"]["branches"] = [
            {
                "name": "pvf_medium",
                "branch_type": "pvf_medium",
                "anchor_ratio": 0.0,
                "mixing_mode": "append",
                "pvf_threshold_score": 5,
                "pvf_min_keep_ratio": 0.2,
            }
        ]
        cfg = AppConfig.model_validate(payload)

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="pvf_medium", generation=1)
        step_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"example_id": "h1", "question": "q", "answer_for_training": "A", "source": "synthetic"},
                {"example_id": "h2", "question": "q2", "answer_for_training": "B", "source": "synthetic"},
            ]
        ).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "h1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
                {"example_id": "h2", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            ]
        ).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        # h2 intentionally absent in judge to emulate row-level judge failure.
        pd.DataFrame(
            [
                {"example_id": "h1", "overall_pedagogical_score": 6, "is_silent_error": False},
            ]
        ).to_parquet(step_dir / "judge_outputs.parquet", index=False)

        runner.run_stage("anchoring", model_name="qwen2.5:0.5b", branch="pvf_medium", generation=1)

        out_df = pd.read_parquet(step_dir / "synthetic_train_next.parquet")
        assert set(out_df["example_id"].tolist()) == {"h1"}
        rejected = pd.read_parquet(step_dir / "rejected_examples.parquet")
        reject_h2 = rejected.loc[rejected["example_id"] == "h2", "pvf_reject_reasons"].astype(str).tolist()
        assert reject_h2 and "missing_judge_row" in reject_h2[0]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_anchoring_stage_soft_pvf_branch_writes_soft_artifacts() -> None:
    base = _mk_local_temp_dir("runner_soft_pvf")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["experiment"]["branches"] = [
            {
                "name": "soft_pvf_medium",
                "branch_type": "soft_pvf_medium",
                "anchor_ratio": 0.0,
                "mixing_mode": "append",
                "soft_pvf_high_quality_threshold": 6,
                "soft_pvf_medium_quality_threshold": 4,
                "soft_pvf_weight_high": 1.0,
                "soft_pvf_weight_medium": 0.5,
                "soft_pvf_weight_low_correct": 0.25,
                "soft_pvf_weight_incorrect": 0.1,
                "soft_pvf_min_keep_ratio": 0.1,
            }
        ]
        cfg = AppConfig.model_validate(payload)

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="soft_pvf_medium", generation=1)
        step_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"example_id": "s1", "question": "q1", "answer_for_training": "A1", "source": "synthetic"},
                {"example_id": "s2", "question": "q2", "answer_for_training": "A2", "source": "synthetic"},
                {"example_id": "s3", "question": "q3", "answer_for_training": "A3", "source": "synthetic"},
            ]
        ).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "s1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
                {"example_id": "s2", "pred_parse_success": False, "accuracy_label": "wrong", "is_correct": False},
                {"example_id": "s3", "pred_parse_success": True, "accuracy_label": "wrong", "is_correct": False},
            ]
        ).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "s1", "overall_pedagogical_score": 7, "is_silent_error": False},
                {"example_id": "s2", "overall_pedagogical_score": 7, "is_silent_error": False},
                {"example_id": "s3", "overall_pedagogical_score": 5, "is_silent_error": False},
            ]
        ).to_parquet(step_dir / "judge_outputs.parquet", index=False)

        runner.run_stage("anchoring", model_name="qwen2.5:0.5b", branch="soft_pvf_medium", generation=1)

        assert (step_dir / "soft_pvf_training_dataset.parquet").exists()
        assert (step_dir / "soft_pvf_decisions.parquet").exists()
        assert (step_dir / "soft_pvf_report.json").exists()
        out_df = pd.read_parquet(step_dir / "synthetic_train_next.parquet")
        assert not out_df.empty
        meta = json.loads((step_dir / "anchor_selection_manifest.json").read_text(encoding="utf-8"))
        assert meta["method"] == "soft_pvf_medium"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_anchoring_stage_pvr_branch_writes_repair_artifacts() -> None:
    base = _mk_local_temp_dir("runner_pvr")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["experiment"]["branches"] = [
            {
                "name": "pvr_repair_medium",
                "branch_type": "pvr_repair_medium",
                "anchor_ratio": 0.0,
                "mixing_mode": "append",
                "pvr_threshold_score": 6,
                "pvr_min_keep_ratio": 0.1,
            }
        ]
        cfg = AppConfig.model_validate(payload)

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="pvr_repair_medium", generation=1)
        step_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"example_id": "h1", "question": "q1", "answer_for_training": "Bad explanation\nFinal answer: 1", "source": "synthetic"},
            ]
        ).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "h1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            ]
        ).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "h1", "overall_pedagogical_score": 4, "is_silent_error": False},
            ]
        ).to_parquet(step_dir / "judge_outputs.parquet", index=False)

        class _FakeRepairClient:
            model_name = "llama-3.1-8b"

            def repair(self, question: str, gold_answer: str, original_response: str) -> str:
                _ = question, original_response
                final = str(gold_answer).split("####", maxsplit=1)[-1].strip()
                return f"Step 1.\nStep 2.\nFinal answer: {final}"

        runner._pvr_repair_client = _FakeRepairClient()  # type: ignore[assignment]
        runner.run_stage("anchoring", model_name="qwen2.5:0.5b", branch="pvr_repair_medium", generation=1)

        assert (step_dir / "pvr_training_dataset.parquet").exists()
        assert (step_dir / "pvr_decisions.parquet").exists()
        assert (step_dir / "pvr_repair_pairs.parquet").exists()
        assert (step_dir / "pvr_report.json").exists()
        out_df = pd.read_parquet(step_dir / "synthetic_train_next.parquet")
        assert set(out_df["example_id"].tolist()) == {"h1"}
        meta = json.loads((step_dir / "anchor_selection_manifest.json").read_text(encoding="utf-8"))
        assert meta["method"] == "pvr_repair_medium"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_anchoring_stage_dbr_branch_writes_budget_artifacts() -> None:
    base = _mk_local_temp_dir("runner_dbr")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["experiment"]["branches"] = [
            {
                "name": "dbr_medium",
                "branch_type": "dbr_medium",
                "anchor_ratio": 0.0,
                "mixing_mode": "append",
                "dbr_budget_parse_failure": 0.0,
                "dbr_budget_silent_error": 0.1,
                "dbr_budget_incorrect_answer": 0.5,
                "dbr_budget_low_reasoning": 0.5,
                "dbr_budget_low_structure": 0.5,
            }
        ]
        cfg = AppConfig.model_validate(payload)

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="dbr_medium", generation=1)
        step_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"example_id": "h1", "question": "short question", "answer_for_training": "A", "source": "synthetic"},
                {"example_id": "h2", "question": "medium question text " * 8, "answer_for_training": "B", "source": "synthetic"},
                {"example_id": "h3", "question": "long question text " * 20, "answer_for_training": "C", "source": "synthetic"},
            ]
        ).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "h1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
                {"example_id": "h2", "pred_parse_success": True, "accuracy_label": "wrong", "is_correct": False},
                {"example_id": "h3", "pred_parse_success": False, "accuracy_label": "parse_failure", "is_correct": False},
            ]
        ).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "example_id": "h1",
                    "overall_pedagogical_score": 6,
                    "is_silent_error": False,
                    "reasoning_soundness": 2,
                    "structure": 2,
                    "clarity": 2,
                },
                {
                    "example_id": "h2",
                    "overall_pedagogical_score": 4,
                    "is_silent_error": False,
                    "reasoning_soundness": 0,
                    "structure": 1,
                    "clarity": 1,
                },
                {
                    "example_id": "h3",
                    "overall_pedagogical_score": 2,
                    "is_silent_error": True,
                    "reasoning_soundness": 0,
                    "structure": 0,
                    "clarity": 0,
                },
            ]
        ).to_parquet(step_dir / "judge_outputs.parquet", index=False)

        runner.run_stage("anchoring", model_name="qwen2.5:0.5b", branch="dbr_medium", generation=1)

        assert (step_dir / "dbr_training_dataset.parquet").exists()
        assert (step_dir / "dbr_decisions.parquet").exists()
        assert (step_dir / "dbr_budget_report.json").exists()
        out_df = pd.read_parquet(step_dir / "synthetic_train_next.parquet")
        assert not out_df.empty
        meta = json.loads((step_dir / "anchor_selection_manifest.json").read_text(encoding="utf-8"))
        assert meta["method"] == "dbr_medium"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_anchoring_stage_pair_lite_branch_writes_repair_artifacts() -> None:
    base = _mk_local_temp_dir("runner_pair_lite")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["experiment"]["branches"] = [
            {
                "name": "pair_lite_medium",
                "branch_type": "pair_lite_medium",
                "anchor_ratio": 0.0,
                "mixing_mode": "append",
                "pair_lite_threshold_score": 6,
                "pair_lite_min_keep_ratio": 0.1,
            }
        ]
        cfg = AppConfig.model_validate(payload)

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="pair_lite_medium", generation=1)
        step_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "example_id": "h1",
                    "question": "q1",
                    "answer_for_training": "Bad explanation\nFinal answer: 1",
                    "source": "synthetic",
                },
            ]
        ).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "example_id": "h1",
                    "pred_parse_success": True,
                    "accuracy_label": "correct",
                    "is_correct": True,
                    "parsed_final_answer": "1",
                    "normalized_predicted": "1",
                },
            ]
        ).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        pd.DataFrame(
            [
                {"example_id": "h1", "overall_pedagogical_score": 4, "is_silent_error": False},
            ]
        ).to_parquet(step_dir / "judge_outputs.parquet", index=False)

        class _FakeRepairClient:
            model_name = "llama-3.1-8b"

            def repair(
                self,
                question: str,
                gold_answer: str,
                original_response: str,
                extracted_final_answer: str,
            ) -> str:
                _ = question, gold_answer, original_response
                return f"Step 1.\nStep 2.\nFinal answer: {extracted_final_answer}"

        runner._pair_lite_repair_client = _FakeRepairClient()  # type: ignore[assignment]
        runner.run_stage("anchoring", model_name="qwen2.5:0.5b", branch="pair_lite_medium", generation=1)

        assert (step_dir / "pair_lite_training_dataset.parquet").exists()
        assert (step_dir / "pair_lite_decisions.parquet").exists()
        assert (step_dir / "pair_lite_repair_pairs.parquet").exists()
        assert (step_dir / "pair_lite_report.json").exists()
        out_df = pd.read_parquet(step_dir / "synthetic_train_next.parquet")
        assert set(out_df["example_id"].tolist()) == {"h1"}
        meta = json.loads((step_dir / "anchor_selection_manifest.json").read_text(encoding="utf-8"))
        assert meta["method"] == "pair_lite_medium"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_generation_stage_writes_csr_candidates_for_csr_branch() -> None:
    base = _mk_local_temp_dir("runner_csr_generation")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["experiment"]["branches"] = [
            {"name": "pure_recycling", "branch_type": "pure_recycling", "anchor_ratio": 0.0, "mixing_mode": "append"},
            {
                "name": "csr_medium",
                "branch_type": "csr_medium",
                "anchor_ratio": 0.0,
                "mixing_mode": "append",
                "csr_num_candidates": 3,
                "csr_candidate_temperature": 0.7,
            },
        ]
        cfg = AppConfig.model_validate(payload)

        class _FakeGenClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, *, model_name: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
                _ = model_name, prompt, top_p, max_tokens
                self.calls += 1
                if temperature >= 0.7:
                    return f"Reasoning alt {self.calls}\nFinal answer: 1"
                return "Reasoning base\nFinal answer: 1"

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        runner._gen_client = _FakeGenClient()  # type: ignore[assignment]

        runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
        runner.run_stage("generation", model_name="qwen2.5:0.5b", branch="csr_medium", generation=1)

        pure_step = runner._step_dir(model_name="qwen2.5:0.5b", branch="pure_recycling", generation=1)
        csr_step = runner._step_dir(model_name="qwen2.5:0.5b", branch="csr_medium", generation=1)
        pure_candidates = pd.read_parquet(pure_step / "csr_candidates.parquet")
        csr_candidates = pd.read_parquet(csr_step / "csr_candidates.parquet")

        assert len(pure_candidates) == 1
        assert set(pure_candidates["candidate_id"].astype(int).tolist()) == {0}
        assert len(csr_candidates) == 3
        assert set(csr_candidates["candidate_id"].astype(int).tolist()) == {0, 1, 2}
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_anchoring_stage_csr_branch_writes_contrastive_artifacts() -> None:
    base = _mk_local_temp_dir("runner_csr_anchoring")
    try:
        cfg = _make_cfg(base)
        payload = cfg.model_dump(mode="python")
        payload["experiment"]["branches"] = [
            {
                "name": "csr_medium",
                "branch_type": "csr_medium",
                "anchor_ratio": 0.0,
                "mixing_mode": "append",
                "csr_num_candidates": 3,
                "csr_min_pair_quality_gap": 2,
            }
        ]
        cfg = AppConfig.model_validate(payload)

        runner = ExperimentRunner(cfg, run_dir=base / "run", stage_executors={})
        step_dir = runner._step_dir(model_name="qwen2.5:0.5b", branch="csr_medium", generation=1)
        step_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [{"example_id": "h1", "question": "q", "answer_for_training": "base", "source": "synthetic"}]
        ).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame(
            [{"example_id": "h1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True}]
        ).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        pd.DataFrame(
            [{"example_id": "h1", "overall_pedagogical_score": 6, "is_silent_error": False}]
        ).to_parquet(step_dir / "judge_outputs.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "run_id": "r",
                    "branch": "csr_medium",
                    "generation": 1,
                    "model_name": "qwen2.5:0.5b",
                    "example_id": "h1",
                    "candidate_id": 0,
                    "question": "q",
                    "prompt_version": "v2",
                    "prompt_text": "p",
                    "raw_response": "Bad explanation\nFinal answer: 9",
                    "parsed_final_answer": "9",
                },
                {
                    "run_id": "r",
                    "branch": "csr_medium",
                    "generation": 1,
                    "model_name": "qwen2.5:0.5b",
                    "example_id": "h1",
                    "candidate_id": 1,
                    "question": "q",
                    "prompt_version": "v2",
                    "prompt_text": "p",
                    "raw_response": "Good explanation\nFinal answer: 1",
                    "parsed_final_answer": "1",
                },
            ]
        ).to_parquet(step_dir / "csr_candidates.parquet", index=False)

        class _FakeJudgeClient:
            def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
                _ = question, gold_answer, rubric_prompt
                if "Good explanation" in model_output:
                    return {
                        "clarity": 2,
                        "structure": 2,
                        "terminology": 2,
                        "reasoning_soundness": 2,
                        "overall_pedagogical_score": 8,
                        "is_silent_error": False,
                        "comment": "good",
                    }
                return {
                    "clarity": 0,
                    "structure": 0,
                    "terminology": 0,
                    "reasoning_soundness": 0,
                    "overall_pedagogical_score": 2,
                    "is_silent_error": True,
                    "comment": "bad",
                }

        runner._judge_client = _FakeJudgeClient()  # type: ignore[assignment]
        runner.run_stage("anchoring", model_name="qwen2.5:0.5b", branch="csr_medium", generation=1)

        assert (step_dir / "csr_candidate_scores.parquet").exists()
        assert (step_dir / "csr_pairs.parquet").exists()
        assert (step_dir / "csr_training_dataset.parquet").exists()
        assert (step_dir / "csr_report.json").exists()
        out_df = pd.read_parquet(step_dir / "synthetic_train_next.parquet")
        assert set(out_df["source"].astype(str).tolist()) == {"csr"}
        assert set(out_df["example_id"].astype(str).tolist()) == {"h1"}
    finally:
        shutil.rmtree(base, ignore_errors=True)
