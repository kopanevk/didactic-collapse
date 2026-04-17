from __future__ import annotations

import json
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
