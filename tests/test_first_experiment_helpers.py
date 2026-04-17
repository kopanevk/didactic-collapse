from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.first_experiment import (
    _export_first_summary_table,
    _export_qualitative_candidates,
    build_first_experiment_config,
    validate_first_experiment_outputs,
)
from didactic_collapse.orchestration.runner import CONTEXT_STAGES, RUN_STAGES


def _cfg(tmp_path: Path) -> AppConfig:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    prompt_dir = tmp_path / "configs" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "judge_system.txt").write_text("judge", encoding="utf-8")

    cfg_dict = {
        "project": {"name": "dc", "seed": 42, "run_tag": "first"},
        "paths": {"data_root": str(data_root), "output_root": str(output_root), "prompt_dir": str(prompt_dir)},
        "models": {"local_models": [{"name": "qwen2.5:0.5b", "role": "subject"}]},
        "judge": {
            "provider": "gemini_openai_compatible",
            "model_name": "gemini-2.5-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key_env": "GEMINI_API_KEY",
            "timeout_sec": 60,
            "max_retries": 3,
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 128},
        "experiment": {
            "generations": 2,
            "branches": [{"name": "pure_recycling", "anchor_ratio": 0.0}, {"name": "anchor_10", "anchor_ratio": 0.1}],
        },
        "dataset": {"source": "gsm8k", "base_train_size": 100, "anchor_pool_size": 200, "heldout_test_size": 100},
        "runtime": {"force_recompute": False, "save_parquet": True, "save_csv": True},
    }
    return AppConfig.model_validate(cfg_dict)


def _stage_record(stage: str, model: str | None, branch: str | None, gen: int | None) -> dict:
    return {
        "stage_name": stage,
        "status": "completed",
        "timestamp_start": "2026-01-01T00:00:00+00:00",
        "timestamp_end": "2026-01-01T00:00:01+00:00",
        "model_name": model,
        "generation": gen,
        "branch": branch,
        "seed": 42,
        "config_hash": "abc",
        "input_artifacts": [],
        "output_artifacts": [],
        "row_count": 1,
        "error_message": None,
    }


def test_build_first_experiment_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    out = build_first_experiment_config(cfg=cfg, data_root=tmp_path / "pilot_data")
    assert out.models.local_models[0].name == "qwen2.5:0.5b"
    assert out.experiment.generations == 2
    assert {b.name for b in out.experiment.branches} == {"pure_recycling", "anchor_10"}


def test_summary_and_qualitative_exports(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 0,
                "example_id": "e1",
                "is_correct": True,
                "overall_pedagogical_score": 1,
                "is_silent_error": True,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "anchor_10",
                "generation": 1,
                "example_id": "e2",
                "is_correct": False,
                "overall_pedagogical_score": 6,
                "is_silent_error": False,
            },
        ]
    )
    csv_path, pq_path, summary_df = _export_first_summary_table(all_eval=df, out_dir=tmp_path)
    q_csv, q_pq = _export_qualitative_candidates(all_eval=df, out_dir=tmp_path)

    assert csv_path.exists() and pq_path.exists()
    assert q_csv.exists() and q_pq.exists()
    assert len(summary_df) == 2


def test_validate_first_experiment_outputs_fails_on_missing_generation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "schema_version": 1,
        "run_id": "r",
        "run_dir": str(run_dir),
        "scope": "run",
        "model_name": None,
        "generation": None,
        "branch": None,
        "seed": 42,
        "config_hash": "abc",
        "stages": {s: _stage_record(s, None, None, None) for s in RUN_STAGES},
    }
    (run_dir / "run_stage_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    model_name = "qwen2.5:0.5b"
    for branch in ["pure_recycling", "anchor_10"]:
        for gen in [0, 1]:
            step_dir = run_dir / model_name.replace(":", "_") / branch / f"gen_{gen}"
            step_dir.mkdir(parents=True, exist_ok=True)
            ctx_manifest = {
                "schema_version": 1,
                "run_id": "r",
                "run_dir": str(run_dir),
                "scope": "context",
                "model_name": model_name,
                "generation": gen,
                "branch": branch,
                "seed": 42,
                "config_hash": "abc",
                "stages": {s: _stage_record(s, model_name, branch, gen) for s in CONTEXT_STAGES},
            }
            (step_dir / "stage_manifest.json").write_text(json.dumps(ctx_manifest), encoding="utf-8")

    summary_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "branch": "pure_recycling",
                "generation": 0,
                "sample_count": 10,
                "accuracy_mean": 0.5,
                "pedagogical_score_mean": 4.0,
                "silent_error_rate": 0.1,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Missing generations"):
        validate_first_experiment_outputs(
            run_dir=run_dir,
            model_name=model_name,
            branches=["pure_recycling", "anchor_10"],
            generations=[0, 1],
            summary_table=summary_df,
        )
