from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.first_experiment import (
    _ensure_real_judge_provider,
    _preflight_real_judge_auth,
    _override_judge_config_for_resume,
    _export_first_summary_table,
    _export_qualitative_candidates,
    build_first_experiment_config,
    validate_first_experiment_outputs,
    verify_first_experiment_artifacts,
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
    assert {b.mixing_mode for b in out.experiment.branches} == {"append"}


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
            },
            {
                "model_name": model_name,
                "branch": "anchor_10",
                "generation": 0,
                "sample_count": 12,
                "accuracy_mean": 0.4,
                "pedagogical_score_mean": 3.8,
                "silent_error_rate": 0.2,
            },
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


def test_real_experiment_does_not_allow_mock_provider(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    raw = cfg.model_dump(mode="python")
    raw["judge"]["provider"] = "mock"
    mock_cfg = AppConfig.model_validate(raw)

    with pytest.raises(RuntimeError, match="requires real judge provider"):
        _ensure_real_judge_provider(mock_cfg)


def test_resume_can_override_judge_provider(tmp_path: Path) -> None:
    snapshot_cfg = _cfg(tmp_path)
    raw = snapshot_cfg.model_dump(mode="python")
    raw["judge"]["provider"] = "cerebras"
    raw["judge"]["api_key_env"] = "CEREBRAS_API_KEY"
    raw["judge"]["base_url"] = "https://api.cerebras.ai/v1"
    raw["judge"]["model_name"] = "llama-3.1-8b"
    requested_cfg = AppConfig.model_validate(raw)

    merged = _override_judge_config_for_resume(snapshot_cfg=snapshot_cfg, requested_cfg=requested_cfg)
    assert merged.judge.provider == "cerebras"
    assert merged.judge.api_key_env == "CEREBRAS_API_KEY"


def test_preflight_real_judge_auth_fails_fast_for_missing_cerebras_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    raw = cfg.model_dump(mode="python")
    raw["judge"]["provider"] = "cerebras"
    raw["judge"]["api_key_env"] = "CEREBRAS_API_KEY"
    raw["judge"]["base_url"] = "https://api.cerebras.ai/v1"
    raw["judge"]["model_name"] = "llama-3.1-8b"
    cerebras_cfg = AppConfig.model_validate(raw)

    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    with pytest.raises(Exception, match="Missing Cerebras API key"):
        _preflight_real_judge_auth(cerebras_cfg)


def test_verify_first_experiment_artifacts_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 0,
                "sample_count": 10,
                "accuracy_mean": 0.5,
                "pedagogical_score_mean": 4.0,
                "silent_error_rate": 0.1,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "anchor_10",
                "generation": 1,
                "sample_count": 10,
                "accuracy_mean": 0.6,
                "pedagogical_score_mean": 4.2,
                "silent_error_rate": 0.2,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 1,
                "sample_count": 10,
                "accuracy_mean": 0.4,
                "pedagogical_score_mean": 3.7,
                "silent_error_rate": 0.15,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "anchor_10",
                "generation": 0,
                "sample_count": 10,
                "accuracy_mean": 0.55,
                "pedagogical_score_mean": 4.1,
                "silent_error_rate": 0.12,
            },
        ]
    )
    summary_csv = run_dir / "tables" / "first_experiment_summary.csv"
    summary_pq = run_dir / "tables" / "first_experiment_summary.parquet"
    summary.to_csv(summary_csv, index=False)
    summary.to_parquet(summary_pq, index=False)

    qualitative = pd.DataFrame(columns=["example_id", "generation", "branch"])
    qual_csv = run_dir / "tables" / "qualitative_silent_error_candidates.csv"
    qual_pq = run_dir / "tables" / "qualitative_silent_error_candidates.parquet"
    qualitative.to_csv(qual_csv, index=False)
    qualitative.to_parquet(qual_pq, index=False)
    qual_meta = run_dir / "tables" / "qualitative_silent_error_candidates.meta.json"
    qual_meta.write_text(json.dumps({"is_empty": True, "row_count": 0}), encoding="utf-8")

    pd.DataFrame([{"x": 1}]).to_parquet(run_dir / "all_eval_merged.parquet", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(run_dir / "tables" / "metrics_by_generation.csv", index=False)
    for plot_name in (
        "accuracy_vs_generation.png",
        "pedagogical_vs_generation.png",
        "silent_error_vs_generation.png",
    ):
        (run_dir / "figures" / plot_name).write_bytes(b"png")

    verify_first_experiment_artifacts(
        run_dir=run_dir,
        summary_csv=summary_csv,
        summary_parquet=summary_pq,
        qualitative_csv=qual_csv,
        qualitative_parquet=qual_pq,
        qualitative_meta_path=qual_meta,
        branches=["pure_recycling", "anchor_10"],
        generations=[0, 1],
    )


def test_verify_first_experiment_artifacts_fails_on_missing_summary_column(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 0,
                "sample_count": 10,
                "accuracy_mean": 0.5,
                "silent_error_rate": 0.1,
            }
        ]
    )
    summary_csv = run_dir / "tables" / "first_experiment_summary.csv"
    summary_pq = run_dir / "tables" / "first_experiment_summary.parquet"
    summary.to_csv(summary_csv, index=False)
    summary.to_parquet(summary_pq, index=False)

    qual_csv = run_dir / "tables" / "qualitative_silent_error_candidates.csv"
    qual_pq = run_dir / "tables" / "qualitative_silent_error_candidates.parquet"
    pd.DataFrame(columns=["example_id"]).to_csv(qual_csv, index=False)
    pd.DataFrame(columns=["example_id"]).to_parquet(qual_pq, index=False)
    qual_meta = run_dir / "tables" / "qualitative_silent_error_candidates.meta.json"
    qual_meta.write_text(json.dumps({"is_empty": True, "row_count": 0}), encoding="utf-8")

    pd.DataFrame([{"x": 1}]).to_parquet(run_dir / "all_eval_merged.parquet", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(run_dir / "tables" / "metrics_by_generation.csv", index=False)
    for plot_name in (
        "accuracy_vs_generation.png",
        "pedagogical_vs_generation.png",
        "silent_error_vs_generation.png",
    ):
        (run_dir / "figures" / plot_name).write_bytes(b"png")

    with pytest.raises(RuntimeError, match="missing required columns"):
        verify_first_experiment_artifacts(
            run_dir=run_dir,
            summary_csv=summary_csv,
            summary_parquet=summary_pq,
            qualitative_csv=qual_csv,
            qualitative_parquet=qual_pq,
            qualitative_meta_path=qual_meta,
            branches=["pure_recycling", "anchor_10"],
            generations=[0, 1],
        )
