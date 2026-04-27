from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.config.settings import load_config
from didactic_collapse.orchestration.first_experiment import validate_first_experiment_outputs
from didactic_collapse.orchestration.runner import CONTEXT_STAGES, RUN_STAGES


def _mk_base_dir(tag: str) -> Path:
    base = Path("outputs/.tmp") / f"{tag}_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    return base


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


def test_baseline_series_config_uses_gen2() -> None:
    cfg = load_config("configs/baseline_series.yaml")
    assert cfg.experiment.generations == 3
    assert [b.name for b in cfg.experiment.branches] == ["pure_recycling", "anchor_10", "anchor_20"]


def test_validate_first_experiment_outputs_supports_gen2_contexts() -> None:
    base = _mk_base_dir("gen2_contract")
    try:
        run_dir = base / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_name = "qwen2.5:0.5b"
        branches = ["pure_recycling", "anchor_10", "anchor_20"]
        generations = [0, 1, 2]

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

        for branch in branches:
            for gen in generations:
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

        rows: list[dict[str, object]] = []
        for branch in branches:
            for gen in generations:
                rows.append(
                    {
                        "model_name": model_name,
                        "branch": branch,
                        "generation": gen,
                        "sample_count": 10,
                        "accuracy_mean": 0.4,
                        "pedagogical_score_mean": 5.0 - 0.3 * gen,
                        "silent_error_rate": 0.2 + 0.05 * gen,
                    }
                )
        summary_df = pd.DataFrame(rows)

        validate_first_experiment_outputs(
            run_dir=run_dir,
            model_name=model_name,
            branches=branches,
            generations=generations,
            summary_table=summary_df,
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)
