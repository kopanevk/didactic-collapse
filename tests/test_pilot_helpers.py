from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.pilot import (
    build_pilot_config,
    prepare_pilot_splits,
    validate_pilot_artifacts,
)
from didactic_collapse.orchestration.runner import CONTEXT_STAGES, RUN_STAGES


def _make_cfg(tmp_path: Path) -> AppConfig:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    prompt_dir = tmp_path / "configs" / "prompts"
    split_dir = data_root / "splits"

    split_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "judge_system.txt").write_text("judge prompt", encoding="utf-8")

    base = pd.DataFrame([{"example_id": f"b{i}", "question": "q", "answer_gold": "1"} for i in range(100)])
    heldout = pd.DataFrame([{"example_id": f"h{i}", "question": "q", "answer_gold": "1"} for i in range(100)])
    anchor = pd.DataFrame([{"example_id": f"a{i}", "question": "q", "answer_gold": "1"} for i in range(120)])
    base.to_parquet(split_dir / "base_train.parquet", index=False)
    heldout.to_parquet(split_dir / "heldout_test.parquet", index=False)
    anchor.to_parquet(split_dir / "anchor_pool.parquet", index=False)
    (split_dir / "split_metadata.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    cfg_dict = {
        "project": {"name": "dc", "seed": 42, "run_tag": "pilottest"},
        "paths": {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "prompt_dir": str(prompt_dir),
        },
        "models": {"local_models": [{"name": "qwen2.5:0.5b", "role": "subject"}]},
        "judge": {
            "provider": "mock",
            "model_name": "mock-judge",
            "base_url": "mock://local",
            "api_key_env": "MOCK_UNUSED",
            "timeout_sec": 10,
            "max_retries": 1,
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 128},
        "experiment": {
            "generations": 1,
            "branches": [{"name": "pure_recycling", "anchor_ratio": 0.0}],
        },
        "dataset": {
            "source": "gsm8k",
            "base_train_size": 50,
            "anchor_pool_size": 80,
            "heldout_test_size": 50,
        },
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


def test_build_pilot_config_fields(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    pilot_cfg = build_pilot_config(cfg=cfg, pilot_data_root=tmp_path / "pilot_data", mock_judge=True)

    assert pilot_cfg.models.local_models[0].name == "qwen2.5:0.5b"
    assert pilot_cfg.experiment.generations == 1
    assert {b.name for b in pilot_cfg.experiment.branches} == {"pure_recycling", "anchor_10"}
    assert pilot_cfg.judge.provider == "mock"


def test_prepare_pilot_splits_deterministic(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    root1 = tmp_path / "pilot1"
    root2 = tmp_path / "pilot2"
    _, n1 = prepare_pilot_splits(cfg=cfg, sample_size=30, pilot_data_root=root1, seed=42)
    _, n2 = prepare_pilot_splits(cfg=cfg, sample_size=30, pilot_data_root=root2, seed=42)

    h1 = pd.read_parquet(root1 / "splits" / "heldout_test.parquet")["example_id"].tolist()
    h2 = pd.read_parquet(root2 / "splits" / "heldout_test.parquet")["example_id"].tolist()

    assert n1 == 30
    assert n2 == 30
    assert h1 == h2


def test_validate_pilot_artifacts_missing(tmp_path: Path) -> None:
    missing = validate_pilot_artifacts(
        run_dir=tmp_path / "run",
        model_name="qwen2.5:0.5b",
        branches=["pure_recycling", "anchor_10"],
        generations=1,
    )
    assert len(missing) > 0


def test_validate_pilot_artifacts_success(tmp_path: Path) -> None:
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

    model_dir = run_dir / "qwen2.5_0.5b"
    for branch in ["pure_recycling", "anchor_10"]:
        step_dir = model_dir / branch / "gen_1"
        step_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"x": 1}]).to_parquet(step_dir / "model_outputs.parquet", index=False)
        pd.DataFrame([{"x": 1}]).to_parquet(step_dir / "answer_extraction.parquet", index=False)
        pd.DataFrame([{"x": 1}]).to_parquet(step_dir / "accuracy_table.parquet", index=False)
        pd.DataFrame([{"x": 1}]).to_parquet(step_dir / "judge_outputs.parquet", index=False)
        pd.DataFrame([{"x": 1}]).to_parquet(step_dir / "eval_merged.parquet", index=False)
        pd.DataFrame([{"x": 1}]).to_parquet(step_dir / "synthetic_base.parquet", index=False)
        pd.DataFrame([{"x": 1}]).to_parquet(step_dir / "synthetic_train_next.parquet", index=False)
        (step_dir / "anchor_selection_manifest.json").write_text("{}", encoding="utf-8")
        (step_dir / "used_anchor_ids.json").write_text("[]", encoding="utf-8")

        context_manifest = {
            "schema_version": 1,
            "run_id": "r",
            "run_dir": str(run_dir),
            "scope": "context",
            "model_name": "qwen2.5:0.5b",
            "generation": 1,
            "branch": branch,
            "seed": 42,
            "config_hash": "abc",
            "stages": {s: _stage_record(s, "qwen2.5:0.5b", branch, 1) for s in CONTEXT_STAGES},
        }
        (step_dir / "stage_manifest.json").write_text(json.dumps(context_manifest), encoding="utf-8")

    pd.DataFrame([{"x": 1}]).to_parquet(run_dir / "all_eval_merged.parquet", index=False)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"x": 1}]).to_csv(run_dir / "tables" / "metrics_by_generation.csv", index=False)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures" / "accuracy_vs_generation.png").write_bytes(b"png")
    (run_dir / "figures" / "pedagogical_vs_generation.png").write_bytes(b"png")
    (run_dir / "figures" / "silent_error_vs_generation.png").write_bytes(b"png")

    missing = validate_pilot_artifacts(
        run_dir=run_dir,
        model_name="qwen2.5:0.5b",
        branches=["pure_recycling", "anchor_10"],
        generations=1,
    )
    assert missing == []
