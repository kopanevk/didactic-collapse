from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from didactic_collapse.analysis.mode_comparison import export_mode_comparison_analysis


def _mk_dir(prefix: str) -> Path:
    base = Path("outputs") / ".tmp" / "unit_mode_compare"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{prefix}_{uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_snapshot(run_dir: Path, *, seed: int, mode: str, model_name: str = "qwen2.5:0.5b") -> None:
    payload = {
        "run_id": run_dir.name,
        "created_at": "2026-04-24T00:00:00",
        "config_hash": "abc",
        "config": {
            "project": {"name": "dc", "seed": seed, "run_tag": run_dir.name},
            "paths": {"data_root": str(run_dir / "data"), "output_root": "outputs", "prompt_dir": "configs/prompts"},
            "models": {"local_models": [{"name": model_name, "role": "subject"}]},
            "experiment": {"mode": mode, "generations": 2, "branches": []},
        },
    }
    (run_dir / "run_config.snapshot.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_inference_run(run_dir: Path, *, seed: int) -> None:
    _write_snapshot(run_dir, seed=seed, mode="inference_recycling_only")
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 0,
                "sample_count": 4,
                "accuracy_mean": 0.5,
                "pedagogical_score_mean": 4.5,
                "silent_error_rate": 0.25,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "anchor_20_append",
                "generation": 0,
                "sample_count": 4,
                "accuracy_mean": 0.75,
                "pedagogical_score_mean": 5.5,
                "silent_error_rate": 0.10,
            },
        ]
    )
    summary.to_csv(tables / "first_experiment_summary.csv", index=False)

    for branch, parse_success in [("pure_recycling", [True, True, False, False]), ("anchor_20_append", [True, True, True, False])]:
        d = run_dir / "qwen2.5_0.5b" / branch / "gen_0"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "model_name": ["qwen2.5:0.5b"] * 4,
                "branch": [branch] * 4,
                "generation": [0] * 4,
                "pred_parse_success": parse_success,
            }
        ).to_parquet(d / "accuracy_table.parquet", index=False)


def _build_training_run(run_dir: Path, *, seed: int) -> None:
    _write_snapshot(run_dir, seed=seed, mode="training_recycling_feasibility")
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "branch": "pure_recycling",
                "generation": 0,
                "sample_count": 4,
                "accuracy_mean": 0.6,
                "pedagogical_score_mean": 4.7,
                "silent_error_rate": 0.20,
                "parse_failure_pred_count": 1,
                "parse_failure_pred_rate": 0.25,
            },
            {
                "branch": "anchor_20_append",
                "generation": 0,
                "sample_count": 4,
                "accuracy_mean": 0.8,
                "pedagogical_score_mean": 5.7,
                "silent_error_rate": 0.08,
                "parse_failure_pred_count": 0,
                "parse_failure_pred_rate": 0.0,
            },
        ]
    ).to_csv(tables / "training_feasibility_summary.csv", index=False)


def test_mode_comparison_export_writes_outputs() -> None:
    root = _mk_dir("mode_export")
    inf = root / "inference_run"
    trn = root / "training_run"
    inf.mkdir(parents=True, exist_ok=True)
    trn.mkdir(parents=True, exist_ok=True)
    _build_inference_run(inf, seed=71)
    _build_training_run(trn, seed=71)

    out_dir = root / "analysis" / "tables"
    artifacts = export_mode_comparison_analysis(
        inference_run_dirs=[inf],
        training_run_dirs=[trn],
        out_dir=out_dir,
    )
    assert artifacts.run_level_csv.exists()
    assert artifacts.seed_stats_csv.exists()
    assert artifacts.mode_deltas_csv.exists()
    deltas = pd.read_csv(artifacts.mode_deltas_csv)
    assert len(deltas) == 2
    assert set(deltas["branch"].tolist()) == {"pure_recycling", "anchor_20_append"}

