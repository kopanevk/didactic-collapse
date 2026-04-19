from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.analysis.compare_runs import (
    build_first_experiment_run_metrics,
    compare_first_experiment_runs,
)


def _mk_run_dir(tag: str) -> Path:
    run_dir = Path("outputs/.tmp") / f"{tag}_{uuid.uuid4().hex}"
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_summary(
    *,
    run_dir: Path,
    accuracy_value: float,
    pedagogical_value: float,
    silent_error_value: float,
    corrected: bool,
) -> None:
    df = pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 0,
                "sample_count": 3,
                "accuracy_mean": accuracy_value,
                "pedagogical_score_mean": pedagogical_value,
                "silent_error_rate": silent_error_value,
            }
        ]
    )
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)
    df.to_parquet(run_dir / "tables" / "first_experiment_summary.parquet", index=False)
    if corrected:
        corr = df.copy()
        corr["accuracy_mean_corrected"] = 0.333333
        corr.to_csv(
            run_dir / "tables" / "first_experiment_summary_corrected_accuracy.csv",
            index=False,
        )


def _write_accuracy_table(*, run_dir: Path, pred_parse_success: list[bool]) -> None:
    out_dir = run_dir / "qwen2.5_0.5b" / "pure_recycling" / "gen_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "model_name": ["qwen2.5:0.5b"] * len(pred_parse_success),
            "branch": ["pure_recycling"] * len(pred_parse_success),
            "generation": [0] * len(pred_parse_success),
            "pred_parse_success": pred_parse_success,
        }
    )
    df.to_parquet(out_dir / "accuracy_table.parquet", index=False)


def test_build_metrics_prefers_corrected_accuracy_when_available() -> None:
    run_dir = _mk_run_dir("compare_corrected")
    try:
        _write_summary(
            run_dir=run_dir,
            accuracy_value=0.0,
            pedagogical_value=6.0,
            silent_error_value=0.2,
            corrected=True,
        )
        _write_accuracy_table(run_dir=run_dir, pred_parse_success=[True, False, True])

        metrics = build_first_experiment_run_metrics(run_dir)
        row = metrics.iloc[0]
        assert float(row["accuracy_mean"]) == pytest.approx(0.333333, rel=1e-6)
        assert float(row["parse_failure_pred_rate"]) == pytest.approx(1.0 / 3.0, rel=1e-6)
        assert str(row["accuracy_source"]) == "corrected"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_compare_first_experiment_runs_exports_deltas() -> None:
    old_run = _mk_run_dir("compare_old")
    new_run = _mk_run_dir("compare_new")
    try:
        _write_summary(
            run_dir=old_run,
            accuracy_value=0.2,
            pedagogical_value=6.0,
            silent_error_value=0.25,
            corrected=False,
        )
        _write_accuracy_table(run_dir=old_run, pred_parse_success=[True, False, False, True])

        _write_summary(
            run_dir=new_run,
            accuracy_value=0.35,
            pedagogical_value=6.2,
            silent_error_value=0.18,
            corrected=False,
        )
        _write_accuracy_table(run_dir=new_run, pred_parse_success=[True, True, False, True])

        artifacts = compare_first_experiment_runs(old_run_dir=old_run, new_run_dir=new_run)
        assert artifacts.csv_path.exists()
        assert artifacts.parquet_path.exists()
        row = artifacts.table.iloc[0]
        assert float(row["delta_accuracy_mean"]) == pytest.approx(0.15, rel=1e-6)
        assert float(row["delta_parse_failure_pred_rate"]) == pytest.approx(-0.25, rel=1e-6)
    finally:
        shutil.rmtree(old_run, ignore_errors=True)
        shutil.rmtree(new_run, ignore_errors=True)
