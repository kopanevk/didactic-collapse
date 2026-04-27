from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.baseline_series import (
    collect_baseline_run_metrics,
    export_baseline_series_analysis,
)
from didactic_collapse.orchestration.baseline_series import parse_seed_list


def _mk_base_dir(tag: str) -> Path:
    base = Path("outputs/.tmp") / f"{tag}_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_mock_run(
    base_dir: Path,
    *,
    run_name: str,
    seed: int,
    acc_shift: float,
    include_anchor20: bool = False,
    include_gen2: bool = False,
) -> Path:
    run_dir = base_dir / "outputs" / "runs" / run_name
    data_root = base_dir / "data" / run_name
    (data_root / "splits").mkdir(parents=True, exist_ok=True)
    heldout = pd.DataFrame(
        [
            {"example_id": "e1", "question": "Q1", "answer_gold": "#### 1"},
            {"example_id": "e2", "question": "Q2", "answer_gold": "#### 2"},
        ]
    )
    heldout.to_parquet(data_root / "splits" / "heldout_test.parquet", index=False)

    snapshot = {
        "config": {
            "project": {"seed": seed},
            "paths": {"data_root": str(data_root)},
        }
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    summary_rows = [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 0,
                "sample_count": 2,
                "accuracy_mean": 0.30 + acc_shift,
                "pedagogical_score_mean": 5.5,
                "silent_error_rate": 0.30,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 1,
                "sample_count": 2,
                "accuracy_mean": 0.25 + acc_shift,
                "pedagogical_score_mean": 5.0,
                "silent_error_rate": 0.25,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "anchor_10",
                "generation": 0,
                "sample_count": 2,
                "accuracy_mean": 0.35 + acc_shift,
                "pedagogical_score_mean": 5.8,
                "silent_error_rate": 0.20,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "anchor_10",
                "generation": 1,
                "sample_count": 2,
                "accuracy_mean": 0.33 + acc_shift,
                "pedagogical_score_mean": 5.4,
                "silent_error_rate": 0.18,
            },
        ]
    if include_gen2:
        summary_rows.extend(
            [
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "pure_recycling",
                    "generation": 2,
                    "sample_count": 2,
                    "accuracy_mean": 0.20 + acc_shift,
                    "pedagogical_score_mean": 4.4,
                    "silent_error_rate": 0.35,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "anchor_10",
                    "generation": 2,
                    "sample_count": 2,
                    "accuracy_mean": 0.30 + acc_shift,
                    "pedagogical_score_mean": 4.9,
                    "silent_error_rate": 0.14,
                },
            ]
        )
    if include_anchor20:
        summary_rows.extend(
            [
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "anchor_20",
                    "generation": 0,
                    "sample_count": 2,
                    "accuracy_mean": 0.40 + acc_shift,
                    "pedagogical_score_mean": 5.9,
                    "silent_error_rate": 0.12,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "anchor_20",
                    "generation": 1,
                    "sample_count": 2,
                    "accuracy_mean": 0.37 + acc_shift,
                    "pedagogical_score_mean": 5.6,
                    "silent_error_rate": 0.10,
                },
            ]
        )
        if include_gen2:
            summary_rows.append(
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "anchor_20",
                    "generation": 2,
                    "sample_count": 2,
                    "accuracy_mean": 0.35 + acc_shift,
                    "pedagogical_score_mean": 5.2,
                    "silent_error_rate": 0.12,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)
    summary.to_parquet(run_dir / "tables" / "first_experiment_summary.parquet", index=False)

    branches = ["pure_recycling", "anchor_10"]
    if include_anchor20:
        branches.append("anchor_20")
    generations = [0, 1] + ([2] if include_gen2 else [])
    for branch in branches:
        for gen in generations:
            step_dir = run_dir / "qwen2.5_0.5b" / branch / f"gen_{gen}"
            step_dir.mkdir(parents=True, exist_ok=True)

            acc_df = pd.DataFrame(
                [
                    {
                        "model_name": "qwen2.5:0.5b",
                        "branch": branch,
                        "generation": gen,
                        "example_id": "e1",
                        "pred_parse_success": True,
                        "is_correct": True,
                    },
                    {
                        "model_name": "qwen2.5:0.5b",
                        "branch": branch,
                        "generation": gen,
                        "example_id": "e2",
                        "pred_parse_success": gen == 1,
                        "is_correct": branch in {"anchor_10", "anchor_20"},
                    },
                ]
            )
            acc_df.to_parquet(step_dir / "accuracy_table.parquet", index=False)

            eval_df = pd.DataFrame(
                [
                    {
                        "model_name": "qwen2.5:0.5b",
                        "branch": branch,
                        "generation": gen,
                        "example_id": "e1",
                        "is_correct": True,
                        "overall_pedagogical_score": {0: 6, 1: 4, 2: 2}[gen],
                        "is_silent_error": False,
                    },
                    {
                        "model_name": "qwen2.5:0.5b",
                        "branch": branch,
                        "generation": gen,
                        "example_id": "e2",
                        "is_correct": branch in {"anchor_10", "anchor_20"},
                        "overall_pedagogical_score": {0: 5, 1: 3, 2: 1}[gen],
                        "is_silent_error": branch == "pure_recycling" and gen >= 1,
                    },
                ]
            )
            eval_df.to_parquet(step_dir / "eval_merged.parquet", index=False)

            model_outputs = pd.DataFrame(
                [
                    {"example_id": "e1", "raw_response": "Final answer: 1"},
                    {"example_id": "e2", "raw_response": "Final answer: 2"},
                ]
            )
            model_outputs.to_parquet(step_dir / "model_outputs.parquet", index=False)

    return run_dir


def test_parse_seed_list() -> None:
    assert parse_seed_list("42, 43,44") == [42, 43, 44]


def test_collect_baseline_run_metrics_includes_seed_and_mode() -> None:
    base = _mk_base_dir("baseline_collect")
    try:
        run1 = _write_mock_run(base, run_name="run_seed42", seed=42, acc_shift=0.0)
        run2 = _write_mock_run(base, run_name="run_seed43", seed=43, acc_shift=0.1)
        df = collect_baseline_run_metrics([run1, run2])
        assert set(df["seed"].tolist()) == {42, 43}
        assert set(df["evaluation_mode"].tolist()) == {"inference_recycling_only"}
        assert {"parse_failure_pred_rate", "accuracy_mean", "pedagogical_score_mean"}.issubset(df.columns)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_export_baseline_series_analysis_writes_outputs() -> None:
    base = _mk_base_dir("baseline_export")
    try:
        run1 = _write_mock_run(
            base,
            run_name="run_seed52",
            seed=52,
            acc_shift=0.0,
            include_anchor20=True,
            include_gen2=True,
        )
        run2 = _write_mock_run(
            base,
            run_name="run_seed53",
            seed=53,
            acc_shift=0.05,
            include_anchor20=True,
            include_gen2=True,
        )
        out_dir = base / "outputs" / "baseline_series" / "tables"
        artifacts = export_baseline_series_analysis(run_dirs=[run1, run2], out_dir=out_dir)

        assert artifacts.run_level_csv.exists()
        assert artifacts.seed_stats_csv.exists()
        assert artifacts.generation_deltas_csv.exists()
        assert artifacts.branch_deltas_csv.exists()
        assert artifacts.qualitative_csv.exists()
        assert artifacts.accuracy_plot.exists()
        assert artifacts.pedagogical_plot.exists()
        assert artifacts.silent_error_plot.exists()

        stats = pd.read_csv(artifacts.seed_stats_csv)
        assert "accuracy_mean_ci_low" in stats.columns
        assert "accuracy_mean_ci_high" in stats.columns
        assert set(stats["evaluation_mode"].tolist()) == {"inference_recycling_only"}
        gen_deltas = pd.read_csv(artifacts.generation_deltas_csv)
        assert {"gen1_minus_gen0", "gen2_minus_gen1", "gen2_minus_gen0"}.issubset(
            set(gen_deltas["delta_generation"].tolist())
        )
        qualitative = pd.read_csv(artifacts.qualitative_csv)
        assert not qualitative.empty
        assert "pedagogical_decline_gen0_to_gen2" in set(qualitative["category"].tolist())
    finally:
        shutil.rmtree(base, ignore_errors=True)
