from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.dbr import export_dbr_analysis


def _make_fake_run(root: Path) -> Path:
    run_dir = root / "dbr_fake_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    snapshot = {"config": {"project": {"seed": 171}, "paths": {"data_root": "data"}}}
    (run_dir / "run_config.snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 2,
                "sample_count": 4,
                "accuracy_mean": 0.50,
                "pedagogical_score_mean": 4.8,
                "silent_error_rate": 0.3,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "soft_pvf_noisy_keep",
                "generation": 2,
                "sample_count": 4,
                "accuracy_mean": 0.55,
                "pedagogical_score_mean": 4.9,
                "silent_error_rate": 0.25,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "dbr_medium",
                "generation": 2,
                "sample_count": 4,
                "accuracy_mean": 0.60,
                "pedagogical_score_mean": 5.0,
                "silent_error_rate": 0.20,
            },
        ]
    )
    summary.to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)

    model_dir = run_dir / "qwen2.5_0.5b"
    for branch in ["pure_recycling", "soft_pvf_noisy_keep", "dbr_medium"]:
        ctx = model_dir / branch / "gen_2"
        ctx.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"model_name": "qwen2.5:0.5b", "branch": branch, "generation": 2, "pred_parse_success": True},
                {"model_name": "qwen2.5:0.5b", "branch": branch, "generation": 2, "pred_parse_success": True},
                {"model_name": "qwen2.5:0.5b", "branch": branch, "generation": 2, "pred_parse_success": False},
                {"model_name": "qwen2.5:0.5b", "branch": branch, "generation": 2, "pred_parse_success": True},
            ]
        ).to_parquet(ctx / "accuracy_table.parquet", index=False)

    dbr_report = {
        "model_name": "qwen2.5:0.5b",
        "branch": "dbr_medium",
        "generation": 2,
        "seed": 171,
        "policy_name": "dbr_medium",
        "total_candidates": 50,
        "target_size": 50,
        "selected_count": 44,
        "selection_rate": 0.88,
        "min_selection_rate": 0.80,
        "budgets": {
            "parse_failure": 0.0,
            "silent_error": 0.1,
            "incorrect_answer": 0.3,
            "low_reasoning": 0.25,
            "low_structure": 0.3,
        },
        "defect_rates_before": {
            "parse_failure": 0.2,
            "silent_error": 0.18,
            "incorrect_answer": 0.42,
            "low_reasoning": 0.34,
            "low_structure": 0.36,
        },
        "defect_rates_after": {
            "parse_failure": 0.0,
            "silent_error": 0.09,
            "incorrect_answer": 0.28,
            "low_reasoning": 0.22,
            "low_structure": 0.27,
        },
        "budget_violations": {},
        "relaxation_steps_used": ["relax_low_structure"],
        "bucket_coverage_before": {"short": 20, "medium": 15, "long": 15},
        "bucket_coverage_after": {"short": 17, "medium": 13, "long": 14},
        "fallback_bucket_count": 0,
    }
    (model_dir / "dbr_medium" / "gen_2" / "dbr_budget_report.json").write_text(
        json.dumps(dbr_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return run_dir


def test_export_dbr_analysis_writes_expected_artifacts() -> None:
    base = Path("outputs/.tmp") / f"dbr_analysis_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = _make_fake_run(base)
        out_dir = base / "analysis" / "tables"
        artifacts = export_dbr_analysis(run_dir=run_dir, out_dir=out_dir)
        required_paths = [
            artifacts.run_level_csv,
            artifacts.generation_deltas_csv,
            artifacts.branch_deltas_csv,
            artifacts.budget_summary_csv,
            artifacts.defect_rates_before_after_csv,
            artifacts.bucket_coverage_csv,
            artifacts.stress_summary_csv,
            artifacts.accuracy_plot,
            artifacts.pedagogical_plot,
            artifacts.silent_error_plot,
            artifacts.keep_rate_plot,
            artifacts.metadata_json,
        ]
        missing = [str(p) for p in required_paths if not p.exists()]
        assert not missing, f"Missing artifacts: {missing}"
        stress = pd.read_csv(artifacts.stress_summary_csv)
        assert {"branch", "generation", "selection_rate", "accuracy_mean", "pedagogical_score_mean"}.issubset(
            set(stress.columns)
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)

