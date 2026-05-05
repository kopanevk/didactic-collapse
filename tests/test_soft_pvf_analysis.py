from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.soft_pvf import export_soft_pvf_analysis


def _make_fake_run(root: Path) -> Path:
    run_dir = root / "soft_pvf_fake_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    snapshot = {"config": {"project": {"seed": 123}, "paths": {"data_root": "data"}}}
    (run_dir / "run_config.snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 0,
                "sample_count": 4,
                "accuracy_mean": 0.5,
                "pedagogical_score_mean": 5.0,
                "silent_error_rate": 0.25,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pvf_medium",
                "generation": 0,
                "sample_count": 4,
                "accuracy_mean": 0.6,
                "pedagogical_score_mean": 5.5,
                "silent_error_rate": 0.2,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "soft_pvf_medium",
                "generation": 0,
                "sample_count": 4,
                "accuracy_mean": 0.65,
                "pedagogical_score_mean": 5.7,
                "silent_error_rate": 0.15,
            },
        ]
    )
    summary.to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)

    model_dir = run_dir / "qwen2.5_0.5b"
    for branch in ["pure_recycling", "pvf_medium", "soft_pvf_medium"]:
        ctx = model_dir / branch / "gen_0"
        ctx.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"model_name": "qwen2.5:0.5b", "branch": branch, "generation": 0, "pred_parse_success": True},
                {"model_name": "qwen2.5:0.5b", "branch": branch, "generation": 0, "pred_parse_success": False},
                {"model_name": "qwen2.5:0.5b", "branch": branch, "generation": 0, "pred_parse_success": True},
                {"model_name": "qwen2.5:0.5b", "branch": branch, "generation": 0, "pred_parse_success": True},
            ]
        ).to_parquet(ctx / "accuracy_table.parquet", index=False)

    pvf_report = {
        "model_name": "qwen2.5:0.5b",
        "branch": "pvf_medium",
        "generation": 0,
        "seed": 123,
        "threshold_score": 5,
        "min_keep_ratio": 0.1,
        "total_candidates": 50,
        "kept_count": 20,
        "rejected_count": 30,
        "keep_rate": 0.4,
        "rejection_reason_counts": {"accuracy_incorrect": 10},
    }
    (model_dir / "pvf_medium" / "gen_0" / "pvf_filter_report.json").write_text(
        json.dumps(pvf_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    decisions = pd.DataFrame(
        [
            {
                "example_id": "a",
                "is_correct": True,
                "pred_parse_success": True,
                "overall_pedagogical_score": 7.0,
                "is_silent_error": False,
                "assigned_weight": 1.0,
                "kept": True,
                "decision_reason": "high_quality",
                "deterministic_score": 0.1,
                "sampling_value": 0.1,
                "policy_name": "soft_pvf_medium",
                "branch": "soft_pvf_medium",
                "generation": 0,
                "seed": 123,
            },
            {
                "example_id": "b",
                "is_correct": False,
                "pred_parse_success": True,
                "overall_pedagogical_score": 3.0,
                "is_silent_error": False,
                "assigned_weight": 0.1,
                "kept": False,
                "decision_reason": "incorrect_low_weight",
                "deterministic_score": 0.8,
                "sampling_value": 0.8,
                "policy_name": "soft_pvf_medium",
                "branch": "soft_pvf_medium",
                "generation": 0,
                "seed": 123,
            },
        ]
    )
    soft_ctx = model_dir / "soft_pvf_medium" / "gen_0"
    decisions.to_parquet(soft_ctx / "soft_pvf_decisions.parquet", index=False)
    soft_report = {
        "model_name": "qwen2.5:0.5b",
        "branch": "soft_pvf_medium",
        "generation": 0,
        "seed": 123,
        "high_quality_threshold": 6,
        "medium_quality_threshold": 4,
        "min_keep_ratio": 0.1,
        "total_candidates": 50,
        "kept_count": 25,
        "rejected_count": 25,
        "keep_rate": 0.5,
        "effective_keep_rate": 0.6,
        "mean_assigned_weight": 0.6,
        "decision_reason_counts": {"high_quality": 1, "incorrect_low_weight": 1},
        "weight_distribution": {"1.00": 1, "0.10": 1},
    }
    (soft_ctx / "soft_pvf_report.json").write_text(
        json.dumps(soft_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return run_dir


def test_export_soft_pvf_analysis_writes_expected_artifacts() -> None:
    base = Path("outputs/.tmp") / f"soft_pvf_analysis_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = _make_fake_run(base)
        out_dir = base / "analysis" / "tables"
        artifacts = export_soft_pvf_analysis(run_dir=run_dir, out_dir=out_dir)

        required_paths = [
            artifacts.run_level_csv,
            artifacts.generation_deltas_csv,
            artifacts.branch_deltas_csv,
            artifacts.policy_summary_csv,
            artifacts.decision_reasons_csv,
            artifacts.keep_rejected_quality_csv,
            artifacts.weight_distribution_csv,
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
        assert {"branch", "generation", "keep_rate", "effective_keep_rate", "mean_assigned_weight"}.issubset(
            set(stress.columns)
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_export_soft_pvf_analysis_supports_custom_prefix() -> None:
    base = Path("outputs/.tmp") / f"soft_pvf_analysis_prefix_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = _make_fake_run(base)
        out_dir = base / "analysis" / "tables"
        artifacts = export_soft_pvf_analysis(
            run_dir=run_dir,
            out_dir=out_dir,
            file_prefix="soft_pvf_policy_tuning",
        )
        assert artifacts.run_level_csv.name == "soft_pvf_policy_tuning_run_level.csv"
        assert artifacts.policy_summary_csv.name == "soft_pvf_policy_tuning_policy_summary.csv"
        assert artifacts.stress_summary_csv.name == "soft_pvf_policy_tuning_stress_summary.csv"
        assert artifacts.run_level_csv.exists()
        assert artifacts.policy_summary_csv.exists()
        assert artifacts.stress_summary_csv.exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)
