from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.pair_lite import export_pair_lite_analysis


def _make_fake_run(root: Path) -> Path:
    run_dir = root / "pair_lite_fake_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.snapshot.json").write_text(
        json.dumps({"config": {"project": {"seed": 654}}}, ensure_ascii=False, indent=2),
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
                "branch": "pair_lite_medium",
                "generation": 0,
                "sample_count": 4,
                "accuracy_mean": 0.6,
                "pedagogical_score_mean": 5.5,
                "silent_error_rate": 0.2,
            },
        ]
    )
    summary.to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)

    model_dir = run_dir / "qwen2.5_0.5b"
    for branch in ["pure_recycling", "pair_lite_medium"]:
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

    pair_ctx = model_dir / "pair_lite_medium" / "gen_0"
    decisions = pd.DataFrame(
        [
            {
                "example_id": "e1",
                "branch": "pair_lite_medium",
                "generation": 0,
                "seed": 654,
                "is_correct": True,
                "pred_parse_success": True,
                "overall_pedagogical_score": 7.0,
                "is_silent_error": False,
                "action_initial": "keep_original",
                "action_final": "keep_original",
                "decision_reason": "high_quality",
                "repair_attempted": False,
                "repair_success": False,
                "original_response_hash": "h1",
                "repaired_response_hash": None,
                "repair_model_name": None,
                "repair_error": None,
            },
            {
                "example_id": "e2",
                "branch": "pair_lite_medium",
                "generation": 0,
                "seed": 654,
                "is_correct": True,
                "pred_parse_success": True,
                "overall_pedagogical_score": 4.0,
                "is_silent_error": False,
                "action_initial": "repair_pedagogy",
                "action_final": "repair_pedagogy",
                "decision_reason": "correct_but_pedagogically_weak",
                "repair_attempted": True,
                "repair_success": True,
                "original_response_hash": "h2",
                "repaired_response_hash": "rh2",
                "repair_model_name": "llama-3.1-8b",
                "repair_error": None,
            },
            {
                "example_id": "e3",
                "branch": "pair_lite_medium",
                "generation": 0,
                "seed": 654,
                "is_correct": False,
                "pred_parse_success": True,
                "overall_pedagogical_score": 3.0,
                "is_silent_error": False,
                "action_initial": "reject",
                "action_final": "reject",
                "decision_reason": "accuracy_incorrect",
                "repair_attempted": False,
                "repair_success": False,
                "original_response_hash": "h3",
                "repaired_response_hash": None,
                "repair_model_name": None,
                "repair_error": None,
            },
        ]
    )
    decisions.to_parquet(pair_ctx / "pair_lite_decisions.parquet", index=False)
    report = {
        "model_name": "qwen2.5:0.5b",
        "branch": "pair_lite_medium",
        "generation": 0,
        "seed": 654,
        "threshold_score": 6,
        "min_keep_ratio": 0.1,
        "target_rows": 3,
        "kept_original_count": 1,
        "repaired_count": 1,
        "rejected_count": 1,
        "repair_attempted_count": 1,
        "repair_success_count": 1,
        "repair_failure_count": 0,
        "keep_original_rate": 0.3333333333,
        "repair_rate": 0.3333333333,
        "reject_rate": 0.3333333333,
        "repair_success_rate": 1.0,
        "decision_reason_counts": {
            "high_quality": 1,
            "correct_but_pedagogically_weak": 1,
            "accuracy_incorrect": 1,
        },
        "repair_model_name": "llama-3.1-8b",
    }
    (pair_ctx / "pair_lite_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def test_export_pair_lite_analysis_writes_expected_outputs() -> None:
    base = Path("outputs/.tmp") / f"pair_lite_analysis_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = _make_fake_run(base)
        out_dir = base / "analysis" / "tables"
        artifacts = export_pair_lite_analysis(run_dir=run_dir, out_dir=out_dir)

        required = [
            artifacts.run_level_csv,
            artifacts.generation_deltas_csv,
            artifacts.branch_deltas_csv,
            artifacts.action_reasons_csv,
            artifacts.repair_success_summary_csv,
            artifacts.keep_repair_reject_quality_csv,
            artifacts.stress_summary_csv,
            artifacts.accuracy_plot,
            artifacts.pedagogical_plot,
            artifacts.silent_error_plot,
            artifacts.keep_rate_plot,
            artifacts.metadata_json,
        ]
        missing = [str(p) for p in required if not p.exists()]
        assert not missing, f"Missing artifacts: {missing}"

        stress = pd.read_csv(artifacts.stress_summary_csv)
        assert {"branch", "generation", "keep_original_rate", "repair_rate", "reject_rate"}.issubset(stress.columns)
    finally:
        shutil.rmtree(base, ignore_errors=True)
