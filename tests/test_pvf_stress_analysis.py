from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.pvf_stress import export_pvf_stress_analysis


def test_export_pvf_stress_analysis_writes_expected_tables() -> None:
    base = Path("outputs/.tmp") / f"pvf_stress_{uuid.uuid4().hex}"
    run_dir = base / "run"
    try:
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "pure_recycling",
                    "generation": 0,
                    "example_id": "e1",
                    "is_correct": True,
                    "overall_pedagogical_score": 6,
                    "is_silent_error": False,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "pvf_medium",
                    "generation": 0,
                    "example_id": "e2",
                    "is_correct": True,
                    "overall_pedagogical_score": 6,
                    "is_silent_error": False,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "pvf_medium",
                    "generation": 1,
                    "example_id": "e3",
                    "is_correct": False,
                    "overall_pedagogical_score": 3,
                    "is_silent_error": True,
                },
            ]
        ).to_parquet(run_dir / "all_eval_merged.parquet", index=False)

        pvf_step = run_dir / "qwen2.5_0.5b" / "pvf_medium" / "gen_1"
        pvf_step.mkdir(parents=True, exist_ok=True)
        (pvf_step / "pvf_filter_report.json").write_text(
            json.dumps(
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "pvf_medium",
                    "generation": 1,
                    "seed": 1,
                    "threshold_score": 5,
                    "min_keep_ratio": 0.1,
                    "total_candidates": 10,
                    "kept_count": 6,
                    "rejected_count": 4,
                    "keep_rate": 0.6,
                    "rejection_reason_counts": {
                        "pred_parse_failure": 2,
                        "accuracy_incorrect": 3,
                        "pedagogical_below_threshold": 4,
                        "silent_error_true": 1,
                    },
                    "kept_accuracy_mean": 1.0,
                    "rejected_accuracy_mean": 0.0,
                    "kept_pedagogical_mean": 6.2,
                    "rejected_pedagogical_mean": 3.1,
                    "kept_silent_error_rate": 0.0,
                    "rejected_silent_error_rate": 1.0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        artifacts = export_pvf_stress_analysis(run_dir=run_dir, out_dir=run_dir / "tables")
        assert artifacts.stress_summary_csv.exists()
        assert artifacts.generation_deltas_csv.exists()
        assert artifacts.reject_reasons_csv.exists()
        assert artifacts.keep_reject_quality_csv.exists()

        summary = pd.read_csv(artifacts.stress_summary_csv)
        assert "keep_rate" in summary.columns
        assert "pedagogical_score_mean" in summary.columns
        reasons = pd.read_csv(artifacts.reject_reasons_csv)
        assert int(reasons["count"].sum()) > 0
    finally:
        shutil.rmtree(base, ignore_errors=True)

