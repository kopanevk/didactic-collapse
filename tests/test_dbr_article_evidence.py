from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pandas as pd

from didactic_collapse.analysis.dbr_article_evidence import (
    compute_collapse_metrics,
    export_dbr_article_evidence,
)


def _mk_run(base: Path, *, seed: int) -> Path:
    run_dir = base / f"dbr_run_seed{seed}"
    model_dir = run_dir / "qwen2.5_0.5b"
    data_root = base / "data"
    (data_root / "splits").mkdir(parents=True, exist_ok=True)
    heldout = pd.DataFrame(
        {
            "example_id": [f"ex_{i}" for i in range(50)],
            "question": [f"q_{i}" for i in range(50)],
            "answer": ["a"] * 50,
        }
    )
    heldout.to_parquet(data_root / "splits" / "heldout_test.parquet", index=False)

    snapshot = {
        "config": {
            "project": {"seed": seed},
            "paths": {"data_root": str(data_root), "output_root": str(base), "prompt_dir": "configs/prompts"},
        }
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    for branch in ("pure_recycling", "dbr_medium"):
        for generation in (0, 1, 2):
            step = model_dir / branch / f"gen_{generation}"
            step.mkdir(parents=True, exist_ok=True)
            eval_df = pd.DataFrame(
                {
                    "example_id": [f"ex_{i}" for i in range(50)],
                    "raw_response": [f"{branch}_resp_{i}" for i in range(50)],
                    "answer_gold": [str(i) for i in range(50)],
                    "pred_parse_success": [True] * 45 + [False] * 5,
                    "is_correct": [True] * 30 + [False] * 20,
                    "overall_pedagogical_score": [6] * 25 + [2] * 25,
                    "is_silent_error": [False] * 40 + [True] * 10,
                    "model_name": ["qwen2.5:0.5b"] * 50,
                }
            )
            judge_df = pd.DataFrame(
                {
                    "example_id": [f"ex_{i}" for i in range(50)],
                    "reasoning_soundness": [2] * 20 + [0] * 30,
                    "structure": [2] * 25 + [0] * 25,
                }
            )
            eval_df.to_parquet(step / "eval_merged.parquet", index=False)
            judge_df.to_parquet(step / "judge_outputs.parquet", index=False)

            if branch == "dbr_medium":
                decisions = pd.DataFrame(
                    {
                        "example_id": [f"ex_{i}" for i in range(50)],
                        "selected": [True] * 45 + [False] * 5,
                        "defect_parse_failure": [False] * 48 + [True] * 2,
                        "defect_incorrect": [False] * 30 + [True] * 20,
                        "defect_silent": [False] * 40 + [True] * 10,
                        "defect_low_reasoning": [False] * 20 + [True] * 30,
                        "defect_low_structure": [False] * 25 + [True] * 25,
                        "severity": [0] * 10 + [2] * 20 + [4] * 20,
                        "question_length_bin": ["medium"] * 50,
                    }
                )
                decisions.to_parquet(step / "dbr_decisions.parquet", index=False)
                decisions[decisions["selected"]].to_parquet(step / "dbr_training_dataset.parquet", index=False)
                report = {
                    "target_size": 50,
                    "selected_count": 45,
                    "selection_rate": 0.9,
                    "min_selection_rate": 0.8,
                    "relaxation_steps_used": ["relax_incorrect_answer"],
                    "defect_rates_before": {
                        "parse_failure": 0.04,
                        "incorrect_answer": 0.4,
                        "silent_error": 0.2,
                        "low_reasoning": 0.6,
                        "low_structure": 0.5,
                    },
                    "defect_rates_after": {
                        "parse_failure": 0.0,
                        "incorrect_answer": 0.35,
                        "silent_error": 0.15,
                        "low_reasoning": 0.55,
                        "low_structure": 0.45,
                    },
                    "budget_violations": {
                        "incorrect_answer": {
                            "allowed_rate": 0.3,
                            "allowed_count": 15,
                            "actual_count": 18,
                            "violation_count": 3,
                        }
                    },
                    "bucket_coverage_before": {"short": 10, "medium": 20, "long": 20},
                    "bucket_coverage_after": {"short": 9, "medium": 18, "long": 18},
                }
                (step / "dbr_budget_report.json").write_text(json.dumps(report), encoding="utf-8")

    return run_dir


def test_compute_collapse_metrics_basic() -> None:
    frame = pd.DataFrame(
        {
            "pred_parse_success": [True, False],
            "is_correct": [True, False],
            "overall_pedagogical_score": [6, 2],
            "is_silent_error": [False, True],
            "reasoning_soundness": [2, 0],
            "structure": [2, 0],
        }
    )
    metrics = compute_collapse_metrics(frame)
    assert metrics["sample_count"] == 2
    assert abs(metrics["accuracy_mean"] - 0.5) < 1e-9
    assert abs(metrics["parse_failure_pred_rate"] - 0.5) < 1e-9


def test_export_dbr_article_evidence_outputs_and_blinding() -> None:
    base = Path("test_workdirs") / f"dbr_article_ev_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run1 = _mk_run(base, seed=211)
        run2 = _mk_run(base, seed=212)
        run3 = _mk_run(base, seed=213)
        out_dir = base / "out"
        artifacts = export_dbr_article_evidence(
            run_dirs=[run1, run2, run3],
            out_dir=out_dir,
            manual_sample_size=30,
            manual_sample_seed=123,
        )

        for path in (
            artifacts.collapse_by_seed_csv,
            artifacts.collapse_summary_csv,
            artifacts.generation_curves_csv,
            artifacts.mechanism_defect_before_after_csv,
            artifacts.mechanism_budget_violations_csv,
            artifacts.mechanism_selection_rate_csv,
            artifacts.mechanism_bucket_coverage_csv,
            artifacts.mechanism_severity_distribution_csv,
            artifacts.manual_audit_template_csv,
            artifacts.manual_audit_template_xlsx,
            artifacts.manual_audit_hidden_key_csv,
            artifacts.integrity_report_json,
        ):
            assert path.exists(), f"Missing output: {path}"

        template = pd.read_csv(artifacts.manual_audit_template_csv)
        hidden = pd.read_csv(artifacts.manual_audit_hidden_key_csv)
        assert "A_branch" not in template.columns
        assert "B_branch" not in template.columns
        assert len(template) == len(hidden)
        assert set(hidden["A_branch"].unique().tolist()).issubset({"pure_recycling", "dbr_medium"})
        assert set(hidden["B_branch"].unique().tolist()).issubset({"pure_recycling", "dbr_medium"})

        mech_sel = pd.read_csv(artifacts.mechanism_selection_rate_csv)
        assert (mech_sel["selected_count_reported"] == mech_sel["selected_count_training_rows"]).all()
    finally:
        shutil.rmtree(base, ignore_errors=True)

