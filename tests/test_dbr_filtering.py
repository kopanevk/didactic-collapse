from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.recycling.defect_budgeted_recycling import (
    DBRPolicy,
    apply_dbr,
    assign_question_length_bin,
    compute_defect_flags,
    compute_severity,
    save_dbr_artifacts,
)


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e1", "question": "short q", "answer_for_training": "a1", "source": "synthetic"},
            {"example_id": "e2", "question": "medium question " * 8, "answer_for_training": "a2", "source": "synthetic"},
            {"example_id": "e3", "question": "long question " * 22, "answer_for_training": "a3", "source": "synthetic"},
            {"example_id": "e4", "question": "short q 2", "answer_for_training": "a4", "source": "synthetic"},
            {"example_id": "e5", "question": "medium 2 " * 8, "answer_for_training": "a5", "source": "synthetic"},
            {"example_id": "e6", "question": "long 2 " * 20, "answer_for_training": "a6", "source": "synthetic"},
        ]
    )


def _accuracy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e2", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e3", "pred_parse_success": False, "accuracy_label": "parse_failure", "is_correct": False},
            {"example_id": "e4", "pred_parse_success": True, "accuracy_label": "wrong", "is_correct": False},
            {"example_id": "e5", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e6", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
        ]
    )


def _judge_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "example_id": "e1",
                "overall_pedagogical_score": 6,
                "is_silent_error": False,
                "reasoning_soundness": 2,
                "structure": 2,
                "clarity": 2,
            },
            {
                "example_id": "e2",
                "overall_pedagogical_score": 5,
                "is_silent_error": False,
                "reasoning_soundness": 1,
                "structure": 1,
                "clarity": 1,
            },
            {
                "example_id": "e3",
                "overall_pedagogical_score": 2,
                "is_silent_error": True,
                "reasoning_soundness": 0,
                "structure": 0,
                "clarity": 0,
            },
            {
                "example_id": "e4",
                "overall_pedagogical_score": 4,
                "is_silent_error": False,
                "reasoning_soundness": 1,
                "structure": 1,
                "clarity": 1,
            },
            {
                "example_id": "e5",
                "overall_pedagogical_score": 3,
                "is_silent_error": False,
                "reasoning_soundness": 0,
                "structure": 2,
                "clarity": 1,
            },
            {
                "example_id": "e6",
                "overall_pedagogical_score": 3,
                "is_silent_error": False,
                "reasoning_soundness": 2,
                "structure": 0,
                "clarity": 1,
            },
        ]
    )


def test_defect_flags_and_severity_are_computed_correctly() -> None:
    row = pd.Series(
        {
            "pred_parse_success": False,
            "is_correct_effective": False,
            "is_silent_error": True,
            "reasoning_soundness": 0,
            "structure": 0,
        }
    )
    flags = compute_defect_flags(row)
    assert flags == {
        "defect_parse_failure": True,
        "defect_incorrect": True,
        "defect_silent": True,
        "defect_low_reasoning": True,
        "defect_low_structure": True,
    }
    assert compute_severity(flags) == 13


def test_question_length_bin_assignment_is_stable() -> None:
    b1, fb1 = assign_question_length_bin(
        question="abc",
        example_id="ex",
        seed=1,
        branch="dbr_medium",
        generation=1,
        short_max_chars=10,
        medium_max_chars=20,
    )
    assert b1 == "short"
    assert fb1 is False
    b2, fb2 = assign_question_length_bin(
        question=None,
        example_id="ex",
        seed=1,
        branch="dbr_medium",
        generation=1,
        short_max_chars=10,
        medium_max_chars=20,
    )
    b3, fb3 = assign_question_length_bin(
        question=None,
        example_id="ex",
        seed=1,
        branch="dbr_medium",
        generation=1,
        short_max_chars=10,
        medium_max_chars=20,
    )
    assert fb2 is True and fb3 is True
    assert b2 == b3


def test_dbr_respects_zero_parse_failure_budget_when_possible() -> None:
    result = apply_dbr(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="dbr_medium",
        generation=1,
        seed=11,
        policy=DBRPolicy(
            target_size_ratio=0.6,
            budget_parse_failure=0.0,
            budget_silent_error=0.5,
            budget_incorrect_answer=0.5,
            budget_low_reasoning=0.5,
            budget_low_structure=0.5,
            allow_parse_failure_fallback=False,
        ),
    )
    selected = result.decisions_df[result.decisions_df["selected"] == True]  # noqa: E712
    assert not selected["defect_parse_failure"].any()


def test_dbr_respects_silent_error_budget_when_possible() -> None:
    result = apply_dbr(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="dbr_medium",
        generation=1,
        seed=11,
        policy=DBRPolicy(
            target_size_ratio=0.5,
            budget_parse_failure=0.0,
            budget_silent_error=0.0,
            budget_incorrect_answer=1.0,
            budget_low_reasoning=1.0,
            budget_low_structure=1.0,
            allow_parse_failure_fallback=False,
        ),
    )
    selected = result.decisions_df[result.decisions_df["selected"] == True]  # noqa: E712
    assert not selected["defect_silent"].any()


def test_dbr_fills_target_with_fallback_and_logs_violations() -> None:
    synthetic = _synthetic_df().drop(index=[2]).reset_index(drop=True)  # remove parse-failure row
    accuracy = _accuracy_df().drop(index=[2]).reset_index(drop=True)
    judge = _judge_df().drop(index=[2]).reset_index(drop=True)

    result = apply_dbr(
        synthetic_df=synthetic,
        accuracy_df=accuracy,
        judge_df=judge,
        model_name="qwen2.5:0.5b",
        branch="dbr_medium",
        generation=1,
        seed=3,
        policy=DBRPolicy(
            target_size_ratio=1.0,
            budget_parse_failure=0.0,
            budget_silent_error=0.0,
            budget_incorrect_answer=0.0,
            budget_low_reasoning=0.0,
            budget_low_structure=0.0,
            allow_parse_failure_fallback=False,
        ),
    )
    assert result.report.selected_count == result.report.target_size
    assert any(
        v.get("violation_count", 0) > 0
        for k, v in result.report.budget_violations.items()
        if isinstance(v, dict)
    )
    selected = result.decisions_df[result.decisions_df["selected"] == True]  # noqa: E712
    assert result.report.relaxation_steps_used or selected["budget_violation_if_selected"].any()


def test_dbr_decision_report_contains_bucket_coverage() -> None:
    result = apply_dbr(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="dbr_medium",
        generation=2,
        seed=9,
        policy=DBRPolicy(target_size_ratio=0.7),
    )
    assert set(result.report.bucket_coverage_before.keys()) == {"short", "medium", "long"}
    assert set(result.report.bucket_coverage_after.keys()) == {"short", "medium", "long"}


def test_dbr_sampling_is_deterministic_across_input_order() -> None:
    s = _synthetic_df()
    a = _accuracy_df()
    j = _judge_df()
    p = DBRPolicy(target_size_ratio=0.8)
    out1 = apply_dbr(
        synthetic_df=s,
        accuracy_df=a,
        judge_df=j,
        model_name="qwen2.5:0.5b",
        branch="dbr_medium",
        generation=1,
        seed=77,
        policy=p,
    )
    out2 = apply_dbr(
        synthetic_df=s.sample(frac=1.0, random_state=1).reset_index(drop=True),
        accuracy_df=a.sample(frac=1.0, random_state=2).reset_index(drop=True),
        judge_df=j.sample(frac=1.0, random_state=3).reset_index(drop=True),
        model_name="qwen2.5:0.5b",
        branch="dbr_medium",
        generation=1,
        seed=77,
        policy=p,
    )
    d1 = out1.decisions_df.sort_values("example_id").reset_index(drop=True)
    d2 = out2.decisions_df.sort_values("example_id").reset_index(drop=True)
    assert d1[["example_id", "selected", "severity", "sampling_value"]].equals(
        d2[["example_id", "selected", "severity", "sampling_value"]]
    )


def test_save_dbr_artifacts_writes_expected_files() -> None:
    result = apply_dbr(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="dbr_medium",
        generation=1,
        seed=1,
        policy=DBRPolicy(target_size_ratio=0.7),
    )
    base = Path("outputs/.tmp") / f"dbr_artifacts_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        training_path = base / "dbr_training_dataset.parquet"
        decisions_path = base / "dbr_decisions.parquet"
        report_path = base / "dbr_budget_report.json"
        save_dbr_artifacts(
            result=result,
            training_path=training_path,
            decisions_path=decisions_path,
            report_path=report_path,
        )
        assert training_path.exists()
        assert decisions_path.exists()
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["policy_name"] == "dbr_medium"
    finally:
        shutil.rmtree(base, ignore_errors=True)
