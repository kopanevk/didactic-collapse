from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.recycling.pedagogical_verification_filter import (
    PVFError,
    PVFPolicy,
    apply_pedagogical_verification_filter,
    save_pvf_artifacts,
)


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e1", "question": "q1", "answer_for_training": "a1", "source": "synthetic"},
            {"example_id": "e2", "question": "q2", "answer_for_training": "a2", "source": "synthetic"},
            {"example_id": "e3", "question": "q3", "answer_for_training": "a3", "source": "synthetic"},
            {"example_id": "e4", "question": "q4", "answer_for_training": "a4", "source": "synthetic"},
        ]
    )


def _accuracy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e1", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e2", "pred_parse_success": False, "accuracy_label": "parse_failure", "is_correct": False},
            {"example_id": "e3", "pred_parse_success": True, "accuracy_label": "wrong", "is_correct": False},
            {"example_id": "e4", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
        ]
    )


def _judge_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e1", "overall_pedagogical_score": 7, "is_silent_error": False},
            {"example_id": "e2", "overall_pedagogical_score": 7, "is_silent_error": False},
            {"example_id": "e3", "overall_pedagogical_score": 7, "is_silent_error": False},
            {"example_id": "e4", "overall_pedagogical_score": 4, "is_silent_error": True},
        ]
    )


def test_pvf_keeps_only_correct_high_pedag_non_silent_parse_success() -> None:
    result = apply_pedagogical_verification_filter(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="pvf_medium",
        generation=1,
        seed=42,
        policy=PVFPolicy(threshold_score=5, min_keep_ratio=0.2),
    )
    assert set(result.filtered_training_df["example_id"].tolist()) == {"e1"}
    assert result.report.kept_count == 1
    assert result.report.rejected_count == 3


def test_pvf_rejects_parse_failures_and_silent_errors_and_low_pedag() -> None:
    result = apply_pedagogical_verification_filter(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="pvf_medium",
        generation=1,
        seed=42,
        policy=PVFPolicy(threshold_score=5, min_keep_ratio=0.2),
    )
    rejected = result.rejected_examples_df.set_index("example_id")
    assert "pred_parse_failure" in str(rejected.loc["e2", "pvf_reject_reasons"])
    assert "accuracy_incorrect" in str(rejected.loc["e3", "pvf_reject_reasons"])
    assert "pedagogical_below_threshold" in str(rejected.loc["e4", "pvf_reject_reasons"])
    assert "silent_error_true" in str(rejected.loc["e4", "pvf_reject_reasons"])


def test_pvf_strict_merge_prevents_row_explosion() -> None:
    acc = _accuracy_df()
    acc = pd.concat([acc, acc.iloc[[0]]], ignore_index=True)
    with pytest.raises(PVFError, match="duplicate example_id"):
        apply_pedagogical_verification_filter(
            synthetic_df=_synthetic_df(),
            accuracy_df=acc,
            judge_df=_judge_df(),
            model_name="qwen2.5:0.5b",
            branch="pvf_medium",
            generation=1,
            seed=42,
            policy=PVFPolicy(threshold_score=5, min_keep_ratio=0.2),
        )


def test_pvf_strict_mode_fails_on_missing_judge_rows() -> None:
    with pytest.raises(PVFError, match="synthetic and judge example_id sets differ"):
        apply_pedagogical_verification_filter(
            synthetic_df=_synthetic_df(),
            accuracy_df=_accuracy_df(),
            judge_df=_judge_df().iloc[:3].copy(),
            model_name="qwen2.5:0.5b",
            branch="pvf_medium",
            generation=1,
            seed=42,
            policy=PVFPolicy(threshold_score=5, min_keep_ratio=0.2),
        )


def test_pvf_report_counts_are_correct() -> None:
    result = apply_pedagogical_verification_filter(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="pvf_medium",
        generation=1,
        seed=42,
        policy=PVFPolicy(threshold_score=5, min_keep_ratio=0.2),
    )
    rep = result.report
    assert rep.total_candidates == 4
    assert rep.kept_count == 1
    assert rep.rejected_count == 3
    assert abs(rep.keep_rate - 0.25) < 1e-9
    assert rep.rejection_reason_counts["pred_parse_failure"] == 1
    assert rep.rejection_reason_counts["accuracy_incorrect"] == 2
    assert rep.rejection_reason_counts["pedagogical_below_threshold"] == 1
    assert rep.rejection_reason_counts["silent_error_true"] == 1


def test_pvf_partial_inputs_reject_missing_judge_rows_when_enabled() -> None:
    judge = _judge_df().iloc[:3].copy()  # drop e4 to simulate row-level judge failure
    result = apply_pedagogical_verification_filter(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=judge,
        model_name="qwen2.5:0.5b",
        branch="pvf_medium",
        generation=1,
        seed=42,
        policy=PVFPolicy(threshold_score=5, min_keep_ratio=0.2),
        allow_partial_inputs=True,
    )
    rejected = result.rejected_examples_df.set_index("example_id")
    assert "missing_judge_row" in str(rejected.loc["e4", "pvf_reject_reasons"])
    assert result.report.rejection_reason_counts["missing_judge_row"] == 1
    assert result.report.total_candidates == 4


def test_pvf_save_artifacts_roundtrip() -> None:
    result = apply_pedagogical_verification_filter(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="pvf_medium",
        generation=1,
        seed=42,
        policy=PVFPolicy(threshold_score=5, min_keep_ratio=0.2),
    )
    base = Path("outputs/.tmp") / f"pvf_artifacts_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        filtered = base / "filtered_training_dataset.parquet"
        rejected = base / "rejected_examples.parquet"
        report = base / "pvf_filter_report.json"
        save_pvf_artifacts(
            result=result,
            filtered_path=filtered,
            rejected_path=rejected,
            report_path=report,
        )
        assert filtered.exists()
        assert rejected.exists()
        assert report.exists()
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["kept_count"] == 1
        assert payload["rejected_count"] == 3
    finally:
        shutil.rmtree(base, ignore_errors=True)
