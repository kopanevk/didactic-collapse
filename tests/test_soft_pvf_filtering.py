from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.recycling.pedagogical_verification_filter import (
    SoftPVFPolicy,
    apply_soft_pvf,
    deterministic_weighted_keep,
    save_soft_pvf_artifacts,
)


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e_parse_fail", "question": "q1", "answer_for_training": "a1", "source": "synthetic"},
            {"example_id": "e_silent", "question": "q2", "answer_for_training": "a2", "source": "synthetic"},
            {"example_id": "e_high", "question": "q3", "answer_for_training": "a3", "source": "synthetic"},
            {"example_id": "e_medium", "question": "q4", "answer_for_training": "a4", "source": "synthetic"},
            {"example_id": "e_low_correct", "question": "q5", "answer_for_training": "a5", "source": "synthetic"},
            {"example_id": "e_incorrect", "question": "q6", "answer_for_training": "a6", "source": "synthetic"},
        ]
    )


def _accuracy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e_parse_fail", "pred_parse_success": False, "accuracy_label": "parse_failure", "is_correct": False},
            {"example_id": "e_silent", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e_high", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e_medium", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e_low_correct", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e_incorrect", "pred_parse_success": True, "accuracy_label": "wrong", "is_correct": False},
        ]
    )


def _judge_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e_parse_fail", "overall_pedagogical_score": 7, "is_silent_error": False},
            {"example_id": "e_silent", "overall_pedagogical_score": 7, "is_silent_error": True},
            {"example_id": "e_high", "overall_pedagogical_score": 6, "is_silent_error": False},
            {"example_id": "e_medium", "overall_pedagogical_score": 4, "is_silent_error": False},
            {"example_id": "e_low_correct", "overall_pedagogical_score": 2, "is_silent_error": False},
            {"example_id": "e_incorrect", "overall_pedagogical_score": 5, "is_silent_error": False},
        ]
    )


def _policy_synthetic_df() -> pd.DataFrame:
    ids = [
        "pf",
        "silent",
        "high_corr",
        "med_high",
        "med_low",
        "low_corr",
        "inc_high",
        "inc_low",
    ]
    return pd.DataFrame(
        [{"example_id": i, "question": f"q_{i}", "answer_for_training": f"a_{i}", "source": "synthetic"} for i in ids]
    )


def _policy_accuracy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "pf", "pred_parse_success": False, "accuracy_label": "parse_failure", "is_correct": False},
            {"example_id": "silent", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "high_corr", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "med_high", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "med_low", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "low_corr", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "inc_high", "pred_parse_success": True, "accuracy_label": "wrong", "is_correct": False},
            {"example_id": "inc_low", "pred_parse_success": True, "accuracy_label": "wrong", "is_correct": False},
        ]
    )


def _policy_judge_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "pf", "overall_pedagogical_score": 7, "is_silent_error": False},
            {"example_id": "silent", "overall_pedagogical_score": 7, "is_silent_error": True},
            {"example_id": "high_corr", "overall_pedagogical_score": 6, "is_silent_error": False},
            {"example_id": "med_high", "overall_pedagogical_score": 4, "is_silent_error": False},
            {"example_id": "med_low", "overall_pedagogical_score": 2, "is_silent_error": False},
            {"example_id": "low_corr", "overall_pedagogical_score": 1, "is_silent_error": False},
            {"example_id": "inc_high", "overall_pedagogical_score": 5, "is_silent_error": False},
            {"example_id": "inc_low", "overall_pedagogical_score": 1, "is_silent_error": False},
        ]
    )


def test_deterministic_weighted_keep_is_reproducible() -> None:
    k1, s1 = deterministic_weighted_keep(
        weight=0.5,
        seed=123,
        branch="soft_pvf_medium",
        generation=1,
        example_id="ex-1",
    )
    k2, s2 = deterministic_weighted_keep(
        weight=0.5,
        seed=123,
        branch="soft_pvf_medium",
        generation=1,
        example_id="ex-1",
    )
    assert k1 == k2
    assert abs(s1 - s2) < 1e-12


def test_soft_pvf_weight_assignment_and_rejections() -> None:
    result = apply_soft_pvf(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_medium",
        generation=1,
        seed=42,
        policy=SoftPVFPolicy(min_keep_ratio=0.1),
    )
    decisions = result.decisions_df.set_index("example_id")

    # Hard exclusions
    assert decisions.loc["e_parse_fail", "assigned_weight"] == 0.0
    assert decisions.loc["e_parse_fail", "kept"] == False  # noqa: E712
    assert decisions.loc["e_parse_fail", "decision_reason"] == "pred_parse_failure"
    assert decisions.loc["e_silent", "assigned_weight"] == 0.0
    assert decisions.loc["e_silent", "kept"] == False  # noqa: E712
    assert decisions.loc["e_silent", "decision_reason"] == "silent_error_true"

    # Quality buckets
    assert decisions.loc["e_high", "assigned_weight"] == 1.0
    assert decisions.loc["e_high", "kept"] == True  # noqa: E712
    assert decisions.loc["e_high", "decision_reason"] == "high_quality"
    assert decisions.loc["e_medium", "assigned_weight"] == 0.5
    assert decisions.loc["e_medium", "decision_reason"] == "medium_quality"
    assert decisions.loc["e_low_correct", "assigned_weight"] == 0.25
    assert decisions.loc["e_low_correct", "decision_reason"] == "low_pedagogy_correct"
    assert decisions.loc["e_incorrect", "assigned_weight"] == 0.1
    assert decisions.loc["e_incorrect", "decision_reason"] == "incorrect_low_weight"
    assert result.report.policy_name == "soft_pvf_medium"


def test_soft_pvf_sampling_stable_across_row_order() -> None:
    s = _synthetic_df()
    a = _accuracy_df()
    j = _judge_df()

    out_a = apply_soft_pvf(
        synthetic_df=s,
        accuracy_df=a,
        judge_df=j,
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_medium",
        generation=2,
        seed=99,
        policy=SoftPVFPolicy(min_keep_ratio=0.1),
    )
    out_b = apply_soft_pvf(
        synthetic_df=s.sample(frac=1.0, random_state=7).reset_index(drop=True),
        accuracy_df=a.sample(frac=1.0, random_state=8).reset_index(drop=True),
        judge_df=j.sample(frac=1.0, random_state=9).reset_index(drop=True),
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_medium",
        generation=2,
        seed=99,
        policy=SoftPVFPolicy(min_keep_ratio=0.1),
    )

    dec_a = out_a.decisions_df.sort_values("example_id").reset_index(drop=True)
    dec_b = out_b.decisions_df.sort_values("example_id").reset_index(drop=True)
    assert dec_a[["example_id", "assigned_weight", "kept", "deterministic_score"]].equals(
        dec_b[["example_id", "assigned_weight", "kept", "deterministic_score"]]
    )


def test_soft_pvf_report_counts_and_artifacts() -> None:
    result = apply_soft_pvf(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_medium",
        generation=1,
        seed=42,
        policy=SoftPVFPolicy(min_keep_ratio=0.1),
    )
    assert result.report.total_candidates == 6
    assert result.report.kept_count >= 1
    assert result.report.rejected_count >= 1
    assert result.report.decision_reason_counts["pred_parse_failure"] == 1
    assert result.report.decision_reason_counts["silent_error_true"] == 1
    assert "1.00" in result.report.weight_distribution

    base = Path("outputs/.tmp") / f"soft_pvf_artifacts_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        training_path = base / "soft_pvf_training_dataset.parquet"
        decisions_path = base / "soft_pvf_decisions.parquet"
        report_path = base / "soft_pvf_report.json"
        save_soft_pvf_artifacts(
            result=result,
            training_path=training_path,
            decisions_path=decisions_path,
            report_path=report_path,
        )
        assert training_path.exists()
        assert decisions_path.exists()
        assert report_path.exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_soft_pvf_lenient_policy_assigns_expected_weights() -> None:
    result = apply_soft_pvf(
        synthetic_df=_policy_synthetic_df(),
        accuracy_df=_policy_accuracy_df(),
        judge_df=_policy_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_lenient",
        generation=1,
        seed=11,
        policy=SoftPVFPolicy(policy_name="soft_pvf_lenient", min_keep_ratio=0.1),
    )
    decisions = result.decisions_df.set_index("example_id")
    assert decisions.loc["pf", "assigned_weight"] == 0.0
    assert decisions.loc["silent", "assigned_weight"] == 0.0
    assert decisions.loc["high_corr", "assigned_weight"] == 1.0
    assert decisions.loc["med_high", "assigned_weight"] == 0.75
    assert decisions.loc["med_low", "assigned_weight"] == 0.5
    assert decisions.loc["low_corr", "assigned_weight"] == 0.25
    assert decisions.loc["inc_high", "assigned_weight"] == 0.25
    assert decisions.loc["inc_low", "assigned_weight"] == 0.1
    assert result.report.policy_name == "soft_pvf_lenient"


def test_soft_pvf_noisy_keep_policy_assigns_expected_weights() -> None:
    result = apply_soft_pvf(
        synthetic_df=_policy_synthetic_df(),
        accuracy_df=_policy_accuracy_df(),
        judge_df=_policy_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_noisy_keep",
        generation=1,
        seed=11,
        policy=SoftPVFPolicy(policy_name="soft_pvf_noisy_keep", min_keep_ratio=0.1),
    )
    decisions = result.decisions_df.set_index("example_id")
    assert decisions.loc["pf", "assigned_weight"] == 0.0
    assert decisions.loc["silent", "assigned_weight"] == 0.0
    assert decisions.loc["high_corr", "assigned_weight"] == 1.0
    assert decisions.loc["med_high", "assigned_weight"] == 0.75
    assert decisions.loc["med_low", "assigned_weight"] == 0.5
    assert decisions.loc["low_corr", "assigned_weight"] == 0.5
    assert decisions.loc["inc_high", "assigned_weight"] == 0.25
    assert decisions.loc["inc_low", "assigned_weight"] == 0.1
    assert result.report.policy_name == "soft_pvf_noisy_keep"


def test_soft_pvf_silent_only_keeps_all_non_silent_parsed_rows() -> None:
    result = apply_soft_pvf(
        synthetic_df=_policy_synthetic_df(),
        accuracy_df=_policy_accuracy_df(),
        judge_df=_policy_judge_df(),
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_silent_only",
        generation=1,
        seed=11,
        policy=SoftPVFPolicy(policy_name="soft_pvf_silent_only", min_keep_ratio=0.1),
    )
    decisions = result.decisions_df.set_index("example_id")
    assert decisions.loc["pf", "kept"] == False  # noqa: E712
    assert decisions.loc["silent", "kept"] == False  # noqa: E712
    for eid in ["high_corr", "med_high", "med_low", "low_corr", "inc_high", "inc_low"]:
        assert decisions.loc[eid, "assigned_weight"] == 1.0
        assert decisions.loc[eid, "kept"] == True  # noqa: E712
    assert result.report.policy_name == "soft_pvf_silent_only"


def test_soft_pvf_lenient_and_noisy_sampling_is_deterministic() -> None:
    s = _policy_synthetic_df()
    a = _policy_accuracy_df()
    j = _policy_judge_df()
    out1 = apply_soft_pvf(
        synthetic_df=s,
        accuracy_df=a,
        judge_df=j,
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_lenient",
        generation=2,
        seed=101,
        policy=SoftPVFPolicy(policy_name="soft_pvf_lenient", min_keep_ratio=0.1),
    )
    out2 = apply_soft_pvf(
        synthetic_df=s.sample(frac=1.0, random_state=1).reset_index(drop=True),
        accuracy_df=a.sample(frac=1.0, random_state=2).reset_index(drop=True),
        judge_df=j.sample(frac=1.0, random_state=3).reset_index(drop=True),
        model_name="qwen2.5:0.5b",
        branch="soft_pvf_lenient",
        generation=2,
        seed=101,
        policy=SoftPVFPolicy(policy_name="soft_pvf_lenient", min_keep_ratio=0.1),
    )
    left = out1.decisions_df.sort_values("example_id").reset_index(drop=True)
    right = out2.decisions_df.sort_values("example_id").reset_index(drop=True)
    assert left[["example_id", "assigned_weight", "kept", "sampling_value"]].equals(
        right[["example_id", "assigned_weight", "kept", "sampling_value"]]
    )
