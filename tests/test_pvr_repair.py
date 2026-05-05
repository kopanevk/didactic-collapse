from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.recycling.pedagogical_verification_repair import (
    PVRError,
    PVRPolicy,
    apply_pvr,
    save_pvr_artifacts,
)


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e_keep", "question": "1+1?", "answer_for_training": "Reasoning\nFinal answer: 2", "source": "synthetic"},
            {"example_id": "e_repair", "question": "3+2?", "answer_for_training": "Hard to read\nFinal answer: 5", "source": "synthetic"},
            {"example_id": "e_incorrect", "question": "2+2?", "answer_for_training": "Wrong\nFinal answer: 5", "source": "synthetic"},
            {"example_id": "e_silent", "question": "4-1?", "answer_for_training": "Looks ok\nFinal answer: 3", "source": "synthetic"},
            {"example_id": "e_parse", "question": "10-3?", "answer_for_training": "No final numeric line", "source": "synthetic"},
        ]
    )


def _accuracy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e_keep", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e_repair", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e_incorrect", "pred_parse_success": True, "accuracy_label": "incorrect", "is_correct": False},
            {"example_id": "e_silent", "pred_parse_success": True, "accuracy_label": "correct", "is_correct": True},
            {"example_id": "e_parse", "pred_parse_success": False, "accuracy_label": "prediction_parse_failure", "is_correct": False},
        ]
    )


def _judge_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e_keep", "overall_pedagogical_score": 7, "is_silent_error": False},
            {"example_id": "e_repair", "overall_pedagogical_score": 4, "is_silent_error": False},
            {"example_id": "e_incorrect", "overall_pedagogical_score": 6, "is_silent_error": False},
            {"example_id": "e_silent", "overall_pedagogical_score": 7, "is_silent_error": True},
            {"example_id": "e_parse", "overall_pedagogical_score": 7, "is_silent_error": False},
        ]
    )


def _gold_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"example_id": "e_keep", "answer_gold": "#### 2"},
            {"example_id": "e_repair", "answer_gold": "#### 5"},
            {"example_id": "e_incorrect", "answer_gold": "#### 4"},
            {"example_id": "e_silent", "answer_gold": "#### 3"},
            {"example_id": "e_parse", "answer_gold": "#### 7"},
        ]
    )


def _repair_ok(question: str, gold_answer: str, original_response: str) -> str:
    _ = question, original_response
    # Gold is GSM8K-style text, keep numeric tail in final line.
    final = str(gold_answer).split("####", maxsplit=1)[-1].strip()
    return (
        "Step 1: identify numbers.\n"
        "Step 2: compute carefully.\n"
        f"Final answer: {final}"
    )


def _repair_fail(question: str, gold_answer: str, original_response: str) -> str:
    _ = question, gold_answer, original_response
    raise RuntimeError("provider timeout")


def test_pvr_policy_actions_and_lineage_columns() -> None:
    out = apply_pvr(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        gold_df=_gold_df(),
        model_name="qwen2.5:0.5b",
        branch="pvr_repair_medium",
        generation=1,
        seed=42,
        policy=PVRPolicy(threshold_score=6, min_keep_ratio=0.2),
        repair_model_name="llama-3.1-8b",
        repair_callable=_repair_ok,
    )
    decisions = out.decisions_df.set_index("example_id")

    assert decisions.loc["e_keep", "action"] == "keep_original"
    assert decisions.loc["e_repair", "action"] == "repair_explanation"
    assert decisions.loc["e_incorrect", "decision_reason"] == "accuracy_incorrect"
    assert decisions.loc["e_silent", "decision_reason"] == "silent_error_true"
    assert decisions.loc["e_parse", "decision_reason"] == "pred_parse_failure"
    assert bool(decisions.loc["e_repair", "repair_attempted"]) is True
    assert bool(decisions.loc["e_repair", "repair_success"]) is True

    training_ids = set(out.training_df["example_id"].tolist())
    assert training_ids == {"e_keep", "e_repair"}
    assert set(out.repair_pairs_df["example_id"].tolist()) == {"e_repair"}
    assert out.report.repair_success_count == 1

    required_cols = {
        "example_id",
        "branch",
        "generation",
        "seed",
        "is_correct",
        "pred_parse_success",
        "overall_pedagogical_score",
        "is_silent_error",
        "action",
        "decision_reason",
        "repair_attempted",
        "repair_success",
        "original_response_hash",
        "repaired_response_hash",
    }
    assert required_cols.issubset(out.decisions_df.columns)


def test_pvr_repair_failure_is_rejected() -> None:
    out = apply_pvr(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        gold_df=_gold_df(),
        model_name="qwen2.5:0.5b",
        branch="pvr_repair_medium",
        generation=1,
        seed=42,
        policy=PVRPolicy(threshold_score=6, min_keep_ratio=0.1),
        repair_model_name="llama-3.1-8b",
        repair_callable=_repair_fail,
    )
    rec = out.decisions_df.set_index("example_id").loc["e_repair"]
    assert rec["action"] == "reject"
    assert rec["decision_reason"] == "repair_failed"
    assert bool(rec["repair_attempted"]) is True
    assert bool(rec["repair_success"]) is False


def test_pvr_save_artifacts_roundtrip() -> None:
    out = apply_pvr(
        synthetic_df=_synthetic_df(),
        accuracy_df=_accuracy_df(),
        judge_df=_judge_df(),
        gold_df=_gold_df(),
        model_name="qwen2.5:0.5b",
        branch="pvr_repair_medium",
        generation=1,
        seed=42,
        policy=PVRPolicy(threshold_score=6, min_keep_ratio=0.2),
        repair_model_name="llama-3.1-8b",
        repair_callable=_repair_ok,
    )
    base = Path("outputs/.tmp") / f"pvr_artifacts_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        training_path = base / "pvr_training_dataset.parquet"
        decisions_path = base / "pvr_decisions.parquet"
        pairs_path = base / "pvr_repair_pairs.parquet"
        report_path = base / "pvr_report.json"
        save_pvr_artifacts(
            result=out,
            training_path=training_path,
            decisions_path=decisions_path,
            repair_pairs_path=pairs_path,
            report_path=report_path,
        )
        assert training_path.exists()
        assert decisions_path.exists()
        assert pairs_path.exists()
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["repair_success_count"] == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_pvr_fails_when_all_rows_rejected() -> None:
    acc = _accuracy_df().copy()
    acc["is_correct"] = False
    acc["accuracy_label"] = "incorrect"
    with pytest.raises(PVRError, match="rejected all rows"):
        apply_pvr(
            synthetic_df=_synthetic_df(),
            accuracy_df=acc,
            judge_df=_judge_df(),
            gold_df=_gold_df(),
            model_name="qwen2.5:0.5b",
            branch="pvr_repair_medium",
            generation=1,
            seed=42,
            policy=PVRPolicy(threshold_score=6, min_keep_ratio=0.0),
            repair_model_name="llama-3.1-8b",
            repair_callable=_repair_ok,
        )
