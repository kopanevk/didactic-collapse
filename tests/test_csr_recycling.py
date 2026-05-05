from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.recycling.contrastive_self_recycling import (
    CSRPolicy,
    apply_csr,
    save_csr_artifacts,
)


def _candidates_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for example_id in ("e1", "e2"):
        for candidate_id in (0, 1, 2):
            rows.append(
                {
                    "example_id": example_id,
                    "candidate_id": candidate_id,
                    "question": f"Q {example_id}",
                    "raw_response": f"response-{example_id}-{candidate_id}",
                }
            )
    return pd.DataFrame(rows)


def _candidate_scores_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # e1: strong contrast
            {
                "example_id": "e1",
                "candidate_id": 0,
                "question": "Q e1",
                "raw_response": "good e1",
                "parsed_final_answer": "1",
                "pred_parse_success": True,
                "is_correct": True,
                "accuracy_label": "correct",
                "is_silent_error": False,
                "overall_pedagogical_score": 7,
                "reasoning_soundness": 2,
                "structure": 2,
                "clarity": 2,
                "terminology": 2,
            },
            {
                "example_id": "e1",
                "candidate_id": 1,
                "question": "Q e1",
                "raw_response": "bad e1",
                "parsed_final_answer": "9",
                "pred_parse_success": True,
                "is_correct": False,
                "accuracy_label": "incorrect",
                "is_silent_error": True,
                "overall_pedagogical_score": 2,
                "reasoning_soundness": 0,
                "structure": 0,
                "clarity": 0,
                "terminology": 0,
            },
            {
                "example_id": "e1",
                "candidate_id": 2,
                "question": "Q e1",
                "raw_response": "mid e1",
                "parsed_final_answer": "1",
                "pred_parse_success": True,
                "is_correct": True,
                "accuracy_label": "correct",
                "is_silent_error": False,
                "overall_pedagogical_score": 5,
                "reasoning_soundness": 1,
                "structure": 1,
                "clarity": 1,
                "terminology": 1,
            },
            # e2: weak contrast under min_pair_quality_gap=2
            {
                "example_id": "e2",
                "candidate_id": 0,
                "question": "Q e2",
                "raw_response": "good e2",
                "parsed_final_answer": "2",
                "pred_parse_success": True,
                "is_correct": True,
                "accuracy_label": "correct",
                "is_silent_error": False,
                "overall_pedagogical_score": 6,
                "reasoning_soundness": 2,
                "structure": 2,
                "clarity": 1,
                "terminology": 1,
            },
            {
                "example_id": "e2",
                "candidate_id": 1,
                "question": "Q e2",
                "raw_response": "near e2",
                "parsed_final_answer": "2",
                "pred_parse_success": True,
                "is_correct": True,
                "accuracy_label": "correct",
                "is_silent_error": False,
                "overall_pedagogical_score": 5,
                "reasoning_soundness": 1,
                "structure": 1,
                "clarity": 1,
                "terminology": 1,
            },
            {
                "example_id": "e2",
                "candidate_id": 2,
                "question": "Q e2",
                "raw_response": "near2 e2",
                "parsed_final_answer": "2",
                "pred_parse_success": True,
                "is_correct": True,
                "accuracy_label": "correct",
                "is_silent_error": False,
                "overall_pedagogical_score": 5,
                "reasoning_soundness": 1,
                "structure": 1,
                "clarity": 1,
                "terminology": 1,
            },
        ]
    )


def test_csr_builds_pairs_and_logs_no_pair_cases() -> None:
    result = apply_csr(
        candidates_df=_candidates_df(),
        candidate_scores_df=_candidate_scores_df(),
        model_name="qwen2.5:0.5b",
        branch="csr_medium",
        generation=1,
        seed=7,
        policy=CSRPolicy(min_pair_quality_gap=2.0),
    )

    assert len(result.training_df) == 1
    assert result.report.total_questions == 2
    assert result.report.pair_count == 1
    assert result.report.no_pair_count == 1
    assert abs(result.report.pair_construction_rate - 0.5) < 1e-9

    paired = result.pairs_df[result.pairs_df["pair_status"] == "paired"]
    assert len(paired) == 1
    assert paired.iloc[0]["example_id"] == "e1"
    defect_tags = json.loads(str(paired.iloc[0]["defect_tags"]))
    assert "incorrect_answer" in defect_tags
    assert "silent_error" in defect_tags

    unpaired = result.pairs_df[result.pairs_df["pair_status"] == "no_pair"]
    assert len(unpaired) == 1
    assert unpaired.iloc[0]["no_pair_reason"] == "quality_gap_below_threshold"

    training_text = str(result.training_df.iloc[0]["answer_for_training"])
    assert "Weak explanation:" in training_text
    assert "Better explanation:" in training_text
    assert "Final answer:" in training_text


def test_csr_quality_gap_threshold_controls_pair_count() -> None:
    strict = apply_csr(
        candidates_df=_candidates_df(),
        candidate_scores_df=_candidate_scores_df(),
        model_name="qwen2.5:0.5b",
        branch="csr_medium",
        generation=1,
        seed=7,
        policy=CSRPolicy(min_pair_quality_gap=2.0),
    )
    loose = apply_csr(
        candidates_df=_candidates_df(),
        candidate_scores_df=_candidate_scores_df(),
        model_name="qwen2.5:0.5b",
        branch="csr_medium",
        generation=1,
        seed=7,
        policy=CSRPolicy(min_pair_quality_gap=0.0),
    )
    assert strict.report.pair_count == 1
    assert loose.report.pair_count == 2


def test_save_csr_artifacts_writes_expected_files() -> None:
    result = apply_csr(
        candidates_df=_candidates_df(),
        candidate_scores_df=_candidate_scores_df(),
        model_name="qwen2.5:0.5b",
        branch="csr_medium",
        generation=1,
        seed=7,
        policy=CSRPolicy(min_pair_quality_gap=2.0),
    )
    base = Path("outputs/.tmp") / f"csr_artifacts_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        save_csr_artifacts(
            result=result,
            candidates_path=base / "csr_candidates.parquet",
            candidate_scores_path=base / "csr_candidate_scores.parquet",
            pairs_path=base / "csr_pairs.parquet",
            training_path=base / "csr_training_dataset.parquet",
            report_path=base / "csr_report.json",
        )
        assert (base / "csr_candidates.parquet").exists()
        assert (base / "csr_candidate_scores.parquet").exists()
        assert (base / "csr_pairs.parquet").exists()
        assert (base / "csr_training_dataset.parquet").exists()
        assert (base / "csr_report.json").exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)
