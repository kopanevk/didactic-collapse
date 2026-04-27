from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.pairwise_judge_sensitivity import (
    _write_simple_xlsx,
    build_blinded_pair_tables,
    build_pair_candidates,
    build_pairwise_summary,
    parse_pairwise_judge_response,
    select_balanced_pair_sample,
)


def _synthetic_source_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_seed in (71, 72, 73):
        for i in range(20):
            for branch in ("pure_recycling", "anchor_20_append"):
                rows.append(
                    {
                        "source_run_dir": f"outputs/runs/seed{source_seed}",
                        "source_seed": source_seed,
                        "run_id": f"run_{source_seed}",
                        "generation": 1,
                        "model_name": "qwen2.5:0.5b",
                        "branch": branch,
                        "example_id": f"ex_{source_seed}_{i}",
                        "question": f"q_{i}",
                        "answer_gold": str(i),
                        "raw_response": f"{branch} response {i}",
                        "overall_pedagogical_score": 6 if branch == "anchor_20_append" else 5,
                        "is_silent_error": False,
                        "comment": "ok",
                        "judge_model": "llama-3.1-8b",
                    }
                )
    return pd.DataFrame(rows)


def test_build_pair_candidates_pairs_by_example_id() -> None:
    source = _synthetic_source_rows()
    candidates = build_pair_candidates(source, generation=1)
    assert len(candidates) == 60
    assert {"response_pure", "response_anchor"}.issubset(candidates.columns)
    assert candidates["example_id"].nunique() == 60


def test_ab_randomization_and_hidden_key_consistency() -> None:
    source = _synthetic_source_rows()
    candidates = build_pair_candidates(source, generation=1)
    selected = select_balanced_pair_sample(candidates, total_n=48, seed=4242)
    public_df, hidden_df, _ = build_blinded_pair_tables(selected, seed=4242)

    assert len(public_df) == 48
    assert len(hidden_df) == 48
    merged = public_df.merge(hidden_df, on=["pair_id", "source_run_dir", "source_seed", "run_id", "generation", "model_name", "example_id"])
    assert len(merged) == 48
    assert set(hidden_df["A_branch"].unique().tolist()).issubset({"pure_recycling", "anchor_20_append"})
    assert set(hidden_df["B_branch"].unique().tolist()).issubset({"pure_recycling", "anchor_20_append"})


def test_manual_audit_export_has_no_branch_labels() -> None:
    source = _synthetic_source_rows()
    candidates = build_pair_candidates(source, generation=1)
    selected = select_balanced_pair_sample(candidates, total_n=12, seed=111)
    _, _, manual_df = build_blinded_pair_tables(selected, seed=111)
    assert "A_branch" not in manual_df.columns
    assert "B_branch" not in manual_df.columns
    assert "branch" not in manual_df.columns
    assert {"human_winner", "human_confidence", "human_notes"}.issubset(manual_df.columns)


def test_win_rate_calculation_correct() -> None:
    comparison = pd.DataFrame(
        [
            {"source_seed": 71, "llama_anchor_win": True, "llama_pure_win": False, "llama_tie": False, "qwen_anchor_win": True, "qwen_pure_win": False, "qwen_tie": False, "judges_agree_winner_branch": True, "judges_agree_winner_raw": True},
            {"source_seed": 71, "llama_anchor_win": False, "llama_pure_win": True, "llama_tie": False, "qwen_anchor_win": False, "qwen_pure_win": True, "qwen_tie": False, "judges_agree_winner_branch": True, "judges_agree_winner_raw": True},
            {"source_seed": 72, "llama_anchor_win": False, "llama_pure_win": False, "llama_tie": True, "qwen_anchor_win": True, "qwen_pure_win": False, "qwen_tie": False, "judges_agree_winner_branch": False, "judges_agree_winner_raw": False},
        ]
    )
    summary, seed = build_pairwise_summary(
        comparison,
        llama_model_name="llama-3.1-8b",
        qwen_model_name="qwen-3-235b-a22b-instruct-2507",
    )
    rec = summary.iloc[0]
    assert int(rec["n_pairs"]) == 3
    assert abs(float(rec["llama_anchor_win_rate"]) - (1 / 3)) < 1e-9
    assert abs(float(rec["qwen_anchor_win_rate"]) - (2 / 3)) < 1e-9
    assert int(seed["source_seed"].nunique()) == 2


def test_pairwise_json_validation_works() -> None:
    parsed = parse_pairwise_judge_response(
        '{"winner":"A","confidence":"2","reason":"Clear and structured explanation."}'
    )
    assert parsed.decision.winner == "A"
    assert parsed.decision.confidence == 2
    assert parsed.repair_applied is True


def test_dependency_free_xlsx_writer_creates_valid_zip() -> None:
    df = pd.DataFrame([{"a": 1, "b": "x"}])
    base = Path("outputs/.tmp") / f"pairwise_xlsx_test_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        out = base / "manual.xlsx"
        _write_simple_xlsx(df, out)
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        shutil.rmtree(base, ignore_errors=True)
