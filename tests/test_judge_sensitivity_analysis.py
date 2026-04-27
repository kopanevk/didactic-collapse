from __future__ import annotations

from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.judge_sensitivity import (
    _safe_out_dir,
    build_judge_sensitivity_comparison,
    build_judge_sensitivity_seed_branch_summary,
    build_judge_sensitivity_summary,
    select_balanced_sensitivity_sample,
    select_confirmatory_seed_branch_sample,
)
from didactic_collapse.config.settings import load_config


def _synthetic_source_rows(n_per_cell: int = 16) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    idx = 0
    for source_seed in (71, 72, 73):
        for branch in ("pure_recycling", "anchor_20_append"):
            for generation in (0, 1):
                for i in range(n_per_cell):
                    old_score = [2, 4, 7, 6][i % 4]
                    old_silent = (i % 5) == 0
                    rows.append(
                        {
                            "source_run_dir": f"outputs/runs/training_confirmatory_anchor20_seed{source_seed}_fake",
                            "source_step_dir": f"step_seed{source_seed}_{branch}_{generation}",
                            "source_seed": source_seed,
                            "run_id": f"run_seed{source_seed}",
                            "branch": branch,
                            "generation": generation,
                            "model_name": "qwen2.5:0.5b",
                            "example_id": f"{source_seed}_{branch}_{generation}_{idx}",
                            "question": f"q_{idx}",
                            "answer_gold": "1",
                            "raw_response": "Final answer: 1",
                            "old_llama_overall_pedagogical_score": old_score,
                            "old_llama_is_silent_error": old_silent,
                        }
                    )
                    idx += 1
    return pd.DataFrame(rows)


def test_select_balanced_sensitivity_sample_balances_branch_generation() -> None:
    source = _synthetic_source_rows(n_per_cell=20)
    selected = select_balanced_sensitivity_sample(source, total_n=40, seed=123)
    assert len(selected) == 40

    counts = (
        selected.groupby(["branch", "generation"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    by_cell = {(r["branch"], int(r["generation"])): int(r["n"]) for _, r in counts.iterrows()}
    assert by_cell[("pure_recycling", 0)] == 10
    assert by_cell[("pure_recycling", 1)] == 10
    assert by_cell[("anchor_20_append", 0)] == 10
    assert by_cell[("anchor_20_append", 1)] == 10


def test_build_judge_sensitivity_summary_computes_expected_fields() -> None:
    comparison = pd.DataFrame(
        [
            {
                "branch": "pure_recycling",
                "generation": 1,
                "old_llama_overall_pedagogical_score": 4,
                "qwen_overall_pedagogical_score": 3,
                "abs_score_delta": 1,
                "old_llama_is_silent_error": False,
                "qwen_is_silent_error": False,
                "silent_error_agreement": True,
                "qwen_repair_applied": False,
            },
            {
                "branch": "anchor_20_append",
                "generation": 1,
                "old_llama_overall_pedagogical_score": 5,
                "qwen_overall_pedagogical_score": 6,
                "abs_score_delta": 1,
                "old_llama_is_silent_error": True,
                "qwen_is_silent_error": False,
                "silent_error_agreement": False,
                "qwen_repair_applied": True,
            },
            {
                "branch": "anchor_20_append",
                "generation": 0,
                "old_llama_overall_pedagogical_score": 6,
                "qwen_overall_pedagogical_score": 6,
                "abs_score_delta": 0,
                "old_llama_is_silent_error": False,
                "qwen_is_silent_error": False,
                "silent_error_agreement": True,
                "qwen_repair_applied": False,
            },
        ]
    )
    summary = build_judge_sensitivity_summary(comparison)
    assert len(summary) == 1
    rec = summary.iloc[0].to_dict()
    assert int(rec["n_examples"]) == 3
    assert float(rec["mean_abs_score_delta"]) == 2 / 3
    assert float(rec["silent_error_agreement_rate"]) == 2 / 3
    assert float(rec["gen1_old_anchor_minus_pure_pedagogical"]) == 1.0
    assert float(rec["gen1_qwen_anchor_minus_pure_pedagogical"]) == 3.0
    assert bool(rec["gen1_branch_conclusion_changed"]) is False


def test_select_confirmatory_seed_branch_sample_balances_seed_branch_gen1() -> None:
    source = _synthetic_source_rows(n_per_cell=12)
    selected = select_confirmatory_seed_branch_sample(
        source,
        total_n=48,
        seed=4242,
        focus_generation=1,
    )
    assert len(selected) == 48
    assert set(selected["generation"].astype(int).tolist()) == {1}

    counts = (
        selected.groupby(["source_seed", "branch", "generation"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    by_cell = {
        (int(r["source_seed"]), str(r["branch"]), int(r["generation"])): int(r["n"])
        for _, r in counts.iterrows()
    }
    for source_seed in (71, 72, 73):
        assert by_cell[(source_seed, "pure_recycling", 1)] == 8
        assert by_cell[(source_seed, "anchor_20_append", 1)] == 8


def test_select_confirmatory_seed_branch_sample_respects_allowed_run_dirs() -> None:
    source = _synthetic_source_rows(n_per_cell=8)
    allowed = [
        "outputs/runs/training_confirmatory_anchor20_seed71_fake",
        "outputs/runs/training_confirmatory_anchor20_seed72_fake",
    ]
    selected = select_confirmatory_seed_branch_sample(
        source,
        total_n=24,
        seed=10,
        focus_generation=1,
        allowed_run_dirs=allowed,
    )
    assert set(selected["source_run_dir"].unique().tolist()).issubset(set(allowed))
    assert set(selected["source_seed"].astype(int).tolist()) == {71, 72}


def test_build_judge_sensitivity_seed_branch_summary_deltas_and_conclusion() -> None:
    comparison = pd.DataFrame(
        [
            {
                "source_seed": 71,
                "branch": "pure_recycling",
                "generation": 1,
                "old_llama_overall_pedagogical_score": 4.0,
                "qwen_overall_pedagogical_score": 5.0,
                "old_llama_is_silent_error": False,
                "qwen_is_silent_error": False,
                "silent_error_agreement": True,
                "score_delta": 1.0,
            },
            {
                "source_seed": 71,
                "branch": "anchor_20_append",
                "generation": 1,
                "old_llama_overall_pedagogical_score": 5.0,
                "qwen_overall_pedagogical_score": 4.0,
                "old_llama_is_silent_error": True,
                "qwen_is_silent_error": False,
                "silent_error_agreement": False,
                "score_delta": -1.0,
            },
        ]
    )
    summary = build_judge_sensitivity_seed_branch_summary(comparison, focus_generation=1)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert int(row["source_seed"]) == 71
    assert float(row["old_llama_anchor_minus_pure"]) == 1.0
    assert float(row["qwen_anchor_minus_pure"]) == -1.0
    assert bool(row["seed_branch_conclusion_changed"]) is True


def test_confirmatory_selector_does_not_mutate_source_dataframe() -> None:
    source = _synthetic_source_rows(n_per_cell=8)
    before = source.copy(deep=True)
    _ = select_confirmatory_seed_branch_sample(source, total_n=48, seed=123, focus_generation=1)
    assert source.equals(before)


def test_comparison_normalizes_source_seed_after_merge_suffixes() -> None:
    selected = pd.DataFrame(
        [
            {
                "source_run_dir": "run_a",
                "source_step_dir": "step_a",
                "source_seed": 71,
                "run_id": "rid",
                "branch": "pure_recycling",
                "generation": 1,
                "model_name": "qwen2.5:0.5b",
                "example_id": "e1",
                "old_llama_overall_pedagogical_score": 4,
                "old_llama_is_silent_error": False,
            }
        ]
    )
    qwen = pd.DataFrame(
        [
            {
                "source_run_dir": "run_a",
                "source_step_dir": "step_a",
                "source_seed": 71,
                "run_id": "rid",
                "branch": "pure_recycling",
                "generation": 1,
                "model_name": "qwen2.5:0.5b",
                "example_id": "e1",
                "qwen_overall_pedagogical_score": 5,
                "qwen_is_silent_error": False,
                "qwen_repair_applied": False,
            }
        ]
    )
    out = build_judge_sensitivity_comparison(selected_sample_df=selected, qwen_results_df=qwen)
    assert "source_seed" in out.columns
    assert int(out.iloc[0]["source_seed"]) == 71


def test_safe_out_dir_rejects_output_inside_source_runs() -> None:
    source_run = Path("outputs/runs/some_run_for_test")
    unsafe_out = source_run / "tables"
    try:
        _safe_out_dir(unsafe_out, [source_run])
    except ValueError as exc:
        assert "cannot be inside source run_dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsafe output directory")


def test_qwen_sensitivity_config_uses_expected_model_name() -> None:
    cfg = load_config("configs/judge_sensitivity_qwen.yaml")
    assert cfg.judge.provider == "cerebras"
    assert cfg.judge.model_name == "qwen-3-235b-a22b-instruct-2507"
