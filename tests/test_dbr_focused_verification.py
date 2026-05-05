from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pandas as pd

from didactic_collapse.analysis.dbr_focused_verification import (
    _compare_against_reference_tables,
    build_blinded_dbr_pair_tables,
    build_qwen_dbr_pairwise_summary,
    recompute_gen2_deltas_by_seed,
    recompute_gen2_metrics,
    select_balanced_dbr_pairs,
)


def _mk_eval_df(*, n: int, acc: float, pedagogy: float, silent: float) -> pd.DataFrame:
    acc_true = int(round(n * acc))
    silent_true = int(round(n * silent))
    rows: list[dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "example_id": f"ex_{i}",
                "model_name": "qwen2.5:0.5b",
                "is_correct": i < acc_true,
                "overall_pedagogical_score": pedagogy,
                "is_silent_error": i < silent_true,
            }
        )
    return pd.DataFrame(rows)


def _write_min_run(base: Path, *, seed: int, pure_acc: float, dbr_acc: float) -> Path:
    run_dir = base / f"run_seed{seed}"
    model_dir = run_dir / "qwen2.5_0.5b"
    pure_gen = model_dir / "pure_recycling" / "gen_2"
    dbr_gen = model_dir / "dbr_medium" / "gen_2"
    pure_gen.mkdir(parents=True, exist_ok=True)
    dbr_gen.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    data_root = base / "data"
    (data_root / "splits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"example_id": [f"ex_{i}" for i in range(50)], "question": ["q"] * 50}).to_parquet(
        data_root / "splits" / "heldout_test.parquet", index=False
    )

    snapshot = {
        "config": {
            "project": {"seed": seed},
            "paths": {"data_root": str(data_root), "output_root": str(base), "prompt_dir": "configs/prompts"},
        }
    }
    (run_dir / "run_config.snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (run_dir / "run_stage_manifest.json").write_text(
        json.dumps(
            {
                "stages": {
                    "data_prep": {"status": "completed"},
                    "aggregate": {"status": "completed"},
                    "plotting": {"status": "completed"},
                }
            }
        ),
        encoding="utf-8",
    )
    stage_manifest = {"stages": {k: {"status": "completed"} for k in ("generation", "answer_extraction", "accuracy", "judge")}}
    (pure_gen / "stage_manifest.json").write_text(json.dumps(stage_manifest), encoding="utf-8")
    (dbr_gen / "stage_manifest.json").write_text(json.dumps(stage_manifest), encoding="utf-8")

    pure_eval = _mk_eval_df(n=50, acc=pure_acc, pedagogy=5.0, silent=0.2)
    dbr_eval = _mk_eval_df(n=50, acc=dbr_acc, pedagogy=5.5, silent=0.1)
    pure_eval.to_parquet(pure_gen / "eval_merged.parquet", index=False)
    dbr_eval.to_parquet(dbr_gen / "eval_merged.parquet", index=False)

    for gen in (pure_gen, dbr_gen):
        minimal = pd.DataFrame({"example_id": [f"ex_{i}" for i in range(50)]})
        minimal.to_parquet(gen / "model_outputs.parquet", index=False)
        minimal.to_parquet(gen / "answer_extraction.parquet", index=False)
        minimal.to_parquet(gen / "accuracy_table.parquet", index=False)
        minimal.to_parquet(gen / "judge_outputs.parquet", index=False)
        pd.DataFrame(columns=["example_id"]).to_parquet(gen / "judge_failures.parquet", index=False)

    decisions = pd.DataFrame(
        {
            "example_id": [f"ex_{i}" for i in range(50)],
            "selected": [True] * 45 + [False] * 5,
            "defect_parse_failure": [False] * 50,
            "defect_incorrect": [False] * 50,
            "defect_silent": [False] * 50,
            "defect_low_reasoning": [False] * 50,
            "defect_low_structure": [False] * 50,
        }
    )
    decisions.to_parquet(dbr_gen / "dbr_decisions.parquet", index=False)
    decisions[decisions["selected"]].to_parquet(dbr_gen / "dbr_training_dataset.parquet", index=False)
    (dbr_gen / "dbr_budget_report.json").write_text(
        json.dumps(
            {
                "selected_count": 45,
                "target_size": 50,
                "selection_rate": 0.9,
                "defect_rates_before": {k: 0.0 for k in ("parse_failure", "incorrect_answer", "silent_error", "low_reasoning", "low_structure")},
                "defect_rates_after": {k: 0.0 for k in ("parse_failure", "incorrect_answer", "silent_error", "low_reasoning", "low_structure")},
                "budget_violations": {},
                "relaxation_steps_used": [],
            }
        ),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "pure_recycling",
                "generation": 2,
                "sample_count": 50,
                "accuracy_mean": pure_acc,
                "pedagogical_score_mean": 5.0,
                "silent_error_rate": 0.2,
            },
            {
                "model_name": "qwen2.5:0.5b",
                "branch": "dbr_medium",
                "generation": 2,
                "sample_count": 50,
                "accuracy_mean": dbr_acc,
                "pedagogical_score_mean": 5.5,
                "silent_error_rate": 0.1,
            },
        ]
    )
    summary.to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)
    return run_dir


def test_recompute_metrics_from_row_level_artifacts() -> None:
    base = Path("test_workdirs") / f"dbr_focus_test_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = _write_min_run(base, seed=211, pure_acc=0.4, dbr_acc=0.5)
        metrics = recompute_gen2_metrics([run_dir])
        assert set(metrics["branch"].tolist()) == {"pure_recycling", "dbr_medium"}
        deltas = recompute_gen2_deltas_by_seed(metrics)
        assert len(deltas) == 1
        assert abs(float(deltas["delta_accuracy_dbr_minus_pure"].iloc[0]) - 0.1) < 1e-9
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_balanced_pair_selection_and_blinding() -> None:
    rows: list[dict[str, object]] = []
    for seed in (211, 212, 213):
        for i in range(20):
            rows.append(
                {
                    "source_run_dir": f"run_{seed}",
                    "source_seed": seed,
                    "run_id": f"run_{seed}",
                    "generation": 2,
                    "model_name": "qwen2.5:0.5b",
                    "example_id": f"ex_{seed}_{i}",
                    "question": "q",
                    "answer_gold": "1",
                    "response_pure": "pure",
                    "response_dbr": "dbr",
                }
            )
    cand = pd.DataFrame(rows)
    selected = select_balanced_dbr_pairs(cand, total_n=48, sample_seed=7)
    counts = selected["source_seed"].value_counts().to_dict()
    assert counts == {211: 16, 212: 16, 213: 16}

    public_df, hidden_df = build_blinded_dbr_pair_tables(selected, sample_seed=7)
    assert len(public_df) == 48
    assert len(hidden_df) == 48
    assert "A_branch" not in public_df.columns
    assert "B_branch" not in public_df.columns
    assert {"A_branch", "B_branch"}.issubset(hidden_df.columns)


def test_pairwise_summary_calculation() -> None:
    decoded = pd.DataFrame(
        [
            {"source_seed": 211, "qwen_dbr_win": True, "qwen_pure_win": False, "qwen_tie": False},
            {"source_seed": 211, "qwen_dbr_win": False, "qwen_pure_win": True, "qwen_tie": False},
            {"source_seed": 212, "qwen_dbr_win": True, "qwen_pure_win": False, "qwen_tie": False},
            {"source_seed": 213, "qwen_dbr_win": False, "qwen_pure_win": False, "qwen_tie": True},
        ]
    )
    summary, seed = build_qwen_dbr_pairwise_summary(decoded, llama_aggregate_pedagogy_delta=0.2)
    assert int(summary.iloc[0]["n_pairs"]) == 4
    assert abs(float(summary.iloc[0]["qwen_dbr_win_rate"]) - 0.5) < 1e-9
    assert int(seed["source_seed"].nunique()) == 3


def test_table_mismatch_detection() -> None:
    base = Path("test_workdirs") / f"dbr_focus_test_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = _write_min_run(base, seed=211, pure_acc=0.4, dbr_acc=0.5)
        outputs_root = base / "outputs"
        ref_dir = outputs_root / "dbr_confirmatory" / "dbr_paper_summary_20260101_000000" / "tables"
        ref_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "section": "gen2_seed_branch_compare",
                    "seed": 211,
                    "generation": 2,
                    "metric": "accuracy",
                    "pure_value": 0.9,  # intentionally wrong
                    "dbr_value": 0.9,
                    "delta_dbr_minus_pure": 0.0,
                }
            ]
        ).to_csv(ref_dir / "dbr_paper_summary.csv", index=False)

        metrics = recompute_gen2_metrics([run_dir])
        deltas = recompute_gen2_deltas_by_seed(metrics)
        findings: list[dict[str, object]] = []
        cmp_df = _compare_against_reference_tables(
            recomputed_metrics=metrics,
            recomputed_deltas=deltas,
            outputs_root=outputs_root,
            findings=findings,
        )
        assert not cmp_df.empty
        assert any(str(x.get("severity")) == "CRITICAL" for x in findings)
    finally:
        shutil.rmtree(base, ignore_errors=True)
