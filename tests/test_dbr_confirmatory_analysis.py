from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.dbr_confirmatory import export_dbr_confirmatory_analysis


def _make_fake_run(root: Path, *, seed: int, tag: str) -> Path:
    run_dir = root / f"{tag}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    snapshot = {"config": {"project": {"seed": seed}, "paths": {"data_root": "data"}}}
    (run_dir / "run_config.snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_rows: list[dict[str, object]] = []
    for gen in [0, 1, 2]:
        summary_rows.extend(
            [
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "pure_recycling",
                    "generation": gen,
                    "sample_count": 6,
                    "accuracy_mean": 0.40 + (0.01 * gen),
                    "pedagogical_score_mean": 5.0 - (0.1 * gen),
                    "silent_error_rate": 0.25 + (0.01 * gen),
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "dbr_medium",
                    "generation": gen,
                    "sample_count": 6,
                    "accuracy_mean": 0.44 + (0.02 * gen),
                    "pedagogical_score_mean": 5.2 + (0.05 * gen),
                    "silent_error_rate": 0.22 - (0.01 * gen),
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "soft_pvf_noisy_keep",
                    "generation": gen,
                    "sample_count": 6,
                    "accuracy_mean": 0.42,
                    "pedagogical_score_mean": 4.9,
                    "silent_error_rate": 0.24,
                },
            ]
        )
    pd.DataFrame(summary_rows).to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)

    model_dir = run_dir / "qwen2.5_0.5b"
    for branch in ["pure_recycling", "dbr_medium", "soft_pvf_noisy_keep"]:
        for gen in [0, 1, 2]:
            ctx = model_dir / branch / f"gen_{gen}"
            ctx.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "model_name": "qwen2.5:0.5b",
                        "branch": branch,
                        "generation": gen,
                        "pred_parse_success": True,
                    }
                    for _ in range(6)
                ]
            ).to_parquet(ctx / "accuracy_table.parquet", index=False)

    # Required for matched Gen2 comparison:
    for branch, acc, ped, sil in [
        ("pure_recycling", [True, True, False, False, True, False], [5, 6, 4, 3, 5, 4], [False, False, True, False, False, True]),
        ("dbr_medium", [True, True, True, False, True, False], [6, 6, 5, 4, 6, 4], [False, False, False, False, False, True]),
    ]:
        gen2_dir = model_dir / branch / "gen_2"
        pd.DataFrame(
            [
                {
                    "example_id": f"ex_{i}",
                    "is_correct": acc[i],
                    "overall_pedagogical_score": ped[i],
                    "is_silent_error": sil[i],
                }
                for i in range(6)
            ]
        ).to_parquet(gen2_dir / "eval_merged.parquet", index=False)

    # DBR budget reports.
    for gen in [0, 1, 2]:
        payload = {
            "model_name": "qwen2.5:0.5b",
            "branch": "dbr_medium",
            "generation": gen,
            "seed": seed,
            "policy_name": "dbr_medium",
            "total_candidates": 50,
            "target_size": 50,
            "selected_count": 43 + gen,
            "selection_rate": (43 + gen) / 50.0,
            "min_selection_rate": 0.80,
            "budgets": {
                "parse_failure": 0.0,
                "silent_error": 0.1,
                "incorrect_answer": 0.3,
                "low_reasoning": 0.25,
                "low_structure": 0.3,
            },
            "defect_rates_before": {
                "parse_failure": 0.2,
                "silent_error": 0.18,
                "incorrect_answer": 0.42,
                "low_reasoning": 0.34,
                "low_structure": 0.36,
            },
            "defect_rates_after": {
                "parse_failure": 0.0,
                "silent_error": 0.09,
                "incorrect_answer": 0.28,
                "low_reasoning": 0.22,
                "low_structure": 0.27,
            },
            "budget_violations": {},
            "relaxation_steps_used": ["relax_low_structure"],
            "bucket_coverage_before": {"short": 20, "medium": 15, "long": 15},
            "bucket_coverage_after": {"short": 17, "medium": 13, "long": 14},
            "fallback_bucket_count": 0,
        }
        report_path = model_dir / "dbr_medium" / f"gen_{gen}" / "dbr_budget_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return run_dir


def test_export_dbr_confirmatory_analysis_writes_expected_outputs() -> None:
    base = Path("outputs/.tmp") / f"dbr_confirmatory_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_a = _make_fake_run(base, seed=211, tag="run")
        run_b = _make_fake_run(base, seed=212, tag="run")
        out_dir = base / "analysis" / "tables"
        artifacts = export_dbr_confirmatory_analysis(run_dirs=[run_a, run_b], out_dir=out_dir)

        required = [
            artifacts.run_level_csv,
            artifacts.seed_stats_csv,
            artifacts.generation_deltas_csv,
            artifacts.branch_deltas_csv,
            artifacts.budget_summary_csv,
            artifacts.defect_rates_csv,
            artifacts.bucket_coverage_csv,
            artifacts.stress_summary_csv,
            artifacts.matched_gen2_csv,
            artifacts.accuracy_plot,
            artifacts.pedagogical_plot,
            artifacts.silent_error_plot,
            artifacts.selection_rate_plot,
            artifacts.defect_rates_plot,
            artifacts.metadata_json,
        ]
        missing = [str(p) for p in required if not p.exists()]
        assert not missing, f"Missing artifacts: {missing}"

        matched = pd.read_csv(artifacts.matched_gen2_csv)
        assert len(matched) == 2
        assert {"delta_accuracy_agg_dbr_minus_pure", "delta_pedagogy_matched_dbr_minus_pure"}.issubset(
            set(matched.columns)
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)

