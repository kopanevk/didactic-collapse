from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.soft_pvf_confirmatory import export_soft_pvf_confirmatory_analysis


def _make_fake_run(root: Path, *, seed: int, tag: str) -> Path:
    run_dir = root / f"{tag}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    snapshot = {"config": {"project": {"seed": seed}, "paths": {"data_root": "data"}}}
    (run_dir / "run_config.snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for branch, acc_g2, ped_g2, sil_g2 in [
        ("pure_recycling", 0.44, 4.80, 0.28),
        ("soft_pvf_medium", 0.40, 5.00, 0.30),
        ("soft_pvf_noisy_keep", 0.39, 4.40, 0.22),
    ]:
        rows.extend(
            [
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": branch,
                    "generation": 0,
                    "sample_count": 50,
                    "accuracy_mean": 0.35,
                    "pedagogical_score_mean": 4.2,
                    "silent_error_rate": 0.30,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": branch,
                    "generation": 1,
                    "sample_count": 50,
                    "accuracy_mean": 0.38,
                    "pedagogical_score_mean": 4.5,
                    "silent_error_rate": 0.29,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": branch,
                    "generation": 2,
                    "sample_count": 50,
                    "accuracy_mean": acc_g2,
                    "pedagogical_score_mean": ped_g2,
                    "silent_error_rate": sil_g2,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)

    model_dir = run_dir / "qwen2.5_0.5b"
    for branch in ["pure_recycling", "soft_pvf_medium", "soft_pvf_noisy_keep"]:
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
                    for _ in range(50)
                ]
            ).to_parquet(ctx / "accuracy_table.parquet", index=False)

    for branch, policy, keep in [
        ("soft_pvf_medium", "soft_pvf_medium", 0.42),
        ("soft_pvf_noisy_keep", "soft_pvf_noisy_keep", 0.40),
    ]:
        for gen in [0, 1, 2]:
            ctx = model_dir / branch / f"gen_{gen}"
            payload = {
                "model_name": "qwen2.5:0.5b",
                "branch": branch,
                "generation": gen,
                "seed": seed,
                "policy_name": policy,
                "total_candidates": 50,
                "kept_count": int(round(50 * keep)),
                "rejected_count": 50 - int(round(50 * keep)),
                "keep_rate": keep,
                "effective_keep_rate": keep + 0.02,
                "mean_assigned_weight": keep + 0.02,
                "decision_reason_counts": {"high_quality": 10, "silent_error_true": 8},
                "weight_distribution": {"1.00": 10, "0.00": 8, "0.50": 12},
                "kept_accuracy_mean": 0.9,
                "rejected_accuracy_mean": 0.1,
                "kept_pedagogical_mean": 7.5,
                "rejected_pedagogical_mean": 2.3,
                "kept_silent_error_rate": 0.0,
                "rejected_silent_error_rate": 0.6,
            }
            (ctx / "soft_pvf_report.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return run_dir


def test_export_soft_pvf_confirmatory_analysis_writes_expected_outputs() -> None:
    base = Path("outputs/.tmp") / f"soft_pvf_confirmatory_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_a = _make_fake_run(base, seed=141, tag="run")
        run_b = _make_fake_run(base, seed=142, tag="run")
        out_dir = base / "analysis" / "tables"
        artifacts = export_soft_pvf_confirmatory_analysis(run_dirs=[run_a, run_b], out_dir=out_dir)

        required = [
            artifacts.run_level_csv,
            artifacts.seed_stats_csv,
            artifacts.generation_deltas_csv,
            artifacts.branch_deltas_csv,
            artifacts.decision_reasons_csv,
            artifacts.keep_reject_quality_csv,
            artifacts.policy_summary_csv,
            artifacts.stress_summary_csv,
            artifacts.accuracy_plot,
            artifacts.pedagogical_plot,
            artifacts.silent_error_plot,
            artifacts.keep_rate_plot,
            artifacts.metadata_json,
        ]
        missing = [str(p) for p in required if not p.exists()]
        assert not missing, f"Missing artifacts: {missing}"

        policy = pd.read_csv(artifacts.policy_summary_csv)
        assert {"branch", "policy_name", "generation", "rank_pedagogy_desc"}.issubset(set(policy.columns))
        assert set(policy["branch"].tolist()) == {
            "pure_recycling",
            "soft_pvf_medium",
            "soft_pvf_noisy_keep",
        }
    finally:
        shutil.rmtree(base, ignore_errors=True)

