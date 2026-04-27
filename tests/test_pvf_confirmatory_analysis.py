from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.pvf_confirmatory import export_pvf_confirmatory_analysis


def _make_fake_run(
    *,
    root: Path,
    run_name: str,
    seed: int,
) -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    snapshot = {
        "config": {
            "project": {"seed": seed},
            "paths": {"data_root": "data"},
        }
    }
    (run_dir / "run_config.snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_name = "qwen2.5:0.5b"
    rows: list[dict[str, object]] = []
    for branch in ["pure_recycling", "pvf_medium"]:
        for generation in [0, 1, 2]:
            rows.append(
                {
                    "model_name": model_name,
                    "branch": branch,
                    "generation": generation,
                    "sample_count": 4,
                    "accuracy_mean": 0.5 if branch == "pure_recycling" else 0.55,
                    "pedagogical_score_mean": 5.0 - 0.2 * generation + (0.2 if branch == "pvf_medium" else 0.0),
                    "silent_error_rate": 0.25 + 0.05 * generation + (-0.05 if branch == "pvf_medium" else 0.0),
                }
            )
            ctx_dir = run_dir / model_name.replace(":", "_") / branch / f"gen_{generation}"
            ctx_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "model_name": model_name,
                        "branch": branch,
                        "generation": generation,
                        "pred_parse_success": True,
                    },
                    {
                        "model_name": model_name,
                        "branch": branch,
                        "generation": generation,
                        "pred_parse_success": False,
                    },
                    {
                        "model_name": model_name,
                        "branch": branch,
                        "generation": generation,
                        "pred_parse_success": True,
                    },
                    {
                        "model_name": model_name,
                        "branch": branch,
                        "generation": generation,
                        "pred_parse_success": True,
                    },
                ]
            ).to_parquet(ctx_dir / "accuracy_table.parquet", index=False)

            if branch == "pvf_medium":
                report = {
                    "model_name": model_name,
                    "branch": branch,
                    "generation": generation,
                    "seed": seed,
                    "threshold_score": 5,
                    "min_keep_ratio": 0.15,
                    "total_candidates": 50,
                    "kept_count": 25 - generation,
                    "rejected_count": 25 + generation,
                    "keep_rate": 0.50 - generation * 0.02,
                    "rejection_reason_counts": {
                        "accuracy_incorrect": 10 + generation,
                        "pedagogical_below_threshold": 8 + generation,
                        "silent_error_true": 5 + generation,
                    },
                    "kept_accuracy_mean": 1.0,
                    "rejected_accuracy_mean": 0.1,
                    "kept_pedagogical_mean": 8.0,
                    "rejected_pedagogical_mean": 2.0,
                    "kept_silent_error_rate": 0.0,
                    "rejected_silent_error_rate": 0.5,
                }
                (ctx_dir / "pvf_filter_report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    pd.DataFrame(rows).to_csv(run_dir / "tables" / "first_experiment_summary.csv", index=False)
    return run_dir


def test_export_pvf_confirmatory_analysis_writes_expected_artifacts() -> None:
    base = Path("outputs/.tmp") / f"pvf_confirmatory_test_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_a = _make_fake_run(root=base, run_name="run_seed_1", seed=1)
        run_b = _make_fake_run(root=base, run_name="run_seed_2", seed=2)
        out_dir = base / "analysis" / "tables"
        artifacts = export_pvf_confirmatory_analysis(
            run_dirs=[run_a, run_b],
            out_dir=out_dir,
            bootstrap_seed=123,
        )

        required_paths = [
            artifacts.run_level_csv,
            artifacts.seed_stats_csv,
            artifacts.generation_deltas_csv,
            artifacts.branch_deltas_csv,
            artifacts.reject_reasons_csv,
            artifacts.keep_reject_quality_csv,
            artifacts.stress_summary_csv,
            artifacts.accuracy_plot,
            artifacts.pedagogical_plot,
            artifacts.silent_error_plot,
            artifacts.keep_rate_plot,
            artifacts.metadata_json,
        ]
        missing = [str(p) for p in required_paths if not p.exists()]
        assert not missing, f"Missing artifacts: {missing}"

        stress = pd.read_csv(artifacts.stress_summary_csv)
        assert {"branch", "generation", "accuracy_mean", "pedagogical_score_mean", "silent_error_rate_mean"}.issubset(
            set(stress.columns)
        )
        pvf_rows = stress[stress["branch"] == "pvf_medium"]
        assert not pvf_rows.empty
        assert pvf_rows["keep_rate_mean"].notna().all()

        reject = pd.read_csv(artifacts.reject_reasons_csv)
        assert {"reason", "count", "branch", "generation", "seed"}.issubset(set(reject.columns))
        assert not reject.empty
    finally:
        shutil.rmtree(base, ignore_errors=True)

