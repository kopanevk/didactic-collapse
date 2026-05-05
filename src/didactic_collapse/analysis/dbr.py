from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from didactic_collapse.analysis.baseline_series import build_branch_deltas, build_generation_deltas
from didactic_collapse.analysis.compare_runs import build_first_experiment_run_metrics


@dataclass(frozen=True)
class DBRAnalysisArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    budget_summary_csv: Path
    budget_summary_parquet: Path
    defect_rates_before_after_csv: Path
    defect_rates_before_after_parquet: Path
    bucket_coverage_csv: Path
    bucket_coverage_parquet: Path
    stress_summary_csv: Path
    stress_summary_parquet: Path
    accuracy_plot: Path
    pedagogical_plot: Path
    silent_error_plot: Path
    keep_rate_plot: Path
    metadata_json: Path


def _seed_from_run_snapshot(run_dir: Path) -> int:
    snapshot_path = run_dir / "run_config.snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run snapshot: {snapshot_path}")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return int(payload["config"]["project"]["seed"])


def _build_run_level(run_dir: Path) -> pd.DataFrame:
    metrics = build_first_experiment_run_metrics(run_dir)
    metrics["run_id"] = run_dir.name
    metrics["run_dir"] = str(run_dir)
    metrics["seed"] = _seed_from_run_snapshot(run_dir)
    metrics["evaluation_mode"] = "inference_recycling_only"
    cols = [
        "run_id",
        "run_dir",
        "seed",
        "model_name",
        "branch",
        "generation",
        "sample_count",
        "accuracy_mean",
        "pedagogical_score_mean",
        "silent_error_rate",
        "parse_failure_pred_count",
        "sample_count_from_accuracy",
        "parse_failure_pred_rate",
        "accuracy_source",
        "evaluation_mode",
    ]
    return metrics[cols].sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


def _safe_generation_deltas(run_level: pd.DataFrame) -> pd.DataFrame:
    if run_level.empty or run_level["generation"].nunique() < 2:
        return pd.DataFrame(
            columns=[
                "run_id",
                "seed",
                "model_name",
                "branch",
                "generation_start",
                "generation_end",
                "delta_generation",
                "evaluation_mode",
                "delta_accuracy_mean",
                "delta_pedagogical_score_mean",
                "delta_silent_error_rate",
                "delta_parse_failure_pred_rate",
            ]
        )
    return build_generation_deltas(run_level)


def _collect_dbr_reports(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*/*/gen_*/dbr_budget_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["report_path"] = str(path)
        rows.append(payload)
    if not rows:
        return pd.DataFrame(
            columns=[
                "branch",
                "generation",
                "target_size",
                "selected_count",
                "selection_rate",
                "min_selection_rate",
                "budgets",
                "defect_rates_before",
                "defect_rates_after",
                "budget_violations",
                "relaxation_steps_used",
                "bucket_coverage_before",
                "bucket_coverage_after",
                "fallback_bucket_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_budget_summary(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty:
        return pd.DataFrame(
            columns=[
                "branch",
                "generation",
                "target_size",
                "selected_count",
                "selection_rate",
                "min_selection_rate",
                "fallback_bucket_count",
                "relaxation_steps_used",
                "budget_violations_json",
            ]
        )
    out = reports[
        [
            "branch",
            "generation",
            "target_size",
            "selected_count",
            "selection_rate",
            "min_selection_rate",
            "fallback_bucket_count",
            "relaxation_steps_used",
            "budget_violations",
        ]
    ].copy()
    out["budget_violations_json"] = out["budget_violations"].map(lambda x: json.dumps(x, ensure_ascii=False))
    out["relaxation_steps_used"] = out["relaxation_steps_used"].map(lambda x: json.dumps(x, ensure_ascii=False))
    out = out.drop(columns=["budget_violations"])
    return out.sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_defect_rates(reports: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if reports.empty:
        return pd.DataFrame(
            columns=["branch", "generation", "defect_name", "rate_before", "rate_after", "delta_after_minus_before"]
        )
    for rec in reports.to_dict(orient="records"):
        branch = str(rec.get("branch"))
        generation = int(rec.get("generation"))
        before = rec.get("defect_rates_before") or {}
        after = rec.get("defect_rates_after") or {}
        for defect_name in sorted(set(before.keys()) | set(after.keys())):
            rb = float(before.get(defect_name, 0.0))
            ra = float(after.get(defect_name, 0.0))
            rows.append(
                {
                    "branch": branch,
                    "generation": generation,
                    "defect_name": defect_name,
                    "rate_before": rb,
                    "rate_after": ra,
                    "delta_after_minus_before": ra - rb,
                }
            )
    return pd.DataFrame(rows).sort_values(["branch", "generation", "defect_name"]).reset_index(drop=True)


def _build_bucket_coverage(reports: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if reports.empty:
        return pd.DataFrame(
            columns=["branch", "generation", "bucket", "count_before", "count_after", "rate_before", "rate_after"]
        )
    for rec in reports.to_dict(orient="records"):
        branch = str(rec.get("branch"))
        generation = int(rec.get("generation"))
        before = rec.get("bucket_coverage_before") or {}
        after = rec.get("bucket_coverage_after") or {}
        total_before = max(1, int(sum(int(v) for v in before.values())))
        total_after = max(1, int(sum(int(v) for v in after.values())))
        for bucket in ("short", "medium", "long"):
            cb = int(before.get(bucket, 0))
            ca = int(after.get(bucket, 0))
            rows.append(
                {
                    "branch": branch,
                    "generation": generation,
                    "bucket": bucket,
                    "count_before": cb,
                    "count_after": ca,
                    "rate_before": float(cb / total_before),
                    "rate_after": float(ca / total_after),
                }
            )
    return pd.DataFrame(rows).sort_values(["branch", "generation", "bucket"]).reset_index(drop=True)


def _build_stress_summary(run_level: pd.DataFrame, budget_summary: pd.DataFrame) -> pd.DataFrame:
    base = run_level[
        [
            "model_name",
            "branch",
            "generation",
            "sample_count",
            "accuracy_mean",
            "pedagogical_score_mean",
            "silent_error_rate",
            "parse_failure_pred_rate",
            "evaluation_mode",
        ]
    ].copy()
    if budget_summary.empty:
        for col in [
            "target_size",
            "selected_count",
            "selection_rate",
            "min_selection_rate",
            "fallback_bucket_count",
            "relaxation_steps_used",
            "budget_violations_json",
        ]:
            base[col] = pd.NA
        return base.sort_values(["branch", "generation"]).reset_index(drop=True)

    merged = base.merge(
        budget_summary[
            [
                "branch",
                "generation",
                "target_size",
                "selected_count",
                "selection_rate",
                "min_selection_rate",
                "fallback_bucket_count",
                "relaxation_steps_used",
                "budget_violations_json",
            ]
        ],
        on=["branch", "generation"],
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(["branch", "generation"]).reset_index(drop=True)


def _plot_metric_by_branch_generation(run_level: pd.DataFrame, *, metric: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for branch, sub in run_level.groupby("branch", as_index=False):
        sub = sub.sort_values("generation")
        plt.plot(sub["generation"], sub[metric], marker="o", label=str(branch))
    plt.xlabel("Generation")
    plt.ylabel(metric)
    plt.title(f"{metric} by branch and generation (DBR)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_keep_rate(budget_summary: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if budget_summary.empty:
        plt.title("DBR keep-rate by generation (no data)")
    else:
        for branch, sub in budget_summary.groupby("branch", as_index=False):
            sub = sub.sort_values("generation")
            plt.plot(sub["generation"], sub["selection_rate"], marker="o", label=str(branch))
        plt.ylim(0.0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title("DBR selection_rate by generation")
        plt.xlabel("Generation")
        plt.ylabel("selection_rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_table(df: pd.DataFrame, *, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def export_dbr_analysis(
    *,
    run_dir: Path,
    out_dir: Path | None = None,
    file_prefix: str = "dbr",
) -> DBRAnalysisArtifacts:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    target_out = out_dir or (run_dir / "tables")
    target_out.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_level = _build_run_level(run_dir)
    generation_deltas = _safe_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)

    reports = _collect_dbr_reports(run_dir)
    budget_summary = _build_budget_summary(reports)
    defect_rates = _build_defect_rates(reports)
    bucket_coverage = _build_bucket_coverage(reports)
    stress_summary = _build_stress_summary(run_level, budget_summary)

    run_level_csv = target_out / f"{file_prefix}_run_level.csv"
    run_level_parquet = target_out / f"{file_prefix}_run_level.parquet"
    generation_deltas_csv = target_out / f"{file_prefix}_generation_deltas.csv"
    generation_deltas_parquet = target_out / f"{file_prefix}_generation_deltas.parquet"
    branch_deltas_csv = target_out / f"{file_prefix}_branch_deltas.csv"
    branch_deltas_parquet = target_out / f"{file_prefix}_branch_deltas.parquet"
    budget_summary_csv = target_out / f"{file_prefix}_budget_summary.csv"
    budget_summary_parquet = target_out / f"{file_prefix}_budget_summary.parquet"
    defect_rates_csv = target_out / f"{file_prefix}_defect_rates_before_after.csv"
    defect_rates_parquet = target_out / f"{file_prefix}_defect_rates_before_after.parquet"
    bucket_coverage_csv = target_out / f"{file_prefix}_bucket_coverage.csv"
    bucket_coverage_parquet = target_out / f"{file_prefix}_bucket_coverage.parquet"
    stress_summary_csv = target_out / f"{file_prefix}_stress_summary.csv"
    stress_summary_parquet = target_out / f"{file_prefix}_stress_summary.parquet"

    _write_table(run_level, csv_path=run_level_csv, parquet_path=run_level_parquet)
    _write_table(generation_deltas, csv_path=generation_deltas_csv, parquet_path=generation_deltas_parquet)
    _write_table(branch_deltas, csv_path=branch_deltas_csv, parquet_path=branch_deltas_parquet)
    _write_table(budget_summary, csv_path=budget_summary_csv, parquet_path=budget_summary_parquet)
    _write_table(defect_rates, csv_path=defect_rates_csv, parquet_path=defect_rates_parquet)
    _write_table(bucket_coverage, csv_path=bucket_coverage_csv, parquet_path=bucket_coverage_parquet)
    _write_table(stress_summary, csv_path=stress_summary_csv, parquet_path=stress_summary_parquet)

    accuracy_plot = figures_dir / f"{file_prefix}_accuracy_by_branch_generation.png"
    pedagogical_plot = figures_dir / f"{file_prefix}_pedagogical_by_branch_generation.png"
    silent_error_plot = figures_dir / f"{file_prefix}_silent_error_by_branch_generation.png"
    keep_rate_plot = figures_dir / f"{file_prefix}_keep_rate_by_generation.png"
    _plot_metric_by_branch_generation(run_level, metric="accuracy_mean", out_path=accuracy_plot)
    _plot_metric_by_branch_generation(run_level, metric="pedagogical_score_mean", out_path=pedagogical_plot)
    _plot_metric_by_branch_generation(run_level, metric="silent_error_rate", out_path=silent_error_plot)
    _plot_keep_rate(budget_summary, out_path=keep_rate_plot)

    metadata = {
        "run_dir": str(run_dir),
        "analysis_out_dir": str(target_out),
        "file_prefix": file_prefix,
        "evaluation_mode": "inference_recycling_only",
        "rows": {
            "run_level": int(len(run_level)),
            "generation_deltas": int(len(generation_deltas)),
            "branch_deltas": int(len(branch_deltas)),
            "budget_summary": int(len(budget_summary)),
            "defect_rates_before_after": int(len(defect_rates)),
            "bucket_coverage": int(len(bucket_coverage)),
            "stress_summary": int(len(stress_summary)),
        },
    }
    metadata_json = target_out / f"{file_prefix}_analysis_metadata.json"
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return DBRAnalysisArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        budget_summary_csv=budget_summary_csv,
        budget_summary_parquet=budget_summary_parquet,
        defect_rates_before_after_csv=defect_rates_csv,
        defect_rates_before_after_parquet=defect_rates_parquet,
        bucket_coverage_csv=bucket_coverage_csv,
        bucket_coverage_parquet=bucket_coverage_parquet,
        stress_summary_csv=stress_summary_csv,
        stress_summary_parquet=stress_summary_parquet,
        accuracy_plot=accuracy_plot,
        pedagogical_plot=pedagogical_plot,
        silent_error_plot=silent_error_plot,
        keep_rate_plot=keep_rate_plot,
        metadata_json=metadata_json,
    )


def summarize_dbr_confirmatory(
    *,
    run_dirs: Sequence[Path],
    out_dir: Path,
    file_prefix: str = "dbr_confirmatory",
) -> pd.DataFrame:
    """Convenience aggregation across multiple DBR runs (single table)."""
    rows: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        run_level = _build_run_level(run_dir)
        run_level["run_dir"] = str(run_dir)
        rows.append(run_level)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / f"{file_prefix}_run_level.csv", index=False)
    combined.to_parquet(out_dir / f"{file_prefix}_run_level.parquet", index=False)
    return combined

