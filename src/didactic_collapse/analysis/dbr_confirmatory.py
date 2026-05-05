from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from didactic_collapse.analysis.baseline_series import (
    EVAL_MODE,
    build_branch_deltas,
    build_generation_deltas,
    build_seed_level_summary,
    collect_baseline_run_metrics,
)


@dataclass(frozen=True)
class DBRConfirmatoryArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    seed_stats_csv: Path
    seed_stats_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    budget_summary_csv: Path
    budget_summary_parquet: Path
    defect_rates_csv: Path
    defect_rates_parquet: Path
    bucket_coverage_csv: Path
    bucket_coverage_parquet: Path
    stress_summary_csv: Path
    stress_summary_parquet: Path
    matched_gen2_csv: Path
    matched_gen2_parquet: Path
    accuracy_plot: Path
    pedagogical_plot: Path
    silent_error_plot: Path
    selection_rate_plot: Path
    defect_rates_plot: Path
    metadata_json: Path


def _load_seed_from_snapshot(run_dir: Path) -> int:
    snapshot_path = run_dir / "run_config.snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run snapshot: {snapshot_path}")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return int(payload["config"]["project"]["seed"])


def _bootstrap_ci(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    s = pd.Series(vals)
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    delta = 1.96 * std / max(1.0, len(vals) ** 0.5)
    return mean - delta, mean + delta


def _collect_dbr_reports(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        seed = _load_seed_from_snapshot(run_dir)
        for path in sorted(run_dir.glob("*/*/gen_*/dbr_budget_report.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "seed": int(seed),
                    "model_name": str(payload.get("model_name", "")),
                    "branch": str(payload.get("branch", "")),
                    "generation": int(payload.get("generation", 0)),
                    "target_size": int(payload.get("target_size", 0)),
                    "selected_count": int(payload.get("selected_count", 0)),
                    "selection_rate": float(payload.get("selection_rate", 0.0)),
                    "min_selection_rate": float(payload.get("min_selection_rate", 0.0)),
                    "budgets": payload.get("budgets", {}),
                    "defect_rates_before": payload.get("defect_rates_before", {}),
                    "defect_rates_after": payload.get("defect_rates_after", {}),
                    "budget_violations": payload.get("budget_violations", {}),
                    "relaxation_steps_used": payload.get("relaxation_steps_used", []),
                    "bucket_coverage_before": payload.get("bucket_coverage_before", {}),
                    "bucket_coverage_after": payload.get("bucket_coverage_after", {}),
                    "fallback_bucket_count": int(payload.get("fallback_bucket_count", 0)),
                    "report_path": str(path),
                    "evaluation_mode": EVAL_MODE,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_dir",
                "seed",
                "model_name",
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
                "report_path",
                "evaluation_mode",
            ]
        )
    return pd.DataFrame(rows).sort_values(["seed", "model_name", "branch", "generation"]).reset_index(drop=True)


def _build_budget_summary(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "generation",
                "seed_count",
                "target_size_mean",
                "selected_count_mean",
                "selection_rate_mean",
                "selection_rate_std",
                "selection_rate_ci_low",
                "selection_rate_ci_high",
                "min_selection_rate_mean",
                "fallback_bucket_count_mean",
                "relaxation_steps_any_json",
                "budget_violations_any_json",
                "evaluation_mode",
            ]
        )

    rows: list[dict[str, object]] = []
    grouped = reports.groupby(["model_name", "branch", "generation"], as_index=False)
    for _, grp in grouped:
        selection_vals = [float(v) for v in grp["selection_rate"].tolist()]
        ci_low, ci_high = _bootstrap_ci(selection_vals)
        rows.append(
            {
                "model_name": str(grp["model_name"].iloc[0]),
                "branch": str(grp["branch"].iloc[0]),
                "generation": int(grp["generation"].iloc[0]),
                "seed_count": int(grp["seed"].nunique()),
                "target_size_mean": float(pd.to_numeric(grp["target_size"], errors="coerce").mean()),
                "selected_count_mean": float(pd.to_numeric(grp["selected_count"], errors="coerce").mean()),
                "selection_rate_mean": float(pd.Series(selection_vals).mean()),
                "selection_rate_std": float(pd.Series(selection_vals).std(ddof=1)) if len(selection_vals) > 1 else 0.0,
                "selection_rate_ci_low": ci_low,
                "selection_rate_ci_high": ci_high,
                "min_selection_rate_mean": float(pd.to_numeric(grp["min_selection_rate"], errors="coerce").mean()),
                "fallback_bucket_count_mean": float(pd.to_numeric(grp["fallback_bucket_count"], errors="coerce").mean()),
                "relaxation_steps_any_json": json.dumps(
                    sorted({step for steps in grp["relaxation_steps_used"] for step in (steps or [])}),
                    ensure_ascii=False,
                ),
                "budget_violations_any_json": json.dumps(
                    [x for x in grp["budget_violations"].tolist()],
                    ensure_ascii=False,
                ),
                "evaluation_mode": EVAL_MODE,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


def _build_defect_rates(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "generation",
                "defect_name",
                "seed_count",
                "rate_before_mean",
                "rate_after_mean",
                "delta_after_minus_before",
            ]
        )

    rows: list[dict[str, object]] = []
    for rec in reports.to_dict(orient="records"):
        before = rec.get("defect_rates_before") or {}
        after = rec.get("defect_rates_after") or {}
        keys = sorted(set(before.keys()) | set(after.keys()))
        for key in keys:
            rows.append(
                {
                    "model_name": str(rec["model_name"]),
                    "branch": str(rec["branch"]),
                    "generation": int(rec["generation"]),
                    "seed": int(rec["seed"]),
                    "defect_name": str(key),
                    "rate_before": float(before.get(key, 0.0)),
                    "rate_after": float(after.get(key, 0.0)),
                }
            )
    exploded = pd.DataFrame(rows)
    out = (
        exploded.groupby(["model_name", "branch", "generation", "defect_name"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            rate_before_mean=("rate_before", "mean"),
            rate_after_mean=("rate_after", "mean"),
        )
        .sort_values(["model_name", "branch", "generation", "defect_name"])
        .reset_index(drop=True)
    )
    out["delta_after_minus_before"] = out["rate_after_mean"] - out["rate_before_mean"]
    return out


def _build_bucket_coverage(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "generation",
                "bucket",
                "seed_count",
                "count_before_mean",
                "count_after_mean",
                "rate_before_mean",
                "rate_after_mean",
            ]
        )

    rows: list[dict[str, object]] = []
    for rec in reports.to_dict(orient="records"):
        before = rec.get("bucket_coverage_before") or {}
        after = rec.get("bucket_coverage_after") or {}
        total_before = max(1, int(sum(int(v) for v in before.values())))
        total_after = max(1, int(sum(int(v) for v in after.values())))
        for bucket in ("short", "medium", "long"):
            cb = int(before.get(bucket, 0))
            ca = int(after.get(bucket, 0))
            rows.append(
                {
                    "model_name": str(rec["model_name"]),
                    "branch": str(rec["branch"]),
                    "generation": int(rec["generation"]),
                    "seed": int(rec["seed"]),
                    "bucket": bucket,
                    "count_before": cb,
                    "count_after": ca,
                    "rate_before": float(cb / total_before),
                    "rate_after": float(ca / total_after),
                }
            )
    exploded = pd.DataFrame(rows)
    out = (
        exploded.groupby(["model_name", "branch", "generation", "bucket"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            count_before_mean=("count_before", "mean"),
            count_after_mean=("count_after", "mean"),
            rate_before_mean=("rate_before", "mean"),
            rate_after_mean=("rate_after", "mean"),
        )
        .sort_values(["model_name", "branch", "generation", "bucket"])
        .reset_index(drop=True)
    )
    return out


def _build_stress_summary(seed_stats: pd.DataFrame, budget_summary: pd.DataFrame) -> pd.DataFrame:
    summary = seed_stats.rename(
        columns={
            "accuracy_mean_mean": "accuracy_mean",
            "accuracy_mean_std": "accuracy_std",
            "accuracy_mean_ci_low": "accuracy_ci_low",
            "accuracy_mean_ci_high": "accuracy_ci_high",
            "pedagogical_score_mean_mean": "pedagogical_score_mean",
            "pedagogical_score_mean_std": "pedagogical_score_std",
            "pedagogical_score_mean_ci_low": "pedagogical_ci_low",
            "pedagogical_score_mean_ci_high": "pedagogical_ci_high",
            "silent_error_rate_mean": "silent_error_rate",
            "silent_error_rate_std": "silent_error_std",
            "silent_error_rate_ci_low": "silent_error_ci_low",
            "silent_error_rate_ci_high": "silent_error_ci_high",
            "parse_failure_pred_rate_mean": "parse_failure_pred_rate",
            "parse_failure_pred_rate_std": "parse_failure_pred_rate_std",
            "parse_failure_pred_rate_ci_low": "parse_failure_pred_rate_ci_low",
            "parse_failure_pred_rate_ci_high": "parse_failure_pred_rate_ci_high",
        }
    ).copy()

    if budget_summary.empty:
        for col in (
            "selection_rate_mean",
            "selection_rate_std",
            "selection_rate_ci_low",
            "selection_rate_ci_high",
            "target_size_mean",
            "selected_count_mean",
            "min_selection_rate_mean",
            "fallback_bucket_count_mean",
            "relaxation_steps_any_json",
            "budget_violations_any_json",
        ):
            summary[col] = pd.NA
        return summary.sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)

    merged = summary.merge(
        budget_summary[
            [
                "model_name",
                "branch",
                "generation",
                "selection_rate_mean",
                "selection_rate_std",
                "selection_rate_ci_low",
                "selection_rate_ci_high",
                "target_size_mean",
                "selected_count_mean",
                "min_selection_rate_mean",
                "fallback_bucket_count_mean",
                "relaxation_steps_any_json",
                "budget_violations_any_json",
            ]
        ],
        on=["model_name", "branch", "generation"],
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


def _build_matched_gen2(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        seed = _load_seed_from_snapshot(run_dir)
        summary_path = run_dir / "tables" / "first_experiment_summary.csv"
        if not summary_path.exists():
            continue
        summary_df = pd.read_csv(summary_path)
        model_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name != "tables" and p.name != "figures"]
        if not model_dirs:
            continue
        # Single-model runs are expected in this project.
        model_dir = model_dirs[0]
        pure_path = model_dir / "pure_recycling" / "gen_2" / "eval_merged.parquet"
        dbr_path = model_dir / "dbr_medium" / "gen_2" / "eval_merged.parquet"
        if not pure_path.exists() or not dbr_path.exists():
            continue
        pure = pd.read_parquet(pure_path)
        dbr = pd.read_parquet(dbr_path)
        req = {"example_id", "is_correct", "overall_pedagogical_score", "is_silent_error"}
        if req.difference(pure.columns) or req.difference(dbr.columns):
            continue
        pure_s = pure[list(req)].rename(
            columns={
                "is_correct": "is_correct_pure",
                "overall_pedagogical_score": "pedagogy_pure",
                "is_silent_error": "silent_pure",
            }
        )
        dbr_s = dbr[list(req)].rename(
            columns={
                "is_correct": "is_correct_dbr",
                "overall_pedagogical_score": "pedagogy_dbr",
                "is_silent_error": "silent_dbr",
            }
        )
        matched = pure_s.merge(dbr_s, on="example_id", how="inner", validate="one_to_one")

        pure_row = summary_df[(summary_df["branch"] == "pure_recycling") & (summary_df["generation"] == 2.0)]
        dbr_row = summary_df[(summary_df["branch"] == "dbr_medium") & (summary_df["generation"] == 2.0)]
        if pure_row.empty or dbr_row.empty:
            continue
        pure_agg = pure_row.iloc[0]
        dbr_agg = dbr_row.iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "seed": int(seed),
                "model_name": str(pure_agg["model_name"]),
                "generation": 2,
                "pure_count": int(len(pure)),
                "dbr_count": int(len(dbr)),
                "matched_count": int(len(matched)),
                "accuracy_pure_agg": float(pure_agg["accuracy_mean"]),
                "accuracy_dbr_agg": float(dbr_agg["accuracy_mean"]),
                "pedagogy_pure_agg": float(pure_agg["pedagogical_score_mean"]),
                "pedagogy_dbr_agg": float(dbr_agg["pedagogical_score_mean"]),
                "silent_pure_agg": float(pure_agg["silent_error_rate"]),
                "silent_dbr_agg": float(dbr_agg["silent_error_rate"]),
                "accuracy_pure_matched": float(matched["is_correct_pure"].astype(bool).mean()) if len(matched) else float("nan"),
                "accuracy_dbr_matched": float(matched["is_correct_dbr"].astype(bool).mean()) if len(matched) else float("nan"),
                "pedagogy_pure_matched": float(matched["pedagogy_pure"].mean()) if len(matched) else float("nan"),
                "pedagogy_dbr_matched": float(matched["pedagogy_dbr"].mean()) if len(matched) else float("nan"),
                "silent_pure_matched": float(matched["silent_pure"].astype(bool).mean()) if len(matched) else float("nan"),
                "silent_dbr_matched": float(matched["silent_dbr"].astype(bool).mean()) if len(matched) else float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_dir",
                "seed",
                "model_name",
                "generation",
                "pure_count",
                "dbr_count",
                "matched_count",
                "accuracy_pure_agg",
                "accuracy_dbr_agg",
                "pedagogy_pure_agg",
                "pedagogy_dbr_agg",
                "silent_pure_agg",
                "silent_dbr_agg",
                "accuracy_pure_matched",
                "accuracy_dbr_matched",
                "pedagogy_pure_matched",
                "pedagogy_dbr_matched",
                "silent_pure_matched",
                "silent_dbr_matched",
            ]
        )
    out = pd.DataFrame(rows)
    out["delta_accuracy_agg_dbr_minus_pure"] = out["accuracy_dbr_agg"] - out["accuracy_pure_agg"]
    out["delta_accuracy_matched_dbr_minus_pure"] = out["accuracy_dbr_matched"] - out["accuracy_pure_matched"]
    out["delta_pedagogy_agg_dbr_minus_pure"] = out["pedagogy_dbr_agg"] - out["pedagogy_pure_agg"]
    out["delta_pedagogy_matched_dbr_minus_pure"] = out["pedagogy_dbr_matched"] - out["pedagogy_pure_matched"]
    out["delta_silent_agg_dbr_minus_pure"] = out["silent_dbr_agg"] - out["silent_pure_agg"]
    out["delta_silent_matched_dbr_minus_pure"] = out["silent_dbr_matched"] - out["silent_pure_matched"]
    out["evaluation_mode"] = EVAL_MODE
    return out.sort_values(["seed"]).reset_index(drop=True)


def _plot_seed_metric(seed_stats_df: pd.DataFrame, *, metric: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_ci_low"
    high_col = f"{metric}_ci_high"
    required = {"generation", "branch", mean_col, low_col, high_col}
    missing = required.difference(seed_stats_df.columns)
    if missing:
        raise ValueError(f"seed_stats_df missing columns for plotting: {sorted(missing)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for branch, sub in seed_stats_df.groupby("branch", as_index=False):
        sub = sub.sort_values("generation")
        x = sub["generation"].astype(float).to_numpy()
        y = sub[mean_col].astype(float).to_numpy()
        low = sub[low_col].astype(float).to_numpy()
        high = sub[high_col].astype(float).to_numpy()
        plt.plot(x, y, marker="o", label=str(branch))
        plt.fill_between(x, low, high, alpha=0.15)
    plt.xlabel("Generation")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_selection_rate(budget_summary_df: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if budget_summary_df.empty:
        plt.text(0.5, 0.5, "No DBR budget data", ha="center", va="center")
    else:
        for branch, sub in budget_summary_df.groupby("branch", as_index=False):
            sub = sub.sort_values("generation")
            x = sub["generation"].astype(float).to_numpy()
            y = sub["selection_rate_mean"].astype(float).to_numpy()
            low = sub["selection_rate_ci_low"].astype(float).to_numpy()
            high = sub["selection_rate_ci_high"].astype(float).to_numpy()
            plt.plot(x, y, marker="o", label=str(branch))
            plt.fill_between(x, low, high, alpha=0.15)
        plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("selection_rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_defect_rates(defect_rates_df: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    if defect_rates_df.empty:
        plt.text(0.5, 0.5, "No defect-rate data", ha="center", va="center")
    else:
        dbr_only = defect_rates_df[defect_rates_df["branch"] == "dbr_medium"].copy()
        if dbr_only.empty:
            dbr_only = defect_rates_df.copy()
        for defect_name, sub in dbr_only.groupby("defect_name", as_index=False):
            sub = sub.sort_values("generation")
            plt.plot(
                sub["generation"].astype(float).to_numpy(),
                sub["rate_before_mean"].astype(float).to_numpy(),
                marker="o",
                linestyle="--",
                label=f"{defect_name}:before",
            )
            plt.plot(
                sub["generation"].astype(float).to_numpy(),
                sub["rate_after_mean"].astype(float).to_numpy(),
                marker="o",
                linestyle="-",
                label=f"{defect_name}:after",
            )
        plt.legend(ncol=2, fontsize=8)
    plt.xlabel("Generation")
    plt.ylabel("defect_rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_dbr_confirmatory_analysis(
    *,
    run_dirs: Sequence[Path],
    out_dir: Path,
) -> DBRConfirmatoryArtifacts:
    if not run_dirs:
        raise ValueError("run_dirs must not be empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_level = collect_baseline_run_metrics(run_dirs)
    seed_stats = build_seed_level_summary(run_level)
    generation_deltas = build_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)

    dbr_reports = _collect_dbr_reports(run_dirs)
    budget_summary = _build_budget_summary(dbr_reports)
    defect_rates = _build_defect_rates(dbr_reports)
    bucket_coverage = _build_bucket_coverage(dbr_reports)
    stress_summary = _build_stress_summary(seed_stats, budget_summary)
    matched_gen2 = _build_matched_gen2(run_dirs)

    run_level_csv = out_dir / "dbr_confirmatory_run_level.csv"
    run_level_parquet = out_dir / "dbr_confirmatory_run_level.parquet"
    seed_stats_csv = out_dir / "dbr_confirmatory_seed_stats.csv"
    seed_stats_parquet = out_dir / "dbr_confirmatory_seed_stats.parquet"
    generation_deltas_csv = out_dir / "dbr_confirmatory_generation_deltas.csv"
    generation_deltas_parquet = out_dir / "dbr_confirmatory_generation_deltas.parquet"
    branch_deltas_csv = out_dir / "dbr_confirmatory_branch_deltas.csv"
    branch_deltas_parquet = out_dir / "dbr_confirmatory_branch_deltas.parquet"
    budget_summary_csv = out_dir / "dbr_confirmatory_budget_summary.csv"
    budget_summary_parquet = out_dir / "dbr_confirmatory_budget_summary.parquet"
    defect_rates_csv = out_dir / "dbr_confirmatory_defect_rates_before_after.csv"
    defect_rates_parquet = out_dir / "dbr_confirmatory_defect_rates_before_after.parquet"
    bucket_coverage_csv = out_dir / "dbr_confirmatory_bucket_coverage.csv"
    bucket_coverage_parquet = out_dir / "dbr_confirmatory_bucket_coverage.parquet"
    stress_summary_csv = out_dir / "dbr_confirmatory_stress_summary.csv"
    stress_summary_parquet = out_dir / "dbr_confirmatory_stress_summary.parquet"
    matched_gen2_csv = out_dir / "dbr_confirmatory_matched_gen2_comparison.csv"
    matched_gen2_parquet = out_dir / "dbr_confirmatory_matched_gen2_comparison.parquet"

    run_level.to_csv(run_level_csv, index=False)
    run_level.to_parquet(run_level_parquet, index=False)
    seed_stats.to_csv(seed_stats_csv, index=False)
    seed_stats.to_parquet(seed_stats_parquet, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    generation_deltas.to_parquet(generation_deltas_parquet, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    branch_deltas.to_parquet(branch_deltas_parquet, index=False)
    budget_summary.to_csv(budget_summary_csv, index=False)
    budget_summary.to_parquet(budget_summary_parquet, index=False)
    defect_rates.to_csv(defect_rates_csv, index=False)
    defect_rates.to_parquet(defect_rates_parquet, index=False)
    bucket_coverage.to_csv(bucket_coverage_csv, index=False)
    bucket_coverage.to_parquet(bucket_coverage_parquet, index=False)
    stress_summary.to_csv(stress_summary_csv, index=False)
    stress_summary.to_parquet(stress_summary_parquet, index=False)
    matched_gen2.to_csv(matched_gen2_csv, index=False)
    matched_gen2.to_parquet(matched_gen2_parquet, index=False)

    figures_dir = out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot = figures_dir / "dbr_confirmatory_accuracy_by_branch_generation.png"
    pedagogical_plot = figures_dir / "dbr_confirmatory_pedagogical_by_branch_generation.png"
    silent_error_plot = figures_dir / "dbr_confirmatory_silent_error_by_branch_generation.png"
    selection_rate_plot = figures_dir / "dbr_confirmatory_selection_rate_by_generation.png"
    defect_rates_plot = figures_dir / "dbr_confirmatory_defect_rates_before_after.png"

    _plot_seed_metric(seed_stats, metric="accuracy_mean", out_path=accuracy_plot)
    _plot_seed_metric(seed_stats, metric="pedagogical_score_mean", out_path=pedagogical_plot)
    _plot_seed_metric(seed_stats, metric="silent_error_rate", out_path=silent_error_plot)
    _plot_selection_rate(budget_summary, out_path=selection_rate_plot)
    _plot_defect_rates(defect_rates, out_path=defect_rates_plot)

    metadata_json = out_dir / "dbr_confirmatory_metadata.json"
    metadata = {
        "evaluation_mode": EVAL_MODE,
        "run_count": int(len(run_dirs)),
        "run_dirs": [str(x) for x in run_dirs],
        "notes": "DBR confirmatory multi-seed analysis (inference_recycling_only).",
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return DBRConfirmatoryArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        seed_stats_csv=seed_stats_csv,
        seed_stats_parquet=seed_stats_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        budget_summary_csv=budget_summary_csv,
        budget_summary_parquet=budget_summary_parquet,
        defect_rates_csv=defect_rates_csv,
        defect_rates_parquet=defect_rates_parquet,
        bucket_coverage_csv=bucket_coverage_csv,
        bucket_coverage_parquet=bucket_coverage_parquet,
        stress_summary_csv=stress_summary_csv,
        stress_summary_parquet=stress_summary_parquet,
        matched_gen2_csv=matched_gen2_csv,
        matched_gen2_parquet=matched_gen2_parquet,
        accuracy_plot=accuracy_plot,
        pedagogical_plot=pedagogical_plot,
        silent_error_plot=silent_error_plot,
        selection_rate_plot=selection_rate_plot,
        defect_rates_plot=defect_rates_plot,
        metadata_json=metadata_json,
    )


__all__ = [
    "DBRConfirmatoryArtifacts",
    "export_dbr_confirmatory_analysis",
]
