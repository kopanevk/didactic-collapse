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
class SoftPVFAnalysisArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    policy_summary_csv: Path
    policy_summary_parquet: Path
    decision_reasons_csv: Path
    decision_reasons_parquet: Path
    keep_rejected_quality_csv: Path
    keep_rejected_quality_parquet: Path
    weight_distribution_csv: Path
    weight_distribution_parquet: Path
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


def _collect_soft_decisions(run_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(run_dir.glob("*/*/gen_*/soft_pvf_decisions.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=[
                "example_id",
                "is_correct",
                "pred_parse_success",
                "overall_pedagogical_score",
                "is_silent_error",
                "assigned_weight",
                "kept",
                "decision_reason",
                "deterministic_score",
                "sampling_value",
                "policy_name",
                "branch",
                "generation",
                "seed",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    if "policy_name" not in out.columns:
        out["policy_name"] = out["branch"].astype(str)
    return out.sort_values(["branch", "generation", "example_id"]).reset_index(drop=True)


def _collect_keep_stats(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*/*/gen_*/pvf_filter_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "branch": payload.get("branch"),
                "generation": int(payload.get("generation")),
                "keep_rate": float(payload.get("keep_rate")),
                "effective_keep_rate": float(payload.get("keep_rate")),
                "mean_assigned_weight": float(payload.get("keep_rate")),
                "kept_count": int(payload.get("kept_count")),
                "rejected_count": int(payload.get("rejected_count")),
                "total_candidates": int(payload.get("total_candidates")),
                "filter_method": "hard_pvf",
                "policy_name": "pvf_medium",
            }
        )
    for path in sorted(run_dir.glob("*/*/gen_*/soft_pvf_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "branch": payload.get("branch"),
                "generation": int(payload.get("generation")),
                "keep_rate": float(payload.get("keep_rate")),
                "effective_keep_rate": float(payload.get("effective_keep_rate")),
                "mean_assigned_weight": float(payload.get("mean_assigned_weight")),
                "kept_count": int(payload.get("kept_count")),
                "rejected_count": int(payload.get("rejected_count")),
                "total_candidates": int(payload.get("total_candidates")),
                "filter_method": "soft_pvf",
                "policy_name": str(payload.get("policy_name", payload.get("branch", "soft_pvf_medium"))),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "branch",
                "generation",
                "keep_rate",
                "effective_keep_rate",
                "mean_assigned_weight",
                "kept_count",
                "rejected_count",
                "total_candidates",
                "filter_method",
                "policy_name",
            ]
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_stress_summary(run_level: pd.DataFrame, keep_stats: pd.DataFrame) -> pd.DataFrame:
    summary = run_level[
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
    if keep_stats.empty:
        summary["keep_rate"] = pd.NA
        summary["effective_keep_rate"] = pd.NA
        summary["mean_assigned_weight"] = pd.NA
        summary["kept_count"] = pd.NA
        summary["rejected_count"] = pd.NA
        summary["total_candidates"] = pd.NA
        summary["filter_method"] = pd.NA
        summary["policy_name"] = pd.NA
        return summary.sort_values(["branch", "generation"]).reset_index(drop=True)
    merged = summary.merge(
        keep_stats,
        on=["branch", "generation"],
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_policy_summary(stress_summary: pd.DataFrame) -> pd.DataFrame:
    if stress_summary.empty:
        return pd.DataFrame(
            columns=[
                "branch",
                "policy_name",
                "generation",
                "sample_count",
                "accuracy_mean",
                "pedagogical_score_mean",
                "silent_error_rate",
                "keep_rate",
                "effective_keep_rate",
                "mean_assigned_weight",
                "rank_accuracy_desc",
                "rank_pedagogy_desc",
                "rank_silent_error_asc",
                "rank_keep_rate_desc",
                "evaluation_mode",
            ]
        )

    max_gen = int(pd.to_numeric(stress_summary["generation"], errors="coerce").max())
    latest = stress_summary[stress_summary["generation"] == max_gen].copy()
    latest["policy_name"] = latest["policy_name"].where(
        latest["policy_name"].notna(),
        latest["branch"],
    )

    latest["rank_accuracy_desc"] = (
        pd.to_numeric(latest["accuracy_mean"], errors="coerce").rank(method="min", ascending=False)
    )
    latest["rank_pedagogy_desc"] = (
        pd.to_numeric(latest["pedagogical_score_mean"], errors="coerce").rank(method="min", ascending=False)
    )
    latest["rank_silent_error_asc"] = (
        pd.to_numeric(latest["silent_error_rate"], errors="coerce").rank(method="min", ascending=True)
    )
    keep_rank_base = pd.to_numeric(latest["keep_rate"], errors="coerce").fillna(-1.0)
    latest["rank_keep_rate_desc"] = keep_rank_base.rank(method="min", ascending=False)

    keep_cols = [
        "branch",
        "policy_name",
        "generation",
        "sample_count",
        "accuracy_mean",
        "pedagogical_score_mean",
        "silent_error_rate",
        "keep_rate",
        "effective_keep_rate",
        "mean_assigned_weight",
        "rank_accuracy_desc",
        "rank_pedagogy_desc",
        "rank_silent_error_asc",
        "rank_keep_rate_desc",
        "evaluation_mode",
    ]
    for col in keep_cols:
        if col not in latest.columns:
            latest[col] = pd.NA
    return latest[keep_cols].sort_values(["rank_pedagogy_desc", "rank_accuracy_desc"]).reset_index(drop=True)


def _build_decision_reasons(decisions_df: pd.DataFrame) -> pd.DataFrame:
    if decisions_df.empty:
        return pd.DataFrame(columns=["branch", "generation", "decision_reason", "count"])
    out = (
        decisions_df.groupby(["branch", "generation", "decision_reason"], as_index=False)
        .agg(count=("example_id", "count"))
        .sort_values(["branch", "generation", "count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return out


def _build_keep_rejected_quality(decisions_df: pd.DataFrame) -> pd.DataFrame:
    if decisions_df.empty:
        return pd.DataFrame(
            columns=[
                "branch",
                "generation",
                "kept_accuracy_mean",
                "rejected_accuracy_mean",
                "kept_pedagogical_mean",
                "rejected_pedagogical_mean",
                "kept_silent_error_rate",
                "rejected_silent_error_rate",
                "keep_rate",
            ]
        )
    rows: list[dict[str, object]] = []
    for (branch, generation), grp in decisions_df.groupby(["branch", "generation"], as_index=False):
        kept = grp[grp["kept"] == True]  # noqa: E712
        rejected = grp[grp["kept"] == False]  # noqa: E712
        rows.append(
            {
                "branch": str(branch),
                "generation": int(generation),
                "kept_accuracy_mean": float(kept["is_correct"].mean()) if not kept.empty else None,
                "rejected_accuracy_mean": float(rejected["is_correct"].mean()) if not rejected.empty else None,
                "kept_pedagogical_mean": float(kept["overall_pedagogical_score"].mean())
                if not kept.empty
                else None,
                "rejected_pedagogical_mean": float(rejected["overall_pedagogical_score"].mean())
                if not rejected.empty
                else None,
                "kept_silent_error_rate": float(kept["is_silent_error"].mean()) if not kept.empty else None,
                "rejected_silent_error_rate": float(rejected["is_silent_error"].mean())
                if not rejected.empty
                else None,
                "keep_rate": float(grp["kept"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_weight_distribution(decisions_df: pd.DataFrame) -> pd.DataFrame:
    if decisions_df.empty:
        return pd.DataFrame(columns=["branch", "generation", "assigned_weight", "count"])
    out = (
        decisions_df.groupby(["branch", "generation", "assigned_weight"], as_index=False)
        .agg(count=("example_id", "count"))
        .sort_values(["branch", "generation", "assigned_weight"])
        .reset_index(drop=True)
    )
    return out


def _plot_metric_by_branch_generation(run_level: pd.DataFrame, *, metric: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for branch, sub in run_level.groupby("branch", as_index=False):
        sub = sub.sort_values("generation")
        plt.plot(sub["generation"], sub[metric], marker="o", label=str(branch))
    plt.xlabel("Generation")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_keep_rate_by_generation(stress_summary: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    valid = stress_summary[stress_summary["keep_rate"].notna()].copy()
    if valid.empty:
        plt.text(0.5, 0.5, "No keep_rate data", ha="center", va="center")
    else:
        for branch, sub in valid.groupby("branch", as_index=False):
            sub = sub.sort_values("generation")
            plt.plot(sub["generation"], sub["keep_rate"], marker="o", label=str(branch))
        plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("keep_rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_soft_pvf_analysis(
    *,
    run_dir: Path,
    out_dir: Path,
    file_prefix: str = "soft_pvf",
) -> SoftPVFAnalysisArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_level = _build_run_level(run_dir)
    generation_deltas = _safe_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)
    decisions = _collect_soft_decisions(run_dir)
    keep_stats = _collect_keep_stats(run_dir)
    stress_summary = _build_stress_summary(run_level, keep_stats)
    policy_summary = _build_policy_summary(stress_summary)
    decision_reasons = _build_decision_reasons(decisions)
    keep_quality = _build_keep_rejected_quality(decisions)
    weight_distribution = _build_weight_distribution(decisions)

    run_level_csv = out_dir / f"{file_prefix}_run_level.csv"
    run_level_parquet = out_dir / f"{file_prefix}_run_level.parquet"
    generation_deltas_csv = out_dir / f"{file_prefix}_generation_deltas.csv"
    generation_deltas_parquet = out_dir / f"{file_prefix}_generation_deltas.parquet"
    branch_deltas_csv = out_dir / f"{file_prefix}_branch_deltas.csv"
    branch_deltas_parquet = out_dir / f"{file_prefix}_branch_deltas.parquet"
    policy_summary_csv = out_dir / f"{file_prefix}_policy_summary.csv"
    policy_summary_parquet = out_dir / f"{file_prefix}_policy_summary.parquet"
    decision_reasons_csv = out_dir / f"{file_prefix}_decision_reasons.csv"
    decision_reasons_parquet = out_dir / f"{file_prefix}_decision_reasons.parquet"
    keep_rejected_quality_csv = out_dir / f"{file_prefix}_keep_vs_rejected_quality.csv"
    keep_rejected_quality_parquet = out_dir / f"{file_prefix}_keep_vs_rejected_quality.parquet"
    weight_distribution_csv = out_dir / f"{file_prefix}_weight_distribution.csv"
    weight_distribution_parquet = out_dir / f"{file_prefix}_weight_distribution.parquet"
    stress_summary_csv = out_dir / f"{file_prefix}_stress_summary.csv"
    stress_summary_parquet = out_dir / f"{file_prefix}_stress_summary.parquet"

    run_level.to_csv(run_level_csv, index=False)
    run_level.to_parquet(run_level_parquet, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    generation_deltas.to_parquet(generation_deltas_parquet, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    branch_deltas.to_parquet(branch_deltas_parquet, index=False)
    policy_summary.to_csv(policy_summary_csv, index=False)
    policy_summary.to_parquet(policy_summary_parquet, index=False)
    decision_reasons.to_csv(decision_reasons_csv, index=False)
    decision_reasons.to_parquet(decision_reasons_parquet, index=False)
    keep_quality.to_csv(keep_rejected_quality_csv, index=False)
    keep_quality.to_parquet(keep_rejected_quality_parquet, index=False)
    weight_distribution.to_csv(weight_distribution_csv, index=False)
    weight_distribution.to_parquet(weight_distribution_parquet, index=False)
    stress_summary.to_csv(stress_summary_csv, index=False)
    stress_summary.to_parquet(stress_summary_parquet, index=False)

    figures_dir = out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot = figures_dir / f"{file_prefix}_accuracy_by_branch_generation.png"
    pedagogical_plot = figures_dir / f"{file_prefix}_pedagogical_by_branch_generation.png"
    silent_error_plot = figures_dir / f"{file_prefix}_silent_error_by_branch_generation.png"
    keep_rate_plot = figures_dir / f"{file_prefix}_keep_rate_by_generation.png"
    _plot_metric_by_branch_generation(run_level, metric="accuracy_mean", out_path=accuracy_plot)
    _plot_metric_by_branch_generation(run_level, metric="pedagogical_score_mean", out_path=pedagogical_plot)
    _plot_metric_by_branch_generation(run_level, metric="silent_error_rate", out_path=silent_error_plot)
    _plot_keep_rate_by_generation(stress_summary, out_path=keep_rate_plot)

    metadata_json = out_dir / f"{file_prefix}_metadata.json"
    metadata = {
        "run_dir": str(run_dir),
        "evaluation_mode": "inference_recycling_only",
        "file_prefix": file_prefix,
        "notes": "Soft PVF analysis over one run; not full retraining.",
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return SoftPVFAnalysisArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        policy_summary_csv=policy_summary_csv,
        policy_summary_parquet=policy_summary_parquet,
        decision_reasons_csv=decision_reasons_csv,
        decision_reasons_parquet=decision_reasons_parquet,
        keep_rejected_quality_csv=keep_rejected_quality_csv,
        keep_rejected_quality_parquet=keep_rejected_quality_parquet,
        weight_distribution_csv=weight_distribution_csv,
        weight_distribution_parquet=weight_distribution_parquet,
        stress_summary_csv=stress_summary_csv,
        stress_summary_parquet=stress_summary_parquet,
        accuracy_plot=accuracy_plot,
        pedagogical_plot=pedagogical_plot,
        silent_error_plot=silent_error_plot,
        keep_rate_plot=keep_rate_plot,
        metadata_json=metadata_json,
    )
