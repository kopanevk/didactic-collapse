from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from didactic_collapse.analysis.baseline_series import (
    build_branch_deltas,
    build_generation_deltas,
    build_seed_level_summary,
    collect_baseline_run_metrics,
)

EVAL_MODE = "inference_recycling_only"


@dataclass(frozen=True)
class SoftPVFConfirmatoryArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    seed_stats_csv: Path
    seed_stats_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    decision_reasons_csv: Path
    decision_reasons_parquet: Path
    keep_reject_quality_csv: Path
    keep_reject_quality_parquet: Path
    policy_summary_csv: Path
    policy_summary_parquet: Path
    stress_summary_csv: Path
    stress_summary_parquet: Path
    accuracy_plot: Path
    pedagogical_plot: Path
    silent_error_plot: Path
    keep_rate_plot: Path
    metadata_json: Path


def _load_run_snapshot(run_dir: Path) -> dict:
    snapshot_path = run_dir / "run_config.snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run snapshot: {snapshot_path}")
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _seed_from_run(run_dir: Path) -> int:
    payload = _load_run_snapshot(run_dir)
    try:
        return int(payload["config"]["project"]["seed"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Cannot extract seed from run snapshot: {run_dir}") from exc


def _bootstrap_ci_mean(
    values: Iterable[float],
    *,
    n_boot: int = 1500,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> tuple[float, float]:
    arr = [float(v) for v in values if pd.notna(v)]
    if not arr:
        return (math.nan, math.nan)
    if len(arr) == 1:
        return (arr[0], arr[0])

    rng = random.Random(rng_seed)
    means: list[float] = []
    n = len(arr)
    for _ in range(max(100, n_boot)):
        sample = [arr[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int((alpha / 2.0) * (len(means) - 1))
    high_idx = int((1.0 - alpha / 2.0) * (len(means) - 1))
    return means[low_idx], means[high_idx]


def _collect_soft_reports(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        seed = _seed_from_run(run_dir)
        for report_path in sorted(run_dir.glob("*/*/gen_*/soft_pvf_report.json")):
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "seed": int(seed),
                    "model_name": str(payload.get("model_name", "")),
                    "branch": str(payload.get("branch", "")),
                    "generation": int(payload.get("generation", 0)),
                    "policy_name": str(payload.get("policy_name", payload.get("branch", "soft_pvf_medium"))),
                    "keep_rate": float(payload.get("keep_rate", 0.0)),
                    "effective_keep_rate": float(payload.get("effective_keep_rate", 0.0)),
                    "mean_assigned_weight": float(payload.get("mean_assigned_weight", 0.0)),
                    "kept_count": int(payload.get("kept_count", 0)),
                    "rejected_count": int(payload.get("rejected_count", 0)),
                    "total_candidates": int(payload.get("total_candidates", 0)),
                    "kept_accuracy_mean": payload.get("kept_accuracy_mean"),
                    "rejected_accuracy_mean": payload.get("rejected_accuracy_mean"),
                    "kept_pedagogical_mean": payload.get("kept_pedagogical_mean"),
                    "rejected_pedagogical_mean": payload.get("rejected_pedagogical_mean"),
                    "kept_silent_error_rate": payload.get("kept_silent_error_rate"),
                    "rejected_silent_error_rate": payload.get("rejected_silent_error_rate"),
                    "decision_reason_counts": payload.get("decision_reason_counts", {}),
                    "report_path": str(report_path),
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
                "policy_name",
                "keep_rate",
                "effective_keep_rate",
                "mean_assigned_weight",
                "kept_count",
                "rejected_count",
                "total_candidates",
                "kept_accuracy_mean",
                "rejected_accuracy_mean",
                "kept_pedagogical_mean",
                "rejected_pedagogical_mean",
                "kept_silent_error_rate",
                "rejected_silent_error_rate",
                "decision_reason_counts",
                "report_path",
                "evaluation_mode",
            ]
        )
    return pd.DataFrame(rows).sort_values(["seed", "model_name", "branch", "generation"]).reset_index(drop=True)


def _build_decision_reasons_table(soft_reports_df: pd.DataFrame) -> pd.DataFrame:
    if soft_reports_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "policy_name",
                "generation",
                "decision_reason",
                "count_total",
                "count_mean",
                "count_std",
                "seed_count",
                "evaluation_mode",
            ]
        )

    exploded_rows: list[dict[str, object]] = []
    for rec in soft_reports_df.to_dict(orient="records"):
        reasons = rec.get("decision_reason_counts", {}) or {}
        if not isinstance(reasons, dict):
            continue
        for reason, count in reasons.items():
            exploded_rows.append(
                {
                    "seed": int(rec["seed"]),
                    "model_name": str(rec["model_name"]),
                    "branch": str(rec["branch"]),
                    "policy_name": str(rec["policy_name"]),
                    "generation": int(rec["generation"]),
                    "decision_reason": str(reason),
                    "count": int(count),
                    "evaluation_mode": EVAL_MODE,
                }
            )
    if not exploded_rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "policy_name",
                "generation",
                "decision_reason",
                "count_total",
                "count_mean",
                "count_std",
                "seed_count",
                "evaluation_mode",
            ]
        )
    exploded = pd.DataFrame(exploded_rows)
    out = (
        exploded.groupby(["model_name", "branch", "policy_name", "generation", "decision_reason"], as_index=False)
        .agg(
            count_total=("count", "sum"),
            count_mean=("count", "mean"),
            count_std=("count", lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["model_name", "branch", "generation", "count_total"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    out["evaluation_mode"] = EVAL_MODE
    return out


def _build_keep_vs_rejected_quality_table(soft_reports_df: pd.DataFrame) -> pd.DataFrame:
    if soft_reports_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "policy_name",
                "generation",
                "seed_count",
                "keep_rate_mean",
                "effective_keep_rate_mean",
                "mean_assigned_weight_mean",
                "kept_accuracy_mean",
                "rejected_accuracy_mean",
                "kept_pedagogical_mean",
                "rejected_pedagogical_mean",
                "kept_silent_error_rate",
                "rejected_silent_error_rate",
                "evaluation_mode",
            ]
        )

    agg = (
        soft_reports_df.groupby(["model_name", "branch", "policy_name", "generation"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            keep_rate_mean=("keep_rate", "mean"),
            effective_keep_rate_mean=("effective_keep_rate", "mean"),
            mean_assigned_weight_mean=("mean_assigned_weight", "mean"),
            kept_accuracy_mean=("kept_accuracy_mean", "mean"),
            rejected_accuracy_mean=("rejected_accuracy_mean", "mean"),
            kept_pedagogical_mean=("kept_pedagogical_mean", "mean"),
            rejected_pedagogical_mean=("rejected_pedagogical_mean", "mean"),
            kept_silent_error_rate=("kept_silent_error_rate", "mean"),
            rejected_silent_error_rate=("rejected_silent_error_rate", "mean"),
        )
        .sort_values(["model_name", "branch", "generation"])
        .reset_index(drop=True)
    )
    agg["evaluation_mode"] = EVAL_MODE
    return agg


def _build_stress_summary(seed_stats_df: pd.DataFrame, soft_reports_df: pd.DataFrame) -> pd.DataFrame:
    summary = seed_stats_df.rename(
        columns={
            "accuracy_mean_mean": "accuracy_mean",
            "accuracy_mean_std": "accuracy_std",
            "accuracy_mean_ci_low": "accuracy_ci_low",
            "accuracy_mean_ci_high": "accuracy_ci_high",
            "pedagogical_score_mean_mean": "pedagogical_score_mean",
            "pedagogical_score_mean_std": "pedagogical_score_std",
            "pedagogical_score_mean_ci_low": "pedagogical_ci_low",
            "pedagogical_score_mean_ci_high": "pedagogical_ci_high",
            "silent_error_rate_mean": "silent_error_rate_mean",
            "silent_error_rate_std": "silent_error_rate_std",
            "silent_error_rate_ci_low": "silent_error_ci_low",
            "silent_error_rate_ci_high": "silent_error_ci_high",
            "parse_failure_pred_rate_mean": "parse_failure_pred_rate_mean",
            "parse_failure_pred_rate_std": "parse_failure_pred_rate_std",
            "parse_failure_pred_rate_ci_low": "parse_failure_pred_rate_ci_low",
            "parse_failure_pred_rate_ci_high": "parse_failure_pred_rate_ci_high",
        }
    ).copy()

    keep_rows: list[dict[str, object]] = []
    if not soft_reports_df.empty:
        grouped = soft_reports_df.groupby(["model_name", "branch", "policy_name", "generation"], as_index=False)
        for _, grp in grouped:
            keep_vals = [float(v) for v in grp["keep_rate"].tolist()]
            eff_vals = [float(v) for v in grp["effective_keep_rate"].tolist()]
            w_vals = [float(v) for v in grp["mean_assigned_weight"].tolist()]
            keep_ci = _bootstrap_ci_mean(keep_vals, rng_seed=101 + int(grp["generation"].iloc[0]))
            keep_rows.append(
                {
                    "model_name": str(grp["model_name"].iloc[0]),
                    "branch": str(grp["branch"].iloc[0]),
                    "policy_name": str(grp["policy_name"].iloc[0]),
                    "generation": int(grp["generation"].iloc[0]),
                    "keep_rate_mean": float(pd.Series(keep_vals).mean()),
                    "keep_rate_std": float(pd.Series(keep_vals).std(ddof=1)) if len(keep_vals) > 1 else 0.0,
                    "keep_rate_ci_low": keep_ci[0],
                    "keep_rate_ci_high": keep_ci[1],
                    "effective_keep_rate_mean": float(pd.Series(eff_vals).mean()),
                    "mean_assigned_weight_mean": float(pd.Series(w_vals).mean()),
                }
            )
    keep_df = pd.DataFrame(keep_rows)

    if keep_df.empty:
        summary["policy_name"] = summary["branch"]
        summary["keep_rate_mean"] = pd.NA
        summary["keep_rate_std"] = pd.NA
        summary["keep_rate_ci_low"] = pd.NA
        summary["keep_rate_ci_high"] = pd.NA
        summary["effective_keep_rate_mean"] = pd.NA
        summary["mean_assigned_weight_mean"] = pd.NA
        return summary.sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)

    merged = summary.merge(
        keep_df,
        on=["model_name", "branch", "generation"],
        how="left",
        validate="one_to_one",
    )
    merged["policy_name"] = merged["policy_name"].where(merged["policy_name"].notna(), merged["branch"])
    return merged.sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


def _build_policy_summary(stress_summary_df: pd.DataFrame) -> pd.DataFrame:
    if stress_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "policy_name",
                "generation",
                "seed_count",
                "accuracy_mean",
                "pedagogical_score_mean",
                "silent_error_rate_mean",
                "keep_rate_mean",
                "effective_keep_rate_mean",
                "mean_assigned_weight_mean",
                "rank_accuracy_desc",
                "rank_pedagogy_desc",
                "rank_silent_error_asc",
                "rank_keep_rate_desc",
                "evaluation_mode",
            ]
        )

    max_gen = int(pd.to_numeric(stress_summary_df["generation"], errors="coerce").max())
    gen_latest = stress_summary_df[stress_summary_df["generation"] == max_gen].copy()
    gen_latest["rank_accuracy_desc"] = pd.to_numeric(
        gen_latest["accuracy_mean"], errors="coerce"
    ).rank(method="min", ascending=False)
    gen_latest["rank_pedagogy_desc"] = pd.to_numeric(
        gen_latest["pedagogical_score_mean"], errors="coerce"
    ).rank(method="min", ascending=False)
    gen_latest["rank_silent_error_asc"] = pd.to_numeric(
        gen_latest["silent_error_rate_mean"], errors="coerce"
    ).rank(method="min", ascending=True)
    gen_latest["rank_keep_rate_desc"] = pd.to_numeric(
        gen_latest["keep_rate_mean"], errors="coerce"
    ).fillna(-1.0).rank(method="min", ascending=False)
    keep_cols = [
        "model_name",
        "branch",
        "policy_name",
        "generation",
        "seed_count",
        "accuracy_mean",
        "pedagogical_score_mean",
        "silent_error_rate_mean",
        "keep_rate_mean",
        "effective_keep_rate_mean",
        "mean_assigned_weight_mean",
        "rank_accuracy_desc",
        "rank_pedagogy_desc",
        "rank_silent_error_asc",
        "rank_keep_rate_desc",
        "evaluation_mode",
    ]
    for col in keep_cols:
        if col not in gen_latest.columns:
            gen_latest[col] = pd.NA
    return gen_latest[keep_cols].sort_values(["rank_pedagogy_desc", "rank_accuracy_desc"]).reset_index(drop=True)


def _plot_seed_metric(seed_stats_df: pd.DataFrame, *, metric: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_ci_low"
    high_col = f"{metric}_ci_high"
    required = {"branch", "generation", mean_col, low_col, high_col}
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


def _plot_keep_rate(stress_summary_df: pd.DataFrame, *, out_path: Path) -> None:
    required = {"branch", "generation", "keep_rate_mean", "keep_rate_ci_low", "keep_rate_ci_high"}
    missing = required.difference(stress_summary_df.columns)
    if missing:
        raise ValueError(f"stress_summary_df missing columns for keep-rate plot: {sorted(missing)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    valid = stress_summary_df[stress_summary_df["keep_rate_mean"].notna()].copy()
    if valid.empty:
        plt.text(0.5, 0.5, "No keep-rate data", ha="center", va="center")
    else:
        for branch, sub in valid.groupby("branch", as_index=False):
            sub = sub.sort_values("generation")
            x = sub["generation"].astype(float).to_numpy()
            y = sub["keep_rate_mean"].astype(float).to_numpy()
            low = sub["keep_rate_ci_low"].astype(float).to_numpy()
            high = sub["keep_rate_ci_high"].astype(float).to_numpy()
            plt.plot(x, y, marker="o", label=str(branch))
            plt.fill_between(x, low, high, alpha=0.15)
        plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("keep_rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_soft_pvf_confirmatory_analysis(
    *,
    run_dirs: Sequence[Path],
    out_dir: Path,
) -> SoftPVFConfirmatoryArtifacts:
    if not run_dirs:
        raise ValueError("run_dirs must not be empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_level = collect_baseline_run_metrics(run_dirs)
    seed_stats = build_seed_level_summary(run_level)
    generation_deltas = build_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)
    soft_reports = _collect_soft_reports(run_dirs)
    decision_reasons = _build_decision_reasons_table(soft_reports)
    keep_reject_quality = _build_keep_vs_rejected_quality_table(soft_reports)
    stress_summary = _build_stress_summary(seed_stats, soft_reports)
    policy_summary = _build_policy_summary(stress_summary)

    run_level_csv = out_dir / "soft_pvf_confirmatory_run_level.csv"
    run_level_parquet = out_dir / "soft_pvf_confirmatory_run_level.parquet"
    seed_stats_csv = out_dir / "soft_pvf_confirmatory_seed_stats.csv"
    seed_stats_parquet = out_dir / "soft_pvf_confirmatory_seed_stats.parquet"
    generation_deltas_csv = out_dir / "soft_pvf_confirmatory_generation_deltas.csv"
    generation_deltas_parquet = out_dir / "soft_pvf_confirmatory_generation_deltas.parquet"
    branch_deltas_csv = out_dir / "soft_pvf_confirmatory_branch_deltas.csv"
    branch_deltas_parquet = out_dir / "soft_pvf_confirmatory_branch_deltas.parquet"
    decision_reasons_csv = out_dir / "soft_pvf_confirmatory_decision_reasons.csv"
    decision_reasons_parquet = out_dir / "soft_pvf_confirmatory_decision_reasons.parquet"
    keep_reject_quality_csv = out_dir / "soft_pvf_confirmatory_keep_vs_rejected_quality.csv"
    keep_reject_quality_parquet = out_dir / "soft_pvf_confirmatory_keep_vs_rejected_quality.parquet"
    policy_summary_csv = out_dir / "soft_pvf_confirmatory_policy_summary.csv"
    policy_summary_parquet = out_dir / "soft_pvf_confirmatory_policy_summary.parquet"
    stress_summary_csv = out_dir / "soft_pvf_confirmatory_stress_summary.csv"
    stress_summary_parquet = out_dir / "soft_pvf_confirmatory_stress_summary.parquet"

    run_level.to_csv(run_level_csv, index=False)
    run_level.to_parquet(run_level_parquet, index=False)
    seed_stats.to_csv(seed_stats_csv, index=False)
    seed_stats.to_parquet(seed_stats_parquet, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    generation_deltas.to_parquet(generation_deltas_parquet, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    branch_deltas.to_parquet(branch_deltas_parquet, index=False)
    decision_reasons.to_csv(decision_reasons_csv, index=False)
    decision_reasons.to_parquet(decision_reasons_parquet, index=False)
    keep_reject_quality.to_csv(keep_reject_quality_csv, index=False)
    keep_reject_quality.to_parquet(keep_reject_quality_parquet, index=False)
    policy_summary.to_csv(policy_summary_csv, index=False)
    policy_summary.to_parquet(policy_summary_parquet, index=False)
    stress_summary.to_csv(stress_summary_csv, index=False)
    stress_summary.to_parquet(stress_summary_parquet, index=False)

    figures_dir = out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot = figures_dir / "soft_pvf_confirmatory_accuracy_by_branch_generation.png"
    pedagogical_plot = figures_dir / "soft_pvf_confirmatory_pedagogical_by_branch_generation.png"
    silent_error_plot = figures_dir / "soft_pvf_confirmatory_silent_error_by_branch_generation.png"
    keep_rate_plot = figures_dir / "soft_pvf_confirmatory_keep_rate_by_generation.png"

    _plot_seed_metric(seed_stats, metric="accuracy_mean", out_path=accuracy_plot)
    _plot_seed_metric(seed_stats, metric="pedagogical_score_mean", out_path=pedagogical_plot)
    _plot_seed_metric(seed_stats, metric="silent_error_rate", out_path=silent_error_plot)
    _plot_keep_rate(stress_summary, out_path=keep_rate_plot)

    metadata_json = out_dir / "soft_pvf_confirmatory_metadata.json"
    metadata = {
        "evaluation_mode": EVAL_MODE,
        "run_count": int(len(run_dirs)),
        "run_dirs": [str(x) for x in run_dirs],
        "notes": "Soft PVF confirmatory multi-seed analysis (inference_recycling_only).",
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return SoftPVFConfirmatoryArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        seed_stats_csv=seed_stats_csv,
        seed_stats_parquet=seed_stats_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        decision_reasons_csv=decision_reasons_csv,
        decision_reasons_parquet=decision_reasons_parquet,
        keep_reject_quality_csv=keep_reject_quality_csv,
        keep_reject_quality_parquet=keep_reject_quality_parquet,
        policy_summary_csv=policy_summary_csv,
        policy_summary_parquet=policy_summary_parquet,
        stress_summary_csv=stress_summary_csv,
        stress_summary_parquet=stress_summary_parquet,
        accuracy_plot=accuracy_plot,
        pedagogical_plot=pedagogical_plot,
        silent_error_plot=silent_error_plot,
        keep_rate_plot=keep_rate_plot,
        metadata_json=metadata_json,
    )


__all__ = [
    "SoftPVFConfirmatoryArtifacts",
    "export_soft_pvf_confirmatory_analysis",
]

