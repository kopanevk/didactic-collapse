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
class PVFConfirmatoryArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    seed_stats_csv: Path
    seed_stats_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    reject_reasons_csv: Path
    reject_reasons_parquet: Path
    keep_reject_quality_csv: Path
    keep_reject_quality_parquet: Path
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


def _collect_pvf_reports(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        seed = _seed_from_run(run_dir)
        for report_path in sorted(run_dir.glob("*/*/gen_*/pvf_filter_report.json")):
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["run_id"] = run_id
            payload["seed"] = seed
            payload["run_dir"] = str(run_dir)
            payload["report_path"] = str(report_path)
            payload["evaluation_mode"] = EVAL_MODE
            rows.append(payload)

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "seed",
                "run_dir",
                "model_name",
                "branch",
                "generation",
                "keep_rate",
                "kept_count",
                "rejected_count",
                "total_candidates",
                "threshold_score",
                "min_keep_ratio",
                "rejection_reason_counts",
                "kept_accuracy_mean",
                "rejected_accuracy_mean",
                "kept_pedagogical_mean",
                "rejected_pedagogical_mean",
                "kept_silent_error_rate",
                "rejected_silent_error_rate",
                "report_path",
                "evaluation_mode",
            ]
        )
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["seed", "run_id", "model_name", "branch", "generation"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _build_reject_reasons_table(reports_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if reports_df.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "seed",
                "model_name",
                "branch",
                "generation",
                "reason",
                "count",
                "evaluation_mode",
                "report_path",
            ]
        )

    for rec in reports_df.to_dict(orient="records"):
        reasons = rec.get("rejection_reason_counts", {}) or {}
        if not isinstance(reasons, dict):
            continue
        for reason, count in reasons.items():
            rows.append(
                {
                    "run_id": rec.get("run_id"),
                    "seed": rec.get("seed"),
                    "model_name": rec.get("model_name"),
                    "branch": rec.get("branch"),
                    "generation": rec.get("generation"),
                    "reason": reason,
                    "count": int(count),
                    "evaluation_mode": EVAL_MODE,
                    "report_path": rec.get("report_path"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "seed",
                "model_name",
                "branch",
                "generation",
                "reason",
                "count",
                "evaluation_mode",
                "report_path",
            ]
        )
    return pd.DataFrame(rows).sort_values(["seed", "branch", "generation", "reason"]).reset_index(drop=True)


def _build_keep_vs_rejected_quality_table(reports_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run_id",
        "seed",
        "model_name",
        "branch",
        "generation",
        "keep_rate",
        "kept_count",
        "rejected_count",
        "total_candidates",
        "threshold_score",
        "min_keep_ratio",
        "kept_accuracy_mean",
        "rejected_accuracy_mean",
        "kept_pedagogical_mean",
        "rejected_pedagogical_mean",
        "kept_silent_error_rate",
        "rejected_silent_error_rate",
        "evaluation_mode",
    ]
    if reports_df.empty:
        return pd.DataFrame(columns=cols)
    out = reports_df[cols].copy()
    return out.sort_values(["seed", "branch", "generation"]).reset_index(drop=True)


def _build_stress_summary(
    *,
    run_level_df: pd.DataFrame,
    pvf_reports_df: pd.DataFrame,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = run_level_df.groupby(["model_name", "branch", "generation"], as_index=False)
    for _, grp in grouped:
        model_name = str(grp["model_name"].iloc[0])
        branch = str(grp["branch"].iloc[0])
        generation = int(grp["generation"].iloc[0])
        acc_vals = [float(x) for x in grp["accuracy_mean"].tolist()]
        ped_vals = [float(x) for x in grp["pedagogical_score_mean"].tolist()]
        sil_vals = [float(x) for x in grp["silent_error_rate"].tolist()]
        sample_vals = [float(x) for x in grp["sample_count"].tolist()]
        acc_ci = _bootstrap_ci_mean(acc_vals, rng_seed=bootstrap_seed + 11 + generation)
        ped_ci = _bootstrap_ci_mean(ped_vals, rng_seed=bootstrap_seed + 23 + generation)
        sil_ci = _bootstrap_ci_mean(sil_vals, rng_seed=bootstrap_seed + 37 + generation)

        row: dict[str, object] = {
            "model_name": model_name,
            "branch": branch,
            "generation": generation,
            "seed_count": int(grp["seed"].nunique()),
            "run_count": int(len(grp)),
            "sample_count_mean": float(pd.Series(sample_vals).mean()),
            "sample_count_std": float(pd.Series(sample_vals).std(ddof=1)) if len(sample_vals) > 1 else 0.0,
            "accuracy_mean": float(pd.Series(acc_vals).mean()),
            "accuracy_std": float(pd.Series(acc_vals).std(ddof=1)) if len(acc_vals) > 1 else 0.0,
            "accuracy_ci_low": acc_ci[0],
            "accuracy_ci_high": acc_ci[1],
            "pedagogical_score_mean": float(pd.Series(ped_vals).mean()),
            "pedagogical_score_std": float(pd.Series(ped_vals).std(ddof=1)) if len(ped_vals) > 1 else 0.0,
            "pedagogical_ci_low": ped_ci[0],
            "pedagogical_ci_high": ped_ci[1],
            "silent_error_rate_mean": float(pd.Series(sil_vals).mean()),
            "silent_error_rate_std": float(pd.Series(sil_vals).std(ddof=1)) if len(sil_vals) > 1 else 0.0,
            "silent_error_ci_low": sil_ci[0],
            "silent_error_ci_high": sil_ci[1],
            "evaluation_mode": EVAL_MODE,
        }

        pvf_slice = pvf_reports_df[
            (pvf_reports_df["model_name"] == model_name)
            & (pvf_reports_df["branch"] == branch)
            & (pvf_reports_df["generation"].astype(int) == generation)
        ]
        if pvf_slice.empty:
            row["keep_rate_mean"] = math.nan
            row["keep_rate_std"] = math.nan
            row["keep_rate_ci_low"] = math.nan
            row["keep_rate_ci_high"] = math.nan
            row["kept_count_mean"] = math.nan
            row["rejected_count_mean"] = math.nan
            row["total_candidates_mean"] = math.nan
        else:
            keep_vals = [float(x) for x in pvf_slice["keep_rate"].tolist()]
            keep_ci = _bootstrap_ci_mean(keep_vals, rng_seed=bootstrap_seed + 47 + generation)
            row["keep_rate_mean"] = float(pd.Series(keep_vals).mean())
            row["keep_rate_std"] = float(pd.Series(keep_vals).std(ddof=1)) if len(keep_vals) > 1 else 0.0
            row["keep_rate_ci_low"] = keep_ci[0]
            row["keep_rate_ci_high"] = keep_ci[1]
            row["kept_count_mean"] = float(pd.Series(pvf_slice["kept_count"]).mean())
            row["rejected_count_mean"] = float(pd.Series(pvf_slice["rejected_count"]).mean())
            row["total_candidates_mean"] = float(pd.Series(pvf_slice["total_candidates"]).mean())
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


def _plot_seed_metric(seed_stats_df: pd.DataFrame, *, metric: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_ci_low"
    high_col = f"{metric}_ci_high"
    required = {"generation", "branch", mean_col, low_col, high_col}
    missing = required.difference(seed_stats_df.columns)
    if missing:
        raise ValueError(f"seed_stats_df missing plot columns: {sorted(missing)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for branch, sub in seed_stats_df.groupby("branch"):
        sub = sub.sort_values("generation")
        x = sub["generation"].to_numpy()
        y = sub[mean_col].to_numpy()
        y_low = sub[low_col].to_numpy()
        y_high = sub[high_col].to_numpy()
        plt.plot(x, y, marker="o", label=str(branch))
        plt.fill_between(x, y_low, y_high, alpha=0.2)
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
        raise ValueError(f"stress_summary_df missing keep-rate plot columns: {sorted(missing)}")

    pvf = stress_summary_df[stress_summary_df["keep_rate_mean"].notna()].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if pvf.empty:
        plt.text(0.5, 0.5, "No PVF keep-rate rows", ha="center", va="center")
    else:
        for branch, sub in pvf.groupby("branch"):
            sub = sub.sort_values("generation")
            x = sub["generation"].to_numpy()
            y = sub["keep_rate_mean"].to_numpy()
            y_low = sub["keep_rate_ci_low"].to_numpy()
            y_high = sub["keep_rate_ci_high"].to_numpy()
            plt.plot(x, y, marker="o", label=str(branch))
            plt.fill_between(x, y_low, y_high, alpha=0.2)
        plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("keep_rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_pvf_confirmatory_analysis(
    *,
    run_dirs: Sequence[Path],
    out_dir: Path,
    bootstrap_seed: int = 42,
) -> PVFConfirmatoryArtifacts:
    if not run_dirs:
        raise ValueError("run_dirs cannot be empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_level = collect_baseline_run_metrics(run_dirs)
    seed_stats = build_seed_level_summary(run_level, bootstrap_seed=bootstrap_seed)
    generation_deltas = build_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)

    pvf_reports = _collect_pvf_reports(run_dirs)
    reject_reasons = _build_reject_reasons_table(pvf_reports)
    keep_reject_quality = _build_keep_vs_rejected_quality_table(pvf_reports)
    stress_summary = _build_stress_summary(
        run_level_df=run_level,
        pvf_reports_df=pvf_reports,
        bootstrap_seed=bootstrap_seed,
    )

    run_level_csv = out_dir / "pvf_confirmatory_run_level.csv"
    run_level_parquet = out_dir / "pvf_confirmatory_run_level.parquet"
    seed_stats_csv = out_dir / "pvf_confirmatory_seed_stats.csv"
    seed_stats_parquet = out_dir / "pvf_confirmatory_seed_stats.parquet"
    generation_deltas_csv = out_dir / "pvf_confirmatory_generation_deltas.csv"
    generation_deltas_parquet = out_dir / "pvf_confirmatory_generation_deltas.parquet"
    branch_deltas_csv = out_dir / "pvf_confirmatory_branch_deltas.csv"
    branch_deltas_parquet = out_dir / "pvf_confirmatory_branch_deltas.parquet"
    reject_reasons_csv = out_dir / "pvf_confirmatory_reject_reasons.csv"
    reject_reasons_parquet = out_dir / "pvf_confirmatory_reject_reasons.parquet"
    keep_reject_quality_csv = out_dir / "pvf_confirmatory_keep_vs_rejected_quality.csv"
    keep_reject_quality_parquet = out_dir / "pvf_confirmatory_keep_vs_rejected_quality.parquet"
    stress_summary_csv = out_dir / "pvf_confirmatory_stress_summary.csv"
    stress_summary_parquet = out_dir / "pvf_confirmatory_stress_summary.parquet"

    run_level.to_csv(run_level_csv, index=False)
    run_level.to_parquet(run_level_parquet, index=False)
    seed_stats.to_csv(seed_stats_csv, index=False)
    seed_stats.to_parquet(seed_stats_parquet, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    generation_deltas.to_parquet(generation_deltas_parquet, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    branch_deltas.to_parquet(branch_deltas_parquet, index=False)
    reject_reasons.to_csv(reject_reasons_csv, index=False)
    reject_reasons.to_parquet(reject_reasons_parquet, index=False)
    keep_reject_quality.to_csv(keep_reject_quality_csv, index=False)
    keep_reject_quality.to_parquet(keep_reject_quality_parquet, index=False)
    stress_summary.to_csv(stress_summary_csv, index=False)
    stress_summary.to_parquet(stress_summary_parquet, index=False)

    figures_dir = out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot = figures_dir / "pvf_confirmatory_accuracy_by_branch_generation.png"
    pedagogical_plot = figures_dir / "pvf_confirmatory_pedagogical_by_branch_generation.png"
    silent_error_plot = figures_dir / "pvf_confirmatory_silent_error_by_branch_generation.png"
    keep_rate_plot = figures_dir / "pvf_confirmatory_keep_rate_by_generation.png"
    _plot_seed_metric(seed_stats, metric="accuracy_mean", out_path=accuracy_plot)
    _plot_seed_metric(seed_stats, metric="pedagogical_score_mean", out_path=pedagogical_plot)
    _plot_seed_metric(seed_stats, metric="silent_error_rate", out_path=silent_error_plot)
    _plot_keep_rate(stress_summary, out_path=keep_rate_plot)

    metadata_json = out_dir / "pvf_confirmatory_metadata.json"
    metadata = {
        "evaluation_mode": EVAL_MODE,
        "run_count": int(len(run_dirs)),
        "run_dirs": [str(x) for x in run_dirs],
        "notes": "Confirmatory PVF stress analysis over completed runs; not full retraining.",
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return PVFConfirmatoryArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        seed_stats_csv=seed_stats_csv,
        seed_stats_parquet=seed_stats_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        reject_reasons_csv=reject_reasons_csv,
        reject_reasons_parquet=reject_reasons_parquet,
        keep_reject_quality_csv=keep_reject_quality_csv,
        keep_reject_quality_parquet=keep_reject_quality_parquet,
        stress_summary_csv=stress_summary_csv,
        stress_summary_parquet=stress_summary_parquet,
        accuracy_plot=accuracy_plot,
        pedagogical_plot=pedagogical_plot,
        silent_error_plot=silent_error_plot,
        keep_rate_plot=keep_rate_plot,
        metadata_json=metadata_json,
    )

