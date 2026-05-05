from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from didactic_collapse.analysis.baseline_series import build_branch_deltas, build_generation_deltas
from didactic_collapse.analysis.compare_runs import build_first_experiment_run_metrics


@dataclass(frozen=True)
class PAIRLiteAnalysisArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    action_reasons_csv: Path
    action_reasons_parquet: Path
    repair_success_summary_csv: Path
    repair_success_summary_parquet: Path
    keep_repair_reject_quality_csv: Path
    keep_repair_reject_quality_parquet: Path
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


def _collect_pair_decisions(run_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(run_dir.glob("*/*/gen_*/pair_lite_decisions.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=[
                "example_id",
                "branch",
                "generation",
                "seed",
                "is_correct",
                "pred_parse_success",
                "overall_pedagogical_score",
                "is_silent_error",
                "action_initial",
                "action_final",
                "decision_reason",
                "repair_attempted",
                "repair_success",
                "original_response_hash",
                "repaired_response_hash",
                "repair_model_name",
                "repair_error",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["branch", "generation", "example_id"]).reset_index(drop=True)


def _collect_pair_reports(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*/*/gen_*/pair_lite_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["report_path"] = str(path)
        rows.append(payload)
    if not rows:
        return pd.DataFrame(
            columns=[
                "branch",
                "generation",
                "target_rows",
                "kept_original_count",
                "repaired_count",
                "rejected_count",
                "repair_attempted_count",
                "repair_success_count",
                "repair_failure_count",
                "keep_original_rate",
                "repair_rate",
                "reject_rate",
                "repair_success_rate",
            ]
        )
    return pd.DataFrame(rows).sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_action_reasons(decisions_df: pd.DataFrame) -> pd.DataFrame:
    if decisions_df.empty:
        return pd.DataFrame(columns=["branch", "generation", "decision_reason", "count"])
    out = (
        decisions_df.groupby(["branch", "generation", "decision_reason"], as_index=False)
        .agg(count=("example_id", "count"))
        .sort_values(["branch", "generation", "count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return out


def _build_repair_success_summary(decisions_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "branch",
        "generation",
        "repair_attempted_count",
        "repair_success_count",
        "repair_failed_count",
        "repair_success_rate",
    ]
    if decisions_df.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, object]] = []
    for (branch, generation), grp in decisions_df.groupby(["branch", "generation"], as_index=False):
        attempted = int(grp["repair_attempted"].astype(bool).sum())
        success = int(grp["repair_success"].astype(bool).sum())
        failed = attempted - success
        rate = (success / attempted) if attempted > 0 else 0.0
        rows.append(
            {
                "branch": str(branch),
                "generation": int(generation),
                "repair_attempted_count": attempted,
                "repair_success_count": success,
                "repair_failed_count": failed,
                "repair_success_rate": float(rate),
            }
        )
    return pd.DataFrame(rows).sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_keep_repair_reject_quality(decisions_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "branch",
        "generation",
        "kept_accuracy_mean",
        "repaired_accuracy_mean",
        "rejected_accuracy_mean",
        "kept_pedagogical_mean",
        "repaired_pedagogical_mean",
        "rejected_pedagogical_mean",
        "kept_silent_error_rate",
        "repaired_silent_error_rate",
        "rejected_silent_error_rate",
        "keep_original_rate",
        "repair_rate",
        "reject_rate",
    ]
    if decisions_df.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, object]] = []
    for (branch, generation), grp in decisions_df.groupby(["branch", "generation"], as_index=False):
        kept = grp[grp["action_final"] == "keep_original"]
        repaired = grp[grp["action_final"] == "repair_pedagogy"]
        rejected = grp[grp["action_final"] == "reject"]
        rows.append(
            {
                "branch": str(branch),
                "generation": int(generation),
                "kept_accuracy_mean": float(kept["is_correct"].mean()) if not kept.empty else None,
                "repaired_accuracy_mean": float(repaired["is_correct"].mean()) if not repaired.empty else None,
                "rejected_accuracy_mean": float(rejected["is_correct"].mean()) if not rejected.empty else None,
                "kept_pedagogical_mean": float(kept["overall_pedagogical_score"].mean())
                if not kept.empty
                else None,
                "repaired_pedagogical_mean": float(repaired["overall_pedagogical_score"].mean())
                if not repaired.empty
                else None,
                "rejected_pedagogical_mean": float(rejected["overall_pedagogical_score"].mean())
                if not rejected.empty
                else None,
                "kept_silent_error_rate": float(kept["is_silent_error"].mean()) if not kept.empty else None,
                "repaired_silent_error_rate": float(repaired["is_silent_error"].mean())
                if not repaired.empty
                else None,
                "rejected_silent_error_rate": float(rejected["is_silent_error"].mean())
                if not rejected.empty
                else None,
                "keep_original_rate": float((grp["action_final"] == "keep_original").mean()),
                "repair_rate": float((grp["action_final"] == "repair_pedagogy").mean()),
                "reject_rate": float((grp["action_final"] == "reject").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_stress_summary(run_level: pd.DataFrame, reports_df: pd.DataFrame) -> pd.DataFrame:
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
    if reports_df.empty:
        for col in [
            "target_rows",
            "kept_original_count",
            "repaired_count",
            "rejected_count",
            "repair_attempted_count",
            "repair_success_count",
            "repair_failure_count",
            "keep_original_rate",
            "repair_rate",
            "reject_rate",
            "repair_success_rate",
        ]:
            base[col] = pd.NA
        return base.sort_values(["branch", "generation"]).reset_index(drop=True)

    merged = base.merge(
        reports_df[
            [
                "branch",
                "generation",
                "target_rows",
                "kept_original_count",
                "repaired_count",
                "rejected_count",
                "repair_attempted_count",
                "repair_success_count",
                "repair_failure_count",
                "keep_original_rate",
                "repair_rate",
                "reject_rate",
                "repair_success_rate",
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
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_keep_rate(stress_summary: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    valid = stress_summary[stress_summary["keep_original_rate"].notna()].copy()
    if valid.empty:
        plt.text(0.5, 0.5, "No keep-rate data", ha="center", va="center")
    else:
        for branch, sub in valid.groupby("branch", as_index=False):
            sub = sub.sort_values("generation")
            plt.plot(sub["generation"], sub["keep_original_rate"], marker="o", label=str(branch))
        plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("keep_original_rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_pair_lite_analysis(*, run_dir: Path, out_dir: Path, file_prefix: str = "pair_lite") -> PAIRLiteAnalysisArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_level = _build_run_level(run_dir)
    generation_deltas = _safe_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)
    decisions = _collect_pair_decisions(run_dir)
    reports = _collect_pair_reports(run_dir)
    action_reasons = _build_action_reasons(decisions)
    repair_success = _build_repair_success_summary(decisions)
    keep_repair_reject = _build_keep_repair_reject_quality(decisions)
    stress_summary = _build_stress_summary(run_level, reports)

    run_level_csv = out_dir / f"{file_prefix}_run_level.csv"
    run_level_parquet = out_dir / f"{file_prefix}_run_level.parquet"
    generation_deltas_csv = out_dir / f"{file_prefix}_generation_deltas.csv"
    generation_deltas_parquet = out_dir / f"{file_prefix}_generation_deltas.parquet"
    branch_deltas_csv = out_dir / f"{file_prefix}_branch_deltas.csv"
    branch_deltas_parquet = out_dir / f"{file_prefix}_branch_deltas.parquet"
    action_reasons_csv = out_dir / f"{file_prefix}_action_reasons.csv"
    action_reasons_parquet = out_dir / f"{file_prefix}_action_reasons.parquet"
    repair_success_summary_csv = out_dir / f"{file_prefix}_repair_success_summary.csv"
    repair_success_summary_parquet = out_dir / f"{file_prefix}_repair_success_summary.parquet"
    keep_repair_reject_quality_csv = out_dir / f"{file_prefix}_keep_repair_reject_quality.csv"
    keep_repair_reject_quality_parquet = out_dir / f"{file_prefix}_keep_repair_reject_quality.parquet"
    stress_summary_csv = out_dir / f"{file_prefix}_stress_summary.csv"
    stress_summary_parquet = out_dir / f"{file_prefix}_stress_summary.parquet"

    run_level.to_csv(run_level_csv, index=False)
    run_level.to_parquet(run_level_parquet, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    generation_deltas.to_parquet(generation_deltas_parquet, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    branch_deltas.to_parquet(branch_deltas_parquet, index=False)
    action_reasons.to_csv(action_reasons_csv, index=False)
    action_reasons.to_parquet(action_reasons_parquet, index=False)
    repair_success.to_csv(repair_success_summary_csv, index=False)
    repair_success.to_parquet(repair_success_summary_parquet, index=False)
    keep_repair_reject.to_csv(keep_repair_reject_quality_csv, index=False)
    keep_repair_reject.to_parquet(keep_repair_reject_quality_parquet, index=False)
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
    _plot_keep_rate(stress_summary, out_path=keep_rate_plot)

    metadata_json = out_dir / f"{file_prefix}_metadata.json"
    metadata = {
        "run_dir": str(run_dir),
        "evaluation_mode": "inference_recycling_only",
        "file_prefix": file_prefix,
        "notes": "PAIR-lite analysis over one run; not full retraining.",
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return PAIRLiteAnalysisArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        action_reasons_csv=action_reasons_csv,
        action_reasons_parquet=action_reasons_parquet,
        repair_success_summary_csv=repair_success_summary_csv,
        repair_success_summary_parquet=repair_success_summary_parquet,
        keep_repair_reject_quality_csv=keep_repair_reject_quality_csv,
        keep_repair_reject_quality_parquet=keep_repair_reject_quality_parquet,
        stress_summary_csv=stress_summary_csv,
        stress_summary_parquet=stress_summary_parquet,
        accuracy_plot=accuracy_plot,
        pedagogical_plot=pedagogical_plot,
        silent_error_plot=silent_error_plot,
        keep_rate_plot=keep_rate_plot,
        metadata_json=metadata_json,
    )
