from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.baseline_series import build_branch_deltas, build_generation_deltas
from didactic_collapse.analysis.compare_runs import build_first_experiment_run_metrics


@dataclass(frozen=True)
class CSRAnalysisArtifacts:
    run_level_csv: Path
    pair_summary_csv: Path
    generation_deltas_csv: Path
    branch_deltas_csv: Path
    best_vs_worst_quality_csv: Path
    stress_summary_csv: Path


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


def _collect_pair_summary(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*/*/gen_*/csr_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model_name": payload.get("model_name"),
                "branch": payload.get("branch"),
                "generation": int(payload.get("generation")),
                "seed": int(payload.get("seed")),
                "policy_name": payload.get("policy_name"),
                "total_questions": int(payload.get("total_questions", 0)),
                "total_candidates": int(payload.get("total_candidates", 0)),
                "pair_count": int(payload.get("pair_count", 0)),
                "pair_construction_rate": float(payload.get("pair_construction_rate", 0.0)),
                "no_pair_count": int(payload.get("no_pair_count", 0)),
                "mean_quality_gap": payload.get("mean_quality_gap"),
                "best_mean_score": payload.get("best_mean_score"),
                "worst_mean_score": payload.get("worst_mean_score"),
                "best_accuracy": payload.get("best_accuracy"),
                "worst_accuracy": payload.get("worst_accuracy"),
                "best_silent_rate": payload.get("best_silent_rate"),
                "worst_silent_rate": payload.get("worst_silent_rate"),
                "no_pair_rate_warned": bool(payload.get("no_pair_rate_warned", False)),
                "no_pair_reasons_json": json.dumps(payload.get("no_pair_reasons", {}), ensure_ascii=False),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "generation",
                "seed",
                "policy_name",
                "total_questions",
                "total_candidates",
                "pair_count",
                "pair_construction_rate",
                "no_pair_count",
                "mean_quality_gap",
                "best_mean_score",
                "worst_mean_score",
                "best_accuracy",
                "worst_accuracy",
                "best_silent_rate",
                "worst_silent_rate",
                "no_pair_rate_warned",
                "no_pair_reasons_json",
            ]
        )
    return pd.DataFrame(rows).sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


def _collect_best_vs_worst_quality(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*/*/gen_*/csr_pairs.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        model_name = path.parents[2].name.replace("_", ":")
        branch = path.parents[1].name
        generation = int(path.parents[0].name.split("_", maxsplit=1)[1])
        paired = df[df["pair_status"] == "paired"].copy() if "pair_status" in df.columns else df.copy()
        if paired.empty:
            rows.append(
                {
                    "model_name": model_name,
                    "branch": branch,
                    "generation": generation,
                    "pair_count": 0,
                    "quality_gap_mean": None,
                    "best_correct_rate": None,
                    "worst_correct_rate": None,
                    "best_silent_rate": None,
                    "worst_silent_rate": None,
                }
            )
            continue
        rows.append(
            {
                "model_name": model_name,
                "branch": branch,
                "generation": generation,
                "pair_count": int(len(paired)),
                "quality_gap_mean": float(pd.to_numeric(paired["quality_gap"], errors="coerce").mean()),
                "best_correct_rate": float(paired["best_is_correct"].astype(bool).mean()),
                "worst_correct_rate": float(paired["worst_is_correct"].astype(bool).mean()),
                "best_silent_rate": float(paired["best_is_silent_error"].astype(bool).mean()),
                "worst_silent_rate": float(paired["worst_is_silent_error"].astype(bool).mean()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "generation",
                "pair_count",
                "quality_gap_mean",
                "best_correct_rate",
                "worst_correct_rate",
                "best_silent_rate",
                "worst_silent_rate",
            ]
        )
    return pd.DataFrame(rows).sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


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


def export_csr_analysis(
    *,
    run_dir: Path,
    out_dir: Path | None = None,
    file_prefix: str = "csr",
) -> CSRAnalysisArtifacts:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    target_out = out_dir or (run_dir / "tables")
    target_out.mkdir(parents=True, exist_ok=True)

    run_level = _build_run_level(run_dir)
    generation_deltas = _safe_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)
    pair_summary = _collect_pair_summary(run_dir)
    best_vs_worst = _collect_best_vs_worst_quality(run_dir)

    stress_summary = run_level.merge(
        pair_summary[
            [
                "model_name",
                "branch",
                "generation",
                "pair_count",
                "pair_construction_rate",
                "no_pair_count",
                "mean_quality_gap",
                "best_accuracy",
                "worst_accuracy",
                "best_silent_rate",
                "worst_silent_rate",
            ]
        ],
        on=["model_name", "branch", "generation"],
        how="left",
        validate="one_to_one",
    )

    run_level_csv = target_out / f"{file_prefix}_run_level.csv"
    pair_summary_csv = target_out / f"{file_prefix}_pair_summary.csv"
    generation_deltas_csv = target_out / f"{file_prefix}_generation_deltas.csv"
    branch_deltas_csv = target_out / f"{file_prefix}_branch_deltas.csv"
    best_vs_worst_csv = target_out / f"{file_prefix}_best_vs_worst_quality.csv"
    stress_summary_csv = target_out / f"{file_prefix}_stress_summary.csv"

    run_level.to_csv(run_level_csv, index=False)
    pair_summary.to_csv(pair_summary_csv, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    best_vs_worst.to_csv(best_vs_worst_csv, index=False)
    stress_summary.to_csv(stress_summary_csv, index=False)

    return CSRAnalysisArtifacts(
        run_level_csv=run_level_csv,
        pair_summary_csv=pair_summary_csv,
        generation_deltas_csv=generation_deltas_csv,
        branch_deltas_csv=branch_deltas_csv,
        best_vs_worst_quality_csv=best_vs_worst_csv,
        stress_summary_csv=stress_summary_csv,
    )
