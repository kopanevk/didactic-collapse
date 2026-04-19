from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RunComparisonArtifacts:
    csv_path: Path
    parquet_path: Path
    table: pd.DataFrame


def _load_summary_with_corrected_accuracy(run_dir: Path) -> pd.DataFrame:
    tables_dir = run_dir / "tables"
    corrected_csv = tables_dir / "first_experiment_summary_corrected_accuracy.csv"
    summary_csv = tables_dir / "first_experiment_summary.csv"
    summary_parquet = tables_dir / "first_experiment_summary.parquet"

    if corrected_csv.exists():
        summary = pd.read_csv(corrected_csv)
        if "accuracy_mean_corrected" not in summary.columns:
            raise RuntimeError(
                "Corrected summary exists but missing accuracy_mean_corrected column: "
                f"{corrected_csv}"
            )
        summary["accuracy_mean_effective"] = summary["accuracy_mean_corrected"]
        summary["accuracy_source"] = "corrected"
    elif summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        if "accuracy_mean" not in summary.columns:
            raise RuntimeError(f"Summary missing accuracy_mean: {summary_csv}")
        summary["accuracy_mean_effective"] = summary["accuracy_mean"]
        summary["accuracy_source"] = "raw_summary"
    elif summary_parquet.exists():
        summary = pd.read_parquet(summary_parquet)
        if "accuracy_mean" not in summary.columns:
            raise RuntimeError(f"Summary missing accuracy_mean: {summary_parquet}")
        summary["accuracy_mean_effective"] = summary["accuracy_mean"]
        summary["accuracy_source"] = "raw_summary"
    else:
        raise FileNotFoundError(
            "Missing first experiment summary table in run. Expected one of: "
            f"{corrected_csv}, {summary_csv}, {summary_parquet}"
        )

    required = {
        "model_name",
        "branch",
        "generation",
        "sample_count",
        "pedagogical_score_mean",
        "silent_error_rate",
        "accuracy_mean_effective",
        "accuracy_source",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise RuntimeError(
            f"Summary table missing required columns for comparison: {sorted(missing)}"
        )

    return summary[
        [
            "model_name",
            "branch",
            "generation",
            "sample_count",
            "accuracy_mean_effective",
            "pedagogical_score_mean",
            "silent_error_rate",
            "accuracy_source",
        ]
    ].copy()


def _compute_parse_failure_pred_by_context(run_dir: Path) -> pd.DataFrame:
    tables: list[dict[str, object]] = []
    accuracy_paths = sorted(run_dir.glob("*/*/gen_*/accuracy_table.parquet"))
    if not accuracy_paths:
        raise FileNotFoundError(f"No accuracy tables found under run dir: {run_dir}")

    for path in accuracy_paths:
        df = pd.read_parquet(path)
        required = {"model_name", "branch", "generation", "pred_parse_success"}
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(
                "Accuracy table missing columns required for parse-failure diagnostics: "
                f"{sorted(missing)} ({path})"
            )
        if df.empty:
            raise RuntimeError(f"Accuracy table is empty: {path}")

        parse_failure_count = int((~df["pred_parse_success"].astype(bool)).sum())
        sample_count = int(len(df))
        tables.append(
            {
                "model_name": str(df["model_name"].iloc[0]),
                "branch": str(df["branch"].iloc[0]),
                "generation": int(df["generation"].iloc[0]),
                "parse_failure_pred_count": parse_failure_count,
                "sample_count_from_accuracy": sample_count,
                "parse_failure_pred_rate": parse_failure_count / sample_count,
            }
        )

    out = pd.DataFrame(tables).sort_values(["model_name", "branch", "generation"])
    if out.duplicated(subset=["model_name", "branch", "generation"]).any():
        dup = int(out.duplicated(subset=["model_name", "branch", "generation"]).sum())
        raise RuntimeError(
            f"Duplicate accuracy contexts found while building parse-failure metrics: {dup}"
        )
    return out


def build_first_experiment_run_metrics(run_dir: Path) -> pd.DataFrame:
    summary = _load_summary_with_corrected_accuracy(run_dir)
    parse_diag = _compute_parse_failure_pred_by_context(run_dir)

    merged = summary.merge(
        parse_diag,
        on=["model_name", "branch", "generation"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(summary):
        missing = len(summary) - len(merged)
        raise RuntimeError(
            f"Failed to align summary with accuracy diagnostics. Missing contexts: {missing}"
        )

    merged = merged.rename(columns={"accuracy_mean_effective": "accuracy_mean"})
    return merged.sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


def compare_first_experiment_runs(
    *,
    old_run_dir: Path,
    new_run_dir: Path,
    out_dir: Path | None = None,
) -> RunComparisonArtifacts:
    old_metrics = build_first_experiment_run_metrics(old_run_dir)
    new_metrics = build_first_experiment_run_metrics(new_run_dir)

    keys = ["model_name", "branch", "generation"]
    merged = old_metrics.merge(
        new_metrics,
        on=keys,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    missing_contexts = merged[merged["_merge"] != "both"]
    if not missing_contexts.empty:
        contexts = missing_contexts[keys + ["_merge"]].to_dict(orient="records")
        raise RuntimeError(
            "Cannot compare runs with different context coverage. "
            f"Mismatched contexts: {contexts}"
        )

    merged = merged.drop(columns=["_merge"])
    merged["delta_accuracy_mean"] = merged["accuracy_mean_new"] - merged["accuracy_mean_old"]
    merged["delta_parse_failure_pred_rate"] = (
        merged["parse_failure_pred_rate_new"] - merged["parse_failure_pred_rate_old"]
    )
    merged["delta_pedagogical_score_mean"] = (
        merged["pedagogical_score_mean_new"] - merged["pedagogical_score_mean_old"]
    )
    merged["delta_silent_error_rate"] = (
        merged["silent_error_rate_new"] - merged["silent_error_rate_old"]
    )
    merged["old_run_dir"] = str(old_run_dir)
    merged["new_run_dir"] = str(new_run_dir)
    merged = merged.sort_values(keys).reset_index(drop=True)

    target_dir = out_dir or (new_run_dir / "tables")
    target_dir.mkdir(parents=True, exist_ok=True)
    old_tag = old_run_dir.name
    new_tag = new_run_dir.name
    stem = f"first_experiment_comparison_{old_tag}_vs_{new_tag}"

    csv_path = target_dir / f"{stem}.csv"
    parquet_path = target_dir / f"{stem}.parquet"
    merged.to_csv(csv_path, index=False)
    merged.to_parquet(parquet_path, index=False)

    return RunComparisonArtifacts(csv_path=csv_path, parquet_path=parquet_path, table=merged)
