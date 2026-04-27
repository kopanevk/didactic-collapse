from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PVFStressArtifacts:
    stress_summary_csv: Path
    stress_summary_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    reject_reasons_csv: Path
    reject_reasons_parquet: Path
    keep_reject_quality_csv: Path
    keep_reject_quality_parquet: Path
    metadata_json: Path


def _collect_eval_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "all_eval_merged.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing all_eval_merged artifact: {path}")
    df = pd.read_parquet(path)
    required = {"branch", "generation", "example_id", "is_correct", "overall_pedagogical_score", "is_silent_error"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"all_eval_merged missing columns: {sorted(missing)}")

    grouped = (
        df.groupby(["branch", "generation"], as_index=False)
        .agg(
            sample_count=("example_id", "count"),
            accuracy_mean=("is_correct", "mean"),
            pedagogical_score_mean=("overall_pedagogical_score", "mean"),
            silent_error_rate=("is_silent_error", "mean"),
        )
        .sort_values(["branch", "generation"])
    )
    return grouped.reset_index(drop=True)


def _load_pvf_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_pvf_reports(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(run_dir.glob("*/*/gen_*/pvf_filter_report.json")):
        payload = _load_pvf_report(report_path)
        payload["report_path"] = str(report_path)
        rows.append(payload)
    if not rows:
        return pd.DataFrame(
            columns=[
                "branch",
                "generation",
                "keep_rate",
                "kept_count",
                "rejected_count",
                "total_candidates",
                "threshold_score",
                "min_keep_ratio",
                "rejection_reason_counts",
            ]
        )
    out = pd.DataFrame(rows)
    return out


def _build_reject_reason_table(reports_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if reports_df.empty:
        return pd.DataFrame(
            columns=["branch", "generation", "reason", "count", "report_path"]
        )
    for rec in reports_df.to_dict(orient="records"):
        reasons = rec.get("rejection_reason_counts", {}) or {}
        if not isinstance(reasons, dict):
            continue
        for reason, count in reasons.items():
            rows.append(
                {
                    "branch": rec.get("branch"),
                    "generation": rec.get("generation"),
                    "reason": reason,
                    "count": int(count),
                    "report_path": rec.get("report_path"),
                }
            )
    return pd.DataFrame(rows)


def _build_keep_reject_quality_table(reports_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "branch",
        "generation",
        "kept_accuracy_mean",
        "rejected_accuracy_mean",
        "kept_pedagogical_mean",
        "rejected_pedagogical_mean",
        "kept_silent_error_rate",
        "rejected_silent_error_rate",
        "keep_rate",
        "kept_count",
        "rejected_count",
        "total_candidates",
    ]
    if reports_df.empty:
        return pd.DataFrame(columns=cols)
    out = reports_df[cols].copy()
    return out.sort_values(["branch", "generation"]).reset_index(drop=True)


def _build_generation_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "branch",
                "generation_start",
                "generation_end",
                "delta_generation",
                "delta_accuracy_mean",
                "delta_pedagogical_score_mean",
                "delta_silent_error_rate",
            ]
        )
    rows: list[dict[str, Any]] = []
    for branch, grp in summary_df.groupby("branch", as_index=False):
        gens = sorted(int(x) for x in grp["generation"].tolist())
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                g0, g1 = gens[i], gens[j]
                start = grp[grp["generation"] == g0].iloc[0]
                end = grp[grp["generation"] == g1].iloc[0]
                rows.append(
                    {
                        "branch": str(branch),
                        "generation_start": int(g0),
                        "generation_end": int(g1),
                        "delta_generation": f"gen{g1}_minus_gen{g0}",
                        "delta_accuracy_mean": float(end["accuracy_mean"]) - float(start["accuracy_mean"]),
                        "delta_pedagogical_score_mean": float(end["pedagogical_score_mean"])
                        - float(start["pedagogical_score_mean"]),
                        "delta_silent_error_rate": float(end["silent_error_rate"])
                        - float(start["silent_error_rate"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["branch", "generation_start", "generation_end"]).reset_index(drop=True)


def export_pvf_stress_analysis(*, run_dir: Path, out_dir: Path) -> PVFStressArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics = _collect_eval_metrics(run_dir)
    pvf_reports = _collect_pvf_reports(run_dir)

    if not pvf_reports.empty:
        merge_cols = [
            "branch",
            "generation",
            "keep_rate",
            "kept_count",
            "rejected_count",
            "total_candidates",
            "threshold_score",
            "min_keep_ratio",
        ]
        summary = eval_metrics.merge(
            pvf_reports[merge_cols],
            on=["branch", "generation"],
            how="left",
            validate="one_to_one",
        )
    else:
        summary = eval_metrics.copy()
        summary["keep_rate"] = pd.NA
        summary["kept_count"] = pd.NA
        summary["rejected_count"] = pd.NA
        summary["total_candidates"] = pd.NA
        summary["threshold_score"] = pd.NA
        summary["min_keep_ratio"] = pd.NA

    generation_deltas = _build_generation_deltas(summary)
    reject_reasons = _build_reject_reason_table(pvf_reports)
    keep_reject_quality = _build_keep_reject_quality_table(pvf_reports)

    stress_summary_csv = out_dir / "didactic_collapse_stress_summary.csv"
    stress_summary_parquet = out_dir / "didactic_collapse_stress_summary.parquet"
    generation_deltas_csv = out_dir / "didactic_collapse_generation_deltas.csv"
    generation_deltas_parquet = out_dir / "didactic_collapse_generation_deltas.parquet"
    reject_reasons_csv = out_dir / "pvf_reject_reasons.csv"
    reject_reasons_parquet = out_dir / "pvf_reject_reasons.parquet"
    keep_reject_quality_csv = out_dir / "pvf_keep_vs_rejected_quality.csv"
    keep_reject_quality_parquet = out_dir / "pvf_keep_vs_rejected_quality.parquet"
    metadata_json = out_dir / "pvf_stress_metadata.json"

    summary.to_csv(stress_summary_csv, index=False)
    summary.to_parquet(stress_summary_parquet, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    generation_deltas.to_parquet(generation_deltas_parquet, index=False)
    reject_reasons.to_csv(reject_reasons_csv, index=False)
    reject_reasons.to_parquet(reject_reasons_parquet, index=False)
    keep_reject_quality.to_csv(keep_reject_quality_csv, index=False)
    keep_reject_quality.to_parquet(keep_reject_quality_parquet, index=False)

    metadata = {
        "run_dir": str(run_dir),
        "evaluation_mode": "inference_recycling_only",
        "note": "PVF stress analysis over existing run artifacts; no generation/training rerun.",
        "rows_in_summary": int(len(summary)),
        "rows_in_pvf_reports": int(len(pvf_reports)),
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return PVFStressArtifacts(
        stress_summary_csv=stress_summary_csv,
        stress_summary_parquet=stress_summary_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        reject_reasons_csv=reject_reasons_csv,
        reject_reasons_parquet=reject_reasons_parquet,
        keep_reject_quality_csv=keep_reject_quality_csv,
        keep_reject_quality_parquet=keep_reject_quality_parquet,
        metadata_json=metadata_json,
    )

