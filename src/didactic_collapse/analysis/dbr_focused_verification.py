from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from didactic_collapse.analysis.pairwise_judge_sensitivity import run_pairwise_judge_for_model
from didactic_collapse.config.settings import AppConfig

_GEN2 = 2
_PURE_BRANCH = "pure_recycling"
_DBR_BRANCH = "dbr_medium"
_TARGET_BRANCHES = (_PURE_BRANCH, _DBR_BRANCH)
_DEFECT_FIELDS = {
    "parse_failure": "defect_parse_failure",
    "incorrect_answer": "defect_incorrect",
    "silent_error": "defect_silent",
    "low_reasoning": "defect_low_reasoning",
    "low_structure": "defect_low_structure",
}


@dataclass(frozen=True)
class DBRRecomputeAuditArtifacts:
    out_dir: Path
    metrics_csv: Path
    deltas_csv: Path
    table_comparison_csv: Path
    budget_check_csv: Path
    findings_json: Path
    report_md: Path
    has_blocking_findings: bool


@dataclass(frozen=True)
class QwenDBRPairwiseArtifacts:
    out_dir: Path
    selected_pairs_csv: Path
    hidden_key_csv: Path
    pairwise_results_csv: Path
    pairwise_comparison_csv: Path
    pairwise_summary_csv: Path
    seed_summary_csv: Path
    metadata_json: Path


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_seed(run_dir: Path) -> int:
    payload = _read_json(run_dir / "run_config.snapshot.json")
    return int(payload["config"]["project"]["seed"])


def _load_data_root(run_dir: Path) -> Path:
    payload = _read_json(run_dir / "run_config.snapshot.json")
    return Path(payload["config"]["paths"]["data_root"])


def _discover_model_dir(run_dir: Path) -> Path:
    model_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name not in {"tables", "figures"}]
    if len(model_dirs) != 1:
        raise RuntimeError(f"Expected exactly 1 model dir in {run_dir}, got {len(model_dirs)}")
    return model_dirs[0]


def _safe_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _coerce_bool_mean(series: pd.Series) -> float:
    return float(series.astype(bool).mean()) if len(series) else float("nan")


def _ci95(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    s = pd.Series(vals, dtype="float64")
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    delta = 1.96 * std / max(1.0, len(vals) ** 0.5)
    return mean - delta, mean + delta


def recompute_gen2_metrics(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        seed = _load_seed(run_dir)
        run_id = run_dir.name
        model_dir = _discover_model_dir(run_dir)
        for branch in _TARGET_BRANCHES:
            eval_path = model_dir / branch / f"gen_{_GEN2}" / "eval_merged.parquet"
            if not eval_path.exists():
                raise FileNotFoundError(f"Missing eval artifact: {eval_path}")
            df = pd.read_parquet(eval_path)
            required = {"example_id", "is_correct", "overall_pedagogical_score", "is_silent_error"}
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(f"Missing columns in {eval_path}: {sorted(missing)}")
            rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "seed": int(seed),
                    "model_name": str(df["model_name"].iloc[0]) if "model_name" in df.columns and len(df) else "",
                    "branch": branch,
                    "generation": _GEN2,
                    "sample_count": int(len(df)),
                    "accuracy_mean": _coerce_bool_mean(df["is_correct"]),
                    "pedagogical_score_mean": float(pd.to_numeric(df["overall_pedagogical_score"], errors="coerce").mean()),
                    "silent_error_rate": _coerce_bool_mean(df["is_silent_error"]),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["seed", "branch"]).reset_index(drop=True)


def recompute_gen2_deltas_by_seed(metrics_df: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "branch", "sample_count", "accuracy_mean", "pedagogical_score_mean", "silent_error_rate"}
    missing = required.difference(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df missing columns: {sorted(missing)}")
    pure = metrics_df[metrics_df["branch"] == _PURE_BRANCH].copy()
    dbr = metrics_df[metrics_df["branch"] == _DBR_BRANCH].copy()
    merged = pure.merge(
        dbr,
        on=["seed"],
        how="inner",
        suffixes=("_pure", "_dbr"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No matched pure/dbr seeds in metrics_df")
    merged["delta_accuracy_dbr_minus_pure"] = merged["accuracy_mean_dbr"] - merged["accuracy_mean_pure"]
    merged["delta_pedagogy_dbr_minus_pure"] = (
        merged["pedagogical_score_mean_dbr"] - merged["pedagogical_score_mean_pure"]
    )
    merged["delta_silent_dbr_minus_pure"] = merged["silent_error_rate_dbr"] - merged["silent_error_rate_pure"]
    return merged[
        [
            "seed",
            "run_dir_pure",
            "sample_count_pure",
            "sample_count_dbr",
            "accuracy_mean_pure",
            "accuracy_mean_dbr",
            "pedagogical_score_mean_pure",
            "pedagogical_score_mean_dbr",
            "silent_error_rate_pure",
            "silent_error_rate_dbr",
            "delta_accuracy_dbr_minus_pure",
            "delta_pedagogy_dbr_minus_pure",
            "delta_silent_dbr_minus_pure",
        ]
    ].rename(columns={"run_dir_pure": "run_dir"}).sort_values("seed").reset_index(drop=True)


def _recompute_defect_rates(decisions_df: pd.DataFrame, *, selected: bool | None) -> dict[str, float]:
    if selected is None:
        frame = decisions_df
    else:
        frame = decisions_df[decisions_df["selected"].astype(bool) == selected]
    if frame.empty:
        return {key: 0.0 for key in _DEFECT_FIELDS}
    return {
        key: float(frame[col].astype(bool).mean()) if col in frame.columns else 0.0
        for key, col in _DEFECT_FIELDS.items()
    }


def _collect_stage_findings(run_dir: Path, findings: list[dict[str, Any]]) -> None:
    run_manifest = run_dir / "run_stage_manifest.json"
    if not run_manifest.exists():
        findings.append(
            {
                "severity": "CRITICAL",
                "category": "manifest",
                "target": str(run_manifest),
                "message": "Missing run_stage_manifest.json",
            }
        )
        return
    run_payload = _read_json(run_manifest)
    run_stages = (run_payload.get("stages") or {}) if isinstance(run_payload, dict) else {}
    for stage_name in ("data_prep", "aggregate", "plotting"):
        status = str((run_stages.get(stage_name) or {}).get("status", "missing"))
        if status != "completed":
            findings.append(
                {
                    "severity": "HIGH",
                    "category": "manifest",
                    "target": str(run_manifest),
                    "message": f"Run stage {stage_name} status={status}",
                }
            )
    for ctx_manifest in sorted(run_dir.glob("*/*/gen_*/stage_manifest.json")):
        payload = _read_json(ctx_manifest)
        for stage_name, rec in (payload.get("stages") or {}).items():
            status = str((rec or {}).get("status", "missing"))
            if status != "completed" and stage_name not in {"synthetic_build", "anchoring"}:
                findings.append(
                    {
                        "severity": "HIGH",
                        "category": "manifest",
                        "target": str(ctx_manifest),
                        "message": f"Context stage {stage_name} status={status}",
                    }
                )


def _collect_row_and_budget_checks(run_dirs: Sequence[Path], findings: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        seed = _load_seed(run_dir)
        model_dir = _discover_model_dir(run_dir)
        for branch in _TARGET_BRANCHES:
            step_dir = model_dir / branch / f"gen_{_GEN2}"
            model_outputs = _safe_parquet(step_dir / "model_outputs.parquet")
            extraction = _safe_parquet(step_dir / "answer_extraction.parquet")
            accuracy = _safe_parquet(step_dir / "accuracy_table.parquet")
            judge = _safe_parquet(step_dir / "judge_outputs.parquet")
            judge_fail = _safe_parquet(step_dir / "judge_failures.parquet")
            eval_merged = _safe_parquet(step_dir / "eval_merged.parquet")
            summary_path = run_dir / "tables" / "first_experiment_summary.csv"
            summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
            summary_row = summary_df[
                (summary_df.get("branch") == branch) & (pd.to_numeric(summary_df.get("generation"), errors="coerce") == _GEN2)
            ]
            summary_sample = int(summary_row["sample_count"].iloc[0]) if not summary_row.empty else -1
            missing_judge_rows = int(len(model_outputs) - (len(judge) + len(judge_fail))) if len(model_outputs) else 0
            missing_accuracy_rows = int(len(model_outputs) - len(accuracy)) if len(model_outputs) else 0

            row = {
                "run_dir": str(run_dir),
                "seed": int(seed),
                "branch": branch,
                "generation": _GEN2,
                "model_outputs_rows": int(len(model_outputs)),
                "answer_extraction_rows": int(len(extraction)),
                "accuracy_rows": int(len(accuracy)),
                "judge_rows": int(len(judge)),
                "judge_failures_rows": int(len(judge_fail)),
                "eval_merged_rows": int(len(eval_merged)),
                "summary_sample_count": int(summary_sample),
                "missing_judge_rows": int(missing_judge_rows),
                "missing_accuracy_rows": int(missing_accuracy_rows),
                "partial_artifact_exists": bool((step_dir / "judge_partial.parquet").exists()),
            }

            if branch == _DBR_BRANCH:
                decisions_path = step_dir / "dbr_decisions.parquet"
                training_path = step_dir / "dbr_training_dataset.parquet"
                report_path = step_dir / "dbr_budget_report.json"
                decisions = _safe_parquet(decisions_path)
                training = _safe_parquet(training_path)
                report = _read_json(report_path) if report_path.exists() else {}

                selected_count_reported = int(report.get("selected_count", -1))
                target_size = int(report.get("target_size", -1))
                selection_rate_reported = float(report.get("selection_rate", float("nan")))
                selected_count_from_decisions = int(decisions["selected"].astype(bool).sum()) if not decisions.empty else 0
                selected_count_from_training = int(len(training))
                selection_rate_from_decisions = (
                    float(selected_count_from_decisions / len(decisions)) if len(decisions) else float("nan")
                )

                before_rec = _recompute_defect_rates(decisions, selected=None)
                after_rec = _recompute_defect_rates(decisions, selected=True)
                before_rep = report.get("defect_rates_before", {}) or {}
                after_rep = report.get("defect_rates_after", {}) or {}

                row.update(
                    {
                        "dbr_target_size": target_size,
                        "dbr_selected_count_reported": selected_count_reported,
                        "dbr_selected_count_from_decisions": selected_count_from_decisions,
                        "dbr_selected_count_from_training": selected_count_from_training,
                        "dbr_selection_rate_reported": selection_rate_reported,
                        "dbr_selection_rate_from_decisions": selection_rate_from_decisions,
                        "dbr_relaxation_steps_used_json": json.dumps(report.get("relaxation_steps_used", []), ensure_ascii=False),
                        "dbr_budget_violations_json": json.dumps(report.get("budget_violations", {}), ensure_ascii=False),
                    }
                )

                for defect_name in _DEFECT_FIELDS:
                    row[f"defect_before_reported_{defect_name}"] = float(before_rep.get(defect_name, 0.0))
                    row[f"defect_before_recomputed_{defect_name}"] = float(before_rec.get(defect_name, 0.0))
                    row[f"defect_after_reported_{defect_name}"] = float(after_rep.get(defect_name, 0.0))
                    row[f"defect_after_recomputed_{defect_name}"] = float(after_rec.get(defect_name, 0.0))

                if selected_count_reported != selected_count_from_training:
                    findings.append(
                        {
                            "severity": "HIGH",
                            "category": "dbr_budget",
                            "target": str(report_path),
                            "message": "selected_count mismatch between budget report and training dataset",
                            "details": {
                                "reported": selected_count_reported,
                                "training_rows": selected_count_from_training,
                            },
                        }
                    )
                if selected_count_reported != selected_count_from_decisions:
                    findings.append(
                        {
                            "severity": "HIGH",
                            "category": "dbr_budget",
                            "target": str(report_path),
                            "message": "selected_count mismatch between budget report and decisions",
                            "details": {
                                "reported": selected_count_reported,
                                "decisions_selected": selected_count_from_decisions,
                            },
                        }
                    )

            if summary_sample >= 0 and summary_sample != len(eval_merged):
                findings.append(
                    {
                        "severity": "HIGH",
                        "category": "row_counts",
                        "target": str(step_dir),
                        "message": "summary sample_count mismatch vs eval_merged rows",
                        "details": {"summary_sample_count": summary_sample, "eval_merged_rows": int(len(eval_merged))},
                    }
                )
            if missing_judge_rows > 0:
                findings.append(
                    {
                        "severity": "HIGH",
                        "category": "row_counts",
                        "target": str(step_dir),
                        "message": "Missing judge rows not accounted by judge_failures",
                        "details": {"missing_judge_rows": missing_judge_rows},
                    }
                )
            if missing_accuracy_rows > 0:
                findings.append(
                    {
                        "severity": "HIGH",
                        "category": "row_counts",
                        "target": str(step_dir),
                        "message": "Missing accuracy rows compared to model_outputs",
                        "details": {"missing_accuracy_rows": missing_accuracy_rows},
                    }
                )

            rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed", "branch"]).reset_index(drop=True)


def _latest_file(base: Path, pattern: str) -> Path | None:
    candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    return candidates[-1] if candidates else None


def _metric_diff_status(a: float, b: float, *, tol: float = 1e-9) -> tuple[str, float]:
    diff = abs(float(a) - float(b))
    return ("ok" if diff <= tol else "mismatch"), diff


def _compare_against_reference_tables(
    *,
    recomputed_metrics: pd.DataFrame,
    recomputed_deltas: pd.DataFrame,
    outputs_root: Path,
    findings: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ref_paper = _latest_file(outputs_root, "dbr_confirmatory/dbr_paper_summary_*/tables/dbr_paper_summary.csv")
    ref_deltas = _latest_file(outputs_root, "dbr_confirmatory/dbr_paper_summary_*/tables/dbr_paper_deltas_by_seed.csv")
    ref_run_level = _latest_file(outputs_root, "dbr_confirmatory/dbr_confirmatory_*/tables/dbr_confirmatory_run_level.csv")
    ref_branch_deltas = _latest_file(outputs_root, "dbr_confirmatory/dbr_confirmatory_*/tables/dbr_confirmatory_branch_deltas.csv")

    if ref_paper and ref_paper.exists():
        paper_df = pd.read_csv(ref_paper)
        gen2_seed_rows = paper_df[
            (paper_df.get("section") == "gen2_seed_branch_compare")
            & (pd.to_numeric(paper_df.get("generation"), errors="coerce") == _GEN2)
        ].copy()
        for rec in gen2_seed_rows.to_dict(orient="records"):
            seed = int(float(rec["seed"]))
            metric = str(rec["metric"])
            seed_metrics = recomputed_metrics[recomputed_metrics["seed"] == seed]
            pure = seed_metrics[seed_metrics["branch"] == _PURE_BRANCH]
            dbr = seed_metrics[seed_metrics["branch"] == _DBR_BRANCH]
            if pure.empty or dbr.empty:
                findings.append(
                    {
                        "severity": "CRITICAL",
                        "category": "table_compare",
                        "target": str(ref_paper),
                        "message": f"Missing recomputed metrics for seed={seed}",
                    }
                )
                continue
            if metric == "accuracy":
                pure_val = float(pure["accuracy_mean"].iloc[0])
                dbr_val = float(dbr["accuracy_mean"].iloc[0])
            elif metric == "pedagogical_score":
                pure_val = float(pure["pedagogical_score_mean"].iloc[0])
                dbr_val = float(dbr["pedagogical_score_mean"].iloc[0])
            elif metric == "silent_error_rate":
                pure_val = float(pure["silent_error_rate"].iloc[0])
                dbr_val = float(dbr["silent_error_rate"].iloc[0])
            else:
                continue
            delta_val = dbr_val - pure_val
            for field_name, ref_val, rec_val in (
                ("pure_value", float(rec["pure_value"]), pure_val),
                ("dbr_value", float(rec["dbr_value"]), dbr_val),
                ("delta_dbr_minus_pure", float(rec["delta_dbr_minus_pure"]), delta_val),
            ):
                status, abs_diff = _metric_diff_status(ref_val, rec_val)
                row = {
                    "reference_table": str(ref_paper),
                    "table_type": "dbr_paper_summary_seed_compare",
                    "seed": seed,
                    "generation": _GEN2,
                    "branch": None,
                    "metric": metric,
                    "field_name": field_name,
                    "reference_value": ref_val,
                    "recomputed_value": rec_val,
                    "abs_diff": abs_diff,
                    "status": status,
                }
                rows.append(row)
                if status == "mismatch":
                    findings.append(
                        {
                            "severity": "CRITICAL",
                            "category": "table_compare",
                            "target": str(ref_paper),
                            "message": "Mismatch vs dbr_paper_summary seed compare",
                            "details": row,
                        }
                    )

    if ref_deltas and ref_deltas.exists():
        delta_df = pd.read_csv(ref_deltas)
        for rec in delta_df.to_dict(orient="records"):
            seed = int(rec["seed"])
            cur = recomputed_deltas[recomputed_deltas["seed"] == seed]
            if cur.empty:
                findings.append(
                    {
                        "severity": "CRITICAL",
                        "category": "table_compare",
                        "target": str(ref_deltas),
                        "message": f"Missing recomputed delta for seed={seed}",
                    }
                )
                continue
            cur_row = cur.iloc[0]
            pairs = (
                ("delta_accuracy_dbr_minus_pure_gen2", float(rec["delta_accuracy_dbr_minus_pure_gen2"]), float(cur_row["delta_accuracy_dbr_minus_pure"])),
                ("delta_pedagogy_dbr_minus_pure_gen2", float(rec["delta_pedagogy_dbr_minus_pure_gen2"]), float(cur_row["delta_pedagogy_dbr_minus_pure"])),
                ("delta_silent_dbr_minus_pure_gen2", float(rec["delta_silent_dbr_minus_pure_gen2"]), float(cur_row["delta_silent_dbr_minus_pure"])),
            )
            for metric_name, ref_val, rec_val in pairs:
                status, abs_diff = _metric_diff_status(ref_val, rec_val)
                row = {
                    "reference_table": str(ref_deltas),
                    "table_type": "dbr_paper_deltas_by_seed",
                    "seed": seed,
                    "generation": _GEN2,
                    "branch": None,
                    "metric": metric_name,
                    "field_name": "value",
                    "reference_value": ref_val,
                    "recomputed_value": rec_val,
                    "abs_diff": abs_diff,
                    "status": status,
                }
                rows.append(row)
                if status == "mismatch":
                    findings.append(
                        {
                            "severity": "CRITICAL",
                            "category": "table_compare",
                            "target": str(ref_deltas),
                            "message": "Mismatch vs dbr_paper_deltas_by_seed",
                            "details": row,
                        }
                    )

    if ref_run_level and ref_run_level.exists():
        run_level_df = pd.read_csv(ref_run_level)
        run_level_df = run_level_df[
            run_level_df["branch"].isin(_TARGET_BRANCHES)
            & (pd.to_numeric(run_level_df["generation"], errors="coerce") == _GEN2)
        ].copy()
        for rec in run_level_df.to_dict(orient="records"):
            seed = int(rec["seed"])
            branch = str(rec["branch"])
            cur = recomputed_metrics[(recomputed_metrics["seed"] == seed) & (recomputed_metrics["branch"] == branch)]
            if cur.empty:
                findings.append(
                    {
                        "severity": "HIGH",
                        "category": "table_compare",
                        "target": str(ref_run_level),
                        "message": f"Reference run_level includes seed/branch not in target recompute: seed={seed}, branch={branch}",
                    }
                )
                continue
            cur_row = cur.iloc[0]
            pairs = (
                ("sample_count", float(rec["sample_count"]), float(cur_row["sample_count"])),
                ("accuracy_mean", float(rec["accuracy_mean"]), float(cur_row["accuracy_mean"])),
                ("pedagogical_score_mean", float(rec["pedagogical_score_mean"]), float(cur_row["pedagogical_score_mean"])),
                ("silent_error_rate", float(rec["silent_error_rate"]), float(cur_row["silent_error_rate"])),
            )
            for metric_name, ref_val, rec_val in pairs:
                status, abs_diff = _metric_diff_status(ref_val, rec_val)
                row = {
                    "reference_table": str(ref_run_level),
                    "table_type": "dbr_confirmatory_run_level",
                    "seed": seed,
                    "generation": _GEN2,
                    "branch": branch,
                    "metric": metric_name,
                    "field_name": "value",
                    "reference_value": ref_val,
                    "recomputed_value": rec_val,
                    "abs_diff": abs_diff,
                    "status": status,
                }
                rows.append(row)
                if status == "mismatch":
                    findings.append(
                        {
                            "severity": "CRITICAL",
                            "category": "table_compare",
                            "target": str(ref_run_level),
                            "message": "Mismatch vs dbr_confirmatory_run_level",
                            "details": row,
                        }
                    )

    if ref_branch_deltas and ref_branch_deltas.exists():
        branch_df = pd.read_csv(ref_branch_deltas)
        branch_df = branch_df[
            (branch_df["comparison"] == "dbr_medium_minus_pure_recycling")
            & (pd.to_numeric(branch_df["generation"], errors="coerce") == _GEN2)
        ].copy()
        for rec in branch_df.to_dict(orient="records"):
            seed = int(rec["seed"])
            cur = recomputed_deltas[recomputed_deltas["seed"] == seed]
            if cur.empty:
                findings.append(
                    {
                        "severity": "HIGH",
                        "category": "table_compare",
                        "target": str(ref_branch_deltas),
                        "message": f"Reference branch_deltas includes seed not in target recompute: seed={seed}",
                    }
                )
                continue
            cur_row = cur.iloc[0]
            pairs = (
                ("delta_accuracy_mean", float(rec["delta_accuracy_mean"]), float(cur_row["delta_accuracy_dbr_minus_pure"])),
                (
                    "delta_pedagogical_score_mean",
                    float(rec["delta_pedagogical_score_mean"]),
                    float(cur_row["delta_pedagogy_dbr_minus_pure"]),
                ),
                ("delta_silent_error_rate", float(rec["delta_silent_error_rate"]), float(cur_row["delta_silent_dbr_minus_pure"])),
            )
            for metric_name, ref_val, rec_val in pairs:
                status, abs_diff = _metric_diff_status(ref_val, rec_val)
                row = {
                    "reference_table": str(ref_branch_deltas),
                    "table_type": "dbr_confirmatory_branch_deltas",
                    "seed": seed,
                    "generation": _GEN2,
                    "branch": _DBR_BRANCH,
                    "metric": metric_name,
                    "field_name": "value",
                    "reference_value": ref_val,
                    "recomputed_value": rec_val,
                    "abs_diff": abs_diff,
                    "status": status,
                }
                rows.append(row)
                if status == "mismatch":
                    findings.append(
                        {
                            "severity": "CRITICAL",
                            "category": "table_compare",
                            "target": str(ref_branch_deltas),
                            "message": "Mismatch vs dbr_confirmatory_branch_deltas",
                            "details": row,
                        }
                    )

    return pd.DataFrame(rows)


def _build_report_md(findings: list[dict[str, Any]], *, out_dir: Path) -> str:
    sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for item in findings:
        sev = str(item.get("severity", "LOW")).upper()
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    lines: list[str] = []
    lines.append("# DBR Recompute Audit Report")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat()}")
    lines.append(f"- out_dir: {out_dir}")
    lines.append("")
    lines.append("## Severity")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        lines.append(f"- {sev}: {sev_counts.get(sev, 0)}")
    lines.append("")
    lines.append("## Findings")
    if not findings:
        lines.append("- none")
    else:
        for item in findings:
            lines.append(
                f"- [{item.get('severity', 'LOW')}] {item.get('category', 'unknown')} :: "
                f"{item.get('target', '-')}: {item.get('message', '')}"
            )
    lines.append("")
    blocking = any(str(x.get("severity", "")).upper() in {"CRITICAL", "HIGH"} for x in findings)
    lines.append("## Verdict")
    if blocking:
        lines.append("- Blocking CRITICAL/HIGH issues found. Stop before Qwen sensitivity.")
    else:
        lines.append("- No CRITICAL/HIGH issues found. Safe to proceed to Qwen pairwise sensitivity.")
    lines.append("")
    return "\n".join(lines)


def run_dbr_recompute_audit(
    *,
    run_dirs: Sequence[Path],
    outputs_root: Path = Path("outputs"),
) -> DBRRecomputeAuditArtifacts:
    if not run_dirs:
        raise ValueError("run_dirs must be non-empty")
    run_dirs = [Path(p) for p in run_dirs]
    out_dir = outputs_root / "audits" / f"dbr_recompute_audit_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    findings: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        _collect_stage_findings(run_dir, findings)

    metrics_df = recompute_gen2_metrics(run_dirs)
    deltas_df = recompute_gen2_deltas_by_seed(metrics_df)
    budget_df = _collect_row_and_budget_checks(run_dirs, findings)
    table_cmp_df = _compare_against_reference_tables(
        recomputed_metrics=metrics_df,
        recomputed_deltas=deltas_df,
        outputs_root=outputs_root,
        findings=findings,
    )

    # Add aggregate mean/std/ci rows for convenience.
    delta_rows: list[dict[str, Any]] = []
    for metric_col in (
        "delta_accuracy_dbr_minus_pure",
        "delta_pedagogy_dbr_minus_pure",
        "delta_silent_dbr_minus_pure",
    ):
        vals = deltas_df[metric_col].astype(float).tolist()
        series = pd.Series(vals, dtype="float64")
        ci_low, ci_high = _ci95(vals)
        delta_rows.append(
            {
                "seed": pd.NA,
                "run_dir": "",
                "sample_count_pure": pd.NA,
                "sample_count_dbr": pd.NA,
                "accuracy_mean_pure": pd.NA,
                "accuracy_mean_dbr": pd.NA,
                "pedagogical_score_mean_pure": pd.NA,
                "pedagogical_score_mean_dbr": pd.NA,
                "silent_error_rate_pure": pd.NA,
                "silent_error_rate_dbr": pd.NA,
                "delta_accuracy_dbr_minus_pure": pd.NA,
                "delta_pedagogy_dbr_minus_pure": pd.NA,
                "delta_silent_dbr_minus_pure": pd.NA,
                "summary_metric": metric_col,
                "summary_mean": float(series.mean()),
                "summary_std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "summary_ci95_low": ci_low,
                "summary_ci95_high": ci_high,
                "summary_n": int(len(series)),
            }
        )
    deltas_out = pd.concat([deltas_df, pd.DataFrame(delta_rows)], ignore_index=True)

    metrics_csv = out_dir / "dbr_recomputed_gen2_metrics.csv"
    deltas_csv = out_dir / "dbr_recomputed_deltas_by_seed.csv"
    table_comparison_csv = out_dir / "dbr_table_comparison.csv"
    budget_check_csv = out_dir / "dbr_budget_recompute_check.csv"
    findings_json = out_dir / "dbr_recompute_findings.json"
    report_md = out_dir / "dbr_recompute_audit_report.md"

    metrics_df.to_csv(metrics_csv, index=False)
    deltas_out.to_csv(deltas_csv, index=False)
    table_cmp_df.to_csv(table_comparison_csv, index=False)
    budget_df.to_csv(budget_check_csv, index=False)

    findings_payload = {
        "generated_at": datetime.now().isoformat(),
        "run_dirs": [str(p) for p in run_dirs],
        "finding_count": len(findings),
        "findings": findings,
    }
    findings_json.write_text(json.dumps(findings_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_build_report_md(findings, out_dir=out_dir), encoding="utf-8")

    has_blocking = any(str(item.get("severity", "")).upper() in {"CRITICAL", "HIGH"} for item in findings)
    return DBRRecomputeAuditArtifacts(
        out_dir=out_dir,
        metrics_csv=metrics_csv,
        deltas_csv=deltas_csv,
        table_comparison_csv=table_comparison_csv,
        budget_check_csv=budget_check_csv,
        findings_json=findings_json,
        report_md=report_md,
        has_blocking_findings=has_blocking,
    )


def build_dbr_pair_candidates(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        seed = _load_seed(run_dir)
        run_id = run_dir.name
        model_dir = _discover_model_dir(run_dir)
        pure_path = model_dir / _PURE_BRANCH / f"gen_{_GEN2}" / "eval_merged.parquet"
        dbr_path = model_dir / _DBR_BRANCH / f"gen_{_GEN2}" / "eval_merged.parquet"
        pure = pd.read_parquet(pure_path)
        dbr = pd.read_parquet(dbr_path)

        data_root = _load_data_root(run_dir)
        heldout_path = data_root / "splits" / "heldout_test.parquet"
        heldout = pd.read_parquet(heldout_path)[["example_id", "question"]].copy()
        heldout["example_id"] = heldout["example_id"].astype(str)

        pure_v = pure[["example_id", "raw_response", "answer_gold"]].copy().rename(
            columns={"raw_response": "response_pure", "answer_gold": "answer_gold_pure"}
        )
        dbr_v = dbr[["example_id", "raw_response", "answer_gold"]].copy().rename(
            columns={"raw_response": "response_dbr", "answer_gold": "answer_gold_dbr"}
        )
        pure_v["example_id"] = pure_v["example_id"].astype(str)
        dbr_v["example_id"] = dbr_v["example_id"].astype(str)
        merged = pure_v.merge(dbr_v, on="example_id", how="inner", validate="one_to_one")
        merged = merged.merge(heldout, on="example_id", how="left", validate="one_to_one")
        if merged["question"].isna().any():
            raise ValueError(f"Missing question after heldout merge for run={run_dir}")
        merged["answer_gold"] = merged["answer_gold_pure"].astype(str)
        mismatch = merged["answer_gold_pure"].astype(str) != merged["answer_gold_dbr"].astype(str)
        if mismatch.any():
            raise ValueError(f"Gold answer mismatch between pure/dbr in run={run_dir}")
        merged["source_seed"] = int(seed)
        merged["source_run_dir"] = str(run_dir)
        merged["run_id"] = run_id
        merged["generation"] = _GEN2
        model_name = str(pure["model_name"].iloc[0]) if "model_name" in pure.columns and len(pure) else ""
        merged["model_name"] = model_name
        rows.extend(merged.to_dict(orient="records"))
    if not rows:
        raise ValueError("No DBR pair candidates found")
    return pd.DataFrame(rows)


def select_balanced_dbr_pairs(
    candidates_df: pd.DataFrame,
    *,
    total_n: int = 48,
    sample_seed: int = 4242,
) -> pd.DataFrame:
    if total_n < 1:
        raise ValueError("total_n must be >= 1")
    required = {"source_seed", "example_id", "question", "answer_gold", "response_pure", "response_dbr"}
    missing = required.difference(candidates_df.columns)
    if missing:
        raise ValueError(f"candidates_df missing columns: {sorted(missing)}")

    seeds = sorted(pd.to_numeric(candidates_df["source_seed"], errors="coerce").dropna().astype(int).unique().tolist())
    if not seeds:
        raise ValueError("No valid source_seed values found")
    base = total_n // len(seeds)
    extra = total_n % len(seeds)
    out_parts: list[pd.DataFrame] = []
    for idx, seed in enumerate(seeds):
        target = base + (1 if idx < extra else 0)
        sub = candidates_df[pd.to_numeric(candidates_df["source_seed"], errors="coerce") == seed].copy()
        if len(sub) < target:
            raise ValueError(f"Not enough matched pairs for seed={seed}; need={target}, got={len(sub)}")
        sub = sub.sample(frac=1.0, random_state=sample_seed + seed).reset_index(drop=True)
        out_parts.append(sub.head(target))
    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["source_seed", "example_id"]).reset_index(drop=True)
    return out


def build_blinded_dbr_pair_tables(
    selected_pairs_df: pd.DataFrame,
    *,
    sample_seed: int = 4242,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "source_run_dir",
        "source_seed",
        "run_id",
        "generation",
        "model_name",
        "example_id",
        "question",
        "answer_gold",
        "response_pure",
        "response_dbr",
    }
    missing = required.difference(selected_pairs_df.columns)
    if missing:
        raise ValueError(f"selected_pairs_df missing columns: {sorted(missing)}")
    rng = random.Random(sample_seed)
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(selected_pairs_df.to_dict(orient="records"), start=1):
        pair_id = f"dbr_pair_{idx:04d}"
        dbr_is_a = rng.random() < 0.5
        response_a = rec["response_dbr"] if dbr_is_a else rec["response_pure"]
        response_b = rec["response_pure"] if dbr_is_a else rec["response_dbr"]
        branch_a = _DBR_BRANCH if dbr_is_a else _PURE_BRANCH
        branch_b = _PURE_BRANCH if dbr_is_a else _DBR_BRANCH
        public_rows.append(
            {
                "pair_id": pair_id,
                "source_run_dir": rec["source_run_dir"],
                "source_seed": int(rec["source_seed"]),
                "run_id": rec["run_id"],
                "generation": int(rec["generation"]),
                "model_name": rec["model_name"],
                "example_id": rec["example_id"],
                "question": rec["question"],
                "answer_gold": rec["answer_gold"],
                "response_A": response_a,
                "response_B": response_b,
            }
        )
        key_rows.append(
            {
                "pair_id": pair_id,
                "source_run_dir": rec["source_run_dir"],
                "source_seed": int(rec["source_seed"]),
                "run_id": rec["run_id"],
                "generation": int(rec["generation"]),
                "model_name": rec["model_name"],
                "example_id": rec["example_id"],
                "A_branch": branch_a,
                "B_branch": branch_b,
            }
        )
    return pd.DataFrame(public_rows), pd.DataFrame(key_rows)


def decode_qwen_dbr_winner(pairwise_results_df: pd.DataFrame, hidden_key_df: pd.DataFrame) -> pd.DataFrame:
    required_results = {"pair_id", "winner", "confidence", "reason", "judge_model", "judge_label"}
    required_key = {"pair_id", "source_seed", "A_branch", "B_branch"}
    miss_r = required_results.difference(pairwise_results_df.columns)
    miss_k = required_key.difference(hidden_key_df.columns)
    if miss_r:
        raise ValueError(f"pairwise_results_df missing columns: {sorted(miss_r)}")
    if miss_k:
        raise ValueError(f"hidden_key_df missing columns: {sorted(miss_k)}")
    merged = hidden_key_df.merge(pairwise_results_df, on="pair_id", how="inner", validate="one_to_one")

    def _winner_branch(row: pd.Series) -> str:
        w = str(row["winner"])
        if w == "A":
            return str(row["A_branch"])
        if w == "B":
            return str(row["B_branch"])
        return "Tie"

    merged["qwen_winner_branch"] = merged.apply(_winner_branch, axis=1)
    merged["qwen_dbr_win"] = merged["qwen_winner_branch"] == _DBR_BRANCH
    merged["qwen_pure_win"] = merged["qwen_winner_branch"] == _PURE_BRANCH
    merged["qwen_tie"] = merged["qwen_winner_branch"] == "Tie"
    return merged


def build_qwen_dbr_pairwise_summary(
    decoded_df: pd.DataFrame,
    *,
    llama_aggregate_pedagogy_delta: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decoded_df.empty:
        raise ValueError("decoded_df is empty")
    dbr_rate = float(decoded_df["qwen_dbr_win"].mean())
    pure_rate = float(decoded_df["qwen_pure_win"].mean())
    tie_rate = float(decoded_df["qwen_tie"].mean())
    margin = dbr_rate - pure_rate
    qwen_conclusion = "mixed_tie"
    if margin > 0:
        qwen_conclusion = "dbr_preferred"
    elif margin < 0:
        qwen_conclusion = "pure_preferred"

    llama_conclusion = "unknown"
    conclusion_changed = pd.NA
    if llama_aggregate_pedagogy_delta is not None:
        if llama_aggregate_pedagogy_delta > 0:
            llama_conclusion = "dbr_preferred"
        elif llama_aggregate_pedagogy_delta < 0:
            llama_conclusion = "pure_preferred"
        else:
            llama_conclusion = "mixed_tie"
        conclusion_changed = bool(llama_conclusion != qwen_conclusion)

    summary_df = pd.DataFrame(
        [
            {
                "n_pairs": int(len(decoded_df)),
                "qwen_dbr_win_rate": dbr_rate,
                "qwen_pure_win_rate": pure_rate,
                "qwen_tie_rate": tie_rate,
                "qwen_dbr_minus_pure_win_rate": margin,
                "qwen_conclusion": qwen_conclusion,
                "llama_aggregate_pedagogy_delta_dbr_minus_pure": llama_aggregate_pedagogy_delta,
                "llama_aggregate_conclusion": llama_conclusion,
                "conclusion_changed_vs_llama_aggregate": conclusion_changed,
            }
        ]
    )

    per_seed_rows: list[dict[str, Any]] = []
    for seed, grp in decoded_df.groupby("source_seed", as_index=False):
        dbr_seed = float(grp["qwen_dbr_win"].mean())
        pure_seed = float(grp["qwen_pure_win"].mean())
        tie_seed = float(grp["qwen_tie"].mean())
        per_seed_rows.append(
            {
                "source_seed": int(seed),
                "n_pairs": int(len(grp)),
                "qwen_dbr_win_rate": dbr_seed,
                "qwen_pure_win_rate": pure_seed,
                "qwen_tie_rate": tie_seed,
                "qwen_dbr_minus_pure_win_rate": dbr_seed - pure_seed,
            }
        )
    seed_summary_df = pd.DataFrame(per_seed_rows).sort_values("source_seed").reset_index(drop=True)
    return summary_df, seed_summary_df


def run_qwen_dbr_pairwise_sensitivity(
    *,
    cfg: AppConfig,
    run_dirs: Sequence[Path],
    sample_size: int = 48,
    sample_seed: int = 4242,
    out_dir: Path | None = None,
    llama_aggregate_pedagogy_delta: float | None = None,
) -> QwenDBRPairwiseArtifacts:
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    if cfg.judge.provider.strip().lower() != "cerebras":
        raise ValueError("Qwen DBR sensitivity requires provider=cerebras")

    run_dirs = [Path(p) for p in run_dirs]
    ts = _now_tag()
    tables_dir = Path(out_dir or (Path(cfg.paths.output_root) / "judge_sensitivity" / f"qwen_dbr_pairwise_{ts}" / "tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_dbr_pair_candidates(run_dirs)
    selected = select_balanced_dbr_pairs(candidates, total_n=sample_size, sample_seed=sample_seed)
    selected_pairs, hidden_key = build_blinded_dbr_pair_tables(selected, sample_seed=sample_seed)

    pairwise_results, pairwise_failures = run_pairwise_judge_for_model(
        selected_pairs_df=selected_pairs,
        cfg=cfg,
        judge_model_name=cfg.judge.model_name,
        judge_label="qwen_dbr_pairwise",
    )

    decoded = decode_qwen_dbr_winner(pairwise_results, hidden_key)
    comparison = decoded.copy()
    if not pairwise_failures.empty:
        comparison = comparison.merge(
            pairwise_failures.rename(columns={"judge_model": "qwen_failure_judge_model"}),
            on="pair_id",
            how="left",
        )
    summary_df, seed_summary_df = build_qwen_dbr_pairwise_summary(
        comparison,
        llama_aggregate_pedagogy_delta=llama_aggregate_pedagogy_delta,
    )

    selected_pairs_csv = tables_dir / "qwen_dbr_selected_pairs.csv"
    hidden_key_csv = tables_dir / "qwen_dbr_hidden_key.csv"
    pairwise_results_csv = tables_dir / "qwen_dbr_pairwise_results.csv"
    pairwise_comparison_csv = tables_dir / "qwen_dbr_pairwise_comparison.csv"
    pairwise_summary_csv = tables_dir / "qwen_dbr_pairwise_summary.csv"
    seed_summary_csv = tables_dir / "qwen_dbr_seed_summary.csv"
    metadata_json = tables_dir / "qwen_dbr_metadata.json"

    selected_pairs.to_csv(selected_pairs_csv, index=False)
    hidden_key.to_csv(hidden_key_csv, index=False)
    pairwise_results.to_csv(pairwise_results_csv, index=False)
    comparison.to_csv(pairwise_comparison_csv, index=False)
    summary_df.to_csv(pairwise_summary_csv, index=False)
    seed_summary_df.to_csv(seed_summary_csv, index=False)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "provider": cfg.judge.provider,
        "model_name": cfg.judge.model_name,
        "sample_size_requested": int(sample_size),
        "sample_size_selected": int(len(selected_pairs)),
        "run_dirs": [str(p) for p in run_dirs],
        "sample_seed": int(sample_seed),
        "branches": [_PURE_BRANCH, _DBR_BRANCH],
        "generation": _GEN2,
        "failures_count": int(len(pairwise_failures)),
        "llama_aggregate_pedagogy_delta_dbr_minus_pure": llama_aggregate_pedagogy_delta,
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return QwenDBRPairwiseArtifacts(
        out_dir=tables_dir.parent,
        selected_pairs_csv=selected_pairs_csv,
        hidden_key_csv=hidden_key_csv,
        pairwise_results_csv=pairwise_results_csv,
        pairwise_comparison_csv=pairwise_comparison_csv,
        pairwise_summary_csv=pairwise_summary_csv,
        seed_summary_csv=seed_summary_csv,
        metadata_json=metadata_json,
    )

