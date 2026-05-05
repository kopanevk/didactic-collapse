from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    category: str
    target: str
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class AuditArtifacts:
    out_dir: Path
    audit_run_manifest_status_csv: Path
    audit_row_counts_csv: Path
    audit_summary_recompute_checks_csv: Path
    audit_analysis_source_checks_csv: Path
    audit_pairwise_checks_csv: Path
    audit_pvf_soft_checks_csv: Path
    audit_findings_json: Path
    audit_report_md: Path


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_parquet_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _safe_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_run_config_snapshot(run_dir: Path) -> dict[str, Any] | None:
    payload = _safe_json(run_dir / "run_config.snapshot.json")
    if payload is None or not isinstance(payload, dict):
        return None
    config = payload.get("config")
    return config if isinstance(config, dict) else None


def _max_generation_from_config(run_dir: Path) -> int | None:
    config = _load_run_config_snapshot(run_dir)
    if config is None:
        return None
    exp = config.get("experiment")
    if not isinstance(exp, dict):
        return None
    generations = exp.get("generations")
    if isinstance(generations, int) and generations > 0:
        return generations - 1
    return None


def _latest_matching(base: Path, pattern: str) -> Path | None:
    candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    return candidates[-1] if candidates else None


def discover_default_targets(outputs_root: Path) -> dict[str, list[Path]]:
    runs_root = outputs_root / "runs"
    targets: dict[str, list[Path]] = {
        "run_dirs": [],
        "analysis_dirs": [],
    }

    # Human anchoring confirmatory seeds.
    for seed in (51, 52, 53):
        p = _latest_matching(runs_root, f"anchoring_confirmatory_append_seed{seed}_*")
        if p is not None:
            targets["run_dirs"].append(p)

    # Training confirmatory anchor20 seeds.
    for seed in (71, 72, 73):
        p = _latest_matching(runs_root, f"training_confirmatory_anchor20_seed{seed}_*")
        if p is not None:
            targets["run_dirs"].append(p)

    # PVF confirmatory seeds.
    for seed in (91, 92, 93):
        p = _latest_matching(runs_root, f"pvf_confirmatory_seed{seed}_*")
        if p is not None:
            targets["run_dirs"].append(p)

    # Soft-PVF confirmatory seeds.
    for seed in (141, 142, 143):
        p = _latest_matching(runs_root, f"soft_pvf_confirmatory_seed{seed}_*")
        if p is not None:
            targets["run_dirs"].append(p)

    # Single-run families explicitly requested.
    for pattern in ("soft_pvf_policy_tuning_*", "soft_pvf_shakedown_*"):
        p = _latest_matching(runs_root, pattern)
        if p is not None:
            targets["run_dirs"].append(p)

    # Analysis dirs.
    for pattern in (
        "judge_sensitivity/qwen3_confirmatory_*",
        "judge_sensitivity/pairwise_confirmatory_*",
        "pvf_confirmatory/*",
        "soft_pvf_confirmatory/*",
        "baseline_series/anchoring_confirmatory_append_*",
    ):
        p = _latest_matching(outputs_root, pattern)
        if p is not None:
            targets["analysis_dirs"].append(p)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    dedup_runs: list[Path] = []
    for p in targets["run_dirs"]:
        if str(p) in seen:
            continue
        seen.add(str(p))
        dedup_runs.append(p)
    targets["run_dirs"] = dedup_runs

    seen.clear()
    dedup_analysis: list[Path] = []
    for p in targets["analysis_dirs"]:
        if str(p) in seen:
            continue
        seen.add(str(p))
        dedup_analysis.append(p)
    targets["analysis_dirs"] = dedup_analysis
    return targets


def _parse_context_from_step_dir(step_dir: Path) -> tuple[str, str, int] | None:
    try:
        model_dir = step_dir.parent.parent.name
        branch = step_dir.parent.name
        gen_text = step_dir.name
        if not gen_text.startswith("gen_"):
            return None
        generation = int(gen_text.split("_", maxsplit=1)[1])
        return model_dir, branch, generation
    except Exception:
        return None


def _collect_manifest_status(run_dir: Path, findings: list[AuditFinding]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    max_generation = _max_generation_from_config(run_dir)
    run_manifest = run_dir / "run_stage_manifest.json"
    if not run_manifest.exists():
        findings.append(
            AuditFinding(
                severity="CRITICAL",
                category="manifest",
                target=str(run_dir),
                message="Missing run_stage_manifest.json",
                details={},
            )
        )
        return pd.DataFrame(rows)

    payload = _safe_json(run_manifest)
    if payload is None:
        findings.append(
            AuditFinding(
                severity="CRITICAL",
                category="manifest",
                target=str(run_manifest),
                message="Corrupted run manifest JSON",
                details={},
            )
        )
        return pd.DataFrame(rows)

    run_stages = (payload.get("stages") or {}) if isinstance(payload, dict) else {}
    for stage_name, rec in run_stages.items():
        status = str((rec or {}).get("status", "unknown"))
        rows.append(
            {
                "run_dir": str(run_dir),
                "scope": "run",
                "model_dir": None,
                "branch": None,
                "generation": None,
                "stage_name": stage_name,
                "status": status,
            }
        )
        if stage_name in {"data_prep", "aggregate", "plotting"} and status != "completed":
            findings.append(
                AuditFinding(
                    severity="CRITICAL",
                    category="manifest",
                    target=str(run_manifest),
                    message=f"Run stage not completed: {stage_name}",
                    details={"status": status},
                )
            )

    context_manifests = sorted(run_dir.glob("*/*/gen_*/stage_manifest.json"))
    if not context_manifests:
        findings.append(
            AuditFinding(
                severity="HIGH",
                category="manifest",
                target=str(run_dir),
                message="No context stage manifests found",
                details={},
            )
        )
        return pd.DataFrame(rows)

    for path in context_manifests:
        step_dir = path.parent
        ctx = _parse_context_from_step_dir(step_dir)
        if ctx is None:
            continue
        model_dir, branch, generation = ctx
        context_payload = _safe_json(path)
        if context_payload is None:
            findings.append(
                AuditFinding(
                    severity="HIGH",
                    category="manifest",
                    target=str(path),
                    message="Corrupted context stage manifest JSON",
                    details={},
                )
            )
            continue
        for stage_name, rec in (context_payload.get("stages") or {}).items():
            status = str((rec or {}).get("status", "unknown"))
            expected_pending_terminal_stage = (
                status == "pending"
                and stage_name in {"synthetic_build", "anchoring"}
                and max_generation is not None
                and generation == max_generation
            )
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "scope": "context",
                    "model_dir": model_dir,
                    "branch": branch,
                    "generation": generation,
                    "stage_name": stage_name,
                    "status": status,
                    "expected_pending_terminal_stage": expected_pending_terminal_stage,
                }
            )
            if status != "completed" and not expected_pending_terminal_stage:
                findings.append(
                    AuditFinding(
                        severity="HIGH",
                        category="manifest",
                        target=str(path),
                        message="Context stage not completed",
                        details={
                            "stage_name": stage_name,
                            "status": status,
                            "model_dir": model_dir,
                            "branch": branch,
                            "generation": generation,
                        },
                    )
                )

    return pd.DataFrame(rows)


def _collect_row_counts_for_run(run_dir: Path, findings: list[AuditFinding]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step_dir in sorted(run_dir.glob("*/*/gen_*")):
        if not step_dir.is_dir():
            continue
        ctx = _parse_context_from_step_dir(step_dir)
        if ctx is None:
            continue
        model_dir, branch, generation = ctx
        counts = {
            "model_outputs_rows": _safe_parquet_count(step_dir / "model_outputs.parquet"),
            "answer_extraction_rows": _safe_parquet_count(step_dir / "answer_extraction.parquet"),
            "accuracy_rows": _safe_parquet_count(step_dir / "accuracy_table.parquet"),
            "judge_rows": _safe_parquet_count(step_dir / "judge_outputs.parquet"),
            "judge_failures_rows": _safe_parquet_count(step_dir / "judge_failures.parquet"),
            "eval_merged_rows": _safe_parquet_count(step_dir / "eval_merged.parquet"),
            "synthetic_base_rows": _safe_parquet_count(step_dir / "synthetic_base.parquet"),
            "synthetic_train_next_rows": _safe_parquet_count(step_dir / "synthetic_train_next.parquet"),
            "pvf_rejected_rows": _safe_parquet_count(step_dir / "rejected_examples.parquet"),
            "soft_pvf_decisions_rows": _safe_parquet_count(step_dir / "soft_pvf_decisions.parquet"),
            "pvr_decisions_rows": _safe_parquet_count(step_dir / "pvr_decisions.parquet"),
            "pvr_repair_pairs_rows": _safe_parquet_count(step_dir / "pvr_repair_pairs.parquet"),
        }

        def _num(name: str) -> int:
            value = counts.get(name)
            return int(value) if value is not None else -1

        extraction_mismatch = (
            counts["model_outputs_rows"] is not None
            and counts["answer_extraction_rows"] is not None
            and counts["model_outputs_rows"] != counts["answer_extraction_rows"]
        )
        accuracy_mismatch = (
            counts["answer_extraction_rows"] is not None
            and counts["accuracy_rows"] is not None
            and counts["answer_extraction_rows"] != counts["accuracy_rows"]
        )
        judge_accounting_mismatch = False
        if counts["model_outputs_rows"] is not None and counts["judge_rows"] is not None:
            judge_fail = counts["judge_failures_rows"] or 0
            judge_accounting_mismatch = (counts["judge_rows"] + judge_fail) != counts["model_outputs_rows"]

        if extraction_mismatch:
            findings.append(
                AuditFinding(
                    severity="HIGH",
                    category="row_counts",
                    target=str(step_dir),
                    message="answer_extraction row count differs from model_outputs",
                    details={"model_outputs_rows": _num("model_outputs_rows"), "answer_extraction_rows": _num("answer_extraction_rows")},
                )
            )
        if accuracy_mismatch:
            findings.append(
                AuditFinding(
                    severity="HIGH",
                    category="row_counts",
                    target=str(step_dir),
                    message="accuracy row count differs from answer_extraction",
                    details={"accuracy_rows": _num("accuracy_rows"), "answer_extraction_rows": _num("answer_extraction_rows")},
                )
            )
        if judge_accounting_mismatch:
            findings.append(
                AuditFinding(
                    severity="HIGH",
                    category="row_counts",
                    target=str(step_dir),
                    message="judge rows + judge failures do not match model_outputs",
                    details={
                        "model_outputs_rows": _num("model_outputs_rows"),
                        "judge_rows": _num("judge_rows"),
                        "judge_failures_rows": _num("judge_failures_rows"),
                    },
                )
            )

        row = {
            "run_dir": str(run_dir),
            "model_dir": model_dir,
            "branch": branch,
            "generation": generation,
            **counts,
            "extraction_mismatch": extraction_mismatch,
            "accuracy_mismatch": accuracy_mismatch,
            "judge_accounting_mismatch": judge_accounting_mismatch,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _recompute_run_summary_checks(run_dir: Path, findings: list[AuditFinding], tol: float = 1e-6) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary_path = run_dir / "tables" / "first_experiment_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(rows)

    summary = _safe_csv(summary_path)
    if summary is None or summary.empty:
        findings.append(
            AuditFinding(
                severity="HIGH",
                category="summary",
                target=str(summary_path),
                message="Summary CSV is missing or unreadable",
                details={},
            )
        )
        return pd.DataFrame(rows)

    eval_parts: list[pd.DataFrame] = []
    for path in sorted(run_dir.glob("*/*/gen_*/eval_merged.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        eval_parts.append(df)
    if not eval_parts:
        findings.append(
            AuditFinding(
                severity="CRITICAL",
                category="summary",
                target=str(run_dir),
                message="No eval_merged.parquet files found for summary recomputation",
                details={},
            )
        )
        return pd.DataFrame(rows)

    eval_df = pd.concat(eval_parts, ignore_index=True)
    required = {"model_name", "branch", "generation", "is_correct", "overall_pedagogical_score", "is_silent_error"}
    missing = required.difference(eval_df.columns)
    if missing:
        findings.append(
            AuditFinding(
                severity="CRITICAL",
                category="summary",
                target=str(run_dir),
                message="eval_merged missing required columns",
                details={"missing_columns": sorted(missing)},
            )
        )
        return pd.DataFrame(rows)

    recomputed = (
        eval_df.groupby(["model_name", "branch", "generation"], as_index=False)
        .agg(
            sample_count_recomputed=("example_id", "count"),
            accuracy_mean_recomputed=("is_correct", "mean"),
            pedagogical_score_mean_recomputed=("overall_pedagogical_score", "mean"),
            silent_error_rate_recomputed=("is_silent_error", "mean"),
        )
    )
    merged = summary.merge(
        recomputed,
        on=["model_name", "branch", "generation"],
        how="outer",
        indicator=True,
    )
    for rec in merged.to_dict(orient="records"):
        ctx = {
            "run_dir": str(run_dir),
            "model_name": rec.get("model_name"),
            "branch": rec.get("branch"),
            "generation": rec.get("generation"),
        }
        if rec.get("_merge") != "both":
            findings.append(
                AuditFinding(
                    severity="CRITICAL",
                    category="summary",
                    target=str(run_dir),
                    message="Summary/recomputed context coverage mismatch",
                    details={**ctx, "merge_status": rec.get("_merge")},
                )
            )
            rows.append({**ctx, "metric": "coverage", "status": "mismatch", "abs_diff": None})
            continue

        for metric, summary_col, recomputed_col in (
            ("sample_count", "sample_count", "sample_count_recomputed"),
            ("accuracy_mean", "accuracy_mean", "accuracy_mean_recomputed"),
            ("pedagogical_score_mean", "pedagogical_score_mean", "pedagogical_score_mean_recomputed"),
            ("silent_error_rate", "silent_error_rate", "silent_error_rate_recomputed"),
        ):
            summary_val = rec.get(summary_col)
            recomputed_val = rec.get(recomputed_col)
            if summary_val is None or recomputed_val is None or pd.isna(summary_val) or pd.isna(recomputed_val):
                status = "missing_value"
                abs_diff = None
            else:
                abs_diff = abs(float(summary_val) - float(recomputed_val))
                status = "ok" if abs_diff <= tol else "mismatch"
            rows.append(
                {
                    **ctx,
                    "metric": metric,
                    "summary_value": summary_val,
                    "recomputed_value": recomputed_val,
                    "abs_diff": abs_diff,
                    "status": status,
                    "tolerance": tol,
                }
            )
            if status == "mismatch":
                findings.append(
                    AuditFinding(
                        severity="CRITICAL",
                        category="summary",
                        target=str(run_dir),
                        message=f"Summary metric mismatch: {metric}",
                        details={**ctx, "summary_value": summary_val, "recomputed_value": recomputed_val, "abs_diff": abs_diff},
                    )
                )

    return pd.DataFrame(rows)


def _analysis_source_checks(
    outputs_root: Path,
    findings: list[AuditFinding],
    analysis_dirs: list[Path] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    checks: list[tuple[str, Path, str]] = []
    candidate_dirs: list[Path]
    if analysis_dirs:
        candidate_dirs = analysis_dirs
    else:
        candidate_dirs = []
        for pattern in (
            "pvf_confirmatory/*",
            "soft_pvf_confirmatory/*",
            "baseline_series/anchoring_confirmatory_append_*",
            "judge_sensitivity/qwen3_confirmatory_*",
        ):
            p = _latest_matching(outputs_root, pattern)
            if p is not None:
                candidate_dirs.append(p)

    for analysis_dir in candidate_dirs:
        norm = str(analysis_dir).replace("\\", "/")
        family: str | None = None
        token: str | None = None
        if "/pvf_confirmatory/" in norm:
            family = "pvf_confirmatory"
            token = "pvf_confirmatory_seed"
        elif "/soft_pvf_confirmatory/" in norm:
            family = "soft_pvf_confirmatory"
            token = "soft_pvf_confirmatory_seed"
        elif "/baseline_series/" in norm and "anchoring_confirmatory_append_" in analysis_dir.name:
            family = "anchoring_confirmatory_series"
            token = "anchoring_confirmatory_append_seed"
        if family is not None and token is not None:
            checks.append((family, analysis_dir, token))

    for family, analysis_dir, expected_token in checks:
        metadata_candidates = sorted((analysis_dir / "tables").glob("*metadata*.json")) + sorted(analysis_dir.glob("tables/*metadata*.json"))
        metadata = _safe_json(metadata_candidates[0]) if metadata_candidates else None
        run_dirs = (metadata or {}).get("run_dirs", [])
        if not isinstance(run_dirs, list):
            run_dirs = []
        has_mixed = False
        for run_ref in run_dirs:
            run_ref_s = str(run_ref)
            ok = expected_token in run_ref_s
            rows.append(
                {
                    "analysis_family": family,
                    "analysis_dir": str(analysis_dir),
                    "source_run_dir": run_ref_s,
                    "expected_token": expected_token,
                    "token_match": ok,
                    "status": "ok" if ok else "mismatch",
                }
            )
            if not ok:
                has_mixed = True
        if has_mixed:
            findings.append(
                AuditFinding(
                    severity="CRITICAL",
                    category="analysis_sources",
                    target=str(analysis_dir),
                    message="Analysis appears to mix wrong run family",
                    details={"expected_token": expected_token, "run_dirs": run_dirs},
                )
            )

    # Judge sensitivity focused check: should use confirmatory training runs only.
    focused_dirs: list[Path] = []
    if analysis_dirs:
        focused_dirs = [p for p in analysis_dirs if "qwen3_confirmatory_" in p.name]
    else:
        focused = _latest_matching(outputs_root, "judge_sensitivity/qwen3_confirmatory_*")
        if focused is not None:
            focused_dirs.append(focused)
    for focused in focused_dirs:
        selected = _safe_csv(focused / "tables" / "selected_sample.csv")
        if selected is not None and not selected.empty and "source_run_dir" in selected.columns:
            allowed_token = "training_confirmatory_anchor20_seed"
            for run_ref in sorted(selected["source_run_dir"].astype(str).unique().tolist()):
                ok = allowed_token in run_ref
                rows.append(
                    {
                        "analysis_family": "judge_sensitivity_qwen_confirmatory",
                        "analysis_dir": str(focused),
                        "source_run_dir": run_ref,
                        "expected_token": allowed_token,
                        "token_match": ok,
                        "status": "ok" if ok else "mismatch",
                    }
                )
                if not ok:
                    findings.append(
                        AuditFinding(
                            severity="HIGH",
                            category="analysis_sources",
                            target=str(focused),
                            message="Qwen confirmatory sample includes non-confirmatory source run",
                            details={"source_run_dir": run_ref},
                        )
                    )

    return pd.DataFrame(rows)


def _pairwise_checks(outputs_root: Path, findings: list[AuditFinding], tol: float = 1e-6) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pair_dir = _latest_matching(outputs_root, "judge_sensitivity/pairwise_confirmatory_*")
    if pair_dir is None:
        return pd.DataFrame(rows)

    tables = pair_dir / "tables"
    selected = _safe_csv(tables / "pairwise_selected_pairs.csv")
    hidden = _safe_csv(tables / "pairwise_hidden_key.csv")
    manual = _safe_csv(tables / "pairwise_manual_audit_template.csv")
    llama = _safe_csv(tables / "pairwise_llama_results.csv")
    qwen = _safe_csv(tables / "pairwise_qwen_results.csv")
    summary = _safe_csv(tables / "pairwise_summary.csv")

    if selected is None or hidden is None or llama is None or qwen is None or summary is None:
        findings.append(
            AuditFinding(
                severity="CRITICAL",
                category="pairwise",
                target=str(pair_dir),
                message="Missing pairwise tables for integrity checks",
                details={},
            )
        )
        return pd.DataFrame(rows)

    selected_ids = set(selected["pair_id"].astype(str).tolist()) if "pair_id" in selected.columns else set()
    hidden_ids = set(hidden["pair_id"].astype(str).tolist()) if "pair_id" in hidden.columns else set()
    llama_ids = set(llama["pair_id"].astype(str).tolist()) if "pair_id" in llama.columns else set()
    qwen_ids = set(qwen["pair_id"].astype(str).tolist()) if "pair_id" in qwen.columns else set()

    for check_name, left, right in (
        ("selected_vs_hidden_pair_ids", selected_ids, hidden_ids),
        ("selected_vs_llama_pair_ids", selected_ids, llama_ids),
        ("selected_vs_qwen_pair_ids", selected_ids, qwen_ids),
    ):
        ok = left == right
        rows.append(
            {
                "pairwise_dir": str(pair_dir),
                "check_name": check_name,
                "status": "ok" if ok else "mismatch",
                "left_count": len(left),
                "right_count": len(right),
            }
        )
        if not ok:
            findings.append(
                AuditFinding(
                    severity="CRITICAL",
                    category="pairwise",
                    target=str(pair_dir),
                    message=f"Pairwise ID mismatch: {check_name}",
                    details={"left_count": len(left), "right_count": len(right)},
                )
            )

    if manual is not None:
        manual_cols = set(manual.columns.astype(str).tolist())
        has_branch_labels = ("A_branch" in manual_cols) or ("B_branch" in manual_cols) or ("branch" in manual_cols)
        rows.append(
            {
                "pairwise_dir": str(pair_dir),
                "check_name": "manual_audit_branch_labels_hidden",
                "status": "ok" if not has_branch_labels else "mismatch",
                "left_count": int(has_branch_labels),
                "right_count": 0,
            }
        )
        if has_branch_labels:
            findings.append(
                AuditFinding(
                    severity="HIGH",
                    category="pairwise",
                    target=str(pair_dir),
                    message="Manual audit template contains explicit branch labels",
                    details={"columns": sorted(manual_cols)},
                )
            )

    # Recompute win rates from row-level bool flags.
    def _mean_bool(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or df.empty:
            return float("nan")
        return float(df[col].astype(bool).mean())

    recomputed = {
        "llama_anchor_win_rate": _mean_bool(llama, "llama_anchor_win"),
        "llama_pure_win_rate": _mean_bool(llama, "llama_pure_win"),
        "llama_tie_rate": _mean_bool(llama, "llama_tie"),
        "qwen_anchor_win_rate": _mean_bool(qwen, "qwen_anchor_win"),
        "qwen_pure_win_rate": _mean_bool(qwen, "qwen_pure_win"),
        "qwen_tie_rate": _mean_bool(qwen, "qwen_tie"),
    }
    summary_row = summary.iloc[0].to_dict()
    for key, recomputed_val in recomputed.items():
        summary_val = summary_row.get(key)
        if summary_val is None or pd.isna(summary_val):
            status = "missing_value"
            abs_diff = None
        else:
            abs_diff = abs(float(summary_val) - float(recomputed_val))
            status = "ok" if abs_diff <= tol else "mismatch"
        rows.append(
            {
                "pairwise_dir": str(pair_dir),
                "check_name": f"summary_{key}",
                "status": status,
                "summary_value": summary_val,
                "recomputed_value": recomputed_val,
                "abs_diff": abs_diff,
            }
        )
        if status == "mismatch":
            findings.append(
                AuditFinding(
                    severity="CRITICAL",
                    category="pairwise",
                    target=str(pair_dir),
                    message=f"Pairwise summary mismatch: {key}",
                    details={"summary_value": summary_val, "recomputed_value": recomputed_val, "abs_diff": abs_diff},
                )
            )

    return pd.DataFrame(rows)


def _pvf_soft_checks(outputs_root: Path, findings: list[AuditFinding], tol: float = 1e-6) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_patterns = [
        "runs/pvf_confirmatory_seed91_*",
        "runs/pvf_confirmatory_seed92_*",
        "runs/pvf_confirmatory_seed93_*",
        "runs/soft_pvf_policy_tuning_*",
        "runs/soft_pvf_shakedown_*",
    ]
    run_dirs = [p for pat in run_patterns if (p := _latest_matching(outputs_root, pat)) is not None]

    for run_dir in run_dirs:
        for step_dir in sorted(run_dir.glob("*/*/gen_*")):
            ctx = _parse_context_from_step_dir(step_dir)
            if ctx is None:
                continue
            model_dir, branch, generation = ctx
            # Hard PVF.
            pvf_report_path = step_dir / "pvf_filter_report.json"
            if pvf_report_path.exists():
                payload = _safe_json(pvf_report_path) or {}
                kept = int(payload.get("kept_count", 0))
                total = int(payload.get("total_candidates", 0))
                keep_rate = float(payload.get("keep_rate", 0.0)) if total > 0 else 0.0
                recomputed = (kept / total) if total > 0 else 0.0
                diff = abs(keep_rate - recomputed)
                status = "ok" if diff <= tol else "mismatch"
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "model_dir": model_dir,
                        "branch": branch,
                        "generation": generation,
                        "method": "pvf",
                        "check_name": "keep_rate_recompute",
                        "status": status,
                        "reported_value": keep_rate,
                        "recomputed_value": recomputed,
                        "abs_diff": diff,
                    }
                )
                if status == "mismatch":
                    findings.append(
                        AuditFinding(
                            severity="HIGH",
                            category="pvf_soft",
                            target=str(pvf_report_path),
                            message="PVF keep_rate mismatch",
                            details={"reported": keep_rate, "recomputed": recomputed, "abs_diff": diff},
                        )
                    )

            # Soft PVF.
            soft_report_path = step_dir / "soft_pvf_report.json"
            soft_decisions_path = step_dir / "soft_pvf_decisions.parquet"
            if soft_report_path.exists() and soft_decisions_path.exists():
                payload = _safe_json(soft_report_path) or {}
                decisions = pd.read_parquet(soft_decisions_path)
                if not decisions.empty:
                    keep_rate_rep = float(payload.get("keep_rate", 0.0))
                    keep_rate_rec = float(decisions["kept"].astype(bool).mean())
                    keep_diff = abs(keep_rate_rep - keep_rate_rec)
                    status = "ok" if keep_diff <= tol else "mismatch"
                    rows.append(
                        {
                            "run_dir": str(run_dir),
                            "model_dir": model_dir,
                            "branch": branch,
                            "generation": generation,
                            "method": "soft_pvf",
                            "check_name": "keep_rate_recompute",
                            "status": status,
                            "reported_value": keep_rate_rep,
                            "recomputed_value": keep_rate_rec,
                            "abs_diff": keep_diff,
                        }
                    )
                    if status == "mismatch":
                        findings.append(
                            AuditFinding(
                                severity="HIGH",
                                category="pvf_soft",
                                target=str(soft_report_path),
                                message="Soft PVF keep_rate mismatch",
                                details={"reported": keep_rate_rep, "recomputed": keep_rate_rec, "abs_diff": keep_diff},
                            )
                        )
                    reason_counts_rep = payload.get("decision_reason_counts", {}) or {}
                    if isinstance(reason_counts_rep, dict):
                        reason_counts_rec = decisions["decision_reason"].value_counts().to_dict()
                        mismatch = False
                        for k, v in reason_counts_rep.items():
                            if int(v) != int(reason_counts_rec.get(k, 0)):
                                mismatch = True
                                break
                        rows.append(
                            {
                                "run_dir": str(run_dir),
                                "model_dir": model_dir,
                                "branch": branch,
                                "generation": generation,
                                "method": "soft_pvf",
                                "check_name": "decision_reason_counts_match",
                                "status": "ok" if not mismatch else "mismatch",
                                "reported_value": len(reason_counts_rep),
                                "recomputed_value": len(reason_counts_rec),
                                "abs_diff": None,
                            }
                        )
                        if mismatch:
                            findings.append(
                                AuditFinding(
                                    severity="MEDIUM",
                                    category="pvf_soft",
                                    target=str(soft_report_path),
                                    message="Soft PVF decision_reason_counts mismatch",
                                    details={"reported": reason_counts_rep, "recomputed": reason_counts_rec},
                                )
                            )
    return pd.DataFrame(rows)


def _render_report_md(
    *,
    findings: list[AuditFinding],
    run_manifest_df: pd.DataFrame,
    row_counts_df: pd.DataFrame,
    summary_checks_df: pd.DataFrame,
    analysis_source_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    pvf_soft_df: pd.DataFrame,
) -> str:
    by_severity: dict[str, int] = {k: 0 for k in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    for f in findings:
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

    lines: list[str] = []
    lines.append("# Artifact Integrity Audit Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat()}")
    lines.append("- Scope: audit only (no rerun of generation/judge/training)")
    lines.append("")
    lines.append("## Severity Summary")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        lines.append(f"- {sev}: {by_severity.get(sev, 0)}")
    lines.append("")
    lines.append("## Table Stats")
    lines.append(f"- run manifest rows: {len(run_manifest_df)}")
    lines.append(f"- row-count rows: {len(row_counts_df)}")
    lines.append(f"- summary checks rows: {len(summary_checks_df)}")
    lines.append(f"- analysis-source checks rows: {len(analysis_source_df)}")
    lines.append(f"- pairwise checks rows: {len(pairwise_df)}")
    lines.append(f"- pvf/soft checks rows: {len(pvf_soft_df)}")
    lines.append("")

    if findings:
        lines.append("## Findings")
        for f in findings:
            lines.append(f"- [{f.severity}] {f.category} :: {f.target} :: {f.message}")
            if f.details:
                lines.append(f"  - details: {json.dumps(f.details, ensure_ascii=False)}")
    else:
        lines.append("## Findings")
        lines.append("- No findings.")
    lines.append("")

    critical_or_high = [f for f in findings if f.severity in {"CRITICAL", "HIGH"}]
    lines.append("## Integrity Verdict")
    if critical_or_high:
        lines.append("- Key claims are NOT fully reliable until CRITICAL/HIGH issues are resolved or excluded.")
    else:
        lines.append("- Key claims are numerically reproducible under current audit checks.")
    lines.append("- Evaluation mode should remain explicitly marked as inference_recycling_only / feasibility where applicable.")
    return "\n".join(lines) + "\n"


def run_artifact_integrity_audit(
    *,
    outputs_root: Path,
    run_dirs: list[Path] | None = None,
    analysis_dirs: list[Path] | None = None,
) -> AuditArtifacts:
    targets = discover_default_targets(outputs_root)
    selected_run_dirs = run_dirs if run_dirs is not None else targets["run_dirs"]
    selected_analysis_dirs = analysis_dirs if analysis_dirs is not None else targets["analysis_dirs"]

    findings: list[AuditFinding] = []
    manifest_tables: list[pd.DataFrame] = []
    row_count_tables: list[pd.DataFrame] = []
    summary_check_tables: list[pd.DataFrame] = []

    for run_dir in selected_run_dirs:
        if not run_dir.exists():
            findings.append(
                AuditFinding(
                    severity="CRITICAL",
                    category="run_dir",
                    target=str(run_dir),
                    message="Configured run_dir does not exist",
                    details={},
                )
            )
            continue
        manifest_tables.append(_collect_manifest_status(run_dir, findings))
        row_count_tables.append(_collect_row_counts_for_run(run_dir, findings))
        summary_check_tables.append(_recompute_run_summary_checks(run_dir, findings))

    run_manifest_df = (
        pd.concat(manifest_tables, ignore_index=True)
        if manifest_tables
        else pd.DataFrame(columns=["run_dir", "scope", "model_dir", "branch", "generation", "stage_name", "status"])
    )
    row_counts_df = pd.concat(row_count_tables, ignore_index=True) if row_count_tables else pd.DataFrame()
    summary_checks_df = (
        pd.concat(summary_check_tables, ignore_index=True)
        if summary_check_tables
        else pd.DataFrame(columns=["run_dir", "model_name", "branch", "generation", "metric", "status"])
    )

    # analysis checks that are based on latest directories under outputs root
    analysis_source_df = _analysis_source_checks(outputs_root, findings, selected_analysis_dirs)
    pairwise_df = _pairwise_checks(outputs_root, findings)
    pvf_soft_df = _pvf_soft_checks(outputs_root, findings)

    out_dir = outputs_root / "audits" / f"artifact_integrity_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_run_manifest_status_csv = out_dir / "audit_run_manifest_status.csv"
    audit_row_counts_csv = out_dir / "audit_row_counts.csv"
    audit_summary_recompute_checks_csv = out_dir / "audit_summary_recompute_checks.csv"
    audit_analysis_source_checks_csv = out_dir / "audit_analysis_source_checks.csv"
    audit_pairwise_checks_csv = out_dir / "audit_pairwise_checks.csv"
    audit_pvf_soft_checks_csv = out_dir / "audit_pvf_soft_checks.csv"
    audit_findings_json = out_dir / "audit_findings.json"
    audit_report_md = out_dir / "audit_report.md"

    run_manifest_df.to_csv(audit_run_manifest_status_csv, index=False)
    row_counts_df.to_csv(audit_row_counts_csv, index=False)
    summary_checks_df.to_csv(audit_summary_recompute_checks_csv, index=False)
    analysis_source_df.to_csv(audit_analysis_source_checks_csv, index=False)
    pairwise_df.to_csv(audit_pairwise_checks_csv, index=False)
    pvf_soft_df.to_csv(audit_pvf_soft_checks_csv, index=False)

    findings_payload = {
        "generated_at": datetime.now().isoformat(),
        "outputs_root": str(outputs_root),
        "selected_run_dirs": [str(p) for p in selected_run_dirs],
        "selected_analysis_dirs": [str(p) for p in selected_analysis_dirs],
        "finding_count": len(findings),
        "findings": [asdict(f) for f in findings],
    }
    audit_findings_json.write_text(json.dumps(findings_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = _render_report_md(
        findings=findings,
        run_manifest_df=run_manifest_df,
        row_counts_df=row_counts_df,
        summary_checks_df=summary_checks_df,
        analysis_source_df=analysis_source_df,
        pairwise_df=pairwise_df,
        pvf_soft_df=pvf_soft_df,
    )
    audit_report_md.write_text(report_md, encoding="utf-8")

    return AuditArtifacts(
        out_dir=out_dir,
        audit_run_manifest_status_csv=audit_run_manifest_status_csv,
        audit_row_counts_csv=audit_row_counts_csv,
        audit_summary_recompute_checks_csv=audit_summary_recompute_checks_csv,
        audit_analysis_source_checks_csv=audit_analysis_source_checks_csv,
        audit_pairwise_checks_csv=audit_pairwise_checks_csv,
        audit_pvf_soft_checks_csv=audit_pvf_soft_checks_csv,
        audit_findings_json=audit_findings_json,
        audit_report_md=audit_report_md,
    )
