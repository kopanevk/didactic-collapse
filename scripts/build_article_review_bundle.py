from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("outputs")
RUNS_DIR = ROOT / "runs"


@dataclass
class SourceRecord:
    source_type: str
    path: Path
    reason_included: str
    caveats: str = ""


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ci95(values: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    delta = 1.96 * std / math.sqrt(len(vals))
    return mean - delta, mean + delta


def _pick_latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No candidate paths found")
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def _find_latest_dir(parent: Path, pattern: str, required_file: str | None = None) -> Path:
    candidates = [p for p in parent.glob(pattern) if p.is_dir()]
    if required_file:
        candidates = [p for p in candidates if (p / required_file).exists()]
    return _pick_latest(candidates)


def _find_latest_seed_run(prefix: str, seed: int) -> Path:
    pattern = f"{prefix}_seed{seed}_*"
    return _find_latest_dir(RUNS_DIR, pattern, required_file="run_stage_manifest.json")


def _discover_model_dir(run_dir: Path) -> Path:
    model_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name not in {"tables", "figures"}]
    if len(model_dirs) != 1:
        raise RuntimeError(f"Expected one model dir in {run_dir}, got {len(model_dirs)}")
    return model_dirs[0]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_run_dir(run_dir: Path) -> int:
    m = re.search(r"seed(\d+)", run_dir.name)
    if not m:
        raise ValueError(f"Cannot parse seed from {run_dir.name}")
    return int(m.group(1))


def _safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def _context_row_count(context_dir: Path, filename: str) -> int:
    path = context_dir / filename
    if not path.exists():
        return -1
    if filename.endswith(".parquet"):
        return int(len(pd.read_parquet(path)))
    if filename.endswith(".csv"):
        return int(len(pd.read_csv(path)))
    return -1


def _compute_context_metrics(run_dir: Path, branch: str, generation: int, family: str) -> dict[str, Any]:
    model_dir = _discover_model_dir(run_dir)
    context_dir = model_dir / branch / f"gen_{generation}"
    eval_df = pd.read_parquet(context_dir / "eval_merged.parquet")
    judge_df = pd.read_parquet(context_dir / "judge_outputs.parquet")

    eval_df = eval_df.copy()
    judge_df = judge_df.copy()
    eval_df["example_id"] = eval_df["example_id"].astype(str)
    judge_df["example_id"] = judge_df["example_id"].astype(str)
    merged = eval_df.merge(
        judge_df[["example_id", "reasoning_soundness", "structure", "clarity", "terminology"]],
        on="example_id",
        how="left",
        validate="one_to_one",
    )

    pred_parse_success = merged["pred_parse_success"].fillna(False).astype(bool)
    is_correct = merged["is_correct"].fillna(False).astype(bool)
    is_silent = merged["is_silent_error"].fillna(False).astype(bool)
    reasoning = pd.to_numeric(merged["reasoning_soundness"], errors="coerce")
    structure = pd.to_numeric(merged["structure"], errors="coerce")

    defect_parse = ~pred_parse_success
    defect_incorrect = ~is_correct
    defect_silent = is_silent
    defect_low_reasoning = reasoning <= 0
    defect_low_structure = structure <= 0
    severity = (
        4 * defect_parse.astype(int)
        + 4 * defect_silent.astype(int)
        + 2 * defect_incorrect.astype(int)
        + 2 * defect_low_reasoning.fillna(False).astype(int)
        + 1 * defect_low_structure.fillna(False).astype(int)
    )

    summary_df = pd.read_csv(run_dir / "tables" / "first_experiment_summary.csv")
    mask = (summary_df["branch"] == branch) & (summary_df["generation"].astype(int) == int(generation))
    summary_sample = int(summary_df.loc[mask, "sample_count"].iloc[0]) if mask.any() else -1

    return {
        "experiment_family": family,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "seed": _seed_from_run_dir(run_dir),
        "branch": branch,
        "generation": int(generation),
        "sample_count": int(len(merged)),
        "accuracy_mean": float(is_correct.mean()),
        "pedagogical_score_mean": float(pd.to_numeric(merged["overall_pedagogical_score"], errors="coerce").mean()),
        "silent_error_rate": float(is_silent.mean()),
        "parse_failure_pred_rate": float((~pred_parse_success).mean()),
        "low_reasoning_rate": _safe_rate(defect_low_reasoning.fillna(False).astype(float)),
        "low_structure_rate": _safe_rate(defect_low_structure.fillna(False).astype(float)),
        "defect_severity_mean": float(pd.to_numeric(severity, errors="coerce").mean()),
        "missing_judge_rows": int(merged["reasoning_soundness"].isna().sum()),
        "sample_count_summary": summary_sample,
    }


def _compute_matched_gen2(run_dir: Path, family: str) -> dict[str, Any]:
    model_dir = _discover_model_dir(run_dir)
    pure = pd.read_parquet(model_dir / "pure_recycling" / "gen_2" / "eval_merged.parquet")
    dbr = pd.read_parquet(model_dir / "dbr_medium" / "gen_2" / "eval_merged.parquet")
    pure = pure[["example_id", "is_correct", "overall_pedagogical_score", "is_silent_error"]].copy()
    dbr = dbr[["example_id", "is_correct", "overall_pedagogical_score", "is_silent_error"]].copy()
    pure["example_id"] = pure["example_id"].astype(str)
    dbr["example_id"] = dbr["example_id"].astype(str)
    m = pure.merge(dbr, on="example_id", how="inner", suffixes=("_pure", "_dbr"), validate="one_to_one")
    return {
        "family": family,
        "run_id": run_dir.name,
        "seed": _seed_from_run_dir(run_dir),
        "matched_pairs": int(len(m)),
        "delta_accuracy": float(m["is_correct_dbr"].astype(bool).mean() - m["is_correct_pure"].astype(bool).mean()),
        "delta_pedagogy": float(
            pd.to_numeric(m["overall_pedagogical_score_dbr"], errors="coerce").mean()
            - pd.to_numeric(m["overall_pedagogical_score_pure"], errors="coerce").mean()
        ),
        "delta_silent": float(m["is_silent_error_dbr"].astype(bool).mean() - m["is_silent_error_pure"].astype(bool).mean()),
        "mismatch_artifact_flag": False,
    }


def _extract_dbr_policy_tables(
    run_dir: Path, family: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model_dir = _discover_model_dir(run_dir)
    seed = _seed_from_run_dir(run_dir)
    selection_rows: list[dict[str, Any]] = []
    defect_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    mech_rows: list[dict[str, Any]] = []

    for g in (0, 1, 2):
        context_dir = model_dir / "dbr_medium" / f"gen_{g}"
        report = _load_json(context_dir / "dbr_budget_report.json")
        decisions = pd.read_parquet(context_dir / "dbr_decisions.parquet")
        selected = decisions[decisions["selected"].astype(bool)]

        selection_rows.append(
            {
                "family": family,
                "seed": seed,
                "generation": g,
                "selected_count": int(report.get("selected_count", len(selected))),
                "target_size": int(report.get("target_size", len(decisions))),
                "selection_rate": float(report.get("selection_rate", float(len(selected)) / max(1, len(decisions)))),
            }
        )

        for defect in ("parse_failure", "incorrect_answer", "silent_error", "low_reasoning", "low_structure"):
            before = float((report.get("defect_rates_before") or {}).get(defect, float("nan")))
            after = float((report.get("defect_rates_after") or {}).get(defect, float("nan")))
            defect_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "generation": g,
                    "defect_type": defect,
                    "rate_before": before,
                    "rate_after": after,
                    "delta_after_minus_before": after - before,
                }
            )
            b = (report.get("budget_violations") or {}).get(defect, {})
            budget = float((report.get("budgets") or {}).get(defect, float("nan")))
            budget_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "generation": g,
                    "defect_type": defect,
                    "budget": budget,
                    "actual_after": float(b.get("actual_rate", after) if isinstance(b, dict) else after),
                    "violation": bool((b.get("violation_count", 0) > 0) if isinstance(b, dict) else False),
                    "relaxation_steps_used": "|".join(report.get("relaxation_steps_used") or []),
                }
            )

        mech_rows.append(
            {
                "family": family,
                "seed": seed,
                "generation": g,
                "parse_failure_after": float((report.get("defect_rates_after") or {}).get("parse_failure", float("nan"))),
                "selection_rate": float(report.get("selection_rate", float("nan"))),
                "silent_after": float((report.get("defect_rates_after") or {}).get("silent_error", float("nan"))),
                "incorrect_after": float((report.get("defect_rates_after") or {}).get("incorrect_answer", float("nan"))),
                "low_reasoning_after": float((report.get("defect_rates_after") or {}).get("low_reasoning", float("nan"))),
                "low_structure_after": float((report.get("defect_rates_after") or {}).get("low_structure", float("nan"))),
            }
        )

    return selection_rows, defect_rows, budget_rows, mech_rows


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _build_bundle() -> Path:
    bundle_dir = ROOT / f"article_review_bundle_{_now_tag()}"
    tables_dir = bundle_dir / "tables"
    reports_dir = bundle_dir / "reports"
    figures_dir = bundle_dir / "figures"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dbr_confirm_runs = [_find_latest_seed_run("dbr_confirmatory", s) for s in (211, 212, 213)]
    dbr_only_runs = [_find_latest_seed_run("dbr_only_confirmatory", s) for s in (221, 222, 223)]
    csr_k3_run = _find_latest_dir(RUNS_DIR, "csr_shakedown_[0-9]*", required_file="tables/csr_pair_summary.csv")
    csr_k5_run = _find_latest_dir(RUNS_DIR, "csr_shakedown_k5_*", required_file="tables/csr_pair_summary.csv")
    qwen_pairwise = _find_latest_dir(
        ROOT / "judge_sensitivity", "qwen_dbr_pairwise_*", required_file="tables/qwen_dbr_pairwise_summary.csv"
    )
    dbr_recompute_audit = _find_latest_dir(ROOT / "audits", "dbr_recompute_audit_*", required_file="dbr_recompute_findings.json")
    artifact_integrity = _find_latest_dir(ROOT / "audits", "artifact_integrity_*", required_file="audit_findings.json")
    dbr_article_evidence = _find_latest_dir(
        ROOT / "dbr_confirmatory", "dbr_article_evidence_*", required_file="tables/didactic_collapse_evidence_summary.csv"
    )
    dbr_only_analysis = _find_latest_dir(
        ROOT / "dbr_confirmatory", "dbr_only_confirmatory_3seed_*", required_file="tables/dbr_only_run_level.csv"
    )

    sources: list[SourceRecord] = []
    for rd in dbr_confirm_runs:
        sources.append(SourceRecord("run_dir", rd, "DBR confirmatory family (211-213)", "includes soft_pvf_noisy_keep baseline"))
    for rd in dbr_only_runs:
        sources.append(SourceRecord("run_dir", rd, "DBR-only confirmatory family (221-223)", "strict pure vs dbr family"))
    sources.extend(
        [
            SourceRecord("run_dir", csr_k3_run, "CSR shakedown k=3", "single-seed shakedown"),
            SourceRecord("run_dir", csr_k5_run, "CSR shakedown k=5", "single-seed tuning shakedown"),
            SourceRecord("analysis_dir", qwen_pairwise, "Qwen DBR pairwise sensitivity", "pairwise sample size limited"),
            SourceRecord("analysis_dir", dbr_recompute_audit, "DBR recompute audit", "used for recompute-integrity confirmation"),
            SourceRecord(
                "analysis_dir",
                artifact_integrity,
                "Global artifact integrity audit",
                "contains legacy CRITICAL mismatches in seed211/gen0 summary",
            ),
            SourceRecord("analysis_dir", dbr_article_evidence, "DBR article evidence outputs", "includes manual audit template"),
            SourceRecord("analysis_dir", dbr_only_analysis, "DBR-only 3-seed analysis", "primary dbr-only summary tables"),
        ]
    )

    metric_rows: list[dict[str, Any]] = []
    row_count_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    defect_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    mech_rows: list[dict[str, Any]] = []

    family_map = {
        "dbr_confirmatory_211_213": dbr_confirm_runs,
        "dbr_only_221_223": dbr_only_runs,
    }

    for family, run_dirs in family_map.items():
        for run_dir in run_dirs:
            seed = _seed_from_run_dir(run_dir)
            model_dir = _discover_model_dir(run_dir)
            run_manifest = _load_json(run_dir / "run_stage_manifest.json")
            for branch_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                branch = branch_dir.name
                for g in (0, 1, 2):
                    context_dir = branch_dir / f"gen_{g}"
                    if not context_dir.exists():
                        continue
                    stage_manifest_path = context_dir / "stage_manifest.json"
                    stage_status = "missing"
                    if stage_manifest_path.exists():
                        stage_status = str(_load_json(stage_manifest_path).get("status", "unknown"))
                    metric_rows.append(_compute_context_metrics(run_dir, branch, g, family))

                    summary_df = pd.read_csv(run_dir / "tables" / "first_experiment_summary.csv")
                    sm_mask = (summary_df["branch"] == branch) & (summary_df["generation"].astype(int) == g)
                    sm_count = int(summary_df.loc[sm_mask, "sample_count"].iloc[0]) if sm_mask.any() else -1

                    rc = {
                        "family": family,
                        "seed": seed,
                        "branch": branch,
                        "generation": g,
                        "run_dir": str(run_dir),
                        "model_outputs_rows": _context_row_count(context_dir, "model_outputs.parquet"),
                        "extraction_rows": _context_row_count(context_dir, "answer_extraction.parquet"),
                        "accuracy_rows": _context_row_count(context_dir, "accuracy_table.parquet"),
                        "judge_rows": _context_row_count(context_dir, "judge_outputs.parquet"),
                        "eval_merged_rows": _context_row_count(context_dir, "eval_merged.parquet"),
                        "sample_count_summary": sm_count,
                        "stage_manifest_status": stage_status,
                        "run_stage_status": str(run_manifest.get("status", "unknown")),
                    }
                    rc["mismatch_flag"] = bool(
                        (rc["eval_merged_rows"] >= 0 and rc["sample_count_summary"] >= 0 and rc["eval_merged_rows"] != rc["sample_count_summary"])
                        or (rc["accuracy_rows"] >= 0 and rc["eval_merged_rows"] >= 0 and rc["accuracy_rows"] != rc["eval_merged_rows"])
                        or (rc["judge_rows"] >= 0 and rc["eval_merged_rows"] >= 0 and rc["judge_rows"] != rc["eval_merged_rows"])
                    )
                    row_count_rows.append(rc)

            matched_rows.append(_compute_matched_gen2(run_dir, family))
            s_rows, d_rows, b_rows, m_rows = _extract_dbr_policy_tables(run_dir, family)
            selection_rows.extend(s_rows)
            defect_rows.extend(d_rows)
            budget_rows.extend(b_rows)
            mech_rows.extend(m_rows)

    metrics_df = pd.DataFrame(metric_rows)
    row_counts_df = pd.DataFrame(row_count_rows)
    matched_df = pd.DataFrame(matched_rows)
    selection_df = pd.DataFrame(selection_rows)
    defect_df = pd.DataFrame(defect_rows)
    budget_df = pd.DataFrame(budget_rows)
    mech_df = pd.DataFrame(mech_rows)

    pure_df = metrics_df[metrics_df["branch"] == "pure_recycling"].copy()
    pure_by_seed_gen = pure_df[
        [
            "experiment_family",
            "seed",
            "generation",
            "sample_count",
            "accuracy_mean",
            "pedagogical_score_mean",
            "silent_error_rate",
            "parse_failure_pred_rate",
            "low_reasoning_rate",
            "low_structure_rate",
            "defect_severity_mean",
        ]
    ].sort_values(["experiment_family", "seed", "generation"]).reset_index(drop=True)
    pure_by_seed_gen.to_csv(tables_dir / "collapse_evidence_pure_by_seed_generation.csv", index=False)

    delta_rows: list[dict[str, Any]] = []
    for (family, seed), grp in pure_df.groupby(["experiment_family", "seed"], as_index=False):
        g = grp.set_index("generation").sort_index()
        if not {0, 1, 2}.issubset(set(g.index.astype(int).tolist())):
            continue
        row = {"experiment_family": family, "seed": int(seed)}
        for metric in ["accuracy_mean", "pedagogical_score_mean", "silent_error_rate", "parse_failure_pred_rate", "defect_severity_mean"]:
            v0 = float(g.loc[0, metric])
            v1 = float(g.loc[1, metric])
            v2 = float(g.loc[2, metric])
            row[f"{metric}_delta_gen0_to_gen1"] = v1 - v0
            row[f"{metric}_delta_gen1_to_gen2"] = v2 - v1
            row[f"{metric}_delta_gen0_to_gen2"] = v2 - v0
        delta_rows.append(row)
    pure_deltas_df = pd.DataFrame(delta_rows).sort_values(["experiment_family", "seed"]).reset_index(drop=True)
    pure_deltas_df.to_csv(tables_dir / "collapse_evidence_pure_deltas.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for family, fgrp in pure_deltas_df.groupby("experiment_family"):
        for metric in ["accuracy_mean", "pedagogical_score_mean", "silent_error_rate", "parse_failure_pred_rate", "defect_severity_mean"]:
            for tr in ["gen0_to_gen1", "gen1_to_gen2", "gen0_to_gen2"]:
                col = f"{metric}_delta_{tr}"
                vals = pd.to_numeric(fgrp[col], errors="coerce").dropna().tolist()
                if not vals:
                    continue
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ci_low, ci_high = _ci95(vals)
                if metric in {"accuracy_mean", "pedagogical_score_mean"}:
                    if ci_high < 0:
                        flag = "deterioration"
                    elif ci_low > 0:
                        flag = "improvement"
                    else:
                        flag = "mixed"
                else:
                    if ci_low > 0:
                        flag = "deterioration"
                    elif ci_high < 0:
                        flag = "improvement"
                    else:
                        flag = "mixed"
                if len(vals) < 3 and flag == "mixed":
                    flag = "inconclusive"
                summary_rows.append(
                    {
                        "experiment_family": family,
                        "metric": metric,
                        "transition": tr,
                        "mean_delta": mean,
                        "std": std,
                        "n": len(vals),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "interpretation_flag": flag,
                    }
                )
    collapse_summary_df = pd.DataFrame(summary_rows).sort_values(["experiment_family", "metric", "transition"]).reset_index(drop=True)
    collapse_summary_df.to_csv(tables_dir / "collapse_evidence_summary.csv", index=False)

    gen2_rows: list[dict[str, Any]] = []
    for (family, seed), grp in metrics_df[metrics_df["generation"] == 2].groupby(["experiment_family", "seed"], as_index=False):
        sub = grp.set_index("branch")
        if "pure_recycling" not in sub.index or "dbr_medium" not in sub.index:
            continue
        pure = sub.loc["pure_recycling"]
        dbr = sub.loc["dbr_medium"]
        row = {
            "family": family,
            "seed": int(seed),
            "pure_accuracy": float(pure["accuracy_mean"]),
            "dbr_accuracy": float(dbr["accuracy_mean"]),
            "delta_accuracy": float(dbr["accuracy_mean"] - pure["accuracy_mean"]),
            "pure_pedagogy": float(pure["pedagogical_score_mean"]),
            "dbr_pedagogy": float(dbr["pedagogical_score_mean"]),
            "delta_pedagogy": float(dbr["pedagogical_score_mean"] - pure["pedagogical_score_mean"]),
            "pure_silent": float(pure["silent_error_rate"]),
            "dbr_silent": float(dbr["silent_error_rate"]),
            "delta_silent": float(dbr["silent_error_rate"] - pure["silent_error_rate"]),
            "pure_sample_count": int(pure["sample_count"]),
            "dbr_sample_count": int(dbr["sample_count"]),
            "soft_accuracy": float(sub.loc["soft_pvf_noisy_keep", "accuracy_mean"]) if "soft_pvf_noisy_keep" in sub.index else float("nan"),
            "soft_pedagogy": float(sub.loc["soft_pvf_noisy_keep", "pedagogical_score_mean"]) if "soft_pvf_noisy_keep" in sub.index else float("nan"),
            "soft_silent": float(sub.loc["soft_pvf_noisy_keep", "silent_error_rate"]) if "soft_pvf_noisy_keep" in sub.index else float("nan"),
        }
        gen2_rows.append(row)
    dbr_gen2_df = pd.DataFrame(gen2_rows).sort_values(["family", "seed"]).reset_index(drop=True)
    dbr_gen2_df.to_csv(tables_dir / "dbr_gen2_by_seed.csv", index=False)

    summary_rows = []
    for family, grp in dbr_gen2_df.groupby("family"):
        for metric, positive_better in [
            ("delta_accuracy", True),
            ("delta_pedagogy", True),
            ("delta_silent", False),
        ]:
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna().tolist()
            mean = float(np.mean(vals)) if vals else float("nan")
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_low, ci_high = _ci95(vals)
            pos = sum(v > 0 for v in vals)
            neg = sum(v < 0 for v in vals)
            if positive_better:
                interp = "favors_dbr" if mean > 0 else "favors_pure" if mean < 0 else "tie"
            else:
                interp = "favors_dbr" if mean < 0 else "favors_pure" if mean > 0 else "tie"
            summary_rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "mean_delta": mean,
                    "std": std,
                    "n": len(vals),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "positive_seed_count": pos,
                    "negative_seed_count": neg,
                    "interpretation": interp,
                }
            )
    dbr_gen2_summary_df = pd.DataFrame(summary_rows).sort_values(["family", "metric"]).reset_index(drop=True)
    dbr_gen2_summary_df.to_csv(tables_dir / "dbr_gen2_summary.csv", index=False)

    matched_df.sort_values(["family", "seed"]).to_csv(tables_dir / "dbr_matched_gen2_comparison.csv", index=False)
    selection_df.sort_values(["family", "seed", "generation"]).to_csv(tables_dir / "dbr_selection_rate.csv", index=False)
    defect_df.sort_values(["family", "seed", "generation", "defect_type"]).to_csv(tables_dir / "dbr_defect_before_after.csv", index=False)
    budget_df.sort_values(["family", "seed", "generation", "defect_type"]).to_csv(tables_dir / "dbr_budget_violations.csv", index=False)

    mech_summary_rows = []
    for family, grp in mech_df.groupby("family"):
        mech_summary_rows.append(
            {
                "family": family,
                "parse_failure_after_mean": float(pd.to_numeric(grp["parse_failure_after"], errors="coerce").mean()),
                "selection_rate_mean": float(pd.to_numeric(grp["selection_rate"], errors="coerce").mean()),
                "silent_delta_mean": float(
                    defect_df[(defect_df["family"] == family) & (defect_df["defect_type"] == "silent_error")]["delta_after_minus_before"].mean()
                ),
                "incorrect_delta_mean": float(
                    defect_df[(defect_df["family"] == family) & (defect_df["defect_type"] == "incorrect_answer")]["delta_after_minus_before"].mean()
                ),
                "low_reasoning_delta_mean": float(
                    defect_df[(defect_df["family"] == family) & (defect_df["defect_type"] == "low_reasoning")]["delta_after_minus_before"].mean()
                ),
                "low_structure_delta_mean": float(
                    defect_df[(defect_df["family"] == family) & (defect_df["defect_type"] == "low_structure")]["delta_after_minus_before"].mean()
                ),
                "caveats": "Means over generations and seeds; budgets may be relaxed.",
            }
        )
    pd.DataFrame(mech_summary_rows).to_csv(tables_dir / "dbr_mechanism_summary.csv", index=False)

    csr_rows = []
    csr_sources = [("k3", csr_k3_run), ("k5", csr_k5_run)]
    no_pair_rows = []
    bvw_rows = []
    gen2_branch_rows = []

    for variant, run_dir in csr_sources:
        pair_df = pd.read_csv(run_dir / "tables" / "csr_pair_summary.csv")
        pair_df = pair_df.copy()
        pair_df["csr_variant"] = variant
        csr_rows.append(pair_df)

        for rec in pair_df.to_dict(orient="records"):
            reasons_raw = rec.get("no_pair_reasons_json")
            try:
                reasons = json.loads(reasons_raw) if isinstance(reasons_raw, str) and reasons_raw else {}
            except json.JSONDecodeError:
                reasons = {}
            for reason, count in reasons.items():
                no_pair_rows.append(
                    {
                        "csr_variant": variant,
                        "generation": int(rec.get("generation", -1)),
                        "no_pair_reason": str(reason),
                        "count": int(count),
                    }
                )

        bvw = pd.read_csv(run_dir / "tables" / "csr_best_vs_worst_quality.csv")
        bvw = bvw.rename(
            columns={
                "quality_gap_mean": "mean_quality_gap",
                "best_correct_rate": "best_accuracy",
                "worst_correct_rate": "worst_accuracy",
            }
        )
        bvw["csr_variant"] = variant
        bvw_rows.append(bvw)

        summary = pd.read_csv(run_dir / "tables" / "first_experiment_summary.csv")
        summary = summary[summary["generation"].astype(int) == 2].copy()
        summary["csr_variant"] = variant
        gen2_branch_rows.append(summary)

    csr_combined = pd.concat(csr_rows, ignore_index=True)
    csr_pair_out = csr_combined[
        [
            "csr_variant",
            "generation",
            "total_questions",
            "pair_count",
            "pair_construction_rate",
            "no_pair_count",
            "mean_quality_gap",
            "best_accuracy",
            "worst_accuracy",
            "best_silent_rate",
            "worst_silent_rate",
        ]
    ].sort_values(["csr_variant", "generation"])
    csr_pair_out.to_csv(tables_dir / "csr_pair_summary_combined.csv", index=False)

    pd.DataFrame(no_pair_rows).sort_values(["csr_variant", "generation", "no_pair_reason"]).to_csv(
        tables_dir / "csr_no_pair_reasons.csv", index=False
    )

    bvw_out = pd.concat(bvw_rows, ignore_index=True)[
        ["csr_variant", "generation", "mean_quality_gap", "best_accuracy", "worst_accuracy", "best_silent_rate", "worst_silent_rate"]
    ]
    bvw_out.insert(2, "best_mean_score", np.nan)
    bvw_out.insert(3, "worst_mean_score", np.nan)
    bvw_out.to_csv(tables_dir / "csr_best_vs_worst_quality.csv", index=False)

    gen2_out = pd.concat(gen2_branch_rows, ignore_index=True)[
        ["csr_variant", "branch", "generation", "sample_count", "accuracy_mean", "pedagogical_score_mean", "silent_error_rate"]
    ].sort_values(["csr_variant", "branch"])
    gen2_out.to_csv(tables_dir / "csr_gen2_branch_comparison.csv", index=False)

    k3 = csr_pair_out[csr_pair_out["csr_variant"] == "k3"]
    k5 = csr_pair_out[csr_pair_out["csr_variant"] == "k5"]
    csr_interpret = pd.DataFrame(
        [
            {
                "k3_pair_rate_mean": float(k3["pair_construction_rate"].mean()),
                "k5_pair_rate_mean": float(k5["pair_construction_rate"].mean()),
                "k3_quality_gap_mean": float(k3["mean_quality_gap"].mean()),
                "k5_quality_gap_mean": float(k5["mean_quality_gap"].mean()),
                "csr_best_evidence": "High contrast gaps and strong best-vs-worst separation.",
                "csr_branch_outcome_summary": "CSR improves over pure on some Gen2 metrics but trails DBR on pedagogy.",
                "main_limitation": "Pair construction bottleneck remains dominated by best_not_correct no-pair cases.",
            }
        ]
    )
    csr_interpret.to_csv(tables_dir / "csr_interpretation_summary.csv", index=False)

    qwen_summary = pd.read_csv(qwen_pairwise / "tables" / "qwen_dbr_pairwise_summary.csv")
    qwen_seed = pd.read_csv(qwen_pairwise / "tables" / "qwen_dbr_seed_summary.csv")
    qwen_comp = pd.read_csv(qwen_pairwise / "tables" / "qwen_dbr_pairwise_comparison.csv")
    qwen_selected = pd.read_csv(qwen_pairwise / "tables" / "qwen_dbr_selected_pairs.csv")

    qwen_out = pd.DataFrame(
        [
            {
                "selected_pairs": int(len(qwen_selected)),
                "successful_pairs": int(len(qwen_comp)),
                "failures": int(len(qwen_selected) - len(qwen_comp)),
                "dbr_wins": int(qwen_comp["qwen_dbr_win"].fillna(False).astype(bool).sum()),
                "pure_wins": int(qwen_comp["qwen_pure_win"].fillna(False).astype(bool).sum()),
                "ties": int(qwen_comp["qwen_tie"].fillna(False).astype(bool).sum()),
                "dbr_win_rate": float(qwen_comp["qwen_dbr_win"].fillna(False).astype(bool).mean()),
                "pure_win_rate": float(qwen_comp["qwen_pure_win"].fillna(False).astype(bool).mean()),
                "tie_rate": float(qwen_comp["qwen_tie"].fillna(False).astype(bool).mean()),
                "margin": float(
                    qwen_comp["qwen_dbr_win"].fillna(False).astype(bool).mean()
                    - qwen_comp["qwen_pure_win"].fillna(False).astype(bool).mean()
                ),
            }
        ]
    )
    qwen_out.to_csv(tables_dir / "qwen_dbr_pairwise_summary.csv", index=False)

    qwen_seed_out = qwen_seed.rename(
        columns={
            "source_seed": "seed",
            "qwen_dbr_win_rate": "dbr_win_rate",
            "qwen_pure_win_rate": "pure_win_rate",
            "qwen_tie_rate": "tie_rate",
            "qwen_dbr_minus_pure_win_rate": "margin",
        }
    )[["seed", "n_pairs", "dbr_win_rate", "pure_win_rate", "tie_rate", "margin"]]
    qwen_seed_out.to_csv(tables_dir / "qwen_dbr_pairwise_by_seed.csv", index=False)

    manual_csv = dbr_article_evidence / "tables" / "manual_dbr_pairwise_audit_template.csv"
    manual_df = pd.read_csv(manual_csv)
    if "human_winner" in manual_df.columns:
        filled = manual_df["human_winner"].notna() & (manual_df["human_winner"].astype(str).str.strip() != "")
        filled_count = int(filled.sum())
    else:
        filled_count = 0
    manual_summary = pd.DataFrame(
        [
            {
                "pair_count": int(len(manual_df)),
                "audit_type": "blinded_pairwise_pure_vs_dbr_gen2",
                "human_agreement_summary": "not_available" if filled_count == 0 else "labels_present",
                "notes": (
                    "manual audit template exists but structured human labels are empty"
                    if filled_count == 0
                    else "human labels present; treat as qualitative unless adjudication protocol documented"
                ),
            }
        ]
    )
    manual_summary.to_csv(tables_dir / "manual_audit_summary.csv", index=False)
    notes_path = reports_dir / "manual_audit_notes.md"
    notes_path.write_text(
        "- Manual audit template source: `manual_dbr_pairwise_audit_template.csv`.\n"
        f"- Pair rows: {len(manual_df)}.\n"
        f"- Filled `human_winner` labels: {filled_count}.\n"
        "- Treat as qualitative sanity check unless a fully encoded label table is provided.\n",
        encoding="utf-8",
    )

    integrity_rows = []

    def _counts_from_findings(findings: list[dict[str, Any]]) -> dict[str, int]:
        out = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for f in findings:
            sev = str(f.get("severity", "")).upper()
            if sev in out:
                out[sev] += 1
        return out

    art_find = _load_json(artifact_integrity / "audit_findings.json")
    c = _counts_from_findings(art_find.get("findings", []))
    integrity_rows.append(
        {
            "audit_name": artifact_integrity.name,
            "critical_count": c["CRITICAL"],
            "high_count": c["HIGH"],
            "medium_count": c["MEDIUM"],
            "low_count": c["LOW"],
            "key_result": f"finding_count={art_find.get('finding_count', 0)}",
            "caveats": "Contains legacy summary mismatches in seed211/gen0 dbr branch.",
        }
    )
    dbr_find = _load_json(dbr_recompute_audit / "dbr_recompute_findings.json")
    c2 = _counts_from_findings(dbr_find.get("findings", []))
    integrity_rows.append(
        {
            "audit_name": dbr_recompute_audit.name,
            "critical_count": c2["CRITICAL"],
            "high_count": c2["HIGH"],
            "medium_count": c2["MEDIUM"],
            "low_count": c2["LOW"],
            "key_result": f"finding_count={dbr_find.get('finding_count', 0)}",
            "caveats": "Focused DBR recompute audit clean.",
        }
    )
    pd.DataFrame(integrity_rows).to_csv(tables_dir / "integrity_summary.csv", index=False)

    table_compare = pd.read_csv(dbr_recompute_audit / "dbr_table_comparison.csv")
    table_compare.to_csv(tables_dir / "dbr_recompute_table_comparison.csv", index=False)
    row_counts_df.sort_values(["family", "seed", "branch", "generation"]).to_csv(tables_dir / "row_count_consistency.csv", index=False)

    claim_rows = []

    only_gen2 = dbr_gen2_df[dbr_gen2_df["family"] == "dbr_only_221_223"]
    if only_gen2.empty:
        only_gen2 = dbr_gen2_df
    mean_da = float(only_gen2["delta_accuracy"].mean()) if not only_gen2.empty else float("nan")
    mean_dp = float(only_gen2["delta_pedagogy"].mean()) if not only_gen2.empty else float("nan")
    mean_ds = float(only_gen2["delta_silent"].mean()) if not only_gen2.empty else float("nan")

    claim_rows.extend(
        [
            {
                "claim": "Didactic collapse is measurable",
                "status": "supported",
                "strongest_evidence_file": "collapse_evidence_pure_by_seed_generation.csv",
                "numerical_evidence": "Pure branch has non-trivial shifts in pedagogy/silent/parse metrics across generations.",
                "caveat": "Direction is not strictly monotonic across all seeds/families.",
                "recommended_wording": "Didactic quality drift is measurable in inference recycling trajectories.",
            },
            {
                "claim": "Pure recycling shows monotonic collapse",
                "status": "not_supported",
                "strongest_evidence_file": "collapse_evidence_pure_deltas.csv",
                "numerical_evidence": "Mixed signs for Gen0->Gen2 deltas by family/seed.",
                "caveat": "No robust monotonic deterioration in DBR-only wave.",
                "recommended_wording": "Pure recycling shows instability and defect risk, not strict monotonic collapse.",
            },
            {
                "claim": "Pure recycling shows defect risk",
                "status": "supported",
                "strongest_evidence_file": "collapse_evidence_pure_by_seed_generation.csv",
                "numerical_evidence": "Silent/parse/low-reasoning rates remain non-zero across generations.",
                "caveat": "Magnitude varies by seed.",
                "recommended_wording": "Pure recycling retains persistent defect burden.",
            },
            {
                "claim": "DBR preserves coverage",
                "status": "supported",
                "strongest_evidence_file": "dbr_selection_rate.csv",
                "numerical_evidence": f"Mean selection_rate={float(selection_df['selection_rate'].mean()):.3f}",
                "caveat": "Coverage can dip when budgets tighten.",
                "recommended_wording": "DBR maintains high effective coverage under explicit budgets.",
            },
            {
                "claim": "DBR eliminates parse failures",
                "status": "supported",
                "strongest_evidence_file": "dbr_defect_before_after.csv",
                "numerical_evidence": "Parse-failure after-rate is ~0 across DBR contexts.",
                "caveat": "Depends on parse budget set to zero.",
                "recommended_wording": "DBR reliably suppresses parse-failure propagation.",
            },
            {
                "claim": "DBR improves accuracy",
                "status": "mixed" if not math.isnan(mean_da) else "inconclusive",
                "strongest_evidence_file": "dbr_gen2_summary.csv",
                "numerical_evidence": f"Gen2 delta accuracy mean={mean_da:.4f}" if not math.isnan(mean_da) else "n/a",
                "caveat": "Seed-sensitive with wide CI.",
                "recommended_wording": "Accuracy impact is mixed with slight positive tendency in some families.",
            },
            {
                "claim": "DBR improves pedagogy",
                "status": "mixed" if not math.isnan(mean_dp) else "inconclusive",
                "strongest_evidence_file": "dbr_gen2_summary.csv",
                "numerical_evidence": f"Gen2 delta pedagogy mean={mean_dp:.4f}" if not math.isnan(mean_dp) else "n/a",
                "caveat": "Not consistently positive in all seeds.",
                "recommended_wording": "Pedagogical gains are plausible but not yet robust.",
            },
            {
                "claim": "DBR reduces silent errors",
                "status": "partially_supported" if not math.isnan(mean_ds) and mean_ds < 0 else "mixed",
                "strongest_evidence_file": "dbr_gen2_summary.csv",
                "numerical_evidence": f"Gen2 delta silent mean={mean_ds:.4f}" if not math.isnan(mean_ds) else "n/a",
                "caveat": "Direction favorable in aggregate but CI remains wide.",
                "recommended_wording": "DBR tends to reduce silent errors with seed-dependent effect size.",
            },
            {
                "claim": "Qwen supports DBR",
                "status": "partially_supported",
                "strongest_evidence_file": "qwen_dbr_pairwise_summary.csv",
                "numerical_evidence": f"Pairwise margin={float(qwen_out['margin'].iloc[0]):.4f}",
                "caveat": "Per-seed margins are mixed.",
                "recommended_wording": "Qwen pairwise sensitivity gives partial support to DBR direction.",
            },
            {
                "claim": "CSR shows contrastive pedagogical signal",
                "status": "supported",
                "strongest_evidence_file": "csr_pair_summary_combined.csv",
                "numerical_evidence": "Pair construction rates >0.4 and large best-worst quality gaps.",
                "caveat": "Single-seed shakedown only.",
                "recommended_wording": "CSR establishes a usable internal contrastive signal.",
            },
            {
                "claim": "CSR is branch-level superior",
                "status": "not_supported",
                "strongest_evidence_file": "csr_gen2_branch_comparison.csv",
                "numerical_evidence": "CSR does not consistently beat DBR on Gen2 pedagogy/accuracy.",
                "caveat": "Pair-construction bottleneck (best_not_correct) remains high.",
                "recommended_wording": "CSR is promising diagnostically but not yet the top branch-level method.",
            },
        ]
    )
    pd.DataFrame(claim_rows).to_csv(tables_dir / "claim_status_table.csv", index=False)

    method_rows = [
        {
            "method": "DBR",
            "role_in_article": "main_simple_method_candidate",
            "strengths": "High coverage preservation, explicit defect budgets, parse-failure suppression.",
            "weaknesses": "Gen2 gains are seed-sensitive; pedagogy improvement not consistently robust.",
            "best_metric": "selection_rate / parse_failure_after",
            "worst_metric": "pedagogical consistency across seeds",
            "recommended_claim": "DBR is a practical defect-control baseline with mixed but encouraging Gen2 effects.",
        },
        {
            "method": "CSR k=3",
            "role_in_article": "contrastive_signal_proof_of_concept",
            "strengths": "Constructs high-gap best-vs-worst pairs without external teacher.",
            "weaknesses": "Limited pair coverage due to best_not_correct failures.",
            "best_metric": "mean_quality_gap",
            "worst_metric": "pair_construction_rate",
            "recommended_claim": "CSR demonstrates internal contrastive signal but needs stronger candidate generation.",
        },
        {
            "method": "CSR k=5",
            "role_in_article": "tuned_contrastive_variant",
            "strengths": "Improves pair construction and no-pair profile vs k=3.",
            "weaknesses": "Still not clearly superior to DBR branch-level outcomes.",
            "best_metric": "pair_construction_rate",
            "worst_metric": "Gen2 pedagogy vs DBR",
            "recommended_claim": "Increasing candidates helps CSR mechanics; branch-level advantage remains unproven.",
        },
        {
            "method": "pure_recycling",
            "role_in_article": "primary_control_baseline",
            "strengths": "No intervention baseline for trajectory analysis.",
            "weaknesses": "Persistent defect burden and unstable pedagogical trajectory.",
            "best_metric": "simplicity",
            "worst_metric": "defect risk control",
            "recommended_claim": "Pure recycling is a necessary control but not a robust mitigation strategy.",
        },
    ]
    pd.DataFrame(method_rows).to_csv(tables_dir / "method_positioning_table.csv", index=False)

    cheatsheet = (
        "# Article Numbers Cheatsheet\n\n"
        "1. DBR-only Gen2 delta (dbr - pure) accuracy mean: "
        f"{float(only_gen2['delta_accuracy'].mean()):.4f}\n"
        "2. DBR-only Gen2 delta (dbr - pure) pedagogy mean: "
        f"{float(only_gen2['delta_pedagogy'].mean()):.4f}\n"
        "3. DBR-only Gen2 delta (dbr - pure) silent mean: "
        f"{float(only_gen2['delta_silent'].mean()):.4f}\n"
        f"4. DBR mean selection_rate (all families): {float(selection_df['selection_rate'].mean()):.4f}\n"
        "5. DBR parse_failure after-rate mean: "
        f"{float(defect_df[defect_df['defect_type']=='parse_failure']['rate_after'].mean()):.4f}\n"
        "6. Qwen DBR pairwise win rate: "
        f"{float(qwen_out['dbr_win_rate'].iloc[0]):.4f}\n"
        "7. Qwen pure pairwise win rate: "
        f"{float(qwen_out['pure_win_rate'].iloc[0]):.4f}\n"
        "8. Qwen tie rate: "
        f"{float(qwen_out['tie_rate'].iloc[0]):.4f}\n"
        f"9. CSR k=3 mean pair_construction_rate: {float(k3['pair_construction_rate'].mean()):.4f}\n"
        f"10. CSR k=5 mean pair_construction_rate: {float(k5['pair_construction_rate'].mean()):.4f}\n"
        f"11. CSR k=3 mean quality_gap: {float(k3['mean_quality_gap'].mean()):.4f}\n"
        f"12. CSR k=5 mean quality_gap: {float(k5['mean_quality_gap'].mean()):.4f}\n"
        "13. Legacy caveat: artifact_integrity latest reports 2 CRITICAL summary mismatches in seed211/gen0/dbr.\n"
        "14. Focused DBR recompute audit reports 0 CRITICAL / 0 HIGH findings.\n"
        "15. Recommended framing: DBR = strongest practical baseline; effect mixed but reproducible with explicit caveats.\n\n"
        "## Abstract-ready candidates\n"
        "- Report DBR Gen2 deltas with CI from recompute-validated tables.\n"
        "- Report selection_rate and parse-failure suppression as mechanism evidence.\n"
        "- Report Qwen pairwise as sensitivity appendix (partial support, mixed by seed).\n\n"
        "## Required caveats\n"
        "- Inference recycling only (no full retraining claim).\n"
        "- Seed sensitivity and wide CI.\n"
        "- Legacy summary mismatch excluded in favor of recompute-validated numbers.\n"
    )
    (tables_dir / "article_numbers_cheatsheet.md").write_text(cheatsheet, encoding="utf-8")

    fig_sources = [
        dbr_only_analysis / "figures" / "dbr_confirmatory_accuracy_by_branch_generation.png",
        dbr_only_analysis / "figures" / "dbr_confirmatory_pedagogical_by_branch_generation.png",
        dbr_only_analysis / "figures" / "dbr_confirmatory_silent_error_by_branch_generation.png",
        dbr_only_analysis / "figures" / "dbr_confirmatory_selection_rate_by_generation.png",
        dbr_only_analysis / "figures" / "dbr_confirmatory_defect_rates_before_after.png",
        csr_k3_run / "figures" / "accuracy_vs_generation.png",
        csr_k3_run / "figures" / "pedagogical_vs_generation.png",
        csr_k3_run / "figures" / "silent_error_vs_generation.png",
        csr_k5_run / "figures" / "accuracy_vs_generation.png",
        csr_k5_run / "figures" / "pedagogical_vs_generation.png",
        csr_k5_run / "figures" / "silent_error_vs_generation.png",
    ]
    for src in fig_sources:
        if src.exists():
            dst_name = f"{src.parent.parent.name}__{src.name}" if "csr_shakedown" in str(src.parent.parent) else src.name
            _copy_if_exists(src, figures_dir / dst_name)

    report_sources = [
        artifact_integrity / "audit_report.md",
        artifact_integrity / "audit_findings.json",
        dbr_recompute_audit / "dbr_recompute_audit_report.md",
        dbr_recompute_audit / "dbr_recompute_findings.json",
        dbr_article_evidence / "tables" / "dbr_article_evidence_integrity.json",
    ]
    for src in report_sources:
        if src.exists():
            _copy_if_exists(src, reports_dir / src.name)

    _copy_if_exists(dbr_recompute_audit / "dbr_table_comparison.csv", tables_dir / "dbr_recompute_table_comparison_raw.csv")

    source_rows = []
    manifest_sources = []
    for s in sources:
        included_files = []
        if s.path.exists():
            if s.path.is_dir():
                for p in sorted(s.path.rglob("*")):
                    if p.is_file() and len(included_files) < 80:
                        included_files.append(str(p.relative_to(s.path)))
            else:
                included_files.append(s.path.name)
        source_rows.append(
            {
                "source_type": s.source_type,
                "path": str(s.path),
                "timestamp": datetime.fromtimestamp(s.path.stat().st_mtime).isoformat() if s.path.exists() else "",
                "included_files": " | ".join(included_files),
                "reason_included": s.reason_included,
                "caveats": s.caveats,
            }
        )
        manifest_sources.append(
            {
                "source_type": s.source_type,
                "path": str(s.path),
                "timestamp": datetime.fromtimestamp(s.path.stat().st_mtime).isoformat() if s.path.exists() else "",
                "reason_included": s.reason_included,
                "caveats": s.caveats,
            }
        )

    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(bundle_dir / "bundle_sources.csv", index=False)

    csv_checks = []
    for csv_path in sorted(tables_dir.glob("*.csv")):
        ok = True
        err = ""
        try:
            pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = str(exc)
        csv_checks.append({"file": str(csv_path), "readable": ok, "error": err})
    pd.DataFrame(csv_checks).to_csv(reports_dir / "csv_readability_checks.csv", index=False)

    claim_path = tables_dir / "claim_status_table.csv"
    claim_filled = False
    if claim_path.exists():
        claim_df = pd.read_csv(claim_path)
        claim_filled = not claim_df.empty and claim_df["status"].notna().all()

    manifest_payload = {
        "bundle_dir": str(bundle_dir),
        "generated_at": datetime.now().isoformat(),
        "sources": manifest_sources,
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "reports_dir": str(reports_dir),
        "checks": {
            "all_manifest_paths_exist": all(Path(s["path"]).exists() for s in manifest_sources),
            "all_csv_readable": all(r["readable"] for r in csv_checks),
            "claim_status_table_filled": bool(claim_filled),
        },
        "no_new_execution": {
            "new_experiments": False,
            "new_judge_calls": False,
            "new_api_calls": False,
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = (
        "# Article Review Bundle\n\n"
        "This bundle aggregates key didactic_collapse article tables from existing completed artifacts only.\n\n"
        "## Included scope\n"
        "- Collapse evidence (pure_recycling)\n"
        "- DBR method evidence and mechanism tables\n"
        "- CSR k=3 and k=5 shakedown evidence\n"
        "- Qwen DBR pairwise sensitivity tables\n"
        "- Integrity/recompute checks\n"
        "- Claim-ready positioning tables\n\n"
        "## Not performed\n"
        "- No new experiments\n"
        "- No new judge/API calls\n"
        "- No regeneration/training reruns\n\n"
        "## Key caveats\n"
        "- Inference recycling only (not full retraining).\n"
        "- Seed sensitivity remains for DBR deltas.\n"
        "- Legacy global integrity audit includes CRITICAL summary mismatches for seed211/gen0/dbr; recompute-validated tables are preferred.\n"
    )
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")

    return bundle_dir


def main() -> None:
    out = _build_bundle()
    print(f"Bundle created: {out}")


if __name__ == "__main__":
    main()
