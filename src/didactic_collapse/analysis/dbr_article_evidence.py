from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from didactic_collapse.analysis.pairwise_judge_sensitivity import _write_simple_xlsx

_PURE = "pure_recycling"
_DBR = "dbr_medium"
_DEFECT_MAP = {
    "parse_failure": "defect_parse_failure",
    "incorrect_answer": "defect_incorrect",
    "silent_error": "defect_silent",
    "low_reasoning": "defect_low_reasoning",
    "low_structure": "defect_low_structure",
}


@dataclass(frozen=True)
class DBRArticleEvidenceArtifacts:
    out_dir: Path
    collapse_by_seed_csv: Path
    collapse_summary_csv: Path
    generation_curves_csv: Path
    mechanism_defect_before_after_csv: Path
    mechanism_budget_violations_csv: Path
    mechanism_selection_rate_csv: Path
    mechanism_bucket_coverage_csv: Path
    mechanism_severity_distribution_csv: Path
    manual_audit_template_csv: Path
    manual_audit_template_xlsx: Path
    manual_audit_hidden_key_csv: Path
    integrity_report_json: Path


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _compute_defect_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["defect_parse_failure"] = ~out["pred_parse_success"].astype(bool)
    out["defect_incorrect"] = ~out["is_correct"].astype(bool)
    out["defect_silent"] = out["is_silent_error"].astype(bool)
    out["defect_low_reasoning"] = pd.to_numeric(out["reasoning_soundness"], errors="coerce").fillna(0) <= 0
    out["defect_low_structure"] = pd.to_numeric(out["structure"], errors="coerce").fillna(0) <= 0
    out["defect_severity"] = (
        4 * out["defect_parse_failure"].astype(int)
        + 4 * out["defect_silent"].astype(int)
        + 2 * out["defect_incorrect"].astype(int)
        + 2 * out["defect_low_reasoning"].astype(int)
        + 1 * out["defect_low_structure"].astype(int)
    )
    return out


def compute_collapse_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "sample_count": 0,
            "accuracy_mean": float("nan"),
            "pedagogical_score_mean": float("nan"),
            "silent_error_rate": float("nan"),
            "parse_failure_pred_rate": float("nan"),
            "low_reasoning_rate": float("nan"),
            "low_structure_rate": float("nan"),
            "defect_severity_mean": float("nan"),
        }
    f = _compute_defect_frame(frame)
    return {
        "sample_count": float(len(f)),
        "accuracy_mean": float(f["is_correct"].astype(bool).mean()),
        "pedagogical_score_mean": float(pd.to_numeric(f["overall_pedagogical_score"], errors="coerce").mean()),
        "silent_error_rate": float(f["is_silent_error"].astype(bool).mean()),
        "parse_failure_pred_rate": float((~f["pred_parse_success"].astype(bool)).mean()),
        "low_reasoning_rate": float(f["defect_low_reasoning"].astype(bool).mean()),
        "low_structure_rate": float(f["defect_low_structure"].astype(bool).mean()),
        "defect_severity_mean": float(pd.to_numeric(f["defect_severity"], errors="coerce").mean()),
    }


def _load_eval_judge_merged(run_dir: Path, *, branch: str, generation: int) -> pd.DataFrame:
    model_dir = _discover_model_dir(run_dir)
    step_dir = model_dir / branch / f"gen_{generation}"
    eval_path = step_dir / "eval_merged.parquet"
    judge_path = step_dir / "judge_outputs.parquet"
    eval_df = pd.read_parquet(eval_path)
    judge_df = pd.read_parquet(judge_path)
    needed_eval = {
        "example_id",
        "raw_response",
        "answer_gold",
        "pred_parse_success",
        "is_correct",
        "overall_pedagogical_score",
        "is_silent_error",
    }
    missing = needed_eval.difference(eval_df.columns)
    if missing:
        raise ValueError(f"Missing eval columns in {eval_path}: {sorted(missing)}")
    for col in ("reasoning_soundness", "structure"):
        if col not in judge_df.columns:
            judge_df[col] = pd.NA
    view = judge_df[["example_id", "reasoning_soundness", "structure"]].copy()
    eval_df["example_id"] = eval_df["example_id"].astype(str)
    view["example_id"] = view["example_id"].astype(str)
    merged = eval_df.merge(view, on="example_id", how="left", validate="one_to_one")
    return merged


def _load_question_map(run_dir: Path) -> pd.DataFrame:
    data_root = _load_data_root(run_dir)
    heldout = pd.read_parquet(data_root / "splits" / "heldout_test.parquet")
    cols = {"example_id", "question"}
    missing = cols.difference(heldout.columns)
    if missing:
        raise ValueError(f"Heldout split missing columns {sorted(missing)} for {run_dir}")
    out = heldout[list(cols)].copy()
    out["example_id"] = out["example_id"].astype(str)
    return out


def _build_manual_dbr_pairs(
    run_dirs: Sequence[Path],
    *,
    sample_size: int,
    sample_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        seed = _load_seed(run_dir)
        pure = _load_eval_judge_merged(run_dir, branch=_PURE, generation=2)
        dbr = _load_eval_judge_merged(run_dir, branch=_DBR, generation=2)
        qmap = _load_question_map(run_dir)
        pure = pure[["example_id", "raw_response", "answer_gold"]].rename(
            columns={"raw_response": "response_pure", "answer_gold": "gold_answer_pure"}
        )
        dbr = dbr[["example_id", "raw_response", "answer_gold"]].rename(
            columns={"raw_response": "response_dbr", "answer_gold": "gold_answer_dbr"}
        )
        pure["example_id"] = pure["example_id"].astype(str)
        dbr["example_id"] = dbr["example_id"].astype(str)
        merged = pure.merge(dbr, on="example_id", how="inner", validate="one_to_one")
        merged = merged.merge(qmap, on="example_id", how="left", validate="one_to_one")
        merged["seed"] = int(seed)
        merged["run_dir"] = str(run_dir)
        merged["gold_answer"] = merged["gold_answer_pure"].astype(str)
        rows.extend(merged.to_dict(orient="records"))
    pool = pd.DataFrame(rows)
    if pool.empty:
        raise ValueError("No matched Gen2 pure/dbr pairs found")
    seeds = sorted(pool["seed"].astype(int).unique().tolist())
    base = sample_size // len(seeds)
    extra = sample_size % len(seeds)
    selected_parts: list[pd.DataFrame] = []
    for idx, seed in enumerate(seeds):
        n = base + (1 if idx < extra else 0)
        sub = pool[pool["seed"] == seed].copy()
        if len(sub) < n:
            raise ValueError(f"Not enough pairs for seed={seed}; need={n}, got={len(sub)}")
        selected_parts.append(sub.sample(frac=1.0, random_state=sample_seed + seed).head(n))
    selected = pd.concat(selected_parts, ignore_index=True).reset_index(drop=True)

    rng = random.Random(sample_seed)
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(selected.to_dict(orient="records"), start=1):
        pair_id = f"dbr_manual_{idx:04d}"
        dbr_is_a = rng.random() < 0.5
        response_a = rec["response_dbr"] if dbr_is_a else rec["response_pure"]
        response_b = rec["response_pure"] if dbr_is_a else rec["response_dbr"]
        branch_a = _DBR if dbr_is_a else _PURE
        branch_b = _PURE if dbr_is_a else _DBR
        public_rows.append(
            {
                "pair_id": pair_id,
                "seed": int(rec["seed"]),
                "example_id": rec["example_id"],
                "question": rec["question"],
                "gold_answer": rec["gold_answer"],
                "response_A": response_a,
                "response_B": response_b,
                "human_winner": pd.NA,
                "human_confidence": pd.NA,
                "human_notes": pd.NA,
            }
        )
        key_rows.append(
            {
                "pair_id": pair_id,
                "seed": int(rec["seed"]),
                "example_id": rec["example_id"],
                "A_branch": branch_a,
                "B_branch": branch_b,
                "run_dir": rec["run_dir"],
            }
        )
    return pd.DataFrame(public_rows), pd.DataFrame(key_rows)


def export_dbr_article_evidence(
    *,
    run_dirs: Sequence[Path],
    out_dir: Path | None = None,
    manual_sample_size: int = 36,
    manual_sample_seed: int = 4242,
) -> DBRArticleEvidenceArtifacts:
    run_dirs = [Path(p) for p in run_dirs]
    if not run_dirs:
        raise ValueError("run_dirs must be non-empty")
    root = out_dir or (Path("outputs") / "dbr_confirmatory" / f"dbr_article_evidence_{_now_tag()}")
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    collapse_rows: list[dict[str, Any]] = []
    mech_defect_rows: list[dict[str, Any]] = []
    mech_budget_rows: list[dict[str, Any]] = []
    mech_selection_rows: list[dict[str, Any]] = []
    mech_bucket_rows: list[dict[str, Any]] = []
    mech_severity_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        seed = _load_seed(run_dir)
        model_dir = _discover_model_dir(run_dir)
        for generation in (0, 1, 2):
            pure = _load_eval_judge_merged(run_dir, branch=_PURE, generation=generation)
            metrics = compute_collapse_metrics(pure)
            collapse_rows.append(
                {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "seed": int(seed),
                    "branch": _PURE,
                    "generation": int(generation),
                    **metrics,
                }
            )

            dbr_step = model_dir / _DBR / f"gen_{generation}"
            decisions = pd.read_parquet(dbr_step / "dbr_decisions.parquet")
            report = _read_json(dbr_step / "dbr_budget_report.json")
            train_rows = len(pd.read_parquet(dbr_step / "dbr_training_dataset.parquet"))
            decisions = decisions.copy()
            for col in _DEFECT_MAP.values():
                if col not in decisions.columns:
                    decisions[col] = False
            if "severity" not in decisions.columns:
                decisions["severity"] = 0
            before = decisions
            after = decisions[decisions["selected"].astype(bool)]
            for defect_name, col in _DEFECT_MAP.items():
                mech_defect_rows.append(
                    {
                        "run_id": run_dir.name,
                        "seed": int(seed),
                        "generation": int(generation),
                        "defect_name": defect_name,
                        "rate_before": float(before[col].astype(bool).mean()) if len(before) else float("nan"),
                        "rate_after": float(after[col].astype(bool).mean()) if len(after) else float("nan"),
                        "delta_after_minus_before": (
                            float(after[col].astype(bool).mean()) - float(before[col].astype(bool).mean())
                            if len(before) and len(after)
                            else float("nan")
                        ),
                    }
                )

            for defect_name, details in (report.get("budget_violations") or {}).items():
                mech_budget_rows.append(
                    {
                        "run_id": run_dir.name,
                        "seed": int(seed),
                        "generation": int(generation),
                        "defect_name": str(defect_name),
                        "allowed_rate": float(details.get("allowed_rate", 0.0)),
                        "allowed_count": int(details.get("allowed_count", 0)),
                        "actual_count": int(details.get("actual_count", 0)),
                        "violation_count": int(details.get("violation_count", 0)),
                    }
                )

            mech_selection_rows.append(
                {
                    "run_id": run_dir.name,
                    "seed": int(seed),
                    "generation": int(generation),
                    "target_size": int(report.get("target_size", 0)),
                    "selected_count_reported": int(report.get("selected_count", 0)),
                    "selected_count_training_rows": int(train_rows),
                    "selection_rate": float(report.get("selection_rate", 0.0)),
                    "min_selection_rate": float(report.get("min_selection_rate", 0.0)),
                    "relaxation_steps_used_json": json.dumps(report.get("relaxation_steps_used", []), ensure_ascii=False),
                }
            )

            before_bucket = report.get("bucket_coverage_before") or {}
            after_bucket = report.get("bucket_coverage_after") or {}
            tb = max(1, int(sum(int(v) for v in before_bucket.values())))
            ta = max(1, int(sum(int(v) for v in after_bucket.values())))
            for bucket in ("short", "medium", "long"):
                cb = int(before_bucket.get(bucket, 0))
                ca = int(after_bucket.get(bucket, 0))
                mech_bucket_rows.append(
                    {
                        "run_id": run_dir.name,
                        "seed": int(seed),
                        "generation": int(generation),
                        "bucket": bucket,
                        "count_before": cb,
                        "count_after": ca,
                        "rate_before": float(cb / tb),
                        "rate_after": float(ca / ta),
                    }
                )

            for selected_flag, sub in decisions.groupby(decisions["selected"].astype(bool), as_index=False):
                mech_severity_rows.append(
                    {
                        "run_id": run_dir.name,
                        "seed": int(seed),
                        "generation": int(generation),
                        "selected": bool(selected_flag),
                        "row_count": int(len(sub)),
                        "severity_mean": float(pd.to_numeric(sub["severity"], errors="coerce").mean()),
                        "severity_p90": float(pd.to_numeric(sub["severity"], errors="coerce").quantile(0.90)),
                    }
                )

    collapse_df = pd.DataFrame(collapse_rows).sort_values(["seed", "generation"]).reset_index(drop=True)
    collapse_summary_rows: list[dict[str, Any]] = []
    for generation, grp in collapse_df.groupby("generation", as_index=False):
        for metric in (
            "accuracy_mean",
            "pedagogical_score_mean",
            "silent_error_rate",
            "parse_failure_pred_rate",
            "low_reasoning_rate",
            "low_structure_rate",
            "defect_severity_mean",
        ):
            vals = pd.to_numeric(grp[metric], errors="coerce").tolist()
            ci_low, ci_high = _ci95(vals)
            series = pd.Series(vals, dtype="float64")
            collapse_summary_rows.append(
                {
                    "generation": int(generation),
                    "metric": metric,
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_seeds": int(grp["seed"].nunique()),
                }
            )
    collapse_summary_df = pd.DataFrame(collapse_summary_rows).sort_values(["metric", "generation"]).reset_index(drop=True)

    curves = collapse_df[
        [
            "seed",
            "generation",
            "accuracy_mean",
            "pedagogical_score_mean",
            "silent_error_rate",
            "parse_failure_pred_rate",
            "low_reasoning_rate",
            "low_structure_rate",
            "defect_severity_mean",
            "sample_count",
        ]
    ].copy()

    mech_defect_df = pd.DataFrame(mech_defect_rows).sort_values(["seed", "generation", "defect_name"]).reset_index(drop=True)
    mech_budget_df = pd.DataFrame(mech_budget_rows).sort_values(["seed", "generation", "defect_name"]).reset_index(drop=True)
    mech_selection_df = pd.DataFrame(mech_selection_rows).sort_values(["seed", "generation"]).reset_index(drop=True)
    mech_bucket_df = pd.DataFrame(mech_bucket_rows).sort_values(["seed", "generation", "bucket"]).reset_index(drop=True)
    mech_severity_df = pd.DataFrame(mech_severity_rows).sort_values(["seed", "generation", "selected"]).reset_index(drop=True)

    manual_df, hidden_df = _build_manual_dbr_pairs(
        run_dirs,
        sample_size=manual_sample_size,
        sample_seed=manual_sample_seed,
    )

    collapse_by_seed_csv = tables / "didactic_collapse_evidence_by_seed.csv"
    collapse_summary_csv = tables / "didactic_collapse_evidence_summary.csv"
    generation_curves_csv = tables / "didactic_collapse_generation_curves.csv"
    mechanism_defect_before_after_csv = tables / "dbr_mechanism_defect_before_after.csv"
    mechanism_budget_violations_csv = tables / "dbr_mechanism_budget_violations.csv"
    mechanism_selection_rate_csv = tables / "dbr_mechanism_selection_rate.csv"
    mechanism_bucket_coverage_csv = tables / "dbr_mechanism_bucket_coverage.csv"
    mechanism_severity_distribution_csv = tables / "dbr_mechanism_severity_distribution.csv"
    manual_audit_template_csv = tables / "manual_dbr_pairwise_audit_template.csv"
    manual_audit_template_xlsx = tables / "manual_dbr_pairwise_audit_template.xlsx"
    manual_audit_hidden_key_csv = tables / "manual_dbr_pairwise_hidden_key.csv"
    integrity_report_json = tables / "dbr_article_evidence_integrity.json"

    collapse_df.to_csv(collapse_by_seed_csv, index=False)
    collapse_summary_df.to_csv(collapse_summary_csv, index=False)
    curves.to_csv(generation_curves_csv, index=False)
    mech_defect_df.to_csv(mechanism_defect_before_after_csv, index=False)
    mech_budget_df.to_csv(mechanism_budget_violations_csv, index=False)
    mech_selection_df.to_csv(mechanism_selection_rate_csv, index=False)
    mech_bucket_df.to_csv(mechanism_bucket_coverage_csv, index=False)
    mech_severity_df.to_csv(mechanism_severity_distribution_csv, index=False)
    manual_df.to_csv(manual_audit_template_csv, index=False)
    hidden_df.to_csv(manual_audit_hidden_key_csv, index=False)
    try:
        manual_df.to_excel(manual_audit_template_xlsx, index=False)
    except Exception:  # noqa: BLE001
        _write_simple_xlsx(manual_df, manual_audit_template_xlsx)

    integrity = {
        "generated_at": datetime.now().isoformat(),
        "run_dirs": [str(p) for p in run_dirs],
        "checks": {
            "manual_template_has_no_branch_labels": all(
                c not in manual_df.columns for c in ("A_branch", "B_branch", "branch")
            ),
            "manual_hidden_rows_equal_template_rows": int(len(hidden_df)) == int(len(manual_df)),
            "collapse_rows": int(len(collapse_df)),
            "mechanism_defect_rows": int(len(mech_defect_df)),
            "mechanism_budget_rows": int(len(mech_budget_df)),
        },
    }
    integrity_report_json.write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")

    return DBRArticleEvidenceArtifacts(
        out_dir=root,
        collapse_by_seed_csv=collapse_by_seed_csv,
        collapse_summary_csv=collapse_summary_csv,
        generation_curves_csv=generation_curves_csv,
        mechanism_defect_before_after_csv=mechanism_defect_before_after_csv,
        mechanism_budget_violations_csv=mechanism_budget_violations_csv,
        mechanism_selection_rate_csv=mechanism_selection_rate_csv,
        mechanism_bucket_coverage_csv=mechanism_bucket_coverage_csv,
        mechanism_severity_distribution_csv=mechanism_severity_distribution_csv,
        manual_audit_template_csv=manual_audit_template_csv,
        manual_audit_template_xlsx=manual_audit_template_xlsx,
        manual_audit_hidden_key_csv=manual_audit_hidden_key_csv,
        integrity_report_json=integrity_report_json,
    )

