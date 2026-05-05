from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DBRError(ValueError):
    """Raised when Defect-Budgeted Recycling invariants are violated."""


@dataclass(frozen=True)
class DBRPolicy:
    policy_name: str = "dbr_medium"
    target_size_ratio: float = 1.0
    min_selection_rate: float = 0.80
    budget_parse_failure: float = 0.00
    budget_silent_error: float = 0.10
    budget_incorrect_answer: float = 0.30
    budget_low_reasoning: float = 0.25
    budget_low_structure: float = 0.30
    allow_parse_failure_fallback: bool = False
    short_max_chars: int = 80
    medium_max_chars: int = 180


class DBRDecision(BaseModel):
    """Per-example DBR selection decision."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    branch: str
    generation: int
    seed: int
    selected: bool
    defect_parse_failure: bool
    defect_incorrect: bool
    defect_silent: bool
    defect_low_reasoning: bool
    defect_low_structure: bool
    severity: int = Field(ge=0)
    question_length_bin: str
    selection_rank: int | None = None
    budget_violation_if_selected: bool
    decision_reason: str
    sampling_value: float = Field(ge=0.0, le=1.0)


class DBRReport(BaseModel):
    """Serializable DBR budget/coverage diagnostics."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    branch: str
    generation: int
    seed: int
    policy_name: str
    total_candidates: int = Field(ge=0)
    target_size: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    selection_rate: float = Field(ge=0.0)
    min_selection_rate: float = Field(ge=0.0, le=1.0)
    budgets: dict[str, float]
    defect_rates_before: dict[str, float]
    defect_rates_after: dict[str, float]
    budget_violations: dict[str, Any]
    relaxation_steps_used: list[str]
    bucket_coverage_before: dict[str, int]
    bucket_coverage_after: dict[str, int]
    fallback_bucket_count: int = Field(ge=0)


@dataclass(frozen=True)
class DBRResult:
    training_df: pd.DataFrame
    decisions_df: pd.DataFrame
    report: DBRReport


_DEFECT_COLS: tuple[str, ...] = (
    "defect_parse_failure",
    "defect_incorrect",
    "defect_silent",
    "defect_low_reasoning",
    "defect_low_structure",
)

_BUDGET_COL_MAP: dict[str, str] = {
    "defect_parse_failure": "parse_failure",
    "defect_silent": "silent_error",
    "defect_incorrect": "incorrect_answer",
    "defect_low_reasoning": "low_reasoning",
    "defect_low_structure": "low_structure",
}


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise DBRError(f"{name} missing required columns: {sorted(missing)}")


def _assert_unique_example_id(df: pd.DataFrame, name: str) -> None:
    if df["example_id"].duplicated().any():
        dup_count = int(df["example_id"].duplicated().sum())
        raise DBRError(f"{name} contains duplicate example_id values: {dup_count}")


def _assert_equal_example_sets(*, synthetic_df: pd.DataFrame, accuracy_df: pd.DataFrame, judge_df: pd.DataFrame) -> None:
    synth_ids = set(synthetic_df["example_id"].astype(str).tolist())
    acc_ids = set(accuracy_df["example_id"].astype(str).tolist())
    judge_ids = set(judge_df["example_id"].astype(str).tolist())

    if synth_ids != acc_ids:
        only_synth = sorted(synth_ids.difference(acc_ids))[:5]
        only_acc = sorted(acc_ids.difference(synth_ids))[:5]
        raise DBRError(
            "Strict merge invariant failed: synthetic and accuracy example_id sets differ. "
            f"only_synthetic_sample={only_synth}, only_accuracy_sample={only_acc}"
        )
    if synth_ids != judge_ids:
        only_synth = sorted(synth_ids.difference(judge_ids))[:5]
        only_judge = sorted(judge_ids.difference(synth_ids))[:5]
        raise DBRError(
            "Strict merge invariant failed: synthetic and judge example_id sets differ. "
            f"only_synthetic_sample={only_synth}, only_judge_sample={only_judge}"
        )


def _deterministic_unit_score(*, seed: int, branch: str, generation: int, example_id: str) -> float:
    payload = f"{seed}|{branch}|{generation}|{example_id}|dbr"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    numerator = int(digest[:12], 16)
    denominator = float(16**12)
    return float(numerator / denominator)


def compute_defect_flags(row: pd.Series) -> dict[str, bool]:
    """Compute DBR defect flags from merged synthetic+accuracy+judge row."""
    pred_parse_success = bool(row.get("pred_parse_success", False))
    is_correct = bool(row.get("is_correct_effective", False))
    is_silent_error = bool(row.get("is_silent_error", True))

    reasoning = row.get("reasoning_soundness")
    structure = row.get("structure")
    reasoning_val = None if pd.isna(reasoning) else float(reasoning)
    structure_val = None if pd.isna(structure) else float(structure)

    return {
        "defect_parse_failure": not pred_parse_success,
        "defect_incorrect": not is_correct,
        "defect_silent": is_silent_error,
        "defect_low_reasoning": (reasoning_val is None) or (reasoning_val <= 0.0),
        "defect_low_structure": (structure_val is None) or (structure_val <= 0.0),
    }


def compute_severity(flags: dict[str, bool]) -> int:
    """Compute severity score from DBR defect flags."""
    return int(
        4 * int(flags["defect_parse_failure"])
        + 4 * int(flags["defect_silent"])
        + 2 * int(flags["defect_incorrect"])
        + 2 * int(flags["defect_low_reasoning"])
        + 1 * int(flags["defect_low_structure"])
    )


def assign_question_length_bin(
    *,
    question: Any,
    example_id: str,
    seed: int,
    branch: str,
    generation: int,
    short_max_chars: int,
    medium_max_chars: int,
) -> tuple[str, bool]:
    """Assign short/medium/long bucket; fallback to deterministic hash if text unavailable."""
    text = "" if question is None else str(question).strip()
    if text:
        length = len(text)
        if length <= int(short_max_chars):
            return "short", False
        if length <= int(medium_max_chars):
            return "medium", False
        return "long", False
    score = _deterministic_unit_score(
        seed=seed,
        branch=branch,
        generation=generation,
        example_id=example_id,
    )
    if score < (1.0 / 3.0):
        return "short", True
    if score < (2.0 / 3.0):
        return "medium", True
    return "long", True


def _validate_policy(policy: DBRPolicy) -> None:
    if not (0.0 < float(policy.target_size_ratio) <= 1.0):
        raise DBRError(f"target_size_ratio must be in (0,1], got {policy.target_size_ratio}")
    if not (0.0 <= float(policy.min_selection_rate) <= 1.0):
        raise DBRError(f"min_selection_rate must be in [0,1], got {policy.min_selection_rate}")
    for name in (
        "budget_parse_failure",
        "budget_silent_error",
        "budget_incorrect_answer",
        "budget_low_reasoning",
        "budget_low_structure",
    ):
        value = float(getattr(policy, name))
        if not (0.0 <= value <= 1.0):
            raise DBRError(f"{name} must be in [0,1], got {value}")
    if int(policy.short_max_chars) <= 0 or int(policy.medium_max_chars) <= 0:
        raise DBRError("short_max_chars and medium_max_chars must be positive")
    if int(policy.medium_max_chars) < int(policy.short_max_chars):
        raise DBRError("medium_max_chars must be >= short_max_chars")


def apply_dbr(
    *,
    synthetic_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    model_name: str,
    branch: str,
    generation: int,
    seed: int,
    policy: DBRPolicy,
    allow_partial_inputs: bool = False,
) -> DBRResult:
    """Apply Defect-Budgeted Recycling selection over synthetic candidate rows."""
    _validate_policy(policy)
    _require_columns(synthetic_df, {"example_id", "question", "answer_for_training", "source"}, "synthetic_df")
    _require_columns(
        accuracy_df,
        {"example_id", "pred_parse_success", "accuracy_label", "is_correct"},
        "accuracy_df",
    )
    _require_columns(
        judge_df,
        {"example_id", "is_silent_error", "reasoning_soundness", "structure", "clarity"},
        "judge_df",
    )
    _assert_unique_example_id(synthetic_df, "synthetic_df")
    _assert_unique_example_id(accuracy_df, "accuracy_df")
    _assert_unique_example_id(judge_df, "judge_df")
    if not allow_partial_inputs:
        _assert_equal_example_sets(synthetic_df=synthetic_df, accuracy_df=accuracy_df, judge_df=judge_df)

    synthetic = synthetic_df.copy()
    accuracy = accuracy_df.copy()
    judge = judge_df.copy()
    synthetic["example_id"] = synthetic["example_id"].astype(str)
    accuracy["example_id"] = accuracy["example_id"].astype(str)
    judge["example_id"] = judge["example_id"].astype(str)

    merged = synthetic.merge(
        accuracy[["example_id", "pred_parse_success", "accuracy_label", "is_correct"]].assign(_has_accuracy=True),
        on="example_id",
        how="left" if allow_partial_inputs else "inner",
        validate="one_to_one",
    ).merge(
        judge[["example_id", "is_silent_error", "reasoning_soundness", "structure", "clarity"]].assign(_has_judge=True),
        on="example_id",
        how="left" if allow_partial_inputs else "inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise DBRError("DBR received zero candidate rows after merge.")

    merged["pred_parse_success"] = merged["pred_parse_success"].fillna(False).astype(bool)
    merged["is_correct"] = merged["is_correct"].fillna(False).astype(bool)
    merged["accuracy_label"] = merged["accuracy_label"].astype(str).str.lower()
    merged["is_correct_effective"] = (merged["is_correct"] == True) | (merged["accuracy_label"] == "correct")  # noqa: E712
    merged["is_silent_error"] = merged["is_silent_error"].fillna(True).astype(bool)
    merged["reasoning_soundness"] = pd.to_numeric(merged["reasoning_soundness"], errors="coerce")
    merged["structure"] = pd.to_numeric(merged["structure"], errors="coerce")
    merged["clarity"] = pd.to_numeric(merged["clarity"], errors="coerce")

    fallback_bucket_count = 0
    bins: list[str] = []
    for rec in merged.to_dict(orient="records"):
        qbin, used_fallback = assign_question_length_bin(
            question=rec.get("question"),
            example_id=str(rec["example_id"]),
            seed=seed,
            branch=branch,
            generation=generation,
            short_max_chars=policy.short_max_chars,
            medium_max_chars=policy.medium_max_chars,
        )
        bins.append(qbin)
        if used_fallback:
            fallback_bucket_count += 1
    merged["question_length_bin"] = bins
    merged["sampling_value"] = merged["example_id"].astype(str).map(
        lambda x: _deterministic_unit_score(
            seed=seed,
            branch=branch,
            generation=generation,
            example_id=x,
        )
    )

    flag_records = [compute_defect_flags(merged.loc[idx]) for idx in merged.index]
    flags_df = pd.DataFrame(flag_records)
    for col in _DEFECT_COLS:
        merged[col] = flags_df[col].astype(bool)
    merged["severity"] = flags_df.apply(lambda r: compute_severity(r.to_dict()), axis=1).astype(int)

    merged = merged.sort_values(
        by=[
            "question_length_bin",
            "severity",
            "reasoning_soundness",
            "structure",
            "clarity",
            "sampling_value",
        ],
        ascending=[True, True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    total_candidates = int(len(merged))
    target_size = int(round(total_candidates * float(policy.target_size_ratio)))
    target_size = max(1, min(total_candidates, target_size))

    budget_rates = {
        "parse_failure": float(policy.budget_parse_failure),
        "silent_error": float(policy.budget_silent_error),
        "incorrect_answer": float(policy.budget_incorrect_answer),
        "low_reasoning": float(policy.budget_low_reasoning),
        "low_structure": float(policy.budget_low_structure),
    }
    budget_allowed_counts = {
        "defect_parse_failure": int(total_candidates if policy.allow_parse_failure_fallback else int(budget_rates["parse_failure"] * target_size)),
        "defect_silent": int(budget_rates["silent_error"] * target_size),
        "defect_incorrect": int(budget_rates["incorrect_answer"] * target_size),
        "defect_low_reasoning": int(budget_rates["low_reasoning"] * target_size),
        "defect_low_structure": int(budget_rates["low_structure"] * target_size),
    }
    selected_counts = {k: 0 for k in _DEFECT_COLS}
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    selection_reason: dict[int, str] = {}
    selected_budget_violation: dict[int, bool] = {}
    relaxation_steps_used: list[str] = []

    bucket_order = ["short", "medium", "long"]
    bucket_indices: dict[str, list[int]] = {
        bucket: merged.index[merged["question_length_bin"] == bucket].tolist() for bucket in bucket_order
    }

    def _row_violations(idx: int) -> list[str]:
        row = merged.loc[idx]
        violations: list[str] = []
        for defect_col in _DEFECT_COLS:
            if bool(row[defect_col]) and (selected_counts[defect_col] + 1) > int(budget_allowed_counts[defect_col]):
                violations.append(defect_col)
        return violations

    def _select_idx(idx: int, *, reason: str, allow_budget_violation: bool) -> bool:
        if idx in selected_set:
            return False
        row = merged.loc[idx]
        if bool(row["defect_parse_failure"]) and not bool(policy.allow_parse_failure_fallback):
            return False
        violations = _row_violations(idx)
        if violations and not allow_budget_violation:
            return False
        selected_set.add(idx)
        selected_indices.append(idx)
        selection_reason[idx] = reason
        selected_budget_violation[idx] = bool(violations)
        for defect_col in _DEFECT_COLS:
            if bool(row[defect_col]):
                selected_counts[defect_col] += 1
        return True

    def _greedy_bucket_pass(reason: str) -> bool:
        progress_any = False
        while len(selected_indices) < target_size:
            added = False
            for bucket in bucket_order:
                for idx in bucket_indices.get(bucket, []):
                    if idx in selected_set:
                        continue
                    if _select_idx(idx, reason=reason, allow_budget_violation=False):
                        added = True
                        progress_any = True
                        break
            if not added:
                break
        return progress_any

    _greedy_bucket_pass("selected_within_budget")

    defect_totals = {col: int(merged[col].astype(bool).sum()) for col in _DEFECT_COLS}
    for defect_col in (
        "defect_low_structure",
        "defect_low_reasoning",
        "defect_incorrect",
        "defect_silent",
        "defect_parse_failure",
    ):
        if len(selected_indices) >= target_size:
            break
        if defect_col == "defect_parse_failure" and not bool(policy.allow_parse_failure_fallback):
            continue
        max_allowed = int(defect_totals[defect_col])
        if int(budget_allowed_counts[defect_col]) >= max_allowed:
            continue
        budget_allowed_counts[defect_col] = max_allowed
        relaxation_steps_used.append(f"relax_{_BUDGET_COL_MAP[defect_col]}")
        _greedy_bucket_pass(f"selected_relaxed_{_BUDGET_COL_MAP[defect_col]}")

    if len(selected_indices) < target_size:
        for idx in merged.index.tolist():
            if len(selected_indices) >= target_size:
                break
            _select_idx(idx, reason="selected_budget_violation_fallback", allow_budget_violation=True)

    if not selected_indices:
        raise DBRError("DBR selected zero rows; cannot build next-generation training dataset.")

    selected_df = merged.loc[selected_indices].copy()
    selected_df["selection_rank"] = range(1, len(selected_df) + 1)
    rank_map = dict(zip(selected_df.index.tolist(), selected_df["selection_rank"].astype(int).tolist(), strict=False))

    decision_rows: list[dict[str, Any]] = []
    for idx, rec in merged.iterrows():
        selected = idx in selected_set
        if selected:
            reason = selection_reason.get(idx, "selected")
            violation_flag = bool(selected_budget_violation.get(idx, False))
            rank = int(rank_map[idx])
        else:
            if len(selected_indices) >= target_size:
                reason = "rejected_target_reached"
            elif bool(rec["defect_parse_failure"]) and not bool(policy.allow_parse_failure_fallback):
                reason = "rejected_parse_forbidden"
            else:
                reason = "rejected_budget_constraints"
            violation_flag = False
            rank = None
        decision = DBRDecision(
            example_id=str(rec["example_id"]),
            branch=branch,
            generation=generation,
            seed=seed,
            selected=selected,
            defect_parse_failure=bool(rec["defect_parse_failure"]),
            defect_incorrect=bool(rec["defect_incorrect"]),
            defect_silent=bool(rec["defect_silent"]),
            defect_low_reasoning=bool(rec["defect_low_reasoning"]),
            defect_low_structure=bool(rec["defect_low_structure"]),
            severity=int(rec["severity"]),
            question_length_bin=str(rec["question_length_bin"]),
            selection_rank=rank,
            budget_violation_if_selected=violation_flag,
            decision_reason=reason,
            sampling_value=float(rec["sampling_value"]),
        )
        decision_rows.append(decision.model_dump(mode="json"))
    decisions_df = pd.DataFrame(decision_rows).sort_values(["selected", "selection_rank", "example_id"], ascending=[False, True, True])

    selected_mask = decisions_df["selected"].astype(bool)
    selected_count = int(selected_mask.sum())
    selection_rate = float(selected_count / target_size) if target_size > 0 else 0.0

    def _mean_for(col: str, frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        return float(frame[col].astype(bool).mean())

    selected_only = merged.loc[selected_indices].copy() if selected_indices else merged.iloc[0:0].copy()
    defect_rates_before = {
        _BUDGET_COL_MAP[col]: _mean_for(col, merged) for col in _DEFECT_COLS
    }
    defect_rates_after = {
        _BUDGET_COL_MAP[col]: _mean_for(col, selected_only) for col in _DEFECT_COLS
    }

    budget_violations: dict[str, Any] = {}
    for defect_col in _DEFECT_COLS:
        key = _BUDGET_COL_MAP[defect_col]
        allowed_rate = budget_rates[key]
        allowed_count = int(int(allowed_rate * target_size))
        actual_count = int(selected_only[defect_col].astype(bool).sum()) if not selected_only.empty else 0
        violation_count = max(0, actual_count - allowed_count)
        budget_violations[key] = {
            "allowed_rate": allowed_rate,
            "allowed_count": allowed_count,
            "actual_count": actual_count,
            "violation_count": violation_count,
        }
    if selection_rate < float(policy.min_selection_rate):
        budget_violations["selection_rate_below_target"] = {
            "min_selection_rate": float(policy.min_selection_rate),
            "selection_rate": selection_rate,
            "shortfall": float(policy.min_selection_rate) - selection_rate,
        }

    bucket_coverage_before = (
        merged["question_length_bin"].value_counts().reindex(["short", "medium", "long"], fill_value=0).to_dict()
    )
    bucket_coverage_after = (
        selected_only["question_length_bin"].value_counts().reindex(["short", "medium", "long"], fill_value=0).to_dict()
    )

    training_df = selected_only[synthetic_df.columns.tolist()].copy()
    report = DBRReport(
        model_name=model_name,
        branch=branch,
        generation=generation,
        seed=seed,
        policy_name=str(policy.policy_name),
        total_candidates=total_candidates,
        target_size=target_size,
        selected_count=selected_count,
        selection_rate=selection_rate,
        min_selection_rate=float(policy.min_selection_rate),
        budgets=budget_rates,
        defect_rates_before=defect_rates_before,
        defect_rates_after=defect_rates_after,
        budget_violations=budget_violations,
        relaxation_steps_used=relaxation_steps_used,
        bucket_coverage_before={k: int(v) for k, v in bucket_coverage_before.items()},
        bucket_coverage_after={k: int(v) for k, v in bucket_coverage_after.items()},
        fallback_bucket_count=int(fallback_bucket_count),
    )
    return DBRResult(
        training_df=training_df.reset_index(drop=True),
        decisions_df=decisions_df.reset_index(drop=True),
        report=report,
    )


def save_dbr_artifacts(
    *,
    result: DBRResult,
    training_path: Path,
    decisions_path: Path,
    report_path: Path,
) -> None:
    training_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.training_df.to_parquet(training_path, index=False)
    result.decisions_df.to_parquet(decisions_path, index=False)
    report_path.write_text(
        json.dumps(result.report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

