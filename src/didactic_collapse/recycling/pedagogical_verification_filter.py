from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PVFError(ValueError):
    """Raised when pedagogical verification filtering invariants are violated."""


@dataclass(frozen=True)
class PVFPolicy:
    threshold_score: int = 5
    min_keep_ratio: float = 0.0


class PVFReport(BaseModel):
    """Serializable report for PVF coverage and rejection diagnostics."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    branch: str
    generation: int
    seed: int
    threshold_score: int = Field(ge=0, le=8)
    min_keep_ratio: float = Field(ge=0.0, le=1.0)
    total_candidates: int = Field(ge=0)
    kept_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    keep_rate: float = Field(ge=0.0, le=1.0)
    rejection_reason_counts: dict[str, int]
    kept_accuracy_mean: float | None = None
    rejected_accuracy_mean: float | None = None
    kept_pedagogical_mean: float | None = None
    rejected_pedagogical_mean: float | None = None
    kept_silent_error_rate: float | None = None
    rejected_silent_error_rate: float | None = None


@dataclass(frozen=True)
class PVFResult:
    filtered_training_df: pd.DataFrame
    rejected_examples_df: pd.DataFrame
    report: PVFReport


@dataclass(frozen=True)
class SoftPVFPolicy:
    policy_name: str = "soft_pvf_medium"
    high_quality_threshold: int = 6
    medium_quality_threshold: int = 4
    weight_high: float = 1.0
    weight_medium: float = 0.5
    weight_low_correct: float = 0.25
    weight_incorrect: float = 0.1
    min_keep_ratio: float = 0.0


class SoftPVFDecision(BaseModel):
    """Per-example decision row for soft PVF sampling."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    is_correct: bool
    pred_parse_success: bool
    overall_pedagogical_score: float | None
    is_silent_error: bool
    assigned_weight: float = Field(ge=0.0, le=1.0)
    kept: bool
    decision_reason: str
    deterministic_score: float = Field(ge=0.0, le=1.0)
    sampling_value: float = Field(ge=0.0, le=1.0)
    policy_name: str
    branch: str
    generation: int
    seed: int


class SoftPVFReport(BaseModel):
    """Serializable report for soft PVF coverage and weighted decision diagnostics."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    branch: str
    generation: int
    seed: int
    policy_name: str
    high_quality_threshold: int = Field(ge=0, le=8)
    medium_quality_threshold: int = Field(ge=0, le=8)
    min_keep_ratio: float = Field(ge=0.0, le=1.0)
    total_candidates: int = Field(ge=0)
    kept_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    keep_rate: float = Field(ge=0.0, le=1.0)
    effective_keep_rate: float = Field(ge=0.0, le=1.0)
    mean_assigned_weight: float = Field(ge=0.0, le=1.0)
    decision_reason_counts: dict[str, int]
    weight_distribution: dict[str, int]
    kept_accuracy_mean: float | None = None
    rejected_accuracy_mean: float | None = None
    kept_pedagogical_mean: float | None = None
    rejected_pedagogical_mean: float | None = None
    kept_silent_error_rate: float | None = None
    rejected_silent_error_rate: float | None = None


@dataclass(frozen=True)
class SoftPVFResult:
    training_df: pd.DataFrame
    decisions_df: pd.DataFrame
    report: SoftPVFReport


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise PVFError(f"{name} missing required columns: {sorted(missing)}")


def _assert_unique_example_id(df: pd.DataFrame, name: str) -> None:
    if df["example_id"].duplicated().any():
        dup_count = int(df["example_id"].duplicated().sum())
        raise PVFError(f"{name} contains duplicate example_id values: {dup_count}")


def _assert_equal_example_sets(*, synthetic_df: pd.DataFrame, accuracy_df: pd.DataFrame, judge_df: pd.DataFrame) -> None:
    synth_ids = set(synthetic_df["example_id"].astype(str).tolist())
    acc_ids = set(accuracy_df["example_id"].astype(str).tolist())
    judge_ids = set(judge_df["example_id"].astype(str).tolist())

    if synth_ids != acc_ids:
        only_synth = sorted(synth_ids.difference(acc_ids))[:5]
        only_acc = sorted(acc_ids.difference(synth_ids))[:5]
        raise PVFError(
            "Strict merge invariant failed: synthetic and accuracy example_id sets differ. "
            f"only_synthetic_sample={only_synth}, only_accuracy_sample={only_acc}"
        )
    if synth_ids != judge_ids:
        only_synth = sorted(synth_ids.difference(judge_ids))[:5]
        only_judge = sorted(judge_ids.difference(synth_ids))[:5]
        raise PVFError(
            "Strict merge invariant failed: synthetic and judge example_id sets differ. "
            f"only_synthetic_sample={only_synth}, only_judge_sample={only_judge}"
        )


def _reason_join(parts: list[str]) -> str:
    return "|".join(parts) if parts else ""


def _deterministic_unit_score(*, seed: int, branch: str, generation: int, example_id: str) -> float:
    payload = f"{seed}|{branch}|{generation}|{example_id}|soft_pvf"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # Use 48 bits for a stable [0,1) pseudo-random value.
    numerator = int(digest[:12], 16)
    denominator = float(16**12)
    return float(numerator / denominator)


def deterministic_weighted_keep(
    *,
    weight: float,
    seed: int,
    branch: str,
    generation: int,
    example_id: str,
) -> tuple[bool, float]:
    """Deterministic Bernoulli keep decision from (seed, branch, generation, example_id)."""
    w = max(0.0, min(1.0, float(weight)))
    score = _deterministic_unit_score(
        seed=int(seed),
        branch=str(branch),
        generation=int(generation),
        example_id=str(example_id),
    )
    if w <= 0.0:
        return False, score
    if w >= 1.0:
        return True, score
    return bool(score < w), score


def _resolve_soft_policy_name(raw: str) -> str:
    value = str(raw).strip().lower()
    supported = {
        "soft_pvf_medium",
        "soft_pvf_lenient",
        "soft_pvf_noisy_keep",
        "soft_pvf_silent_only",
    }
    if value not in supported:
        raise PVFError(f"Unsupported soft PVF policy_name: {raw!r}. Supported: {sorted(supported)}")
    return value


def _assign_soft_weight(
    *,
    pred_parse_success: bool,
    is_silent_error: bool,
    is_correct: bool,
    pedagogical_score: float | None,
    missing_accuracy: bool,
    missing_judge: bool,
    policy: SoftPVFPolicy,
) -> tuple[float, str]:
    if missing_accuracy:
        return 0.0, "missing_accuracy_row"
    if missing_judge:
        return 0.0, "missing_judge_row"
    if not pred_parse_success:
        return 0.0, "pred_parse_failure"
    if is_silent_error:
        return 0.0, "silent_error_true"

    score = float(pedagogical_score) if pedagogical_score is not None else float("-inf")
    policy_name = _resolve_soft_policy_name(policy.policy_name)

    if policy_name == "soft_pvf_medium":
        if is_correct and score >= float(policy.high_quality_threshold):
            return float(policy.weight_high), "high_quality"
        if is_correct and score >= float(policy.medium_quality_threshold):
            return float(policy.weight_medium), "medium_quality"
        if is_correct:
            return float(policy.weight_low_correct), "low_pedagogy_correct"
        return float(policy.weight_incorrect), "incorrect_low_weight"

    if policy_name == "soft_pvf_lenient":
        if is_correct and score >= 6.0:
            return 1.0, "high_quality"
        if is_correct and score >= 4.0:
            return 0.75, "medium_high_quality"
        if is_correct and score >= 2.0:
            return 0.5, "medium_low_quality"
        if is_correct:
            return 0.25, "low_pedagogy_correct"
        if (not is_correct) and score >= 5.0:
            return 0.25, "pedagogical_but_incorrect"
        return 0.1, "incorrect_low_weight"

    if policy_name == "soft_pvf_noisy_keep":
        if is_correct and score >= 5.0:
            return 1.0, "correct_high_pedagogy"
        if is_correct and score >= 3.0:
            return 0.75, "correct_medium_pedagogy"
        if is_correct:
            return 0.5, "correct_low_pedagogy"
        if (not is_correct) and score >= 5.0:
            return 0.25, "incorrect_but_high_pedagogy"
        return 0.1, "incorrect_low_weight"

    if policy_name == "soft_pvf_silent_only":
        return 1.0, "non_silent_parsed_keep"

    raise PVFError(f"Unsupported soft PVF policy_name: {policy_name}")


def apply_pedagogical_verification_filter(
    *,
    synthetic_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    model_name: str,
    branch: str,
    generation: int,
    seed: int,
    policy: PVFPolicy,
    allow_partial_inputs: bool = False,
) -> PVFResult:
    """Filter synthetic rows for next generation using correctness + pedagogy + silent-error constraints."""
    if not (0 <= int(policy.threshold_score) <= 8):
        raise PVFError(f"threshold_score must be in [0,8], got {policy.threshold_score}")
    if not (0.0 <= float(policy.min_keep_ratio) <= 1.0):
        raise PVFError(f"min_keep_ratio must be in [0,1], got {policy.min_keep_ratio}")

    _require_columns(synthetic_df, {"example_id", "question", "answer_for_training", "source"}, "synthetic_df")
    _require_columns(
        accuracy_df,
        {"example_id", "pred_parse_success", "accuracy_label", "is_correct"},
        "accuracy_df",
    )
    _require_columns(
        judge_df,
        {"example_id", "overall_pedagogical_score", "is_silent_error"},
        "judge_df",
    )

    _assert_unique_example_id(synthetic_df, "synthetic_df")
    _assert_unique_example_id(accuracy_df, "accuracy_df")
    _assert_unique_example_id(judge_df, "judge_df")
    if not allow_partial_inputs:
        _assert_equal_example_sets(
            synthetic_df=synthetic_df,
            accuracy_df=accuracy_df,
            judge_df=judge_df,
        )

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
        judge[["example_id", "overall_pedagogical_score", "is_silent_error"]].assign(_has_judge=True),
        on="example_id",
        how="left" if allow_partial_inputs else "inner",
        validate="one_to_one",
    )

    missing_accuracy = merged["_has_accuracy"].isna() if "_has_accuracy" in merged.columns else pd.Series(False, index=merged.index)
    missing_judge = merged["_has_judge"].isna() if "_has_judge" in merged.columns else pd.Series(False, index=merged.index)

    merged["pred_parse_success"] = merged["pred_parse_success"].fillna(False).astype(bool)
    merged["is_correct"] = merged["is_correct"].fillna(False).astype(bool)
    merged["accuracy_label"] = merged["accuracy_label"].astype(str).str.lower()
    merged["overall_pedagogical_score"] = pd.to_numeric(merged["overall_pedagogical_score"], errors="coerce")
    merged["is_silent_error"] = merged["is_silent_error"].fillna(True).astype(bool)

    invalid_score_mask = merged["overall_pedagogical_score"].isna() & ~missing_judge
    if invalid_score_mask.any():
        bad_ids = merged.loc[invalid_score_mask, "example_id"].astype(str).head(5).tolist()
        raise PVFError(
            "overall_pedagogical_score contains non-numeric values after strict merge. "
            f"sample_example_ids={bad_ids}"
        )

    pass_parse = merged["pred_parse_success"] == True  # noqa: E712
    pass_accuracy = (merged["is_correct"] == True) | (merged["accuracy_label"] == "correct")  # noqa: E712
    pass_pedagogy = merged["overall_pedagogical_score"] >= int(policy.threshold_score)
    pass_non_silent = merged["is_silent_error"] == False  # noqa: E712

    keep_mask = pass_parse & pass_accuracy & pass_pedagogy & pass_non_silent
    merged["pvf_keep"] = keep_mask
    merged["pvf_reject_reason_parse"] = ~pass_parse
    merged["pvf_reject_reason_accuracy"] = ~pass_accuracy
    merged["pvf_reject_reason_pedagogy"] = ~pass_pedagogy
    merged["pvf_reject_reason_silent_error"] = ~pass_non_silent
    merged["pvf_reject_reason_missing_accuracy"] = missing_accuracy
    merged["pvf_reject_reason_missing_judge"] = missing_judge

    reject_reasons: list[str] = []
    for rec in merged.to_dict(orient="records"):
        reasons: list[str] = []
        if bool(rec.get("pvf_reject_reason_missing_accuracy", False)):
            reasons.append("missing_accuracy_row")
        if bool(rec.get("pvf_reject_reason_missing_judge", False)):
            reasons.append("missing_judge_row")
        if bool(rec["pvf_reject_reason_parse"]):
            reasons.append("pred_parse_failure")
        if bool(rec["pvf_reject_reason_accuracy"]):
            reasons.append("accuracy_incorrect")
        if bool(rec["pvf_reject_reason_pedagogy"]):
            reasons.append("pedagogical_below_threshold")
        if bool(rec["pvf_reject_reason_silent_error"]):
            reasons.append("silent_error_true")
        reject_reasons.append(_reason_join(reasons))
    merged["pvf_reject_reasons"] = reject_reasons

    filtered = merged.loc[merged["pvf_keep"], ["example_id", "question", "answer_for_training", "source"]].copy()
    rejected = merged.loc[~merged["pvf_keep"]].copy()

    total = int(len(merged))
    kept = int(len(filtered))
    rejected_count = int(len(rejected))
    keep_rate = (kept / total) if total > 0 else 0.0

    if kept == 0:
        raise PVFError(
            "PVF rejected all synthetic rows; cannot build next-generation training dataset."
        )
    if keep_rate < float(policy.min_keep_ratio):
        raise PVFError(
            "PVF keep_rate below minimum threshold. "
            f"keep_rate={keep_rate:.4f}, min_keep_ratio={policy.min_keep_ratio:.4f}, "
            f"kept={kept}, total={total}"
        )

    reason_counts: dict[str, int] = {
        "missing_accuracy_row": int((merged["pvf_reject_reason_missing_accuracy"] == True).sum()),  # noqa: E712
        "missing_judge_row": int((merged["pvf_reject_reason_missing_judge"] == True).sum()),  # noqa: E712
        "pred_parse_failure": int((merged["pvf_reject_reason_parse"] == True).sum()),  # noqa: E712
        "accuracy_incorrect": int((merged["pvf_reject_reason_accuracy"] == True).sum()),  # noqa: E712
        "pedagogical_below_threshold": int((merged["pvf_reject_reason_pedagogy"] == True).sum()),  # noqa: E712
        "silent_error_true": int((merged["pvf_reject_reason_silent_error"] == True).sum()),  # noqa: E712
    }

    kept_rows = merged.loc[merged["pvf_keep"]].copy()
    rejected_rows = merged.loc[~merged["pvf_keep"]].copy()

    report = PVFReport(
        model_name=model_name,
        branch=branch,
        generation=int(generation),
        seed=int(seed),
        threshold_score=int(policy.threshold_score),
        min_keep_ratio=float(policy.min_keep_ratio),
        total_candidates=total,
        kept_count=kept,
        rejected_count=rejected_count,
        keep_rate=float(keep_rate),
        rejection_reason_counts=reason_counts,
        kept_accuracy_mean=float(kept_rows["is_correct"].mean()) if not kept_rows.empty else None,
        rejected_accuracy_mean=float(rejected_rows["is_correct"].mean()) if not rejected_rows.empty else None,
        kept_pedagogical_mean=float(kept_rows["overall_pedagogical_score"].mean()) if not kept_rows.empty else None,
        rejected_pedagogical_mean=float(rejected_rows["overall_pedagogical_score"].mean())
        if not rejected_rows.empty
        else None,
        kept_silent_error_rate=float(kept_rows["is_silent_error"].mean()) if not kept_rows.empty else None,
        rejected_silent_error_rate=float(rejected_rows["is_silent_error"].mean())
        if not rejected_rows.empty
        else None,
    )

    return PVFResult(
        filtered_training_df=filtered.reset_index(drop=True),
        rejected_examples_df=rejected.reset_index(drop=True),
        report=report,
    )


def apply_soft_pvf(
    *,
    synthetic_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    model_name: str,
    branch: str,
    generation: int,
    seed: int,
    policy: SoftPVFPolicy,
    allow_partial_inputs: bool = False,
) -> SoftPVFResult:
    """Apply pedagogical quality-weighted recycling with deterministic weighted sampling."""
    policy_name = _resolve_soft_policy_name(policy.policy_name)
    if not (0 <= int(policy.medium_quality_threshold) <= 8):
        raise PVFError(
            f"medium_quality_threshold must be in [0,8], got {policy.medium_quality_threshold}"
        )
    if not (0 <= int(policy.high_quality_threshold) <= 8):
        raise PVFError(f"high_quality_threshold must be in [0,8], got {policy.high_quality_threshold}")
    if int(policy.high_quality_threshold) < int(policy.medium_quality_threshold):
        raise PVFError(
            "high_quality_threshold must be >= medium_quality_threshold "
            f"(got {policy.high_quality_threshold} < {policy.medium_quality_threshold})"
        )
    if not (0.0 <= float(policy.min_keep_ratio) <= 1.0):
        raise PVFError(f"min_keep_ratio must be in [0,1], got {policy.min_keep_ratio}")

    _require_columns(synthetic_df, {"example_id", "question", "answer_for_training", "source"}, "synthetic_df")
    _require_columns(
        accuracy_df,
        {"example_id", "pred_parse_success", "accuracy_label", "is_correct"},
        "accuracy_df",
    )
    _require_columns(
        judge_df,
        {"example_id", "overall_pedagogical_score", "is_silent_error"},
        "judge_df",
    )

    _assert_unique_example_id(synthetic_df, "synthetic_df")
    _assert_unique_example_id(accuracy_df, "accuracy_df")
    _assert_unique_example_id(judge_df, "judge_df")
    if not allow_partial_inputs:
        _assert_equal_example_sets(
            synthetic_df=synthetic_df,
            accuracy_df=accuracy_df,
            judge_df=judge_df,
        )

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
        judge[["example_id", "overall_pedagogical_score", "is_silent_error"]].assign(_has_judge=True),
        on="example_id",
        how="left" if allow_partial_inputs else "inner",
        validate="one_to_one",
    )

    missing_accuracy = (
        merged["_has_accuracy"].isna() if "_has_accuracy" in merged.columns else pd.Series(False, index=merged.index)
    )
    missing_judge = merged["_has_judge"].isna() if "_has_judge" in merged.columns else pd.Series(False, index=merged.index)

    merged["pred_parse_success"] = merged["pred_parse_success"].fillna(False).astype(bool)
    merged["is_correct"] = merged["is_correct"].fillna(False).astype(bool)
    merged["accuracy_label"] = merged["accuracy_label"].astype(str).str.lower()
    merged["overall_pedagogical_score"] = pd.to_numeric(merged["overall_pedagogical_score"], errors="coerce")
    merged["is_silent_error"] = merged["is_silent_error"].fillna(True).astype(bool)

    invalid_score_mask = merged["overall_pedagogical_score"].isna() & ~missing_judge
    if invalid_score_mask.any():
        bad_ids = merged.loc[invalid_score_mask, "example_id"].astype(str).head(5).tolist()
        raise PVFError(
            "overall_pedagogical_score contains non-numeric values after strict merge. "
            f"sample_example_ids={bad_ids}"
        )

    effective_correct = (merged["is_correct"] == True) | (merged["accuracy_label"] == "correct")  # noqa: E712
    merged["is_correct_effective"] = effective_correct

    decisions: list[dict[str, Any]] = []
    for rec in merged.to_dict(orient="records"):
        example_id = str(rec["example_id"])
        weight, reason = _assign_soft_weight(
            pred_parse_success=bool(rec["pred_parse_success"]),
            is_silent_error=bool(rec["is_silent_error"]),
            is_correct=bool(rec["is_correct_effective"]),
            pedagogical_score=(
                None
                if pd.isna(rec.get("overall_pedagogical_score"))
                else float(rec.get("overall_pedagogical_score"))
            ),
            missing_accuracy=bool(pd.isna(rec.get("_has_accuracy"))),
            missing_judge=bool(pd.isna(rec.get("_has_judge"))),
            policy=policy,
        )
        kept, sampling_value = deterministic_weighted_keep(
            weight=weight,
            seed=seed,
            branch=branch,
            generation=generation,
            example_id=example_id,
        )
        decision = SoftPVFDecision(
            example_id=example_id,
            is_correct=bool(rec["is_correct_effective"]),
            pred_parse_success=bool(rec["pred_parse_success"]),
            overall_pedagogical_score=(
                None
                if pd.isna(rec.get("overall_pedagogical_score"))
                else float(rec.get("overall_pedagogical_score"))
            ),
            is_silent_error=bool(rec["is_silent_error"]),
            assigned_weight=float(weight),
            kept=bool(kept),
            decision_reason=str(reason),
            deterministic_score=float(sampling_value),
            sampling_value=float(sampling_value),
            policy_name=policy_name,
            branch=str(branch),
            generation=int(generation),
            seed=int(seed),
        )
        decisions.append(decision.model_dump(mode="python"))

    decisions_df = pd.DataFrame(decisions).sort_values("example_id").reset_index(drop=True)
    keep_ids = set(decisions_df.loc[decisions_df["kept"] == True, "example_id"].astype(str).tolist())  # noqa: E712
    training_df = (
        merged.loc[merged["example_id"].astype(str).isin(keep_ids), ["example_id", "question", "answer_for_training", "source"]]
        .copy()
        .sort_values("example_id")
        .reset_index(drop=True)
    )

    total = int(len(decisions_df))
    kept_count = int(decisions_df["kept"].sum())
    rejected_count = int(total - kept_count)
    keep_rate = (kept_count / total) if total > 0 else 0.0
    mean_assigned_weight = float(decisions_df["assigned_weight"].mean()) if total > 0 else 0.0
    effective_keep_rate = float(mean_assigned_weight)
    if kept_count == 0:
        raise PVFError("Soft PVF kept zero rows; cannot build next-generation training dataset.")
    if keep_rate < float(policy.min_keep_ratio):
        raise PVFError(
            "Soft PVF keep_rate below minimum threshold. "
            f"keep_rate={keep_rate:.4f}, min_keep_ratio={policy.min_keep_ratio:.4f}, "
            f"kept={kept_count}, total={total}"
        )

    enriched = merged.merge(
        decisions_df[["example_id", "kept", "assigned_weight", "decision_reason"]],
        on="example_id",
        how="left",
        validate="one_to_one",
    )
    kept_rows = enriched.loc[enriched["kept"] == True].copy()  # noqa: E712
    rejected_rows = enriched.loc[enriched["kept"] == False].copy()  # noqa: E712

    reason_counts = (
        decisions_df["decision_reason"].value_counts(dropna=False).sort_index().astype(int).to_dict()
    )
    weight_distribution = (
        decisions_df["assigned_weight"].map(lambda x: f"{float(x):.2f}")
        .value_counts(dropna=False)
        .sort_index()
        .astype(int)
        .to_dict()
    )

    report = SoftPVFReport(
        model_name=model_name,
        branch=branch,
        generation=int(generation),
        seed=int(seed),
        policy_name=policy_name,
        high_quality_threshold=int(policy.high_quality_threshold),
        medium_quality_threshold=int(policy.medium_quality_threshold),
        min_keep_ratio=float(policy.min_keep_ratio),
        total_candidates=total,
        kept_count=kept_count,
        rejected_count=rejected_count,
        keep_rate=float(keep_rate),
        effective_keep_rate=float(effective_keep_rate),
        mean_assigned_weight=float(mean_assigned_weight),
        decision_reason_counts=reason_counts,
        weight_distribution=weight_distribution,
        kept_accuracy_mean=float(kept_rows["is_correct_effective"].mean()) if not kept_rows.empty else None,
        rejected_accuracy_mean=float(rejected_rows["is_correct_effective"].mean())
        if not rejected_rows.empty
        else None,
        kept_pedagogical_mean=float(kept_rows["overall_pedagogical_score"].mean()) if not kept_rows.empty else None,
        rejected_pedagogical_mean=float(rejected_rows["overall_pedagogical_score"].mean())
        if not rejected_rows.empty
        else None,
        kept_silent_error_rate=float(kept_rows["is_silent_error"].mean()) if not kept_rows.empty else None,
        rejected_silent_error_rate=float(rejected_rows["is_silent_error"].mean())
        if not rejected_rows.empty
        else None,
    )

    return SoftPVFResult(
        training_df=training_df,
        decisions_df=decisions_df,
        report=report,
    )


def save_pvf_artifacts(
    *,
    result: PVFResult,
    filtered_path: Path,
    rejected_path: Path,
    report_path: Path,
) -> None:
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result.filtered_training_df.to_parquet(filtered_path, index=False)
    result.rejected_examples_df.to_parquet(rejected_path, index=False)
    report_path.write_text(
        json.dumps(result.report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_soft_pvf_artifacts(
    *,
    result: SoftPVFResult,
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
