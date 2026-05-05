from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CSRError(ValueError):
    """Raised when CSR invariants are violated."""


@dataclass(frozen=True)
class CSRPolicy:
    policy_name: str = "csr_medium"
    num_candidates: int = 3
    candidate_temperature: float = 0.7
    min_pair_quality_gap: float = 2.0
    require_best_correct: bool = True
    require_best_non_silent: bool = True
    allow_worst_incorrect: bool = True
    allow_worst_silent: bool = True
    max_no_pair_rate_warn: float = 0.60


class CSRReport(BaseModel):
    """Serializable report for CSR pair construction quality diagnostics."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    branch: str
    generation: int
    seed: int
    policy_name: str
    total_questions: int = Field(ge=0)
    total_candidates: int = Field(ge=0)
    pair_count: int = Field(ge=0)
    pair_construction_rate: float = Field(ge=0.0, le=1.0)
    no_pair_count: int = Field(ge=0)
    no_pair_reasons: dict[str, int]
    mean_quality_gap: float | None = None
    best_mean_score: float | None = None
    worst_mean_score: float | None = None
    best_accuracy: float | None = None
    worst_accuracy: float | None = None
    best_silent_rate: float | None = None
    worst_silent_rate: float | None = None
    num_candidates: int = Field(ge=1)
    min_pair_quality_gap: float = Field(ge=0.0)
    require_best_correct: bool
    require_best_non_silent: bool
    allow_worst_incorrect: bool
    allow_worst_silent: bool
    max_no_pair_rate_warn: float = Field(ge=0.0, le=1.0)
    no_pair_rate_warned: bool = False


@dataclass(frozen=True)
class CSRResult:
    candidates_df: pd.DataFrame
    candidate_scores_df: pd.DataFrame
    pairs_df: pd.DataFrame
    training_df: pd.DataFrame
    report: CSRReport


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise CSRError(f"{name} missing required columns: {sorted(missing)}")


def _effective_is_correct(*, is_correct: Any, accuracy_label: Any) -> bool:
    if bool(is_correct):
        return True
    return str(accuracy_label).strip().lower() == "correct"


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return bool(v)


def _float_or(v: Any, default: float) -> float:
    if pd.isna(v):
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _score_for_sort(rec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(_bool(rec.get("pred_parse_success"))),
        int(_bool(rec.get("is_correct_effective"))),
        int(not _bool(rec.get("is_silent_error"))),
        _float_or(rec.get("overall_pedagogical_score"), -1.0),
        _float_or(rec.get("reasoning_soundness"), -1.0),
        _float_or(rec.get("structure"), -1.0),
        -int(rec.get("candidate_id", 0)),
    )


def _worst_score_for_sort(rec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(_bool(rec.get("is_silent_error"))),
        int(not _bool(rec.get("is_correct_effective"))),
        int(not _bool(rec.get("pred_parse_success"))),
        -_float_or(rec.get("overall_pedagogical_score"), 99.0),
        -_float_or(rec.get("reasoning_soundness"), 99.0),
        -_float_or(rec.get("structure"), 99.0),
        int(rec.get("candidate_id", 0)),
    )


def _collect_defect_tags(rec: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if not _bool(rec.get("is_correct_effective")):
        tags.append("incorrect_answer")
    if not _bool(rec.get("pred_parse_success")):
        tags.append("parse_failure")
    if _bool(rec.get("is_silent_error")):
        tags.append("silent_error")
    if _float_or(rec.get("clarity"), -1.0) <= 0.0:
        tags.append("low_clarity")
    if _float_or(rec.get("structure"), -1.0) <= 0.0:
        tags.append("low_structure")
    if _float_or(rec.get("reasoning_soundness"), -1.0) <= 0.0:
        tags.append("low_reasoning")
    if _float_or(rec.get("terminology"), -1.0) <= 0.0:
        tags.append("low_terminology")
    return tags


def _build_contrastive_answer_for_training(
    *,
    question: str,
    weak_response: str,
    defect_tags: list[str],
    better_response: str,
    final_answer: str,
) -> str:
    tags_lines = "\n".join(f"- {x}" for x in defect_tags) if defect_tags else "- none_detected"
    return (
        f"Question:\n{question}\n\n"
        f"Weak explanation:\n{weak_response}\n\n"
        f"Detected pedagogical problems:\n{tags_lines}\n\n"
        f"Better explanation:\n{better_response}\n\n"
        f"Final answer:\n{final_answer}"
    )


def _validate_policy(policy: CSRPolicy) -> None:
    if int(policy.num_candidates) < 1:
        raise CSRError(f"num_candidates must be >= 1, got {policy.num_candidates}")
    if not (0.0 <= float(policy.max_no_pair_rate_warn) <= 1.0):
        raise CSRError(
            f"max_no_pair_rate_warn must be in [0,1], got {policy.max_no_pair_rate_warn}"
        )
    if float(policy.min_pair_quality_gap) < 0.0:
        raise CSRError(f"min_pair_quality_gap must be >= 0, got {policy.min_pair_quality_gap}")


def apply_csr(
    *,
    candidates_df: pd.DataFrame,
    candidate_scores_df: pd.DataFrame,
    model_name: str,
    branch: str,
    generation: int,
    seed: int,
    policy: CSRPolicy,
) -> CSRResult:
    """Build contrastive self-recycling pairs and training rows from scored candidates."""
    _validate_policy(policy)
    _require_columns(
        candidates_df,
        {"example_id", "candidate_id", "question", "raw_response"},
        "candidates_df",
    )
    _require_columns(
        candidate_scores_df,
        {
            "example_id",
            "candidate_id",
            "question",
            "raw_response",
            "parsed_final_answer",
            "pred_parse_success",
            "is_correct",
            "accuracy_label",
            "is_silent_error",
            "overall_pedagogical_score",
            "reasoning_soundness",
            "structure",
            "clarity",
            "terminology",
        },
        "candidate_scores_df",
    )
    if candidate_scores_df.empty:
        raise CSRError("CSR received empty candidate_scores_df")
    if candidate_scores_df.duplicated(subset=["example_id", "candidate_id"]).any():
        dup = int(candidate_scores_df.duplicated(subset=["example_id", "candidate_id"]).sum())
        raise CSRError(f"candidate_scores_df contains duplicate (example_id,candidate_id): {dup}")

    scored = candidate_scores_df.copy()
    scored["example_id"] = scored["example_id"].astype(str)
    scored["candidate_id"] = pd.to_numeric(scored["candidate_id"], errors="coerce").fillna(-1).astype(int)
    scored["pred_parse_success"] = scored["pred_parse_success"].fillna(False).astype(bool)
    scored["is_correct"] = scored["is_correct"].fillna(False).astype(bool)
    scored["accuracy_label"] = scored["accuracy_label"].astype(str)
    scored["is_correct_effective"] = scored.apply(
        lambda r: _effective_is_correct(is_correct=r.get("is_correct"), accuracy_label=r.get("accuracy_label")),
        axis=1,
    )
    scored["is_silent_error"] = scored["is_silent_error"].fillna(True).astype(bool)
    for col in ["overall_pedagogical_score", "reasoning_soundness", "structure", "clarity", "terminology"]:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    pair_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    no_pair_reasons: dict[str, int] = {}
    best_scores: list[float] = []
    worst_scores: list[float] = []
    best_correct_flags: list[bool] = []
    worst_correct_flags: list[bool] = []
    best_silent_flags: list[bool] = []
    worst_silent_flags: list[bool] = []

    for example_id, grp in scored.groupby("example_id", as_index=False):
        records = grp.to_dict(orient="records")
        if not records:
            no_pair_reasons["no_candidates"] = no_pair_reasons.get("no_candidates", 0) + 1
            pair_rows.append(
                {
                    "example_id": str(example_id),
                    "best_candidate_id": None,
                    "worst_candidate_id": None,
                    "best_score": None,
                    "worst_score": None,
                    "quality_gap": None,
                    "best_is_correct": None,
                    "worst_is_correct": None,
                    "best_is_silent_error": None,
                    "worst_is_silent_error": None,
                    "best_parse_success": None,
                    "worst_parse_success": None,
                    "defect_tags": json.dumps([], ensure_ascii=False),
                    "pair_status": "no_pair",
                    "no_pair_reason": "no_candidates",
                }
            )
            continue

        best = max(records, key=_score_for_sort)
        worst = max(records, key=_worst_score_for_sort)
        if int(best["candidate_id"]) == int(worst["candidate_id"]) and len(records) > 1:
            second_worst = sorted(records, key=_worst_score_for_sort, reverse=True)[1]
            worst = second_worst

        reason: str | None = None
        if int(best["candidate_id"]) == int(worst["candidate_id"]):
            reason = "best_equals_worst"
        if reason is None and policy.require_best_correct and (not _bool(best.get("is_correct_effective"))):
            reason = "best_not_correct"
        if reason is None and policy.require_best_non_silent and _bool(best.get("is_silent_error")):
            reason = "best_silent_error"

        best_score = _float_or(best.get("overall_pedagogical_score"), -1.0)
        worst_score = _float_or(worst.get("overall_pedagogical_score"), -1.0)
        quality_gap = float(best_score - worst_score)
        has_quality_gap = quality_gap >= float(policy.min_pair_quality_gap)
        has_hard_contrast = (
            _bool(best.get("is_correct_effective"))
            and (not _bool(best.get("is_silent_error")))
            and (
                (not _bool(worst.get("is_correct_effective")))
                or _bool(worst.get("is_silent_error"))
                or (not _bool(worst.get("pred_parse_success")))
            )
        )
        if reason is None and (not has_quality_gap) and (not has_hard_contrast):
            reason = "quality_gap_below_threshold"

        if reason is None and (not policy.allow_worst_incorrect) and (not _bool(worst.get("is_correct_effective"))):
            reason = "worst_incorrect_not_allowed"
        if reason is None and (not policy.allow_worst_silent) and _bool(worst.get("is_silent_error")):
            reason = "worst_silent_not_allowed"

        best_final = str(best.get("parsed_final_answer") or "").strip()
        if reason is None and not best_final:
            reason = "best_final_answer_missing"

        defect_tags = _collect_defect_tags(worst)
        if reason is not None:
            no_pair_reasons[reason] = no_pair_reasons.get(reason, 0) + 1
            pair_rows.append(
                {
                    "example_id": str(example_id),
                    "best_candidate_id": int(best["candidate_id"]),
                    "worst_candidate_id": int(worst["candidate_id"]),
                    "best_score": best_score,
                    "worst_score": worst_score,
                    "quality_gap": quality_gap,
                    "best_is_correct": bool(best["is_correct_effective"]),
                    "worst_is_correct": bool(worst["is_correct_effective"]),
                    "best_is_silent_error": bool(best["is_silent_error"]),
                    "worst_is_silent_error": bool(worst["is_silent_error"]),
                    "best_parse_success": bool(best["pred_parse_success"]),
                    "worst_parse_success": bool(worst["pred_parse_success"]),
                    "defect_tags": json.dumps(defect_tags, ensure_ascii=False),
                    "pair_status": "no_pair",
                    "no_pair_reason": reason,
                }
            )
            continue

        answer_for_training = _build_contrastive_answer_for_training(
            question=str(best.get("question", "")),
            weak_response=str(worst.get("raw_response", "")),
            defect_tags=defect_tags,
            better_response=str(best.get("raw_response", "")),
            final_answer=best_final,
        )
        training_rows.append(
            {
                "example_id": str(example_id),
                "question": str(best.get("question", "")),
                "answer_for_training": answer_for_training,
                "best_response": str(best.get("raw_response", "")),
                "worst_response": str(worst.get("raw_response", "")),
                "defect_tags": json.dumps(defect_tags, ensure_ascii=False),
                "final_answer": best_final,
                "source": "csr",
                "branch": branch,
                "generation": int(generation),
                "seed": int(seed),
            }
        )
        pair_rows.append(
            {
                "example_id": str(example_id),
                "best_candidate_id": int(best["candidate_id"]),
                "worst_candidate_id": int(worst["candidate_id"]),
                "best_score": best_score,
                "worst_score": worst_score,
                "quality_gap": quality_gap,
                "best_is_correct": bool(best["is_correct_effective"]),
                "worst_is_correct": bool(worst["is_correct_effective"]),
                "best_is_silent_error": bool(best["is_silent_error"]),
                "worst_is_silent_error": bool(worst["is_silent_error"]),
                "best_parse_success": bool(best["pred_parse_success"]),
                "worst_parse_success": bool(worst["pred_parse_success"]),
                "defect_tags": json.dumps(defect_tags, ensure_ascii=False),
                "pair_status": "paired",
                "no_pair_reason": "",
            }
        )
        best_scores.append(best_score)
        worst_scores.append(worst_score)
        best_correct_flags.append(bool(best["is_correct_effective"]))
        worst_correct_flags.append(bool(worst["is_correct_effective"]))
        best_silent_flags.append(bool(best["is_silent_error"]))
        worst_silent_flags.append(bool(worst["is_silent_error"]))

    pairs_df = pd.DataFrame(pair_rows).sort_values("example_id").reset_index(drop=True)
    training_df = pd.DataFrame(training_rows).sort_values("example_id").reset_index(drop=True)
    if training_df.empty:
        raise CSRError(
            "CSR produced zero valid contrastive pairs; cannot build next-generation training dataset."
        )

    total_questions = int(scored["example_id"].nunique())
    pair_count = int(len(training_df))
    no_pair_count = int(total_questions - pair_count)
    pair_construction_rate = float(pair_count / total_questions) if total_questions > 0 else 0.0
    no_pair_rate = float(no_pair_count / total_questions) if total_questions > 0 else 1.0

    report = CSRReport(
        model_name=model_name,
        branch=branch,
        generation=int(generation),
        seed=int(seed),
        policy_name=str(policy.policy_name),
        total_questions=total_questions,
        total_candidates=int(len(candidates_df)),
        pair_count=pair_count,
        pair_construction_rate=pair_construction_rate,
        no_pair_count=no_pair_count,
        no_pair_reasons=no_pair_reasons,
        mean_quality_gap=float(pd.Series(best_scores).sub(pd.Series(worst_scores)).mean())
        if best_scores and worst_scores
        else None,
        best_mean_score=float(pd.Series(best_scores).mean()) if best_scores else None,
        worst_mean_score=float(pd.Series(worst_scores).mean()) if worst_scores else None,
        best_accuracy=float(pd.Series(best_correct_flags).mean()) if best_correct_flags else None,
        worst_accuracy=float(pd.Series(worst_correct_flags).mean()) if worst_correct_flags else None,
        best_silent_rate=float(pd.Series(best_silent_flags).mean()) if best_silent_flags else None,
        worst_silent_rate=float(pd.Series(worst_silent_flags).mean()) if worst_silent_flags else None,
        num_candidates=int(policy.num_candidates),
        min_pair_quality_gap=float(policy.min_pair_quality_gap),
        require_best_correct=bool(policy.require_best_correct),
        require_best_non_silent=bool(policy.require_best_non_silent),
        allow_worst_incorrect=bool(policy.allow_worst_incorrect),
        allow_worst_silent=bool(policy.allow_worst_silent),
        max_no_pair_rate_warn=float(policy.max_no_pair_rate_warn),
        no_pair_rate_warned=bool(no_pair_rate > float(policy.max_no_pair_rate_warn)),
    )

    return CSRResult(
        candidates_df=candidates_df.reset_index(drop=True),
        candidate_scores_df=scored.reset_index(drop=True),
        pairs_df=pairs_df,
        training_df=training_df,
        report=report,
    )


def save_csr_artifacts(
    *,
    result: CSRResult,
    candidates_path: Path,
    candidate_scores_path: Path,
    pairs_path: Path,
    training_path: Path,
    report_path: Path,
) -> None:
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_scores_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result.candidates_df.to_parquet(candidates_path, index=False)
    result.candidate_scores_df.to_parquet(candidate_scores_path, index=False)
    result.pairs_df.to_parquet(pairs_path, index=False)
    result.training_df.to_parquet(training_path, index=False)
    report_path.write_text(
        json.dumps(result.report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
