from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

from didactic_collapse.pipeline.extract_answer import (
    ExtractedAnswer,
    extract_final_answer_result,
    normalize_extracted_answer,
)


@dataclass(frozen=True)
class AccuracyResult:
    """Typed scoring result for one prediction-gold pair."""

    extracted_answer: str | None
    normalized_predicted: str | None
    normalized_gold: str | None
    pred_parse_success: bool
    gold_parse_success: bool
    is_correct: bool
    accuracy_label: str
    parse_failure_reason: str | None


def _to_fraction(token: str) -> Fraction:
    """Convert numeric token to exact Fraction (int/decimal/fraction)."""
    if "/" in token:
        left, right = token.split("/", maxsplit=1)
        numerator = int(left)
        denominator = int(right)
        if denominator == 0:
            raise ValueError("Zero denominator")
        return Fraction(numerator, denominator)

    dec = Decimal(token)
    return Fraction(dec)


def _safe_parse_numeric(token: str | None) -> Fraction | None:
    if token is None:
        return None
    try:
        return _to_fraction(token)
    except (ValueError, InvalidOperation):
        return None


def score_prediction(
    *,
    model_output: str,
    gold_answer: str,
    parsed_final_answer: str | None = None,
) -> AccuracyResult:
    """Score factual accuracy for a single model output against gold answer.

    Rules:
    - Parse prediction via extraction pipeline (or provided parsed_final_answer)
    - Parse gold via safe numeric normalization
    - Compare exact numeric equivalence in Fraction space
    - Prefer explicit parse failure over uncertain guess
    """
    pred_extraction: ExtractedAnswer
    if parsed_final_answer is not None:
        normalized = normalize_extracted_answer(parsed_final_answer)
        if normalized is None:
            pred_extraction = ExtractedAnswer(
                extracted_answer=parsed_final_answer,
                normalized_answer=None,
                parse_success=False,
                parse_strategy="provided_parsed_answer",
                parse_failure_reason="provided_answer_not_numeric",
            )
        else:
            pred_extraction = ExtractedAnswer(
                extracted_answer=parsed_final_answer,
                normalized_answer=normalized,
                parse_success=True,
                parse_strategy="provided_parsed_answer",
                parse_failure_reason=None,
            )
    else:
        pred_extraction = extract_final_answer_result(model_output)

    normalized_gold = normalize_extracted_answer(gold_answer)
    gold_value = _safe_parse_numeric(normalized_gold)
    if normalized_gold is None or gold_value is None:
        return AccuracyResult(
            extracted_answer=pred_extraction.extracted_answer,
            normalized_predicted=pred_extraction.normalized_answer,
            normalized_gold=normalized_gold,
            pred_parse_success=pred_extraction.parse_success,
            gold_parse_success=False,
            is_correct=False,
            accuracy_label="gold_parse_failure",
            parse_failure_reason="gold_not_numeric",
        )

    pred_value = _safe_parse_numeric(pred_extraction.normalized_answer)
    if not pred_extraction.parse_success or pred_value is None:
        return AccuracyResult(
            extracted_answer=pred_extraction.extracted_answer,
            normalized_predicted=pred_extraction.normalized_answer,
            normalized_gold=normalized_gold,
            pred_parse_success=False,
            gold_parse_success=True,
            is_correct=False,
            accuracy_label="prediction_parse_failure",
            parse_failure_reason=pred_extraction.parse_failure_reason or "prediction_not_numeric",
        )

    is_correct = pred_value == gold_value
    return AccuracyResult(
        extracted_answer=pred_extraction.extracted_answer,
        normalized_predicted=pred_extraction.normalized_answer,
        normalized_gold=normalized_gold,
        pred_parse_success=True,
        gold_parse_success=True,
        is_correct=is_correct,
        accuracy_label="correct" if is_correct else "incorrect",
        parse_failure_reason=None,
    )


def evaluate_accuracy(outputs_df: pd.DataFrame, gold_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Compute robust accuracy table with explicit parse diagnostics."""
    merged = outputs_df.merge(gold_df[["example_id", "answer_gold"]], on="example_id", how="left")

    scored_rows: list[dict[str, Any]] = []
    for rec in merged.to_dict(orient="records"):
        result = score_prediction(
            model_output=str(rec.get("raw_response", "")),
            gold_answer=str(rec.get("answer_gold", "")),
            parsed_final_answer=rec.get("parsed_final_answer"),
        )

        row = dict(rec)
        row["extracted_answer"] = result.extracted_answer
        row["normalized_predicted"] = result.normalized_predicted
        row["normalized_gold"] = result.normalized_gold
        row["pred_parse_success"] = result.pred_parse_success
        row["gold_parse_success"] = result.gold_parse_success
        row["accuracy_label"] = result.accuracy_label
        row["accuracy_parse_failure_reason"] = result.parse_failure_reason
        row["is_correct"] = result.is_correct
        scored_rows.append(row)

    out_df = pd.DataFrame(scored_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_df
