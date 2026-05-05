from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import re
from typing import Any, Callable

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from didactic_collapse.pipeline.extract_answer import extract_final_answer_result, normalize_extracted_answer

logger = logging.getLogger(__name__)


class PAIRLiteError(ValueError):
    """Raised when PAIR-lite invariants are violated."""


class PAIRLiteRepairCallError(RuntimeError):
    """Raised when repair provider call fails after retry budget."""


@dataclass(frozen=True)
class PAIRLitePolicy:
    threshold_score: int = 6
    min_keep_ratio: float = 0.0
    max_repaired_chars: int = 12_000


class PAIRLiteDecision(BaseModel):
    """Per-example decision for PAIR-lite branch."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    branch: str
    generation: int
    seed: int
    is_correct: bool
    pred_parse_success: bool
    overall_pedagogical_score: float | None
    is_silent_error: bool
    action_initial: str
    action_final: str
    decision_reason: str
    repair_attempted: bool
    repair_success: bool
    original_response_hash: str
    repaired_response_hash: str | None = None
    repair_model_name: str | None = None
    repair_error: str | None = None


class PAIRLiteReport(BaseModel):
    """Serializable report for PAIR-lite coverage/repair diagnostics."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    branch: str
    generation: int
    seed: int
    threshold_score: int = Field(ge=0, le=8)
    min_keep_ratio: float = Field(ge=0.0, le=1.0)
    target_rows: int = Field(ge=0)
    kept_original_count: int = Field(ge=0)
    repaired_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    repair_attempted_count: int = Field(ge=0)
    repair_success_count: int = Field(ge=0)
    repair_failure_count: int = Field(ge=0)
    keep_original_rate: float = Field(ge=0.0, le=1.0)
    repair_rate: float = Field(ge=0.0, le=1.0)
    reject_rate: float = Field(ge=0.0, le=1.0)
    repair_success_rate: float = Field(ge=0.0, le=1.0)
    decision_reason_counts: dict[str, int]
    repair_model_name: str


@dataclass(frozen=True)
class PAIRLiteResult:
    training_df: pd.DataFrame
    decisions_df: pd.DataFrame
    repair_pairs_df: pd.DataFrame
    report: PAIRLiteReport


RepairCallable = Callable[[str, str, str, str], str]

_FINAL_LINE_RE = re.compile(r"^\s*final\s+answer\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _mask_secret(secret: str) -> str:
    if len(secret) <= 6:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-2:]}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise PAIRLiteError(f"{name} missing required columns: {sorted(missing)}")


def _assert_unique_example_id(df: pd.DataFrame, name: str) -> None:
    if df["example_id"].duplicated().any():
        dup_count = int(df["example_id"].duplicated().sum())
        raise PAIRLiteError(f"{name} contains duplicate example_id values: {dup_count}")


def _assert_equal_example_sets(
    *,
    synthetic_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> None:
    synth_ids = set(synthetic_df["example_id"].astype(str).tolist())
    acc_ids = set(accuracy_df["example_id"].astype(str).tolist())
    judge_ids = set(judge_df["example_id"].astype(str).tolist())
    gold_ids = set(gold_df["example_id"].astype(str).tolist())

    if synth_ids != acc_ids:
        only_synth = sorted(synth_ids.difference(acc_ids))[:5]
        only_acc = sorted(acc_ids.difference(synth_ids))[:5]
        raise PAIRLiteError(
            "Strict merge invariant failed: synthetic and accuracy example_id sets differ. "
            f"only_synthetic_sample={only_synth}, only_accuracy_sample={only_acc}"
        )
    if synth_ids != judge_ids:
        only_synth = sorted(synth_ids.difference(judge_ids))[:5]
        only_judge = sorted(judge_ids.difference(synth_ids))[:5]
        raise PAIRLiteError(
            "Strict merge invariant failed: synthetic and judge example_id sets differ. "
            f"only_synthetic_sample={only_synth}, only_judge_sample={only_judge}"
        )
    if synth_ids != gold_ids:
        only_synth = sorted(synth_ids.difference(gold_ids))[:5]
        only_gold = sorted(gold_ids.difference(synth_ids))[:5]
        raise PAIRLiteError(
            "Strict merge invariant failed: synthetic and gold example_id sets differ. "
            f"only_synthetic_sample={only_synth}, only_gold_sample={only_gold}"
        )


def _parse_retry_after_seconds(retry_after_value: str | None) -> float | None:
    if retry_after_value is None:
        return None
    raw = retry_after_value.strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


def _classify_retryable(exc: Exception) -> tuple[bool, str, float | None]:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, socket.timeout)):
        return True, "timeout", None
    if isinstance(exc, httpx.TransportError):
        return True, "transport_error", None
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            retry_after = _parse_retry_after_seconds(exc.response.headers.get("Retry-After"))
            return True, "http_429_rate_limited", retry_after
        if 500 <= code <= 599:
            return True, f"http_{code}_server_error", None
        return False, f"http_{code}_non_retryable", None
    return False, exc.__class__.__name__, None


class CerebrasPAIRLiteRepairClient:
    """Minimal OpenAI-compatible client for PAIR-lite pedagogical repair calls."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key_env: str,
        timeout_sec: int,
        max_retries: int,
        request_delay_sec: float,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model_name = str(model_name)
        self.max_retries = int(max_retries)
        self.request_delay_sec = max(0.0, float(request_delay_sec))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)

        api_key = (os.getenv(api_key_env) or "").strip()
        if not api_key:
            raise PAIRLiteError(f"Missing repair API key env: {api_key_env}")

        self._http = httpx.Client(
            timeout=httpx.Timeout(connect=15.0, read=float(timeout_sec), write=30.0, pool=30.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        logger.info(
            "pair_lite_repair_client_ready provider=cerebras model=%s base_url=%s key=%s",
            self.model_name,
            self.base_url,
            _mask_secret(api_key),
        )

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PAIRLiteError(
                "Malformed repair API response: missing choices[0].message.content"
            ) from exc

        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join(parts)
        return str(content)

    def repair(
        self,
        question: str,
        gold_answer: str,
        original_response: str,
        extracted_final_answer: str,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You rewrite math explanations to improve pedagogy.\n"
                        "Return only rewritten explanation text, no markdown fences.\n"
                        "Do not introduce new facts, do not solve a different problem.\n"
                        "Keep final answer exactly unchanged.\n"
                        "The last line must be exactly: Final answer: <same extracted final answer>\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Gold answer (reference only):\n{gold_answer}\n\n"
                        f"Required exact final answer value:\n{extracted_final_answer}\n\n"
                        f"Original model output:\n{original_response}\n\n"
                        "Rewrite requirements:\n"
                        "1) Keep the final answer exactly the same as required value.\n"
                        "2) Improve clarity and step-by-step structure.\n"
                        "3) Use correct terminology.\n"
                        "4) Remove unsupported reasoning jumps.\n"
                        "5) Do not add new facts.\n"
                        "6) Final line must be: Final answer: <same extracted final answer>\n"
                        "7) No markdown fences, no extra preface/suffix.\n"
                    ),
                },
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        max_attempts = max(1, self.max_retries + 1)
        attempt = 1
        while True:
            try:
                resp = self._http.post(f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = self._extract_content(data)
                if self.request_delay_sec > 0:
                    time.sleep(self.request_delay_sec)
                return content
            except Exception as exc:  # noqa: BLE001
                retryable, category, retry_after = _classify_retryable(exc)
                if (not retryable) or attempt >= max_attempts:
                    raise PAIRLiteRepairCallError(
                        f"PAIR-lite repair provider call failed category={category}: {exc}"
                    ) from exc
                sleep_sec = min(
                    30.0,
                    (retry_after if retry_after is not None else (1.5 * (2 ** max(0, attempt - 1))))
                    + random.uniform(0.0, 0.6),
                )
                logger.warning(
                    "pair_lite_repair_retry provider=cerebras model=%s category=%s attempt=%d/%d sleep_sec=%.2f",
                    self.model_name,
                    category,
                    attempt,
                    max_attempts,
                    sleep_sec,
                )
                time.sleep(max(0.0, sleep_sec))
                attempt += 1


def _resolve_expected_final_answer(
    *,
    parsed_final_answer: Any,
    normalized_predicted: Any,
    original_response: str,
) -> str:
    if parsed_final_answer is not None and not pd.isna(parsed_final_answer):
        norm = normalize_extracted_answer(str(parsed_final_answer))
        if norm is not None:
            return norm
    if normalized_predicted is not None and not pd.isna(normalized_predicted):
        norm = normalize_extracted_answer(str(normalized_predicted))
        if norm is not None:
            return norm
    extraction = extract_final_answer_result(original_response)
    if extraction.parse_success and extraction.normalized_answer is not None:
        return extraction.normalized_answer
    raise PAIRLiteError(
        "Could not resolve extracted final answer for repair candidate."
    )


def validate_repaired_response(
    *,
    repaired_response: str,
    extracted_final_answer: str,
    max_repaired_chars: int,
) -> str:
    """Strict validation for repaired explanation before it is used downstream."""
    text = str(repaired_response or "").strip()
    if not text:
        raise PAIRLiteError("Repaired response is empty")
    if len(text) > int(max_repaired_chars):
        raise PAIRLiteError(
            f"Repaired response too long: len={len(text)} > max_repaired_chars={max_repaired_chars}"
        )

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise PAIRLiteError("Repaired response has no non-empty lines")

    last = lines[-1]
    m = _FINAL_LINE_RE.match(last)
    if not m:
        raise PAIRLiteError("Repaired response must end with 'Final answer: <number>' line")

    expected_norm = normalize_extracted_answer(str(extracted_final_answer))
    if expected_norm is None:
        raise PAIRLiteError("Expected extracted final answer is not numeric")
    repaired_norm = normalize_extracted_answer(m.group(1))
    if repaired_norm is None:
        raise PAIRLiteError("Final answer line in repaired response is not numeric")
    if repaired_norm != expected_norm:
        raise PAIRLiteError(
            "Repaired response final answer changed. "
            f"expected_norm={expected_norm}, repaired_norm={repaired_norm}"
        )
    return text


def repair_pedagogical_explanation(
    *,
    question: str,
    gold_answer: str,
    original_response: str,
    extracted_final_answer: str,
    policy: PAIRLitePolicy,
    repair_callable: RepairCallable,
) -> str:
    repaired = repair_callable(
        str(question),
        str(gold_answer),
        str(original_response),
        str(extracted_final_answer),
    )
    return validate_repaired_response(
        repaired_response=repaired,
        extracted_final_answer=extracted_final_answer,
        max_repaired_chars=int(policy.max_repaired_chars),
    )


def apply_pair_lite(
    *,
    synthetic_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    model_name: str,
    branch: str,
    generation: int,
    seed: int,
    policy: PAIRLitePolicy,
    repair_model_name: str,
    repair_callable: RepairCallable,
    allow_partial_inputs: bool = False,
) -> PAIRLiteResult:
    """Apply PAIR-lite policy: keep high-quality correct rows, repair weak correct rows, reject unsafe rows."""
    if not (0 <= int(policy.threshold_score) <= 8):
        raise PAIRLiteError(f"threshold_score must be in [0,8], got {policy.threshold_score}")
    if not (0.0 <= float(policy.min_keep_ratio) <= 1.0):
        raise PAIRLiteError(f"min_keep_ratio must be in [0,1], got {policy.min_keep_ratio}")

    _require_columns(synthetic_df, {"example_id", "question", "answer_for_training", "source"}, "synthetic_df")
    _require_columns(
        accuracy_df,
        {"example_id", "pred_parse_success", "accuracy_label", "is_correct"},
        "accuracy_df",
    )
    _require_columns(judge_df, {"example_id", "overall_pedagogical_score", "is_silent_error"}, "judge_df")
    _require_columns(gold_df, {"example_id", "answer_gold"}, "gold_df")

    _assert_unique_example_id(synthetic_df, "synthetic_df")
    _assert_unique_example_id(accuracy_df, "accuracy_df")
    _assert_unique_example_id(judge_df, "judge_df")
    _assert_unique_example_id(gold_df, "gold_df")
    if not allow_partial_inputs:
        _assert_equal_example_sets(
            synthetic_df=synthetic_df,
            accuracy_df=accuracy_df,
            judge_df=judge_df,
            gold_df=gold_df,
        )

    synthetic = synthetic_df.copy()
    accuracy = accuracy_df.copy()
    judge = judge_df.copy()
    gold = gold_df.copy()
    for frame in (synthetic, accuracy, judge, gold):
        frame["example_id"] = frame["example_id"].astype(str)

    acc_cols = ["example_id", "pred_parse_success", "accuracy_label", "is_correct"]
    if "parsed_final_answer" in accuracy.columns:
        acc_cols.append("parsed_final_answer")
    if "normalized_predicted" in accuracy.columns:
        acc_cols.append("normalized_predicted")

    merged = synthetic.merge(
        accuracy[acc_cols].assign(_has_accuracy=True),
        on="example_id",
        how="left" if allow_partial_inputs else "inner",
        validate="one_to_one",
    ).merge(
        judge[["example_id", "overall_pedagogical_score", "is_silent_error"]].assign(_has_judge=True),
        on="example_id",
        how="left" if allow_partial_inputs else "inner",
        validate="one_to_one",
    ).merge(
        gold[["example_id", "answer_gold"]].assign(_has_gold=True),
        on="example_id",
        how="left" if allow_partial_inputs else "inner",
        validate="one_to_one",
    )

    merged["pred_parse_success"] = merged["pred_parse_success"].fillna(False).astype(bool)
    merged["is_correct"] = merged["is_correct"].fillna(False).astype(bool)
    merged["accuracy_label"] = merged["accuracy_label"].astype(str).str.lower()
    merged["overall_pedagogical_score"] = pd.to_numeric(merged["overall_pedagogical_score"], errors="coerce")
    merged["is_silent_error"] = merged["is_silent_error"].fillna(True).astype(bool)
    merged["answer_gold"] = merged["answer_gold"].astype(str)
    merged["_has_accuracy"] = merged["_has_accuracy"].fillna(False).astype(bool)
    merged["_has_judge"] = merged["_has_judge"].fillna(False).astype(bool)
    merged["_has_gold"] = merged["_has_gold"].fillna(False).astype(bool)

    decisions: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    repair_pairs: list[dict[str, Any]] = []

    for rec in merged.to_dict(orient="records"):
        example_id = str(rec["example_id"])
        question = str(rec.get("question", ""))
        original_response = str(rec.get("answer_for_training", ""))
        answer_gold = str(rec.get("answer_gold", ""))
        original_hash = _sha256_text(original_response)

        effective_is_correct = bool(rec["is_correct"]) or str(rec["accuracy_label"]) == "correct"
        parse_success = bool(rec["pred_parse_success"])
        is_silent = bool(rec["is_silent_error"])
        pedagogy = (
            None
            if pd.isna(rec.get("overall_pedagogical_score"))
            else float(rec.get("overall_pedagogical_score"))
        )
        missing_accuracy = not bool(rec["_has_accuracy"])
        missing_judge = not bool(rec["_has_judge"])
        missing_gold = not bool(rec["_has_gold"])

        action_initial = "reject"
        action_final = "reject"
        reason = ""
        repair_attempted = False
        repair_success = False
        repaired_response: str | None = None
        repaired_hash: str | None = None
        repair_error: str | None = None
        extracted_final_answer: str | None = None

        if missing_accuracy:
            reason = "missing_accuracy_row"
        elif missing_judge:
            reason = "missing_judge_row"
        elif missing_gold:
            reason = "missing_gold_row"
        elif not parse_success:
            reason = "pred_parse_failure"
        elif is_silent:
            reason = "silent_error_true"
        elif not effective_is_correct:
            reason = "accuracy_incorrect"
        elif pedagogy is not None and pedagogy >= int(policy.threshold_score):
            action_initial = "keep_original"
            action_final = "keep_original"
            reason = "high_quality"
            training_rows.append(
                {
                    "example_id": example_id,
                    "question": question,
                    "answer_for_training": original_response,
                    "source": "pair_lite",
                    "action": action_final,
                    "decision_reason": reason,
                    "branch": str(branch),
                    "generation": int(generation),
                    "seed": int(seed),
                    "model_name": str(model_name),
                }
            )
        else:
            action_initial = "repair_pedagogy"
            repair_attempted = True
            try:
                extracted_final_answer = _resolve_expected_final_answer(
                    parsed_final_answer=rec.get("parsed_final_answer"),
                    normalized_predicted=rec.get("normalized_predicted"),
                    original_response=original_response,
                )
                repaired_response = repair_pedagogical_explanation(
                    question=question,
                    gold_answer=answer_gold,
                    original_response=original_response,
                    extracted_final_answer=extracted_final_answer,
                    policy=policy,
                    repair_callable=repair_callable,
                )
                repaired_hash = _sha256_text(repaired_response)
                action_final = "repair_pedagogy"
                reason = "correct_but_pedagogically_weak"
                repair_success = True
                training_rows.append(
                    {
                        "example_id": example_id,
                        "question": question,
                        "answer_for_training": repaired_response,
                        "source": "pair_lite",
                        "action": action_final,
                        "decision_reason": reason,
                        "branch": str(branch),
                        "generation": int(generation),
                        "seed": int(seed),
                        "model_name": str(model_name),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                action_final = "reject"
                reason = "repair_failed"
                repair_success = False
                repair_error = str(exc)[:500]
                logger.warning(
                    "pair_lite_repair_failed example_id=%s category=%s message=%s",
                    example_id,
                    exc.__class__.__name__,
                    str(exc)[:280],
                )

            repair_pairs.append(
                {
                    "example_id": example_id,
                    "question": question,
                    "gold_answer": answer_gold,
                    "extracted_final_answer": extracted_final_answer,
                    "original_response": original_response,
                    "repaired_response": repaired_response,
                    "repair_model_name": repair_model_name,
                    "repair_status": "success" if repair_success else "failed",
                    "repair_error": repair_error,
                }
            )

        decision = PAIRLiteDecision(
            example_id=example_id,
            branch=str(branch),
            generation=int(generation),
            seed=int(seed),
            is_correct=bool(effective_is_correct),
            pred_parse_success=bool(parse_success),
            overall_pedagogical_score=pedagogy,
            is_silent_error=bool(is_silent),
            action_initial=action_initial,
            action_final=action_final,
            decision_reason=reason,
            repair_attempted=bool(repair_attempted),
            repair_success=bool(repair_success),
            original_response_hash=original_hash,
            repaired_response_hash=repaired_hash,
            repair_model_name=repair_model_name if repair_attempted else None,
            repair_error=repair_error,
        )
        decisions.append(decision.model_dump(mode="python"))

    training_df = (
        pd.DataFrame(
            training_rows,
            columns=[
                "example_id",
                "question",
                "answer_for_training",
                "source",
                "action",
                "decision_reason",
                "branch",
                "generation",
                "seed",
                "model_name",
            ],
        )
        .drop_duplicates(subset=["example_id"], keep="last")
        .sort_values("example_id")
        .reset_index(drop=True)
    )
    decisions_df = pd.DataFrame(decisions).sort_values("example_id").reset_index(drop=True)
    repair_pairs_df = (
        pd.DataFrame(
            repair_pairs,
            columns=[
                "example_id",
                "question",
                "gold_answer",
                "extracted_final_answer",
                "original_response",
                "repaired_response",
                "repair_model_name",
                "repair_status",
                "repair_error",
            ],
        )
        .sort_values("example_id")
        .reset_index(drop=True)
    )

    total = int(len(decisions_df))
    kept_original_count = int((decisions_df["action_final"] == "keep_original").sum()) if total > 0 else 0
    repaired_count = int((decisions_df["action_final"] == "repair_pedagogy").sum()) if total > 0 else 0
    rejected_count = int((decisions_df["action_final"] == "reject").sum()) if total > 0 else 0
    repair_attempted_count = int(decisions_df["repair_attempted"].sum()) if total > 0 else 0
    repair_success_count = int(decisions_df["repair_success"].sum()) if total > 0 else 0
    repair_failure_count = repair_attempted_count - repair_success_count

    keep_rate = (kept_original_count / total) if total > 0 else 0.0
    repair_rate = (repaired_count / total) if total > 0 else 0.0
    reject_rate = (rejected_count / total) if total > 0 else 0.0
    repair_success_rate = (
        (repair_success_count / repair_attempted_count) if repair_attempted_count > 0 else 0.0
    )

    if training_df.empty:
        raise PAIRLiteError("PAIR-lite rejected all rows (or all repairs failed); cannot build next-generation dataset.")
    if (len(training_df) / max(1, total)) < float(policy.min_keep_ratio):
        raise PAIRLiteError(
            "PAIR-lite keep ratio below min_keep_ratio. "
            f"keep_rate={len(training_df)/max(1,total):.4f}, min_keep_ratio={policy.min_keep_ratio:.4f}"
        )

    report = PAIRLiteReport(
        model_name=str(model_name),
        branch=str(branch),
        generation=int(generation),
        seed=int(seed),
        threshold_score=int(policy.threshold_score),
        min_keep_ratio=float(policy.min_keep_ratio),
        target_rows=total,
        kept_original_count=kept_original_count,
        repaired_count=repaired_count,
        rejected_count=rejected_count,
        repair_attempted_count=repair_attempted_count,
        repair_success_count=repair_success_count,
        repair_failure_count=repair_failure_count,
        keep_original_rate=float(keep_rate),
        repair_rate=float(repair_rate),
        reject_rate=float(reject_rate),
        repair_success_rate=float(repair_success_rate),
        decision_reason_counts=decisions_df["decision_reason"].value_counts().sort_index().astype(int).to_dict(),
        repair_model_name=str(repair_model_name),
    )

    return PAIRLiteResult(
        training_df=training_df,
        decisions_df=decisions_df,
        repair_pairs_df=repair_pairs_df,
        report=report,
    )


def save_pair_lite_artifacts(
    *,
    result: PAIRLiteResult,
    training_path: Path,
    decisions_path: Path,
    repair_pairs_path: Path,
    report_path: Path,
) -> None:
    training_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    repair_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result.training_df.to_parquet(training_path, index=False)
    result.decisions_df.to_parquet(decisions_path, index=False)
    result.repair_pairs_df.to_parquet(repair_pairs_path, index=False)
    report_path.write_text(
        json.dumps(result.report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
