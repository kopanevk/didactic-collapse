from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd

from didactic_collapse.clients.base import JudgeClient

logger = logging.getLogger(__name__)

_FAILURE_COLUMNS = [
    "run_id",
    "branch",
    "generation",
    "model_name",
    "example_id",
    "judge_provider",
    "judge_model",
    "error_category",
    "error_message",
]


def _stable_judge_input_key(
    *,
    question: str,
    gold_answer: str,
    raw_response: str,
    rubric_prompt: str,
    judge_provider: str,
    judge_model: str,
) -> str:
    payload = json.dumps(
        {
            "question": question,
            "gold_answer": gold_answer,
            "raw_response": raw_response,
            "rubric_prompt": rubric_prompt,
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "schema_version": "judge_rubric_v1",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_effective_request_delay_sec(*, judge_provider: str, request_delay_sec: float) -> float:
    """Resolve provider-aware request pacing delay.

    Supports process-level override for Cerebras judge calls so long-running
    resume runs can be tuned without changing config hash / manifests.
    """
    effective_delay = float(request_delay_sec)
    provider_normalized = judge_provider.strip().lower()
    if provider_normalized != "cerebras":
        return max(0.0, effective_delay)

    raw_override = os.getenv("DC_CEREBRAS_MIN_JUDGE_DELAY_SEC")
    if not raw_override:
        return max(0.0, effective_delay)
    try:
        override = float(raw_override)
    except ValueError:
        logger.warning(
            "invalid_cerebras_judge_delay_override value=%s env=DC_CEREBRAS_MIN_JUDGE_DELAY_SEC",
            raw_override,
        )
        return max(0.0, effective_delay)

    if override < 0:
        logger.warning(
            "negative_cerebras_judge_delay_override value=%s env=DC_CEREBRAS_MIN_JUDGE_DELAY_SEC",
            raw_override,
        )
        return max(0.0, effective_delay)
    effective_delay = max(effective_delay, override)
    if effective_delay > request_delay_sec:
        logger.info(
            "cerebras_judge_delay_override_applied configured_delay_sec=%.3f override_delay_sec=%.3f effective_delay_sec=%.3f",
            request_delay_sec,
            override,
            effective_delay,
        )
    return effective_delay


def _dedup_by_example_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "example_id" not in df.columns:
        raise ValueError("Artifact missing example_id column for deduplication")
    return df.drop_duplicates(subset=["example_id"], keep="last").reset_index(drop=True)


def _load_existing_success_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["example_id"])
    loaded = pd.read_parquet(path)
    if loaded.empty:
        return pd.DataFrame(columns=["example_id"])
    if "example_id" not in loaded.columns:
        raise ValueError(f"judge_partial artifact missing example_id: {path}")
    return _dedup_by_example_id(loaded)


def _load_existing_failure_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_FAILURE_COLUMNS)
    loaded = pd.read_parquet(path)
    if loaded.empty:
        return pd.DataFrame(columns=_FAILURE_COLUMNS)
    if "example_id" not in loaded.columns:
        raise ValueError(f"judge_failures artifact missing example_id: {path}")
    return _dedup_by_example_id(loaded)


def _build_failures_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_FAILURE_COLUMNS)
    out = pd.DataFrame(rows)
    for col in _FAILURE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[_FAILURE_COLUMNS].copy()


def _build_unresolved_failures_df(
    *,
    existing_failures: pd.DataFrame,
    failure_rows_buffer: list[dict[str, Any]],
    success_example_ids: set[str],
) -> pd.DataFrame:
    failures_df = pd.concat(
        [existing_failures, _build_failures_df(failure_rows_buffer)],
        ignore_index=True,
    )
    failures_df = _dedup_by_example_id(failures_df)
    if success_example_ids and "example_id" in failures_df.columns:
        failures_df = failures_df[~failures_df["example_id"].astype(str).isin(success_example_ids)].copy()
    return _dedup_by_example_id(failures_df)


def _write_progress_metadata(
    *,
    metadata_path: Path,
    total_rows: int,
    completed_rows: int,
    failed_rows: int,
    skipped_from_checkpoint: int,
    processed_this_run: int,
    partial_save_every_n: int,
    max_row_failures: int,
    continue_on_row_error: bool,
    unique_judge_inputs: int,
    duplicate_count: int,
    api_calls_saved_by_dedupe: int,
    request_delay_sec: float,
    runtime_stats: dict[str, Any] | None,
) -> None:
    runtime_stats = runtime_stats or {}
    payload = {
        "stage": "judge",
        "total_rows": int(total_rows),
        "completed_rows": int(completed_rows),
        "failed_rows": int(failed_rows),
        "skipped_from_checkpoint": int(skipped_from_checkpoint),
        "processed_this_run": int(processed_this_run),
        "partial_save_every_n": int(partial_save_every_n),
        "max_row_failures": int(max_row_failures),
        "continue_on_row_error": bool(continue_on_row_error),
        "request_delay_sec": float(request_delay_sec),
        "unique_judge_inputs": int(unique_judge_inputs),
        "duplicate_count": int(duplicate_count),
        "api_calls_saved_by_dedupe": int(api_calls_saved_by_dedupe),
        "cache_hits": int(runtime_stats.get("cache_hits", 0)),
        "cache_misses": int(runtime_stats.get("cache_misses", 0)),
        "api_calls": int(runtime_stats.get("api_calls", 0)),
        "rate_limit_retries": int(runtime_stats.get("rate_limit_retries", 0)),
        "total_sleep_sec_due_to_rate_limit": float(
            runtime_stats.get("total_sleep_sec_due_to_rate_limit", 0.0)
        ),
        "pacing_sleeps": int(runtime_stats.get("pacing_sleeps", 0)),
        "total_sleep_sec_due_to_pacing": float(runtime_stats.get("total_sleep_sec_due_to_pacing", 0.0)),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_judging(
    *,
    client: JudgeClient,
    generations_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    judge_provider: str,
    judge_model: str,
    rubric_prompt: str,
    out_path: Path,
    request_delay_sec: float = 0.0,
    partial_path: Path | None = None,
    failures_path: Path | None = None,
    metadata_path: Path | None = None,
    dedupe_map_path: Path | None = None,
    partial_save_every_n: int = 10,
    max_row_failures: int = 5,
    continue_on_row_error: bool = True,
) -> pd.DataFrame:
    required_gen = {"run_id", "branch", "generation", "model_name", "example_id", "raw_response"}
    required_q = {"example_id", "question", "answer_gold"}
    missing_gen = required_gen.difference(generations_df.columns)
    missing_q = required_q.difference(questions_df.columns)
    if missing_gen:
        raise ValueError(f"generations_df missing required columns: {sorted(missing_gen)}")
    if missing_q:
        raise ValueError(f"questions_df missing required columns: {sorted(missing_q)}")
    if partial_save_every_n <= 0:
        raise ValueError("partial_save_every_n must be > 0")
    if max_row_failures < 0:
        raise ValueError("max_row_failures must be >= 0")

    try:
        merged = generations_df.merge(
            questions_df[["example_id", "question", "answer_gold"]],
            on="example_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as exc:
        dup_gen_ids = (
            generations_df.loc[generations_df["example_id"].duplicated(), "example_id"].astype(str).head(5).tolist()
        )
        dup_question_ids = (
            questions_df.loc[questions_df["example_id"].duplicated(), "example_id"].astype(str).head(5).tolist()
        )
        raise ValueError(
            "Judge merge cardinality violation on example_id (expected one_to_one). "
            f"sample_duplicate_generation_ids={dup_gen_ids}, sample_duplicate_question_ids={dup_question_ids}"
        ) from exc
    missing_ref_mask = merged["_merge"] != "both"
    if missing_ref_mask.any():
        missing_ids = merged.loc[missing_ref_mask, "example_id"].astype(str).head(5).tolist()
        raise ValueError(
            "Judge merge produced rows without reference question/gold answer. "
            f"missing_count={int(missing_ref_mask.sum())}, sample_example_ids={missing_ids}"
        )
    merged = merged.drop(columns=["_merge"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = partial_path or out_path.with_name("judge_partial.parquet")
    failures_path = failures_path or out_path.with_name("judge_failures.parquet")
    metadata_path = metadata_path or out_path.with_name("judge_progress.json")
    dedupe_map_path = dedupe_map_path or out_path.with_name("judge_dedupe_map.parquet")

    existing_success = _load_existing_success_rows(partial_path)
    existing_failures = _load_existing_failure_rows(failures_path)
    skipped_ids = set(existing_success["example_id"].astype(str).tolist())

    merged_work = merged.copy()
    merged_work["example_id"] = merged_work["example_id"].astype(str)
    to_process = merged_work[~merged_work["example_id"].isin(skipped_ids)].copy()
    to_process["judge_input_key"] = to_process.apply(
        lambda row: _stable_judge_input_key(
            question=str(row["question"]),
            gold_answer=str(row["answer_gold"]),
            raw_response=str(row["raw_response"]),
            rubric_prompt=rubric_prompt,
            judge_provider=judge_provider,
            judge_model=judge_model,
        ),
        axis=1,
    )
    dedupe_map_df = to_process[
        ["example_id", "judge_input_key", "branch", "generation", "model_name"]
    ].copy()
    if dedupe_map_df.empty:
        pd.DataFrame(columns=["example_id", "judge_input_key", "branch", "generation", "model_name"]).to_parquet(
            dedupe_map_path, index=False
        )
    else:
        dedupe_map_df.to_parquet(dedupe_map_path, index=False)

    groups_by_input: dict[str, list[dict[str, Any]]] = {}
    for rec in to_process.to_dict(orient="records"):
        key = str(rec["judge_input_key"])
        groups_by_input.setdefault(key, []).append(rec)

    unique_judge_inputs = len(groups_by_input)
    duplicate_count = max(0, len(to_process) - unique_judge_inputs)
    api_calls_saved_by_dedupe = duplicate_count

    provider_normalized = judge_provider.strip().lower()
    effective_request_delay_sec = _resolve_effective_request_delay_sec(
        judge_provider=judge_provider,
        request_delay_sec=request_delay_sec,
    )
    should_pace = effective_request_delay_sec > 0 and provider_normalized not in {"mock", "stub", "mock_judge"}
    if should_pace:
        logger.info(
            "judge_request_pacing_enabled provider=%s model=%s delay_sec=%.3f batch_size=%d",
            judge_provider,
            judge_model,
            effective_request_delay_sec,
            unique_judge_inputs,
        )
    logger.info(
        "judge_row_resume_state total=%d skipped_from_checkpoint=%d remaining=%d unique_inputs=%d duplicates=%d partial=%s failures=%s dedupe_map=%s",
        len(merged_work),
        len(skipped_ids),
        len(to_process),
        unique_judge_inputs,
        duplicate_count,
        partial_path,
        failures_path,
        dedupe_map_path,
    )

    success_rows_buffer: list[dict[str, Any]] = []
    failure_rows_buffer: list[dict[str, Any]] = []
    processed_this_run = 0
    if hasattr(client, "reset_runtime_stats"):
        try:
            client.reset_runtime_stats()
        except Exception:  # noqa: BLE001
            logger.debug("judge_client_reset_runtime_stats_failed", exc_info=True)

    def flush_partial() -> tuple[pd.DataFrame, pd.DataFrame]:
        success_df = pd.concat(
            [existing_success, pd.DataFrame(success_rows_buffer)], ignore_index=True
        )
        success_df = _dedup_by_example_id(success_df)
        success_example_ids = set(success_df["example_id"].astype(str).tolist())

        failures_df = _build_unresolved_failures_df(
            existing_failures=existing_failures,
            failure_rows_buffer=failure_rows_buffer,
            success_example_ids=success_example_ids,
        )

        success_df.to_parquet(partial_path, index=False)
        failures_df.to_parquet(failures_path, index=False)
        runtime_stats: dict[str, Any] | None = None
        if hasattr(client, "get_runtime_stats_snapshot"):
            try:
                snapshot = client.get_runtime_stats_snapshot()
                if isinstance(snapshot, dict):
                    runtime_stats = snapshot
            except Exception:  # noqa: BLE001
                logger.debug("judge_client_get_runtime_stats_snapshot_failed", exc_info=True)
        _write_progress_metadata(
            metadata_path=metadata_path,
            total_rows=len(merged_work),
            completed_rows=len(success_df),
            failed_rows=len(failures_df),
            skipped_from_checkpoint=len(skipped_ids),
            processed_this_run=processed_this_run,
            partial_save_every_n=partial_save_every_n,
            max_row_failures=max_row_failures,
            continue_on_row_error=continue_on_row_error,
            unique_judge_inputs=unique_judge_inputs,
            duplicate_count=duplicate_count,
            api_calls_saved_by_dedupe=api_calls_saved_by_dedupe,
            request_delay_sec=effective_request_delay_sec,
            runtime_stats=runtime_stats,
        )
        return success_df, failures_df

    for idx, grouped in enumerate(groups_by_input.values(), start=1):
        if should_pace and idx > 1:
            time.sleep(effective_request_delay_sec)
        processed_this_run += len(grouped)
        rec = grouped[0]

        try:
            score = client.score(
                question=str(rec["question"]),
                gold_answer=str(rec["answer_gold"]),
                model_output=str(rec["raw_response"]),
                rubric_prompt=rubric_prompt,
            )
            for one in grouped:
                row = {
                    "run_id": one["run_id"],
                    "branch": one["branch"],
                    "generation": one["generation"],
                    "model_name": one["model_name"],
                    "example_id": str(one["example_id"]),
                    "judge_provider": judge_provider,
                    "judge_model": judge_model,
                }
                row.update(score)
                success_rows_buffer.append(row)
        except Exception as exc:  # noqa: BLE001
            for one in grouped:
                failure_rows_buffer.append(
                    {
                        "run_id": one["run_id"],
                        "branch": one["branch"],
                        "generation": one["generation"],
                        "model_name": one["model_name"],
                        "example_id": str(one["example_id"]),
                        "judge_provider": judge_provider,
                        "judge_model": judge_model,
                        "error_category": exc.__class__.__name__,
                        "error_message": str(exc)[:500],
                    }
                )
            logger.warning(
                "judge_row_failed example_id=%s duplicate_group_size=%d category=%s message=%s",
                rec["example_id"],
                len(grouped),
                exc.__class__.__name__,
                str(exc)[:280],
            )
            if not continue_on_row_error:
                flush_partial()
                raise

            success_example_ids = {
                str(v)
                for v in pd.concat([existing_success, pd.DataFrame(success_rows_buffer)], ignore_index=True)[
                    "example_id"
                ].tolist()
            }
            current_failure_count = len(
                _build_unresolved_failures_df(
                    existing_failures=existing_failures,
                    failure_rows_buffer=failure_rows_buffer,
                    success_example_ids=success_example_ids,
                )
            )
            if current_failure_count > max_row_failures:
                flush_partial()
                raise RuntimeError(
                    "Judge row failure threshold exceeded: "
                    f"failed={current_failure_count}, max_row_failures={max_row_failures}"
                )

        if processed_this_run % partial_save_every_n == 0:
            success_df, failures_df = flush_partial()
            logger.info(
                "judge_progress processed=%d/%d failed=%d completed=%d",
                processed_this_run,
                len(to_process),
                len(failures_df),
                len(success_df),
            )

    final_success_df, final_failures_df = flush_partial()

    if len(final_failures_df) > max_row_failures:
        raise RuntimeError(
            "Judge row failure threshold exceeded: "
            f"failed={len(final_failures_df)}, max_row_failures={max_row_failures}"
        )

    final_success_df.to_parquet(out_path, index=False)
    logger.info(
        "judge_stage_summary total=%d completed=%d failed=%d skipped=%d unique_inputs=%d duplicate_count=%d final=%s partial=%s failures=%s dedupe_map=%s",
        len(merged_work),
        len(final_success_df),
        len(final_failures_df),
        len(skipped_ids),
        unique_judge_inputs,
        duplicate_count,
        out_path,
        partial_path,
        failures_path,
        dedupe_map_path,
    )
    return final_success_df
