from __future__ import annotations

import json
import logging
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
) -> None:
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

    existing_success = _load_existing_success_rows(partial_path)
    existing_failures = _load_existing_failure_rows(failures_path)
    skipped_ids = (
        set(existing_success["example_id"].astype(str).tolist())
        | set(existing_failures["example_id"].astype(str).tolist())
    )

    merged_work = merged.copy()
    merged_work["example_id"] = merged_work["example_id"].astype(str)
    to_process = merged_work[~merged_work["example_id"].isin(skipped_ids)].copy()

    provider_normalized = judge_provider.strip().lower()
    should_pace = request_delay_sec > 0 and provider_normalized not in {"mock", "stub", "mock_judge"}
    if should_pace:
        logger.info(
            "judge_request_pacing_enabled provider=%s model=%s delay_sec=%.3f batch_size=%d",
            judge_provider,
            judge_model,
            request_delay_sec,
            len(to_process),
        )
    logger.info(
        "judge_row_resume_state total=%d skipped_from_checkpoint=%d remaining=%d partial=%s failures=%s",
        len(merged_work),
        len(skipped_ids),
        len(to_process),
        partial_path,
        failures_path,
    )

    success_rows_buffer: list[dict[str, Any]] = []
    failure_rows_buffer: list[dict[str, Any]] = []
    processed_this_run = 0

    def flush_partial() -> tuple[pd.DataFrame, pd.DataFrame]:
        success_df = pd.concat(
            [existing_success, pd.DataFrame(success_rows_buffer)], ignore_index=True
        )
        success_df = _dedup_by_example_id(success_df)

        failures_df = pd.concat(
            [existing_failures, _build_failures_df(failure_rows_buffer)], ignore_index=True
        )
        failures_df = _dedup_by_example_id(failures_df)

        success_df.to_parquet(partial_path, index=False)
        failures_df.to_parquet(failures_path, index=False)
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
        )
        return success_df, failures_df

    for idx, rec in enumerate(to_process.to_dict(orient="records"), start=1):
        if should_pace and idx > 1:
            time.sleep(request_delay_sec)
        processed_this_run += 1

        try:
            score = client.score(
                question=str(rec["question"]),
                gold_answer=str(rec["answer_gold"]),
                model_output=str(rec["raw_response"]),
                rubric_prompt=rubric_prompt,
            )
            row = {
                "run_id": rec["run_id"],
                "branch": rec["branch"],
                "generation": rec["generation"],
                "model_name": rec["model_name"],
                "example_id": str(rec["example_id"]),
                "judge_provider": judge_provider,
                "judge_model": judge_model,
            }
            row.update(score)
            success_rows_buffer.append(row)
        except Exception as exc:  # noqa: BLE001
            failure_rows_buffer.append(
                {
                    "run_id": rec["run_id"],
                    "branch": rec["branch"],
                    "generation": rec["generation"],
                    "model_name": rec["model_name"],
                    "example_id": str(rec["example_id"]),
                    "judge_provider": judge_provider,
                    "judge_model": judge_model,
                    "error_category": exc.__class__.__name__,
                    "error_message": str(exc)[:500],
                }
            )
            logger.warning(
                "judge_row_failed example_id=%s category=%s message=%s",
                rec["example_id"],
                exc.__class__.__name__,
                str(exc)[:280],
            )
            if not continue_on_row_error:
                flush_partial()
                raise

            current_failure_count = len(existing_failures) + len(failure_rows_buffer)
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
        "judge_stage_summary total=%d completed=%d failed=%d skipped=%d final=%s partial=%s failures=%s",
        len(merged_work),
        len(final_success_df),
        len(final_failures_df),
        len(skipped_ids),
        out_path,
        partial_path,
        failures_path,
    )
    return final_success_df
