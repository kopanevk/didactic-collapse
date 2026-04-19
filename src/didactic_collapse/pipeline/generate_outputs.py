from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from didactic_collapse.clients.base import GenerationClient
from didactic_collapse.pipeline.extract_answer import extract_final_answer

logger = logging.getLogger(__name__)

_GEN_FAILURE_COLUMNS = [
    "run_id",
    "branch",
    "generation",
    "model_name",
    "example_id",
    "error_category",
    "error_message",
]


def build_generation_prompt(*, question: str, prompt_version: str) -> str:
    """Build generation prompt by explicit version for auditability."""
    q = str(question)
    if prompt_version.lower() in {"v2", "v2_strict_final", "strict_final_answer"}:
        return (
            "Solve the math word problem step by step.\n"
            "Keep the explanation concise and numerical.\n"
            "The final line MUST be exactly in this format:\n"
            "Final answer: <number>\n"
            "Do not add any text after the final line.\n\n"
            f"Question: {q}"
        )

    # Legacy behavior kept for backward-compatible runs.
    return f"Solve step-by-step and conclude with 'Final answer: ...'.\n\nQuestion: {q}"


def _dedup_by_example_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=["example_id"], keep="last").reset_index(drop=True)


def _load_partial(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["example_id"])
    loaded = pd.read_parquet(path)
    if loaded.empty:
        return pd.DataFrame(columns=["example_id"])
    if "example_id" not in loaded.columns:
        raise ValueError(f"generation partial artifact missing example_id: {path}")
    return _dedup_by_example_id(loaded)


def _load_failures(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_GEN_FAILURE_COLUMNS)
    loaded = pd.read_parquet(path)
    if loaded.empty:
        return pd.DataFrame(columns=_GEN_FAILURE_COLUMNS)
    if "example_id" not in loaded.columns:
        raise ValueError(f"generation failures artifact missing example_id: {path}")
    return _dedup_by_example_id(loaded)


def _build_failures_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_GEN_FAILURE_COLUMNS)
    df = pd.DataFrame(rows)
    for col in _GEN_FAILURE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[_GEN_FAILURE_COLUMNS].copy()


def _write_generation_metadata(
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
        "stage": "generation",
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


def run_generation(
    *,
    client: GenerationClient,
    examples_df: pd.DataFrame,
    model_name: str,
    branch: str,
    generation: int,
    run_id: str,
    prompt_version: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    out_path: Path,
    partial_path: Path | None = None,
    failures_path: Path | None = None,
    metadata_path: Path | None = None,
    partial_save_every_n: int = 10,
    max_row_failures: int = 5,
    continue_on_row_error: bool = True,
) -> pd.DataFrame:
    required_cols = {"example_id", "question"}
    missing = required_cols.difference(examples_df.columns)
    if missing:
        raise ValueError(f"examples_df missing required columns: {sorted(missing)}")
    if examples_df["example_id"].duplicated().any():
        dup_count = int(examples_df["example_id"].duplicated().sum())
        raise ValueError(f"examples_df contains duplicate example_id values: {dup_count}")
    if partial_save_every_n <= 0:
        raise ValueError("partial_save_every_n must be > 0")
    if max_row_failures < 0:
        raise ValueError("max_row_failures must be >= 0")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = partial_path or out_path.with_name("generation_partial.parquet")
    failures_path = failures_path or out_path.with_name("generation_failures.parquet")
    metadata_path = metadata_path or out_path.with_name("generation_progress.json")

    existing_success = _load_partial(partial_path)
    existing_failures = _load_failures(failures_path)
    skipped_ids = set(existing_success["example_id"].astype(str).tolist()) | set(
        existing_failures["example_id"].astype(str).tolist()
    )

    work_df = examples_df.copy()
    work_df["example_id"] = work_df["example_id"].astype(str)
    to_process = work_df[~work_df["example_id"].isin(skipped_ids)].copy()
    logger.info(
        "generation_row_resume_state total=%d skipped_from_checkpoint=%d remaining=%d partial=%s failures=%s",
        len(work_df),
        len(skipped_ids),
        len(to_process),
        partial_path,
        failures_path,
    )

    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    processed_this_run = 0

    def flush_partial() -> tuple[pd.DataFrame, pd.DataFrame]:
        success_df = pd.concat([existing_success, pd.DataFrame(success_rows)], ignore_index=True)
        success_df = _dedup_by_example_id(success_df)

        failures_df = pd.concat([existing_failures, _build_failures_df(failure_rows)], ignore_index=True)
        failures_df = _dedup_by_example_id(failures_df)

        success_df.to_parquet(partial_path, index=False)
        failures_df.to_parquet(failures_path, index=False)
        _write_generation_metadata(
            metadata_path=metadata_path,
            total_rows=len(work_df),
            completed_rows=len(success_df),
            failed_rows=len(failures_df),
            skipped_from_checkpoint=len(skipped_ids),
            processed_this_run=processed_this_run,
            partial_save_every_n=partial_save_every_n,
            max_row_failures=max_row_failures,
            continue_on_row_error=continue_on_row_error,
        )
        return success_df, failures_df

    for rec in to_process.to_dict(orient="records"):
        processed_this_run += 1
        try:
            question = str(rec["question"])
            prompt = build_generation_prompt(question=question, prompt_version=prompt_version)
            raw_response = client.generate(
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            success_rows.append(
                {
                    "run_id": run_id,
                    "branch": branch,
                    "generation": generation,
                    "model_name": model_name,
                    "example_id": rec["example_id"],
                    "prompt_version": prompt_version,
                    "prompt_text": prompt,
                    "raw_response": raw_response,
                    "parsed_final_answer": extract_final_answer(raw_response),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(
                {
                    "run_id": run_id,
                    "branch": branch,
                    "generation": generation,
                    "model_name": model_name,
                    "example_id": rec["example_id"],
                    "error_category": exc.__class__.__name__,
                    "error_message": str(exc)[:500],
                }
            )
            logger.warning(
                "generation_row_failed example_id=%s category=%s message=%s",
                rec["example_id"],
                exc.__class__.__name__,
                str(exc)[:280],
            )
            if not continue_on_row_error:
                flush_partial()
                raise
            current_failures = len(existing_failures) + len(failure_rows)
            if current_failures > max_row_failures:
                flush_partial()
                raise RuntimeError(
                    "Generation row failure threshold exceeded: "
                    f"failed={current_failures}, max_row_failures={max_row_failures}"
                )

        if processed_this_run % partial_save_every_n == 0:
            success_df, failures_df = flush_partial()
            logger.info(
                "generation_progress processed=%d/%d failed=%d completed=%d",
                processed_this_run,
                len(to_process),
                len(failures_df),
                len(success_df),
            )

    final_success, final_failures = flush_partial()
    if len(final_failures) > max_row_failures:
        raise RuntimeError(
            "Generation row failure threshold exceeded: "
            f"failed={len(final_failures)}, max_row_failures={max_row_failures}"
        )

    if final_success["example_id"].duplicated().any():
        dup_count = int(final_success["example_id"].duplicated().sum())
        raise ValueError(f"generation produced duplicate example_id rows: {dup_count}")

    final_success.to_parquet(out_path, index=False)
    logger.info(
        "generation_stage_summary total=%d completed=%d failed=%d skipped=%d final=%s partial=%s failures=%s",
        len(work_df),
        len(final_success),
        len(final_failures),
        len(skipped_ids),
        out_path,
        partial_path,
        failures_path,
    )
    return final_success
