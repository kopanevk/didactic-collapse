from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from didactic_collapse.clients.judge_client import build_cerebras_judge_client
from didactic_collapse.config.settings import AppConfig
from didactic_collapse.prompts.prompt_registry import load_judge_prompt

_TARGET_BRANCHES = {"pure_recycling", "anchor_20_append"}
_TARGET_GENERATIONS = {0, 1}
_CELL_KEYS = [("pure_recycling", 0), ("pure_recycling", 1), ("anchor_20_append", 0), ("anchor_20_append", 1)]
_BUCKET_ORDER = ("silent_error_candidate", "low_pedagogical", "high_pedagogical", "ordinary")
_QWEN_DEFAULT_TAG = "qwen3_235b"
_QWEN_CONFIRMATORY_TAG = "qwen3_confirmatory"


@dataclass(frozen=True)
class JudgeSensitivityArtifacts:
    out_dir: Path
    selected_sample_csv: Path
    selected_sample_parquet: Path
    qwen_rejudge_results_csv: Path
    qwen_rejudge_results_parquet: Path
    qwen_rejudge_failures_csv: Path
    qwen_rejudge_failures_parquet: Path
    comparison_csv: Path
    comparison_parquet: Path
    summary_csv: Path
    summary_parquet: Path
    branch_summary_csv: Path
    branch_summary_parquet: Path
    seed_branch_summary_csv: Path
    seed_branch_summary_parquet: Path
    scatter_plot: Path
    delta_hist_plot: Path
    branch_bar_plot: Path
    metadata_json: Path


def _read_snapshot(run_dir: Path) -> dict[str, Any]:
    snap = run_dir / "run_config.snapshot.json"
    if not snap.exists():
        raise FileNotFoundError(f"Missing run snapshot: {snap}")
    return json.loads(snap.read_text(encoding="utf-8"))


def _safe_out_dir(out_dir: Path, run_dirs: Sequence[Path]) -> None:
    out_resolved = out_dir.resolve()
    for run in run_dirs:
        run_resolved = run.resolve()
        if out_resolved == run_resolved or run_resolved in out_resolved.parents:
            raise ValueError(
                "Output dir for judge sensitivity cannot be inside source run_dir. "
                f"out_dir={out_resolved}, run_dir={run_resolved}"
            )


def _seed_from_snapshot_payload(payload: dict[str, Any], run_dir: Path) -> int:
    try:
        return int(payload["config"]["project"]["seed"])
    except Exception:  # noqa: BLE001
        match = re.search(r"seed(\d+)", run_dir.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"Cannot determine source seed for run_dir={run_dir}")


def _load_context_rows_from_run(run_dir: Path) -> pd.DataFrame:
    payload = _read_snapshot(run_dir)
    source_seed = _seed_from_snapshot_payload(payload, run_dir)
    data_root = Path(payload["config"]["paths"]["data_root"])
    heldout_path = data_root / "splits" / "heldout_test.parquet"
    if not heldout_path.exists():
        raise FileNotFoundError(f"Missing heldout split for run: {heldout_path}")
    heldout = pd.read_parquet(heldout_path)
    heldout_needed = {"example_id", "question", "answer_gold"}
    missing_heldout = heldout_needed.difference(heldout.columns)
    if missing_heldout:
        raise ValueError(f"Heldout split missing columns {sorted(missing_heldout)} in {heldout_path}")
    heldout = heldout[list(heldout_needed)].copy()
    heldout["example_id"] = heldout["example_id"].astype(str)

    frames: list[pd.DataFrame] = []
    for gen_dir in sorted(run_dir.glob("*/*/gen_*")):
        outputs_path = gen_dir / "model_outputs.parquet"
        judge_path = gen_dir / "judge_outputs.parquet"
        if not outputs_path.exists() or not judge_path.exists():
            continue

        outputs = pd.read_parquet(outputs_path)
        judge = pd.read_parquet(judge_path)

        outputs_required = {
            "run_id",
            "branch",
            "generation",
            "model_name",
            "example_id",
            "raw_response",
        }
        missing_outputs = outputs_required.difference(outputs.columns)
        if missing_outputs:
            raise ValueError(f"model_outputs missing columns {sorted(missing_outputs)} in {outputs_path}")

        judge_required = {"example_id", "overall_pedagogical_score", "is_silent_error"}
        missing_judge = judge_required.difference(judge.columns)
        if missing_judge:
            raise ValueError(f"judge_outputs missing columns {sorted(missing_judge)} in {judge_path}")

        judge_cols = [
            "example_id",
            "judge_provider",
            "judge_model",
            "clarity",
            "structure",
            "terminology",
            "reasoning_soundness",
            "overall_pedagogical_score",
            "is_silent_error",
            "comment",
        ]
        for col in judge_cols:
            if col not in judge.columns:
                judge[col] = pd.NA
        judge_view = judge[judge_cols].copy()

        outputs["example_id"] = outputs["example_id"].astype(str)
        judge_view["example_id"] = judge_view["example_id"].astype(str)

        merged = outputs.merge(
            judge_view,
            on="example_id",
            how="left",
            validate="one_to_one",
        )
        merged = merged.merge(
            heldout,
            on="example_id",
            how="left",
            validate="one_to_one",
        )
        missing_q = merged["question"].isna() | merged["answer_gold"].isna()
        if missing_q.any():
            sample_ids = merged.loc[missing_q, "example_id"].astype(str).head(5).tolist()
            raise ValueError(
                "Missing question/answer_gold after merge for sensitivity source rows. "
                f"run_dir={run_dir}, sample_example_ids={sample_ids}"
            )

        merged["source_run_dir"] = str(run_dir)
        merged["source_step_dir"] = str(gen_dir)
        merged["source_seed"] = int(source_seed)
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["generation"] = out["generation"].astype(int)
    out = out[out["branch"].isin(_TARGET_BRANCHES) & out["generation"].isin(_TARGET_GENERATIONS)].copy()
    if out.empty:
        return out
    out = out.rename(
        columns={
            "judge_provider": "old_judge_provider",
            "judge_model": "old_judge_model",
            "clarity": "old_llama_clarity",
            "structure": "old_llama_structure",
            "terminology": "old_llama_terminology",
            "reasoning_soundness": "old_llama_reasoning_soundness",
            "overall_pedagogical_score": "old_llama_overall_pedagogical_score",
            "is_silent_error": "old_llama_is_silent_error",
            "comment": "old_llama_comment",
        }
    )
    return out.reset_index(drop=True)


def load_sensitivity_source_rows(run_dirs: Sequence[Path]) -> pd.DataFrame:
    if not run_dirs:
        raise ValueError("run_dirs must be non-empty")
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Source run_dir does not exist: {run_dir}")
        df = _load_context_rows_from_run(run_dir)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError("No valid source rows found for sensitivity check")
    source = pd.concat(frames, ignore_index=True)
    source["source_seed"] = pd.to_numeric(source.get("source_seed"), errors="coerce").astype("Int64")
    source["old_llama_overall_pedagogical_score"] = pd.to_numeric(
        source["old_llama_overall_pedagogical_score"], errors="coerce"
    )
    # Keep rows with old judge signal for direct old-vs-new comparison.
    source = source[
        source["old_llama_overall_pedagogical_score"].notna()
        & source["old_llama_is_silent_error"].notna()
    ].copy()
    if source.empty:
        raise ValueError("Source rows have no old judge fields for comparison")
    return source.reset_index(drop=True)


def _bucketize(row: pd.Series) -> str:
    old_silent = bool(row["old_llama_is_silent_error"])
    old_score = float(row["old_llama_overall_pedagogical_score"])
    if old_silent:
        return "silent_error_candidate"
    if old_score <= 3:
        return "low_pedagogical"
    if old_score >= 6:
        return "high_pedagogical"
    return "ordinary"


def _bucket_round_robin_sample(cell_df: pd.DataFrame, *, target: int) -> pd.DataFrame:
    if target <= 0:
        return pd.DataFrame(columns=cell_df.columns)

    buckets = {
        name: cell_df[cell_df["sensitivity_bucket"] == name].copy().reset_index(drop=True)
        for name in _BUCKET_ORDER
    }
    picks: list[pd.Series] = []
    idx_map = {name: 0 for name in _BUCKET_ORDER}
    while len(picks) < target:
        progressed = False
        for name in _BUCKET_ORDER:
            b = buckets[name]
            i = idx_map[name]
            if i < len(b) and len(picks) < target:
                picks.append(b.iloc[i])
                idx_map[name] += 1
                progressed = True
        if not progressed:
            break
    if not picks:
        return pd.DataFrame(columns=cell_df.columns)
    return pd.DataFrame(picks)


def select_balanced_sensitivity_sample(
    source_df: pd.DataFrame,
    *,
    total_n: int = 48,
    seed: int = 42,
) -> pd.DataFrame:
    if total_n <= 0:
        raise ValueError("total_n must be > 0")
    required = {
        "branch",
        "generation",
        "example_id",
        "question",
        "answer_gold",
        "raw_response",
        "old_llama_overall_pedagogical_score",
        "old_llama_is_silent_error",
    }
    missing = required.difference(source_df.columns)
    if missing:
        raise ValueError(f"source_df missing required columns: {sorted(missing)}")

    work = source_df.copy()
    work["generation"] = work["generation"].astype(int)
    work = work[work["branch"].isin(_TARGET_BRANCHES) & work["generation"].isin(_TARGET_GENERATIONS)].copy()
    if work.empty:
        raise ValueError("No rows in source_df for target branch/generation grid")
    work["sensitivity_bucket"] = work.apply(_bucketize, axis=1)

    base = total_n // len(_CELL_KEYS)
    extra = total_n % len(_CELL_KEYS)
    target_per_cell: dict[tuple[str, int], int] = {}
    for idx, key in enumerate(_CELL_KEYS):
        target_per_cell[key] = base + (1 if idx < extra else 0)

    rng = pd.Series(range(len(work))).sample(frac=1.0, random_state=seed)
    work = work.iloc[rng.index].reset_index(drop=True)

    selected_chunks: list[pd.DataFrame] = []
    shortfall = 0
    for cell in _CELL_KEYS:
        branch, generation = cell
        cell_df = work[(work["branch"] == branch) & (work["generation"] == generation)].copy()
        target = target_per_cell[cell]
        if cell_df.empty:
            raise ValueError(f"Balanced sampling failed: no rows for cell {cell}")
        picked_df = _bucket_round_robin_sample(cell_df, target=target)
        if len(picked_df) < target:
            shortfall += target - len(picked_df)
        selected_chunks.append(picked_df)

    selected = pd.concat(selected_chunks, ignore_index=True)
    selected = selected.drop_duplicates(
        subset=["source_run_dir", "source_step_dir", "example_id"], keep="first"
    ).reset_index(drop=True)

    if shortfall > 0:
        already = set(
            selected[["source_run_dir", "source_step_dir", "example_id"]]
            .astype(str)
            .agg("|".join, axis=1)
            .tolist()
        )
        work_key = work[["source_run_dir", "source_step_dir", "example_id"]].astype(str).agg("|".join, axis=1)
        remainder = work[~work_key.isin(already)].copy()
        if not remainder.empty:
            fill = remainder.head(shortfall)
            selected = pd.concat([selected, fill], ignore_index=True)

    if len(selected) > total_n:
        selected = selected.head(total_n).copy()
    if len(selected) == 0:
        raise ValueError("Balanced sample selection produced zero rows")
    return selected.reset_index(drop=True)


def select_confirmatory_seed_branch_sample(
    source_df: pd.DataFrame,
    *,
    total_n: int = 48,
    seed: int = 42,
    focus_generation: int = 1,
    allowed_run_dirs: Sequence[str] | None = None,
) -> pd.DataFrame:
    if total_n <= 0:
        raise ValueError("total_n must be > 0")
    required = {
        "source_run_dir",
        "source_seed",
        "branch",
        "generation",
        "example_id",
        "question",
        "answer_gold",
        "raw_response",
        "old_llama_overall_pedagogical_score",
        "old_llama_is_silent_error",
    }
    missing = required.difference(source_df.columns)
    if missing:
        raise ValueError(f"source_df missing required columns: {sorted(missing)}")

    work = source_df.copy()
    if allowed_run_dirs:
        allowed = {str(x) for x in allowed_run_dirs}
        work = work[work["source_run_dir"].astype(str).isin(allowed)].copy()
    if work.empty:
        raise ValueError("No rows available after allowed_run_dirs filtering")

    work["generation"] = pd.to_numeric(work["generation"], errors="coerce").astype("Int64")
    work["source_seed"] = pd.to_numeric(work["source_seed"], errors="coerce").astype("Int64")
    work = work[
        work["branch"].isin(_TARGET_BRANCHES)
        & (work["generation"] == int(focus_generation))
        & work["source_seed"].notna()
    ].copy()
    if work.empty:
        raise ValueError(
            f"No rows available for confirmatory sampling at generation={focus_generation}"
        )

    work["sensitivity_bucket"] = work.apply(_bucketize, axis=1)
    work = work.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    valid_seeds = sorted(
        int(s)
        for s in work["source_seed"].dropna().astype(int).unique().tolist()
        if all(
            not work[(work["source_seed"] == int(s)) & (work["branch"] == b)].empty
            for b in sorted(_TARGET_BRANCHES)
        )
    )
    if not valid_seeds:
        raise ValueError("No seeds with both target branches for confirmatory sampling")

    branches = ["pure_recycling", "anchor_20_append"]
    cell_keys = [(seed_value, branch) for seed_value in valid_seeds for branch in branches]
    if total_n < len(cell_keys):
        raise ValueError(
            "total_n too small for balanced seed-branch confirmatory sampling. "
            f"total_n={total_n}, required_min={len(cell_keys)}"
        )

    base = total_n // len(cell_keys)
    extra = total_n % len(cell_keys)

    selected_parts: list[pd.DataFrame] = []
    for idx, (seed_value, branch) in enumerate(cell_keys):
        target = base + (1 if idx < extra else 0)
        if target == 0:
            continue
        cell_df = work[(work["source_seed"] == seed_value) & (work["branch"] == branch)].copy()
        if len(cell_df) < target:
            raise ValueError(
                "Insufficient rows for balanced confirmatory cell. "
                f"seed={seed_value}, branch={branch}, available={len(cell_df)}, target={target}"
            )
        picked = _bucket_round_robin_sample(cell_df, target=target)
        if len(picked) < target:
            raise ValueError(
                "Could not satisfy confirmatory bucketed sampling target. "
                f"seed={seed_value}, branch={branch}, selected={len(picked)}, target={target}"
            )
        selected_parts.append(picked)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.drop_duplicates(
        subset=["source_run_dir", "source_step_dir", "example_id"], keep="first"
    ).reset_index(drop=True)
    if len(selected) != total_n:
        raise ValueError(
            f"Confirmatory sample size mismatch: selected={len(selected)}, expected={total_n}"
        )
    return selected


def _score_one_with_repair(
    *,
    client: Any,
    rubric_prompt: str,
    question: str,
    gold_answer: str,
    raw_response: str,
) -> dict[str, Any]:
    if hasattr(client, "score_typed"):
        parsed = client.score_typed(
            question=question,
            gold_answer=gold_answer,
            model_output=raw_response,
            rubric_prompt=rubric_prompt,
        )
        score = parsed.score.model_dump()
        return {
            **score,
            "qwen_repair_applied": bool(parsed.repair_applied),
            "qwen_repair_actions": "|".join(parsed.repair_actions),
        }

    score = client.score(
        question=question,
        gold_answer=gold_answer,
        model_output=raw_response,
        rubric_prompt=rubric_prompt,
    )
    return {
        **score,
        "qwen_repair_applied": False,
        "qwen_repair_actions": "",
    }


def run_qwen_rejudge(
    *,
    sample_df: pd.DataFrame,
    cfg: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"example_id", "question", "answer_gold", "raw_response"}
    missing = required.difference(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    if cfg.judge.provider.strip().lower() != "cerebras":
        raise ValueError("Qwen sensitivity re-judge currently supports provider=cerebras only")

    client = build_cerebras_judge_client(
        model_name=cfg.judge.model_name,
        base_url=cfg.judge.base_url,
        api_key_env=cfg.judge.api_key_env,
        timeout_sec=cfg.judge.timeout_sec,
        max_retries=cfg.judge.max_retries,
    )
    rubric_prompt = load_judge_prompt(cfg.paths.prompt_dir)

    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for rec in sample_df.to_dict(orient="records"):
        try:
            score = _score_one_with_repair(
                client=client,
                rubric_prompt=rubric_prompt,
                question=str(rec["question"]),
                gold_answer=str(rec["answer_gold"]),
                raw_response=str(rec["raw_response"]),
            )
            result_rows.append(
                {
                    "source_run_dir": rec.get("source_run_dir"),
                    "source_step_dir": rec.get("source_step_dir"),
                    "source_seed": rec.get("source_seed"),
                    "run_id": rec.get("run_id"),
                    "branch": rec.get("branch"),
                    "generation": rec.get("generation"),
                    "model_name": rec.get("model_name"),
                    "example_id": rec.get("example_id"),
                    "qwen_judge_provider": cfg.judge.provider,
                    "qwen_judge_model": cfg.judge.model_name,
                    "qwen_clarity": score["clarity"],
                    "qwen_structure": score["structure"],
                    "qwen_terminology": score["terminology"],
                    "qwen_reasoning_soundness": score["reasoning_soundness"],
                    "qwen_overall_pedagogical_score": score["overall_pedagogical_score"],
                    "qwen_is_silent_error": score["is_silent_error"],
                    "qwen_comment": score["comment"],
                    "qwen_repair_applied": score["qwen_repair_applied"],
                    "qwen_repair_actions": score["qwen_repair_actions"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(
                {
                    "source_run_dir": rec.get("source_run_dir"),
                    "source_step_dir": rec.get("source_step_dir"),
                    "source_seed": rec.get("source_seed"),
                    "run_id": rec.get("run_id"),
                    "branch": rec.get("branch"),
                    "generation": rec.get("generation"),
                    "model_name": rec.get("model_name"),
                    "example_id": rec.get("example_id"),
                    "error_category": exc.__class__.__name__,
                    "error_message": str(exc)[:500],
                }
            )

    result_cols = [
        "source_run_dir",
        "source_step_dir",
        "source_seed",
        "run_id",
        "branch",
        "generation",
        "model_name",
        "example_id",
        "qwen_judge_provider",
        "qwen_judge_model",
        "qwen_clarity",
        "qwen_structure",
        "qwen_terminology",
        "qwen_reasoning_soundness",
        "qwen_overall_pedagogical_score",
        "qwen_is_silent_error",
        "qwen_comment",
        "qwen_repair_applied",
        "qwen_repair_actions",
    ]
    failure_cols = [
        "source_run_dir",
        "source_step_dir",
        "source_seed",
        "run_id",
        "branch",
        "generation",
        "model_name",
        "example_id",
        "error_category",
        "error_message",
    ]
    results = pd.DataFrame(result_rows, columns=result_cols)
    failures = pd.DataFrame(failure_rows, columns=failure_cols)
    if results.empty:
        raise RuntimeError(
            "Qwen re-judge produced zero successful rows. "
            f"failures={len(failures)}"
        )
    return results, failures


def build_judge_sensitivity_comparison(
    *,
    selected_sample_df: pd.DataFrame,
    qwen_results_df: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["source_run_dir", "source_step_dir", "example_id"]
    for key in key_cols:
        if key not in selected_sample_df.columns or key not in qwen_results_df.columns:
            raise ValueError(f"Missing join key for comparison: {key}")

    merged = selected_sample_df.merge(
        qwen_results_df,
        on=key_cols + ["run_id", "branch", "generation", "model_name"],
        how="inner",
        validate="one_to_one",
    )
    if "source_seed" not in merged.columns:
        left_seed = "source_seed_x"
        right_seed = "source_seed_y"
        if left_seed in merged.columns and right_seed in merged.columns:
            merged["source_seed"] = merged[left_seed].fillna(merged[right_seed])
        elif left_seed in merged.columns:
            merged["source_seed"] = merged[left_seed]
        elif right_seed in merged.columns:
            merged["source_seed"] = merged[right_seed]

    merged["old_llama_overall_pedagogical_score"] = pd.to_numeric(
        merged["old_llama_overall_pedagogical_score"], errors="coerce"
    )
    merged["qwen_overall_pedagogical_score"] = pd.to_numeric(
        merged["qwen_overall_pedagogical_score"], errors="coerce"
    )
    merged["score_delta"] = (
        merged["qwen_overall_pedagogical_score"] - merged["old_llama_overall_pedagogical_score"]
    )
    merged["abs_score_delta"] = merged["score_delta"].abs()
    merged["silent_error_agreement"] = (
        merged["old_llama_is_silent_error"].astype(bool)
        == merged["qwen_is_silent_error"].astype(bool)
    )
    return merged


def _corr_safe(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or len(b) < 3:
        return math.nan
    return float(a.corr(b))


def build_judge_sensitivity_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")

    old = comparison_df["old_llama_overall_pedagogical_score"]
    qwen = comparison_df["qwen_overall_pedagogical_score"]
    summary = {
        "n_examples": int(len(comparison_df)),
        "mean_old_llama_score": float(old.mean()),
        "mean_qwen_score": float(qwen.mean()),
        "mean_abs_score_delta": float(comparison_df["abs_score_delta"].mean()),
        "score_correlation": _corr_safe(old, qwen),
        "silent_error_agreement_rate": float(comparison_df["silent_error_agreement"].mean()),
        "qwen_repair_applied_rate": float(comparison_df["qwen_repair_applied"].astype(bool).mean()),
    }

    # Branch conclusion check on Gen-1 (anchor minus pure).
    gen1 = comparison_df[comparison_df["generation"].astype(int) == 1].copy()
    if not gen1.empty:
        old_means = gen1.groupby("branch")["old_llama_overall_pedagogical_score"].mean()
        qwen_means = gen1.groupby("branch")["qwen_overall_pedagogical_score"].mean()
        if "anchor_20_append" in old_means.index and "pure_recycling" in old_means.index:
            old_delta = float(old_means["anchor_20_append"] - old_means["pure_recycling"])
            qwen_delta = float(qwen_means["anchor_20_append"] - qwen_means["pure_recycling"])
            summary["gen1_old_anchor_minus_pure_pedagogical"] = old_delta
            summary["gen1_qwen_anchor_minus_pure_pedagogical"] = qwen_delta
            summary["gen1_branch_conclusion_changed"] = bool(
                (old_delta >= 0 and qwen_delta < 0) or (old_delta < 0 and qwen_delta >= 0)
            )
        else:
            summary["gen1_old_anchor_minus_pure_pedagogical"] = math.nan
            summary["gen1_qwen_anchor_minus_pure_pedagogical"] = math.nan
            summary["gen1_branch_conclusion_changed"] = pd.NA
    else:
        summary["gen1_old_anchor_minus_pure_pedagogical"] = math.nan
        summary["gen1_qwen_anchor_minus_pure_pedagogical"] = math.nan
        summary["gen1_branch_conclusion_changed"] = pd.NA

    return pd.DataFrame([summary])


def build_judge_sensitivity_branch_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (branch, generation), grp in comparison_df.groupby(["branch", "generation"], as_index=False):
        rows.append(
            {
                "branch": str(branch),
                "generation": int(generation),
                "sample_count": int(len(grp)),
                "old_llama_pedagogical_mean": float(grp["old_llama_overall_pedagogical_score"].mean()),
                "qwen_pedagogical_mean": float(grp["qwen_overall_pedagogical_score"].mean()),
                "old_llama_silent_error_rate": float(grp["old_llama_is_silent_error"].astype(bool).mean()),
                "qwen_silent_error_rate": float(grp["qwen_is_silent_error"].astype(bool).mean()),
                "score_delta_mean": float(grp["score_delta"].mean()),
                "silent_error_agreement_rate": float(grp["silent_error_agreement"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["generation", "branch"]).reset_index(drop=True)


def build_judge_sensitivity_seed_branch_summary(
    comparison_df: pd.DataFrame,
    *,
    focus_generation: int = 1,
) -> pd.DataFrame:
    required = {
        "source_seed",
        "branch",
        "generation",
        "old_llama_overall_pedagogical_score",
        "qwen_overall_pedagogical_score",
        "old_llama_is_silent_error",
        "qwen_is_silent_error",
        "silent_error_agreement",
        "score_delta",
    }
    missing = required.difference(comparison_df.columns)
    if missing:
        raise ValueError(f"comparison_df missing columns for seed-branch summary: {sorted(missing)}")

    gen_df = comparison_df[comparison_df["generation"].astype(int) == int(focus_generation)].copy()
    if gen_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for seed_value, seed_grp in gen_df.groupby("source_seed", as_index=False):
        pure = seed_grp[seed_grp["branch"] == "pure_recycling"]
        anchor = seed_grp[seed_grp["branch"] == "anchor_20_append"]
        if pure.empty or anchor.empty:
            continue

        old_pure = float(pure["old_llama_overall_pedagogical_score"].mean())
        old_anchor = float(anchor["old_llama_overall_pedagogical_score"].mean())
        qwen_pure = float(pure["qwen_overall_pedagogical_score"].mean())
        qwen_anchor = float(anchor["qwen_overall_pedagogical_score"].mean())
        old_delta = old_anchor - old_pure
        qwen_delta = qwen_anchor - qwen_pure
        rows.append(
            {
                "source_seed": int(seed_value),
                "generation": int(focus_generation),
                "sample_count_seed_generation": int(len(seed_grp)),
                "sample_count_pure": int(len(pure)),
                "sample_count_anchor": int(len(anchor)),
                "old_llama_pure_mean": old_pure,
                "old_llama_anchor_mean": old_anchor,
                "old_llama_anchor_minus_pure": old_delta,
                "qwen_pure_mean": qwen_pure,
                "qwen_anchor_mean": qwen_anchor,
                "qwen_anchor_minus_pure": qwen_delta,
                "old_llama_pure_silent_error_rate": float(pure["old_llama_is_silent_error"].astype(bool).mean()),
                "old_llama_anchor_silent_error_rate": float(
                    anchor["old_llama_is_silent_error"].astype(bool).mean()
                ),
                "qwen_pure_silent_error_rate": float(pure["qwen_is_silent_error"].astype(bool).mean()),
                "qwen_anchor_silent_error_rate": float(anchor["qwen_is_silent_error"].astype(bool).mean()),
                "seed_silent_error_agreement_rate": float(seed_grp["silent_error_agreement"].mean()),
                "seed_score_delta_mean": float(seed_grp["score_delta"].mean()),
                "seed_branch_conclusion_changed": bool(
                    (old_delta >= 0 and qwen_delta < 0) or (old_delta < 0 and qwen_delta >= 0)
                ),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["source_seed"]).reset_index(drop=True)


def _plot_scatter(comparison_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(
        comparison_df["old_llama_overall_pedagogical_score"],
        comparison_df["qwen_overall_pedagogical_score"],
        alpha=0.7,
    )
    plt.xlabel("Old Llama Judge Score")
    plt.ylabel("Qwen3 Judge Score")
    plt.title("Judge Sensitivity: Old vs Qwen")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_delta_hist(comparison_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(comparison_df["score_delta"], bins=12)
    plt.xlabel("Score Delta (Qwen - Old)")
    plt.ylabel("Count")
    plt.title("Judge Score Delta Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_branch_bars(branch_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if branch_df.empty:
        return
    plt.figure(figsize=(8, 4))
    labels = branch_df.apply(lambda r: f"{r['branch']}|g{int(r['generation'])}", axis=1)
    x = range(len(branch_df))
    plt.bar([i - 0.15 for i in x], branch_df["old_llama_pedagogical_mean"], width=0.3, label="old_llama")
    plt.bar([i + 0.15 for i in x], branch_df["qwen_pedagogical_mean"], width=0.3, label="qwen3_235b")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("Pedagogical Mean")
    plt.title("Branch/Generation Pedagogical Mean by Judge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_qwen_judge_sensitivity(
    *,
    cfg: AppConfig,
    run_dirs: Sequence[Path],
    sample_size: int = 48,
    sample_seed: int = 42,
    sampling_strategy: str = "balanced_grid",
    focus_generation: int = 1,
    out_dir: Path | None = None,
) -> JudgeSensitivityArtifacts:
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    run_dirs = [Path(p) for p in run_dirs]
    if not run_dirs:
        raise ValueError("run_dirs must be non-empty")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tag = _QWEN_CONFIRMATORY_TAG if sampling_strategy == "confirmatory_gen1_seed_branch" else _QWEN_DEFAULT_TAG
    base_out = out_dir or (cfg.paths.output_root / "judge_sensitivity" / f"{out_tag}_{ts}" / "tables")
    base_out = Path(base_out)
    tables_dir = base_out
    figures_dir = tables_dir.parent / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _safe_out_dir(tables_dir.parent, run_dirs)

    source_df = load_sensitivity_source_rows(run_dirs)
    if sampling_strategy == "balanced_grid":
        selected_sample = select_balanced_sensitivity_sample(source_df, total_n=sample_size, seed=sample_seed)
    elif sampling_strategy == "confirmatory_gen1_seed_branch":
        selected_sample = select_confirmatory_seed_branch_sample(
            source_df,
            total_n=sample_size,
            seed=sample_seed,
            focus_generation=focus_generation,
            allowed_run_dirs=[str(x) for x in run_dirs],
        )
    else:
        raise ValueError(f"Unsupported sampling_strategy={sampling_strategy}")

    qwen_results, qwen_failures = run_qwen_rejudge(sample_df=selected_sample, cfg=cfg)
    comparison = build_judge_sensitivity_comparison(
        selected_sample_df=selected_sample,
        qwen_results_df=qwen_results,
    )
    summary = build_judge_sensitivity_summary(comparison)
    branch_summary = build_judge_sensitivity_branch_summary(comparison)
    seed_branch_summary = build_judge_sensitivity_seed_branch_summary(
        comparison,
        focus_generation=focus_generation,
    )

    selected_sample_csv = tables_dir / "selected_sample.csv"
    selected_sample_parquet = tables_dir / "selected_sample.parquet"
    qwen_results_csv = tables_dir / "qwen_rejudge_results.csv"
    qwen_results_parquet = tables_dir / "qwen_rejudge_results.parquet"
    qwen_failures_csv = tables_dir / "qwen_rejudge_failures.csv"
    qwen_failures_parquet = tables_dir / "qwen_rejudge_failures.parquet"
    comparison_csv = tables_dir / "judge_sensitivity_comparison.csv"
    comparison_parquet = tables_dir / "judge_sensitivity_comparison.parquet"
    summary_csv = tables_dir / "judge_sensitivity_summary.csv"
    summary_parquet = tables_dir / "judge_sensitivity_summary.parquet"
    branch_summary_csv = tables_dir / "judge_sensitivity_branch_summary.csv"
    branch_summary_parquet = tables_dir / "judge_sensitivity_branch_summary.parquet"
    seed_branch_summary_csv = tables_dir / "judge_sensitivity_seed_branch_summary.csv"
    seed_branch_summary_parquet = tables_dir / "judge_sensitivity_seed_branch_summary.parquet"
    metadata_json = tables_dir / "judge_sensitivity_metadata.json"

    selected_sample.to_csv(selected_sample_csv, index=False)
    selected_sample.to_parquet(selected_sample_parquet, index=False)
    qwen_results.to_csv(qwen_results_csv, index=False)
    qwen_results.to_parquet(qwen_results_parquet, index=False)
    qwen_failures.to_csv(qwen_failures_csv, index=False)
    qwen_failures.to_parquet(qwen_failures_parquet, index=False)
    comparison.to_csv(comparison_csv, index=False)
    comparison.to_parquet(comparison_parquet, index=False)
    summary.to_csv(summary_csv, index=False)
    summary.to_parquet(summary_parquet, index=False)
    branch_summary.to_csv(branch_summary_csv, index=False)
    branch_summary.to_parquet(branch_summary_parquet, index=False)
    seed_branch_summary.to_csv(seed_branch_summary_csv, index=False)
    seed_branch_summary.to_parquet(seed_branch_summary_parquet, index=False)

    scatter_plot = figures_dir / "judge_sensitivity_old_vs_qwen_scatter.png"
    delta_hist_plot = figures_dir / "judge_sensitivity_score_delta_hist.png"
    branch_bar_plot = figures_dir / "judge_sensitivity_branch_mean_comparison.png"
    _plot_scatter(comparison, scatter_plot)
    _plot_delta_hist(comparison, delta_hist_plot)
    _plot_branch_bars(branch_summary, branch_bar_plot)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "judge_provider": cfg.judge.provider,
        "judge_model_name": cfg.judge.model_name,
        "sample_size_requested": sample_size,
        "sample_size_selected": int(len(selected_sample)),
        "sample_seed": sample_seed,
        "sampling_strategy": sampling_strategy,
        "focus_generation": int(focus_generation),
        "source_run_dirs": [str(p) for p in run_dirs],
        "evaluation_mode": "robustness_sensitivity_check_only",
        "note": "No generation/training rerun. Re-judge of existing raw responses only.",
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return JudgeSensitivityArtifacts(
        out_dir=tables_dir.parent,
        selected_sample_csv=selected_sample_csv,
        selected_sample_parquet=selected_sample_parquet,
        qwen_rejudge_results_csv=qwen_results_csv,
        qwen_rejudge_results_parquet=qwen_results_parquet,
        qwen_rejudge_failures_csv=qwen_failures_csv,
        qwen_rejudge_failures_parquet=qwen_failures_parquet,
        comparison_csv=comparison_csv,
        comparison_parquet=comparison_parquet,
        summary_csv=summary_csv,
        summary_parquet=summary_parquet,
        branch_summary_csv=branch_summary_csv,
        branch_summary_parquet=branch_summary_parquet,
        seed_branch_summary_csv=seed_branch_summary_csv,
        seed_branch_summary_parquet=seed_branch_summary_parquet,
        scatter_plot=scatter_plot,
        delta_hist_plot=delta_hist_plot,
        branch_bar_plot=branch_bar_plot,
        metadata_json=metadata_json,
    )
