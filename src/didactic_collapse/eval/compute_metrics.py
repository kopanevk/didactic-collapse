from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_math_answer(x: str | None) -> str:
    if x is None:
        return ""
    return str(x).strip().replace(",", "")


def compute_eval_table(outputs_df: pd.DataFrame, judge_df: pd.DataFrame, gold_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    merged = outputs_df.merge(gold_df[["example_id", "answer_gold"]], on="example_id", how="left")
    merged = merged.merge(
        judge_df[["example_id", "overall_pedagogical_score", "is_silent_error"]],
        on="example_id",
        how="left",
    )
    merged["is_correct"] = (
        merged["parsed_final_answer"].map(normalize_math_answer)
        == merged["answer_gold"].map(normalize_math_answer)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    return merged
