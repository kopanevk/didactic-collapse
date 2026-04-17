from __future__ import annotations

from pathlib import Path

import pandas as pd

from didactic_collapse.clients.base import JudgeClient


def run_judging(
    *,
    client: JudgeClient,
    generations_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    judge_provider: str,
    judge_model: str,
    rubric_prompt: str,
    out_path: Path,
) -> pd.DataFrame:
    merged = generations_df.merge(
        questions_df[["example_id", "question", "answer_gold"]],
        on="example_id",
        how="left",
        validate="many_to_one",
    )

    rows: list[dict] = []
    for rec in merged.to_dict(orient="records"):
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
            "example_id": rec["example_id"],
            "judge_provider": judge_provider,
            "judge_model": judge_model,
        }
        row.update(score)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_df
