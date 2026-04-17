from __future__ import annotations

from pathlib import Path

import pandas as pd

from didactic_collapse.clients.base import GenerationClient
from didactic_collapse.pipeline.extract_answer import extract_final_answer


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
) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in examples_df.to_dict(orient="records"):
        question = str(rec["question"])
        prompt = f"Solve step-by-step and conclude with 'Final answer: ...'.\n\nQuestion: {question}"
        raw_response = client.generate(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        rows.append(
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

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_df
