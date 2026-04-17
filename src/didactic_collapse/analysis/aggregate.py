from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_metrics(eval_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    agg = (
        eval_df.groupby(["branch", "generation", "model_name"], as_index=False)
        .agg(
            accuracy=("is_correct", "mean"),
            pedagogical_score_mean=("overall_pedagogical_score", "mean"),
            silent_error_rate=("is_silent_error", "mean"),
            n=("example_id", "count"),
        )
        .sort_values(["branch", "generation", "model_name"])
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    return agg
