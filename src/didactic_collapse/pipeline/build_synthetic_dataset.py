from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_next_generation_train_set(
    *,
    synthetic_outputs_df: pd.DataFrame,
    anchor_pool_df: pd.DataFrame,
    anchor_ratio: float,
    seed: int,
    out_path: Path,
) -> pd.DataFrame:
    n_synth = len(synthetic_outputs_df)
    n_anchor = int(round(n_synth * anchor_ratio))

    synth_rows = synthetic_outputs_df[["example_id", "question", "raw_response"]].rename(
        columns={"raw_response": "answer_for_training"}
    )
    synth_rows["source"] = "synthetic"

    anchor_rows = anchor_pool_df.sample(n=min(n_anchor, len(anchor_pool_df)), random_state=seed).copy()
    anchor_rows = anchor_rows[["example_id", "question", "answer_gold"]].rename(
        columns={"answer_gold": "answer_for_training"}
    )
    anchor_rows["source"] = "human_anchor"

    out_df = pd.concat([synth_rows, anchor_rows], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_df
