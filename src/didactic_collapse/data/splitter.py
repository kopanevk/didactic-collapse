from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitArtifacts:
    base_train_path: Path
    anchor_pool_path: Path
    heldout_test_path: Path


def create_splits(raw_df: pd.DataFrame, out_dir: Path, seed: int, base_train_size: int, anchor_pool_size: int, heldout_test_size: int) -> SplitArtifacts:
    """Create deterministic non-overlapping splits for base train, anchor pool, heldout test."""
    shuffled = raw_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total_needed = base_train_size + anchor_pool_size + heldout_test_size
    if len(shuffled) < total_needed:
        raise ValueError(f"Not enough rows. Need {total_needed}, got {len(shuffled)}")

    base_train = shuffled.iloc[:base_train_size].copy()
    anchor_pool = shuffled.iloc[base_train_size : base_train_size + anchor_pool_size].copy()
    heldout_test = shuffled.iloc[
        base_train_size + anchor_pool_size : base_train_size + anchor_pool_size + heldout_test_size
    ].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    base_train_path = out_dir / "base_train.parquet"
    anchor_pool_path = out_dir / "anchor_pool.parquet"
    heldout_test_path = out_dir / "heldout_test.parquet"

    base_train.to_parquet(base_train_path, index=False)
    anchor_pool.to_parquet(anchor_pool_path, index=False)
    heldout_test.to_parquet(heldout_test_path, index=False)

    return SplitArtifacts(base_train_path, anchor_pool_path, heldout_test_path)
