from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd


@dataclass(frozen=True)
class SplitArtifacts:
    base_train_path: Path
    anchor_pool_path: Path
    heldout_test_path: Path


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"example_id", "question", "answer_gold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _deterministic_shuffle(raw_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle rows deterministically using only example_id ordering and seed."""
    canonical = raw_df.sort_values("example_id", kind="mergesort").reset_index(drop=True)
    order = list(range(len(canonical)))
    random.Random(seed).shuffle(order)
    return canonical.iloc[order].reset_index(drop=True)


def assert_disjoint_by_example_id(
    base_train_df: pd.DataFrame, anchor_pool_df: pd.DataFrame, heldout_test_df: pd.DataFrame
) -> None:
    """Raise explicit error if any split overlap by example_id."""
    base_ids = set(base_train_df["example_id"])
    anchor_ids = set(anchor_pool_df["example_id"])
    heldout_ids = set(heldout_test_df["example_id"])

    overlap_base_anchor = base_ids.intersection(anchor_ids)
    overlap_base_heldout = base_ids.intersection(heldout_ids)
    overlap_anchor_heldout = anchor_ids.intersection(heldout_ids)

    if overlap_base_anchor or overlap_base_heldout or overlap_anchor_heldout:
        raise ValueError(
            "Split invariant violation: overlap detected by example_id. "
            f"base∩anchor={len(overlap_base_anchor)}, "
            f"base∩heldout={len(overlap_base_heldout)}, "
            f"anchor∩heldout={len(overlap_anchor_heldout)}"
        )


def create_splits(
    raw_df: pd.DataFrame,
    out_dir: Path,
    seed: int,
    base_train_size: int,
    anchor_pool_size: int,
    heldout_test_size: int,
) -> SplitArtifacts:
    """Create deterministic non-overlapping splits for base train, anchor pool, heldout test."""
    _ensure_required_columns(raw_df)
    if raw_df["example_id"].duplicated().any():
        n_dup = int(raw_df["example_id"].duplicated().sum())
        raise ValueError(f"Input dataframe has duplicate example_id values: {n_dup}")

    shuffled = _deterministic_shuffle(raw_df, seed=seed)
    total_needed = base_train_size + anchor_pool_size + heldout_test_size
    if len(shuffled) < total_needed:
        raise ValueError(f"Not enough rows. Need {total_needed}, got {len(shuffled)}")

    base_train = shuffled.iloc[:base_train_size].copy()
    anchor_pool = shuffled.iloc[base_train_size : base_train_size + anchor_pool_size].copy()
    heldout_test = shuffled.iloc[
        base_train_size + anchor_pool_size : base_train_size + anchor_pool_size + heldout_test_size
    ].copy()

    assert_disjoint_by_example_id(base_train, anchor_pool, heldout_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_train_path = out_dir / "base_train.parquet"
    anchor_pool_path = out_dir / "anchor_pool.parquet"
    heldout_test_path = out_dir / "heldout_test.parquet"

    base_train.to_parquet(base_train_path, index=False)
    anchor_pool.to_parquet(anchor_pool_path, index=False)
    heldout_test.to_parquet(heldout_test_path, index=False)

    return SplitArtifacts(base_train_path, anchor_pool_path, heldout_test_path)
