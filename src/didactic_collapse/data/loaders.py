from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from didactic_collapse.data.splitter import assert_disjoint_by_example_id

REQUIRED_COLUMNS = ("example_id", "question", "answer_gold")


@dataclass(frozen=True)
class ProcessedSplits:
    base_train: pd.DataFrame
    anchor_pool: pd.DataFrame
    heldout_test: pd.DataFrame
    metadata: dict[str, Any]


def generate_stable_example_id(*, question: str, answer_gold: str, dataset_name: str) -> str:
    """Generate stable ID based on semantic content.

    Deterministic across runs and independent of row ordering.
    """
    canonical = f"{dataset_name.strip()}\n{question.strip()}\n{answer_gold.strip()}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{dataset_name}_{digest[:16]}"


def _to_canonical_df(df: pd.DataFrame, dataset_name: str, source_split: str) -> pd.DataFrame:
    """Map HF GSM8K fields to canonical schema and attach stable IDs."""
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Expected columns 'question' and 'answer' in GSM8K source.")

    out = pd.DataFrame(
        {
            "question": df["question"].astype(str),
            "answer_gold": df["answer"].astype(str),
        }
    )
    out["dataset_name"] = dataset_name
    out["source_split"] = source_split
    out["example_id"] = out.apply(
        lambda r: generate_stable_example_id(
            question=str(r["question"]),
            answer_gold=str(r["answer_gold"]),
            dataset_name=dataset_name,
        ),
        axis=1,
    )

    if out["example_id"].duplicated().any():
        n_dup = int(out["example_id"].duplicated().sum())
        raise ValueError(
            "Stable example_id collision/duplication detected in normalized GSM8K data. "
            f"duplicates={n_dup}"
        )

    return out[["example_id", "question", "answer_gold", "dataset_name", "source_split"]]


def load_gsm8k_from_hf(
    *,
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load GSM8K from Hugging Face datasets with optional local cache."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, cache_dir=str(cache_dir) if cache_dir else None)

    parts: list[pd.DataFrame] = []
    for split_name in ("train", "test"):
        if split_name not in ds:
            continue
        split_df = ds[split_name].to_pandas()
        parts.append(_to_canonical_df(split_df, dataset_name=dataset_name, source_split=split_name))

    if not parts:
        raise ValueError("No train/test splits found in dataset response.")

    full = pd.concat(parts, ignore_index=True)
    if full["example_id"].duplicated().any():
        n_dup = int(full["example_id"].duplicated().sum())
        raise ValueError(f"Duplicate example_id values after merge(train,test): {n_dup}")
    return full


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist normalized dataset into data/processed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def write_split_metadata(
    *,
    metadata_path: Path,
    dataset_name: str,
    dataset_config: str,
    seed: int,
    base_train_size: int,
    anchor_pool_size: int,
    heldout_test_size: int,
    total_rows: int,
) -> None:
    """Write metadata for reproducibility and auditability."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": dataset_name,
        "source_config": dataset_config,
        "seed": seed,
        "requested_sizes": {
            "base_train": base_train_size,
            "anchor_pool": anchor_pool_size,
            "heldout_test": heldout_test_size,
        },
        "total_rows": total_rows,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_split_integrity(
    *,
    base_train_df: pd.DataFrame,
    anchor_pool_df: pd.DataFrame,
    heldout_test_df: pd.DataFrame,
) -> None:
    """Validate split invariants: required columns + strict no-overlap by example_id."""
    for name, df in (
        ("base_train", base_train_df),
        ("anchor_pool", anchor_pool_df),
        ("heldout_test", heldout_test_df),
    ):
        missing = set(REQUIRED_COLUMNS).difference(df.columns)
        if missing:
            raise ValueError(f"Split '{name}' is missing required columns: {sorted(missing)}")
        if df["example_id"].duplicated().any():
            n_dup = int(df["example_id"].duplicated().sum())
            raise ValueError(f"Split '{name}' contains duplicate example_id values: {n_dup}")

    assert_disjoint_by_example_id(base_train_df, anchor_pool_df, heldout_test_df)


def load_processed_splits(split_dir: Path) -> ProcessedSplits:
    """Load previously prepared split artifacts and metadata.

    Expects files:
    - base_train.parquet
    - anchor_pool.parquet
    - heldout_test.parquet
    - split_metadata.json (optional but recommended)
    """
    base_path = split_dir / "base_train.parquet"
    anchor_path = split_dir / "anchor_pool.parquet"
    heldout_path = split_dir / "heldout_test.parquet"
    metadata_path = split_dir / "split_metadata.json"

    for p in (base_path, anchor_path, heldout_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    base_df = pd.read_parquet(base_path)
    anchor_df = pd.read_parquet(anchor_path)
    heldout_df = pd.read_parquet(heldout_path)
    validate_split_integrity(
        base_train_df=base_df,
        anchor_pool_df=anchor_df,
        heldout_test_df=heldout_df,
    )

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return ProcessedSplits(
        base_train=base_df,
        anchor_pool=anchor_df,
        heldout_test=heldout_df,
        metadata=metadata,
    )
