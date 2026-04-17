from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from didactic_collapse.data.loaders import (
    load_gsm8k_from_hf,
    save_processed_dataset,
    validate_split_integrity,
    write_split_metadata,
)
from didactic_collapse.data.splitter import create_splits

app = typer.Typer(help="Prepare GSM8K from Hugging Face and build deterministic split artifacts")


@app.command()
def run(
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    cache_dir: Path = Path("data/raw/hf_cache"),
    processed_path: Path = Path("data/processed/gsm8k_normalized.parquet"),
    split_dir: Path = Path("data/splits"),
    seed: int = 42,
    base_train_size: int = 3000,
    anchor_pool_size: int = 1000,
    heldout_test_size: int = 1000,
) -> None:
    """Load GSM8K, normalize with stable IDs, split, validate, and save metadata."""
    df = load_gsm8k_from_hf(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        cache_dir=cache_dir,
    )
    save_processed_dataset(df, processed_path)

    artifacts = create_splits(
        raw_df=df,
        out_dir=split_dir,
        seed=seed,
        base_train_size=base_train_size,
        anchor_pool_size=anchor_pool_size,
        heldout_test_size=heldout_test_size,
    )

    base_train_df = pd.read_parquet(artifacts.base_train_path)
    anchor_pool_df = pd.read_parquet(artifacts.anchor_pool_path)
    heldout_test_df = pd.read_parquet(artifacts.heldout_test_path)

    validate_split_integrity(
        base_train_df=base_train_df,
        anchor_pool_df=anchor_pool_df,
        heldout_test_df=heldout_test_df,
    )

    write_split_metadata(
        metadata_path=split_dir / "split_metadata.json",
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seed=seed,
        base_train_size=base_train_size,
        anchor_pool_size=anchor_pool_size,
        heldout_test_size=heldout_test_size,
        total_rows=len(df),
    )

    typer.echo("GSM8K preparation completed successfully.")
    typer.echo(f"Processed dataset: {processed_path}")
    typer.echo(f"Splits directory: {split_dir}")


if __name__ == "__main__":
    app()
