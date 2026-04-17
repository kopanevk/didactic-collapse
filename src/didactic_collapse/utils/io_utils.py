from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def save_tabular(df: pd.DataFrame, path_base: Path, save_csv: bool, save_parquet: bool) -> None:
    ensure_dir(path_base.parent)
    if save_csv:
        df.to_csv(path_base.with_suffix(".csv"), index=False)
    if save_parquet:
        df.to_parquet(path_base.with_suffix(".parquet"), index=False)
