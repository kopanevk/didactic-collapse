from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metric_by_generation(df: pd.DataFrame, metric: str, out_file: Path) -> None:
    required = {"generation", "branch", "model_name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"plot input missing required columns: {sorted(missing)}")
    if metric not in df.columns:
        raise ValueError(f"plot metric column not found: {metric}")
    if df.empty:
        raise ValueError("plot input is empty")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x="generation", y=metric, hue="branch", style="model_name", marker="o")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
