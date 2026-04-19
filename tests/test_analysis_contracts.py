from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.analysis.aggregate import aggregate_metrics
from didactic_collapse.analysis.plots import plot_metric_by_generation


def test_aggregate_metrics_fails_on_missing_columns() -> None:
    df = pd.DataFrame([{"branch": "pure", "generation": 0}])
    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_metrics(df, Path("outputs/.tmp/metrics.csv"))


def test_plot_metric_by_generation_fails_on_missing_metric() -> None:
    df = pd.DataFrame(
        [
            {"generation": 0, "branch": "pure", "model_name": "qwen", "accuracy": 1.0},
        ]
    )
    with pytest.raises(ValueError, match="metric column not found"):
        plot_metric_by_generation(df, "pedagogical_score_mean", Path("outputs/.tmp/plot.png"))
