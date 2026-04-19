from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.orchestration.first_experiment import (
    _export_first_summary_table,
    _export_qualitative_candidates,
)


def test_first_summary_export_fails_on_missing_columns() -> None:
    df = pd.DataFrame([{"model_name": "qwen"}])
    with pytest.raises(RuntimeError, match="missing columns"):
        _export_first_summary_table(all_eval=df, out_dir=Path("outputs/.tmp/first_summary"))


def test_first_qualitative_export_fails_on_missing_columns() -> None:
    df = pd.DataFrame([{"generation": 0}])
    with pytest.raises(RuntimeError, match="missing columns"):
        _export_qualitative_candidates(all_eval=df, out_dir=Path("outputs/.tmp/first_qual"))
