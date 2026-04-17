from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StagePaths:
    model_outputs: Path
    judge_outputs: Path
    accuracy_table: Path
    eval_merged: Path
    synthetic_train_next: Path


def resolve_stage_paths(base_dir: Path) -> StagePaths:
    """Canonical artifact layout for one (model, branch, generation) stage."""
    return StagePaths(
        model_outputs=base_dir / "model_outputs.parquet",
        judge_outputs=base_dir / "judge_outputs.parquet",
        accuracy_table=base_dir / "accuracy_table.parquet",
        eval_merged=base_dir / "eval_merged.parquet",
        synthetic_train_next=base_dir / "synthetic_train_next.parquet",
    )
