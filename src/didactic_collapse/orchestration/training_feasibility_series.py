from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from didactic_collapse.analysis.mode_comparison import (
    ModeComparisonArtifacts,
    export_mode_comparison_analysis,
)
from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.baseline_series import parse_seed_list
from didactic_collapse.orchestration.training_feasibility import (
    TrainingFeasibilitySummary,
    run_training_recycling_feasibility,
)


@dataclass(frozen=True)
class TrainingFeasibilitySeriesSummary:
    seeds: list[int]
    sample_size: int
    run_dirs: list[Path]
    analysis_dir: Path | None
    artifacts: ModeComparisonArtifacts | None
    runs: list[TrainingFeasibilitySummary]


def _cfg_for_seed(cfg: AppConfig, *, seed: int) -> AppConfig:
    raw = cfg.model_dump(mode="python")
    raw["project"]["seed"] = seed
    raw["project"]["run_tag"] = f"{cfg.project.run_tag}_seed{seed}"
    return AppConfig.model_validate(raw)


def run_training_feasibility_series(
    *,
    cfg: AppConfig,
    sample_size: int,
    seeds: Sequence[int],
    analysis_dir: Path | None = None,
    compare_with_inference_runs: Sequence[Path] | None = None,
) -> TrainingFeasibilitySeriesSummary:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if not seeds:
        raise ValueError("seeds must not be empty")

    runs: list[TrainingFeasibilitySummary] = []
    for seed in seeds:
        seed_cfg = _cfg_for_seed(cfg, seed=int(seed))
        runs.append(run_training_recycling_feasibility(cfg=seed_cfg, sample_size=sample_size))

    run_dirs = [r.run_dir for r in runs]
    artifacts: ModeComparisonArtifacts | None = None
    out_dir: Path | None = None
    if compare_with_inference_runs:
        out_dir = analysis_dir or (
            cfg.paths.output_root
            / "mode_comparison"
            / f"{cfg.project.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            / "tables"
        )
        artifacts = export_mode_comparison_analysis(
            inference_run_dirs=[Path(x) for x in compare_with_inference_runs],
            training_run_dirs=run_dirs,
            out_dir=out_dir,
        )

    return TrainingFeasibilitySeriesSummary(
        seeds=[int(x) for x in seeds],
        sample_size=sample_size,
        run_dirs=run_dirs,
        analysis_dir=None if out_dir is None else out_dir.parent,
        artifacts=artifacts,
        runs=runs,
    )


__all__ = [
    "TrainingFeasibilitySeriesSummary",
    "run_training_feasibility_series",
    "parse_seed_list",
]

