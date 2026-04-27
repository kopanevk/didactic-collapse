from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from didactic_collapse.analysis.baseline_series import (
    BaselineSeriesArtifacts,
    export_baseline_series_analysis,
)
from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.first_experiment import FirstExperimentSummary, run_first_experiment


@dataclass(frozen=True)
class BaselineSeriesSummary:
    seeds: list[int]
    sample_size: int
    run_dirs: list[Path]
    analysis_dir: Path
    artifacts: BaselineSeriesArtifacts
    runs: list[FirstExperimentSummary]


def parse_seed_list(seed_text: str) -> list[int]:
    raw = [x.strip() for x in seed_text.split(",") if x.strip()]
    if not raw:
        raise ValueError("Seed list is empty")
    seeds = [int(x) for x in raw]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seed values are not allowed: {seeds}")
    return seeds


def _cfg_for_seed(cfg: AppConfig, *, seed: int) -> AppConfig:
    raw = cfg.model_dump(mode="python")
    raw["project"]["seed"] = seed
    raw["project"]["run_tag"] = f"{cfg.project.run_tag}_seed{seed}"
    return AppConfig.model_validate(raw)


def run_baseline_series(
    *,
    cfg: AppConfig,
    sample_size: int,
    seeds: Sequence[int],
    analysis_dir: Path | None = None,
) -> BaselineSeriesSummary:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if not seeds:
        raise ValueError("seeds must not be empty")

    runs: list[FirstExperimentSummary] = []
    for seed in seeds:
        seed_cfg = _cfg_for_seed(cfg, seed=int(seed))
        runs.append(run_first_experiment(cfg=seed_cfg, sample_size=sample_size))

    run_dirs = [r.run_dir for r in runs]
    out_dir = analysis_dir or (
        cfg.paths.output_root
        / "baseline_series"
        / f"{cfg.project.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        / "tables"
    )
    artifacts = export_baseline_series_analysis(run_dirs=run_dirs, out_dir=out_dir)
    return BaselineSeriesSummary(
        seeds=[int(x) for x in seeds],
        sample_size=sample_size,
        run_dirs=run_dirs,
        analysis_dir=out_dir.parent,
        artifacts=artifacts,
        runs=runs,
    )
