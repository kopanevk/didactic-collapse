from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from didactic_collapse.analysis.dbr_confirmatory import (
    DBRConfirmatoryArtifacts,
    export_dbr_confirmatory_analysis,
)
from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.baseline_series import parse_seed_list
from didactic_collapse.orchestration.first_experiment import (
    FirstExperimentSummary,
    resume_first_experiment,
    run_first_experiment,
)


@dataclass(frozen=True)
class DBRConfirmatorySummary:
    seeds: list[int]
    sample_size: int
    run_dirs: list[Path]
    analysis_dir: Path
    artifacts: DBRConfirmatoryArtifacts
    runs: list[FirstExperimentSummary]


def _cfg_for_seed(cfg: AppConfig, *, seed: int) -> AppConfig:
    raw = cfg.model_dump(mode="python")
    raw["project"]["seed"] = seed
    raw["project"]["run_tag"] = f"{cfg.project.run_tag}_seed{seed}"
    return AppConfig.model_validate(raw)


def _find_latest_run_for_tag(*, output_root: Path, run_tag_seed: str) -> Path | None:
    runs_root = output_root / "runs"
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.glob(f"{run_tag_seed}_*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _is_run_manifest_completed(run_dir: Path) -> bool:
    manifest_path = run_dir / "run_stage_manifest.json"
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = payload.get("stages", {})
    if not isinstance(stages, dict):
        return False
    for stage_name in ("data_prep", "aggregate", "plotting"):
        stage = stages.get(stage_name, {})
        if not isinstance(stage, dict):
            return False
        if str(stage.get("status", "")).lower() != "completed":
            return False
    model_root = run_dir / "qwen2.5_0.5b"
    if not model_root.exists():
        return False
    context_stage_files = list(model_root.rglob("stage_manifest.json"))
    if not context_stage_files:
        return False
    required_context_stages = (
        "generation",
        "answer_extraction",
        "accuracy",
        "judge",
        "synthetic_build",
        "anchoring",
    )
    for stage_file in context_stage_files:
        stage_payload = json.loads(stage_file.read_text(encoding="utf-8"))
        stage_map = stage_payload.get("stages", {})
        if not isinstance(stage_map, dict):
            return False
        for stage_name in required_context_stages:
            stage = stage_map.get(stage_name, {})
            if not isinstance(stage, dict):
                return False
            if str(stage.get("status", "")).lower() != "completed":
                return False
    return True


def _run_or_resume_seed(
    *,
    seed_cfg: AppConfig,
    sample_size: int,
) -> FirstExperimentSummary:
    run_tag_seed = seed_cfg.project.run_tag
    existing = _find_latest_run_for_tag(
        output_root=seed_cfg.paths.output_root,
        run_tag_seed=run_tag_seed,
    )
    if existing is not None:
        if _is_run_manifest_completed(existing):
            return resume_first_experiment(cfg=seed_cfg, run_dir=existing)
        return resume_first_experiment(cfg=seed_cfg, run_dir=existing)
    return run_first_experiment(cfg=seed_cfg, sample_size=sample_size)


def run_dbr_confirmatory_series(
    *,
    cfg: AppConfig,
    sample_size: int,
    seeds: Sequence[int],
    analysis_dir: Path | None = None,
) -> DBRConfirmatorySummary:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if not seeds:
        raise ValueError("seeds must not be empty")

    runs: list[FirstExperimentSummary] = []
    for seed in seeds:
        seed_cfg = _cfg_for_seed(cfg, seed=int(seed))
        runs.append(_run_or_resume_seed(seed_cfg=seed_cfg, sample_size=sample_size))

    run_dirs = [r.run_dir for r in runs]
    out_dir = analysis_dir or (
        cfg.paths.output_root
        / "dbr_confirmatory"
        / f"{cfg.project.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        / "tables"
    )
    artifacts = export_dbr_confirmatory_analysis(run_dirs=run_dirs, out_dir=out_dir)
    return DBRConfirmatorySummary(
        seeds=[int(x) for x in seeds],
        sample_size=sample_size,
        run_dirs=run_dirs,
        analysis_dir=out_dir.parent,
        artifacts=artifacts,
        runs=runs,
    )


__all__ = [
    "DBRConfirmatorySummary",
    "parse_seed_list",
    "run_dbr_confirmatory_series",
]
