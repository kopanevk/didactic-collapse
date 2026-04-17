from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.runner import CONTEXT_STAGES, RUN_STAGES, ExperimentRunner, StageManifest


@dataclass(frozen=True)
class FirstExperimentSummary:
    run_dir: Path
    data_root_used: Path
    model_name: str
    branches: list[str]
    generations: list[int]
    sample_size_requested: int
    sample_size_used: int
    total_examples_scored: int
    summary_table_path_csv: Path
    summary_table_path_parquet: Path
    qualitative_path_csv: Path
    qualitative_path_parquet: Path


def _load_manifest(path: Path) -> StageManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StageManifest.model_validate(payload)


def _ensure_real_judge_provider(cfg: AppConfig) -> None:
    provider = cfg.judge.provider.strip().lower()
    if provider in {"mock", "stub", "mock_judge"}:
        raise RuntimeError(
            "First experiment requires real judge provider. "
            "Current provider is mock/stub, which is non-scientific."
        )


def prepare_first_experiment_splits(
    *,
    cfg: AppConfig,
    sample_size: int,
    experiment_data_root: Path,
    seed: int,
) -> int:
    """Create small deterministic subset for first real experiment."""
    source_dir = cfg.paths.data_root / "splits"
    required = [
        source_dir / "base_train.parquet",
        source_dir / "anchor_pool.parquet",
        source_dir / "heldout_test.parquet",
        source_dir / "split_metadata.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared data splits. Run data preparation first. Missing: " + ", ".join(missing)
        )

    base_train = pd.read_parquet(source_dir / "base_train.parquet")
    anchor_pool = pd.read_parquet(source_dir / "anchor_pool.parquet")
    heldout_test = pd.read_parquet(source_dir / "heldout_test.parquet")

    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    n = min(sample_size, len(heldout_test))
    if n < 2:
        raise ValueError("Insufficient heldout samples for first experiment")

    out_dir = experiment_data_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_small = heldout_test.sample(n=n, random_state=seed).sort_values("example_id").reset_index(drop=True)
    base_small = base_train.sample(n=min(max(n, 20), len(base_train)), random_state=seed + 1).sort_values(
        "example_id"
    ).reset_index(drop=True)
    anchor_small = anchor_pool.sample(n=min(max(n * 2, 40), len(anchor_pool)), random_state=seed + 2).sort_values(
        "example_id"
    ).reset_index(drop=True)

    base_small.to_parquet(out_dir / "base_train.parquet", index=False)
    anchor_small.to_parquet(out_dir / "anchor_pool.parquet", index=False)
    heldout_small.to_parquet(out_dir / "heldout_test.parquet", index=False)

    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "mode": "first_real_experiment",
        "sample_size_requested": sample_size,
        "sample_size_used": n,
        "seed": seed,
        "source_split_dir": str(source_dir),
    }
    (out_dir / "split_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def build_first_experiment_config(*, cfg: AppConfig, data_root: Path) -> AppConfig:
    raw = cfg.model_dump(mode="python")
    raw["paths"]["data_root"] = str(data_root)
    raw["models"]["local_models"] = [{"name": "qwen2.5:0.5b", "role": "subject"}]
    raw["experiment"]["branches"] = [
        {"name": "pure_recycling", "anchor_ratio": 0.0},
        {"name": "anchor_10", "anchor_ratio": 0.10},
    ]
    raw["experiment"]["generations"] = 2
    return AppConfig.model_validate(raw)


def _export_first_summary_table(*, all_eval: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, pd.DataFrame]:
    table = (
        all_eval.groupby(["model_name", "branch", "generation"], as_index=False)
        .agg(
            sample_count=("example_id", "count"),
            accuracy_mean=("is_correct", "mean"),
            pedagogical_score_mean=("overall_pedagogical_score", "mean"),
            silent_error_rate=("is_silent_error", "mean"),
        )
        .sort_values(["model_name", "branch", "generation"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "first_experiment_summary.csv"
    parquet_path = out_dir / "first_experiment_summary.parquet"
    table.to_csv(csv_path, index=False)
    table.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path, table


def _export_qualitative_candidates(*, all_eval: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = all_eval[
        (all_eval["is_correct"] == True)  # noqa: E712
        & (all_eval["overall_pedagogical_score"] <= 2)
        & (all_eval["is_silent_error"] == True)  # noqa: E712
    ].copy()
    candidates = candidates.sort_values(["generation", "branch", "overall_pedagogical_score", "example_id"])

    csv_path = out_dir / "qualitative_silent_error_candidates.csv"
    parquet_path = out_dir / "qualitative_silent_error_candidates.parquet"
    candidates.to_csv(csv_path, index=False)
    candidates.to_parquet(parquet_path, index=False)

    status = {
        "created_at": datetime.now().isoformat(),
        "is_empty": bool(candidates.empty),
        "row_count": int(len(candidates)),
    }
    (out_dir / "qualitative_silent_error_candidates.meta.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, parquet_path


def validate_first_experiment_outputs(
    *,
    run_dir: Path,
    model_name: str,
    branches: list[str],
    generations: list[int],
    summary_table: pd.DataFrame,
) -> None:
    if summary_table.empty:
        raise RuntimeError("First experiment summary table is empty")

    observed_branches = set(summary_table["branch"].unique().tolist())
    observed_generations = set(summary_table["generation"].unique().tolist())

    missing_branches = set(branches).difference(observed_branches)
    if missing_branches:
        raise RuntimeError(f"Missing branches in summary: {sorted(missing_branches)}")

    missing_generations = set(generations).difference(observed_generations)
    if missing_generations:
        raise RuntimeError(f"Missing generations in summary: {sorted(missing_generations)}")

    run_manifest = _load_manifest(run_dir / "run_stage_manifest.json")
    for stage in RUN_STAGES:
        if run_manifest.stages[stage].status.value != "completed":
            raise RuntimeError(f"Run stage not completed: {stage}")

    for branch in branches:
        for gen in generations:
            step_manifest_path = run_dir / model_name.replace(":", "_") / branch / f"gen_{gen}" / "stage_manifest.json"
            if not step_manifest_path.exists():
                raise RuntimeError(f"Missing context manifest: {step_manifest_path}")
            ctx_manifest = _load_manifest(step_manifest_path)
            for stage in CONTEXT_STAGES:
                if ctx_manifest.stages[stage].status.value != "completed":
                    raise RuntimeError(
                        f"Context stage not completed for branch={branch} gen={gen}: {stage}"
                    )


def run_first_experiment(*, cfg: AppConfig, sample_size: int) -> FirstExperimentSummary:
    _ensure_real_judge_provider(cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_data_root = cfg.paths.output_root / "first_experiment_inputs" / ts
    exp_data_root.mkdir(parents=True, exist_ok=True)

    used_sample_size = prepare_first_experiment_splits(
        cfg=cfg,
        sample_size=sample_size,
        experiment_data_root=exp_data_root,
        seed=cfg.project.seed,
    )

    exp_cfg = build_first_experiment_config(cfg=cfg, data_root=exp_data_root)
    runner = ExperimentRunner(exp_cfg)

    model_name = exp_cfg.models.local_models[0].name
    branches = [b.name for b in exp_cfg.experiment.branches]
    generations = [0, 1]

    runner.save_run_metadata()
    runner.run_stage("data_prep")
    for branch in branches:
        for gen in generations:
            for stage_name in CONTEXT_STAGES:
                runner.run_stage(
                    stage_name,
                    model_name=model_name,
                    branch=branch,
                    generation=gen,
                    seed=exp_cfg.project.seed,
                    force=exp_cfg.runtime.force_recompute,
                )
    runner.run_stage("aggregate", force=exp_cfg.runtime.force_recompute)
    runner.run_stage("plotting", force=exp_cfg.runtime.force_recompute)

    all_eval = pd.read_parquet(runner.ctx.run_dir / "all_eval_merged.parquet")
    summary_csv, summary_parquet, summary_df = _export_first_summary_table(
        all_eval=all_eval,
        out_dir=runner.ctx.run_dir / "tables",
    )
    qual_csv, qual_parquet = _export_qualitative_candidates(
        all_eval=all_eval,
        out_dir=runner.ctx.run_dir / "tables",
    )

    validate_first_experiment_outputs(
        run_dir=runner.ctx.run_dir,
        model_name=model_name,
        branches=branches,
        generations=generations,
        summary_table=summary_df,
    )

    return FirstExperimentSummary(
        run_dir=runner.ctx.run_dir,
        data_root_used=exp_data_root,
        model_name=model_name,
        branches=branches,
        generations=generations,
        sample_size_requested=sample_size,
        sample_size_used=used_sample_size,
        total_examples_scored=int(len(all_eval)),
        summary_table_path_csv=summary_csv,
        summary_table_path_parquet=summary_parquet,
        qualitative_path_csv=qual_csv,
        qualitative_path_parquet=qual_parquet,
    )
