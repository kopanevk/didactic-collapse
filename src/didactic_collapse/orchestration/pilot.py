from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.runner import (
    CONTEXT_STAGES,
    ExperimentRunner,
    RUN_STAGES,
    StageContext,
    StageExecutionResult,
    StageManifest,
)


@dataclass(frozen=True)
class PilotSummary:
    run_dir: Path
    pilot_data_root: Path
    model_name: str
    branches: list[str]
    generations: int
    sample_size_requested: int
    sample_size_used: int
    total_examples_scored: int
    artifacts_valid: bool
    missing_artifacts: list[str]
    completed_stages: int
    skip_or_resume_events: int


def _pick_model_name(cfg: AppConfig) -> str:
    for spec in cfg.models.local_models:
        if spec.name == "qwen2.5:0.5b":
            return spec.name
    return cfg.models.local_models[0].name


def prepare_pilot_splits(
    *,
    cfg: AppConfig,
    sample_size: int,
    pilot_data_root: Path,
    seed: int,
) -> tuple[Path, int]:
    """Create small deterministic pilot splits under pilot_data_root/splits."""
    source_split_dir = cfg.paths.data_root / "splits"
    base_train = pd.read_parquet(source_split_dir / "base_train.parquet")
    anchor_pool = pd.read_parquet(source_split_dir / "anchor_pool.parquet")
    heldout_test = pd.read_parquet(source_split_dir / "heldout_test.parquet")

    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    n_heldout = min(sample_size, len(heldout_test))
    n_base = min(max(sample_size, 10), len(base_train))
    n_anchor = min(max(sample_size * 2, 20), len(anchor_pool))

    heldout_small = heldout_test.sample(n=n_heldout, random_state=seed).sort_values("example_id").reset_index(drop=True)
    base_small = base_train.sample(n=n_base, random_state=seed + 1).sort_values("example_id").reset_index(drop=True)
    anchor_small = anchor_pool.sample(n=n_anchor, random_state=seed + 2).sort_values("example_id").reset_index(drop=True)

    out_split_dir = pilot_data_root / "splits"
    out_split_dir.mkdir(parents=True, exist_ok=True)
    base_small.to_parquet(out_split_dir / "base_train.parquet", index=False)
    anchor_small.to_parquet(out_split_dir / "anchor_pool.parquet", index=False)
    heldout_small.to_parquet(out_split_dir / "heldout_test.parquet", index=False)

    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "mode": "pilot",
        "sample_size_requested": sample_size,
        "sample_size_used": n_heldout,
        "seed": seed,
        "source_split_dir": str(source_split_dir),
    }
    (out_split_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pilot_data_root, n_heldout


def build_pilot_config(
    *,
    cfg: AppConfig,
    pilot_data_root: Path,
    mock_judge: bool,
) -> AppConfig:
    """Create thin pilot preset without changing main config file."""
    raw = cfg.model_dump(mode="python")

    raw["paths"]["data_root"] = str(pilot_data_root)
    raw["project"]["run_tag"] = f"{cfg.project.run_tag}_pilot"
    raw["models"]["local_models"] = [{"name": _pick_model_name(cfg), "role": "subject"}]
    raw["experiment"]["generations"] = 1
    raw["experiment"]["branches"] = [
        {"name": "pure_recycling", "anchor_ratio": 0.0},
        {"name": "anchor_10", "anchor_ratio": 0.10},
    ]
    raw["sampling"]["temperature"] = 0.0
    raw["sampling"]["top_p"] = 1.0
    raw["sampling"]["max_tokens"] = min(int(raw["sampling"]["max_tokens"]), 128)

    if mock_judge:
        raw["judge"]["provider"] = "mock"
        raw["judge"]["model_name"] = "mock-judge"
        raw["judge"]["base_url"] = "mock://local"
        raw["judge"]["api_key_env"] = "MOCK_UNUSED"

    return AppConfig.model_validate(raw)


def _load_manifest(path: Path) -> StageManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StageManifest.model_validate(payload)


def validate_pilot_artifacts(*, run_dir: Path, model_name: str, branches: list[str], generations: int) -> list[str]:
    """Return missing/invalid artifact list. Empty list means validation passed."""
    missing: list[str] = []

    run_manifest_path = run_dir / "run_stage_manifest.json"
    if not run_manifest_path.exists():
        missing.append(str(run_manifest_path))
    else:
        _ = _load_manifest(run_manifest_path)

    for gen in range(1, generations + 1):
        for branch in branches:
            step_dir = run_dir / model_name.replace(":", "_") / branch / f"gen_{gen}"
            expected = [
                step_dir / "model_outputs.parquet",
                step_dir / "answer_extraction.parquet",
                step_dir / "accuracy_table.parquet",
                step_dir / "judge_outputs.parquet",
                step_dir / "eval_merged.parquet",
                step_dir / "synthetic_base.parquet",
                step_dir / "synthetic_train_next.parquet",
                step_dir / "anchor_selection_manifest.json",
                step_dir / "used_anchor_ids.json",
                step_dir / "stage_manifest.json",
            ]
            for path in expected:
                if not path.exists():
                    missing.append(str(path))

            manifest_path = step_dir / "stage_manifest.json"
            if manifest_path.exists():
                manifest = _load_manifest(manifest_path)
                for stage in CONTEXT_STAGES:
                    rec = manifest.stages[stage]
                    if not rec.config_hash or rec.model_name is None or rec.branch is None or rec.generation is None:
                        missing.append(f"lineage_missing:{manifest_path}:{stage}")

    run_expected = [
        run_dir / "all_eval_merged.parquet",
        run_dir / "tables" / "metrics_by_generation.csv",
        run_dir / "figures" / "accuracy_vs_generation.png",
        run_dir / "figures" / "pedagogical_vs_generation.png",
        run_dir / "figures" / "silent_error_vs_generation.png",
    ]
    for path in run_expected:
        if not path.exists():
            missing.append(str(path))

    return missing


def run_pilot(
    *,
    cfg: AppConfig,
    sample_size: int,
    mock_judge: bool,
    dry_run: bool = False,
) -> PilotSummary:
    """Run fast end-to-end pilot using existing stage engine."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pilot_data_root = cfg.paths.output_root / "pilot_inputs" / ts
    pilot_data_root.mkdir(parents=True, exist_ok=True)

    _, used_sample_size = prepare_pilot_splits(
        cfg=cfg,
        sample_size=sample_size,
        pilot_data_root=pilot_data_root,
        seed=cfg.project.seed,
    )

    use_mock_judge = True if dry_run else mock_judge
    pilot_cfg = build_pilot_config(cfg=cfg, pilot_data_root=pilot_data_root, mock_judge=use_mock_judge)

    stage_executors: dict[str, Any] = {}
    if dry_run:
        stage_executors = {"generation": _dry_run_generation_executor}

    runner = ExperimentRunner(pilot_cfg, stage_executors=stage_executors)
    runner.run_full()

    model_name = pilot_cfg.models.local_models[0].name
    branches = [b.name for b in pilot_cfg.experiment.branches]
    generations = pilot_cfg.experiment.generations

    missing = validate_pilot_artifacts(
        run_dir=runner.ctx.run_dir,
        model_name=model_name,
        branches=branches,
        generations=generations,
    )

    all_eval = pd.read_parquet(runner.ctx.run_dir / "all_eval_merged.parquet")

    completed = 0
    run_manifest = _load_manifest(runner.ctx.run_dir / "run_stage_manifest.json")
    completed += sum(1 for s in RUN_STAGES if run_manifest.stages[s].status.value == "completed")
    for gen in range(1, generations + 1):
        for branch in branches:
            step_manifest = _load_manifest(
                runner.ctx.run_dir / model_name.replace(":", "_") / branch / f"gen_{gen}" / "stage_manifest.json"
            )
            completed += sum(1 for s in CONTEXT_STAGES if step_manifest.stages[s].status.value == "completed")

    return PilotSummary(
        run_dir=runner.ctx.run_dir,
        pilot_data_root=pilot_data_root,
        model_name=model_name,
        branches=branches,
        generations=generations,
        sample_size_requested=sample_size,
        sample_size_used=used_sample_size,
        total_examples_scored=int(len(all_eval)),
        artifacts_valid=(len(missing) == 0),
        missing_artifacts=missing,
        completed_stages=completed,
        skip_or_resume_events=0,
    )


def _dry_run_generation_executor(context: StageContext) -> StageExecutionResult:
    """Lightweight local generation substitute for smoke tests only."""
    heldout = pd.read_parquet(context.artifacts["heldout_test"])
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(heldout.to_dict(orient="records")):
        rows.append(
            {
                "run_id": "dry_run",
                "branch": context.branch,
                "generation": context.generation,
                "model_name": context.model_name,
                "example_id": rec["example_id"],
                "prompt_version": "dry_run",
                "prompt_text": str(rec.get("question", "")),
                "raw_response": f"Final answer: {idx}",
                "parsed_final_answer": str(idx),
            }
        )
    out_df = pd.DataFrame(rows)
    context.artifacts["model_outputs"].parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(context.artifacts["model_outputs"], index=False)
    return StageExecutionResult(row_count=int(len(out_df)))
