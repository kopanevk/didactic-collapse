from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.first_experiment import (
    _ensure_real_judge_provider,
    _override_judge_config_for_resume,
    _preflight_real_judge_auth,
    build_first_experiment_config,
    prepare_first_experiment_splits,
)
from didactic_collapse.orchestration.runner import CONTEXT_STAGES, ExperimentRunner

logger = logging.getLogger(__name__)

FEASIBILITY_MODE = "training_recycling_feasibility"
GEN0 = 0
GEN1 = 1
TRAINING_PLAN_FILE = "training_feasibility_plan.json"


class TrainingFeasibilityError(RuntimeError):
    """Raised when train-stage feasibility contract is violated."""


@dataclass(frozen=True)
class TrainStageRequest:
    run_dir: Path
    branch: str
    seed: int
    generation_from: int
    generation_to: int
    base_model_name: str
    target_model_name: str
    training_data_path: Path
    output_dir: Path


@dataclass(frozen=True)
class TrainStageResult:
    backend: str
    trained_model_name: str
    metadata_path: Path
    is_stub: bool


class TrainStageBackend(Protocol):
    def run(self, request: TrainStageRequest) -> TrainStageResult:
        ...


class CommandTrainStageBackend:
    def __init__(self, *, command_template: str, timeout_sec: int, result_filename: str) -> None:
        self._command_template = command_template
        self._timeout_sec = timeout_sec
        self._result_filename = result_filename

    def run(self, request: TrainStageRequest) -> TrainStageResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        command = self._command_template.format(
            run_dir=str(request.run_dir),
            branch=request.branch,
            seed=request.seed,
            generation_from=request.generation_from,
            generation_to=request.generation_to,
            base_model=request.base_model_name,
            target_model=request.target_model_name,
            train_path=str(request.training_data_path),
            output_dir=str(request.output_dir),
        )
        logger.info(
            "train_backend_command_start branch=%s input=%s output_dir=%s command=%s",
            request.branch,
            request.training_data_path,
            request.output_dir,
            command,
        )

        proc = subprocess.run(  # noqa: S602
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self._timeout_sec,
            cwd=Path.cwd(),
            check=False,
        )
        (request.output_dir / "trainer_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (request.output_dir / "trainer_stderr.log").write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            raise TrainingFeasibilityError(
                "Train-stage command failed. "
                f"branch={request.branch}, return_code={proc.returncode}, "
                f"stderr_preview={(proc.stderr or '')[:240]}"
            )

        metadata_path = request.output_dir / self._result_filename
        if not metadata_path.exists():
            raise TrainingFeasibilityError(
                "Train-stage command finished without required result metadata. "
                f"Expected file: {metadata_path}"
            )
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        trained_model_name = str(payload.get("trained_model_name", "")).strip()
        if not trained_model_name:
            raise TrainingFeasibilityError(
                f"Training result metadata missing non-empty trained_model_name: {metadata_path}"
            )
        logger.info(
            "train_backend_command_done branch=%s trained_model_name=%s metadata=%s",
            request.branch,
            trained_model_name,
            metadata_path,
        )
        return TrainStageResult(
            backend="command",
            trained_model_name=trained_model_name,
            metadata_path=metadata_path,
            is_stub=bool(payload.get("is_stub", False)),
        )


class StubTrainStageBackend:
    def __init__(self, *, result_filename: str) -> None:
        self._result_filename = result_filename

    def run(self, request: TrainStageRequest) -> TrainStageResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = request.output_dir / self._result_filename
        payload = {
            "created_at": datetime.now().isoformat(),
            "is_stub": True,
            "trained_model_name": request.target_model_name,
            "base_model_name": request.base_model_name,
            "branch": request.branch,
            "training_data_path": str(request.training_data_path),
            "note": "stub_backend_non_scientific",
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return TrainStageResult(
            backend="stub",
            trained_model_name=request.target_model_name,
            metadata_path=metadata_path,
            is_stub=True,
        )


@dataclass(frozen=True)
class TrainingRecord:
    branch: str
    generation_from: int
    generation_to: int
    base_model_name: str
    target_model_name: str
    trained_model_name: str
    backend: str
    training_data_path: Path
    metadata_path: Path
    is_stub: bool


@dataclass(frozen=True)
class TrainingFeasibilitySummary:
    run_dir: Path
    data_root_used: Path
    base_model_name: str
    branches: list[str]
    sample_size_requested: int
    sample_size_used: int
    summary_table_path_csv: Path
    summary_table_path_parquet: Path
    qualitative_path_csv: Path
    qualitative_path_parquet: Path
    training_records_csv: Path
    training_records_parquet: Path
    evaluation_mode: str


def _ensure_mode(cfg: AppConfig) -> None:
    if cfg.experiment.mode != FEASIBILITY_MODE:
        raise TrainingFeasibilityError(
            "Config is not in train feasibility mode. "
            f"Expected experiment.mode={FEASIBILITY_MODE}, got {cfg.experiment.mode}"
        )
    if cfg.experiment.generations < 2:
        raise TrainingFeasibilityError(
            "Train feasibility requires at least 2 generations (Gen-0 -> train -> Gen-1)."
        )


def _sanitize_name(text: str) -> str:
    allowed = [ch if (ch.isalnum() or ch in {"_", "-", ".", ":"}) else "_" for ch in text]
    return "".join(allowed).strip("_")


def _build_target_model_name(
    *,
    template: str,
    base_model_name: str,
    branch: str,
    seed: int,
    generation_from: int,
    generation_to: int,
) -> str:
    return template.format(
        base_model=_sanitize_name(base_model_name),
        branch=_sanitize_name(branch),
        seed=seed,
        generation_from=generation_from,
        generation_to=generation_to,
    )


def _build_backend(cfg: AppConfig) -> TrainStageBackend:
    if cfg.training.backend == "command":
        template = (cfg.training.command_template or "").strip()
        if not template:
            raise TrainingFeasibilityError(
                "training.backend=command requires non-empty training.command_template"
            )
        return CommandTrainStageBackend(
            command_template=template,
            timeout_sec=cfg.training.command_timeout_sec,
            result_filename=cfg.training.result_filename,
        )
    if cfg.training.backend == "stub":
        if not cfg.training.allow_stub_for_smoke:
            raise TrainingFeasibilityError(
                "Stub training backend is disabled for scientific runs. "
                "Set training.allow_stub_for_smoke=true only for non-scientific smoke checks."
            )
        return StubTrainStageBackend(result_filename=cfg.training.result_filename)
    raise TrainingFeasibilityError(f"Unsupported training backend: {cfg.training.backend}")


def _load_or_init_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "records": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_record_by_branch(plan: dict[str, Any], branch: str) -> dict[str, Any] | None:
    records = plan.get("records", [])
    if not isinstance(records, list):
        raise TrainingFeasibilityError(f"Corrupted training plan records: {type(records)}")
    for item in records:
        if isinstance(item, dict) and item.get("branch") == branch:
            return item
    return None


def _upsert_plan_record(plan: dict[str, Any], payload: dict[str, Any]) -> None:
    records = plan.setdefault("records", [])
    if not isinstance(records, list):
        raise TrainingFeasibilityError("Corrupted training plan records")
    for idx, item in enumerate(records):
        if isinstance(item, dict) and item.get("branch") == payload["branch"]:
            records[idx] = payload
            return
    records.append(payload)


def _save_plan(path: Path, plan: dict[str, Any]) -> None:
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def _limit_training_rows(*, df: pd.DataFrame, max_train_rows: int | None, seed: int) -> pd.DataFrame:
    if max_train_rows is None or len(df) <= max_train_rows:
        return df
    return df.sample(n=max_train_rows, random_state=seed).sort_values("example_id").reset_index(drop=True)


def _export_feasibility_summary(
    *,
    all_eval: pd.DataFrame,
    out_dir: Path,
    base_model_name: str,
) -> tuple[Path, Path, pd.DataFrame]:
    required = {
        "branch",
        "generation",
        "example_id",
        "model_name",
        "is_correct",
        "overall_pedagogical_score",
        "is_silent_error",
    }
    missing = required.difference(all_eval.columns)
    if missing:
        raise TrainingFeasibilityError(
            f"Cannot build train-feasibility summary: missing columns {sorted(missing)}"
        )

    grouped = (
        all_eval.groupby(["branch", "generation"], as_index=False)
        .agg(
            sample_count=("example_id", "count"),
            accuracy_mean=("is_correct", "mean"),
            pedagogical_score_mean=("overall_pedagogical_score", "mean"),
            silent_error_rate=("is_silent_error", "mean"),
        )
        .sort_values(["branch", "generation"])
        .reset_index(drop=True)
    )
    model_map = (
        all_eval.groupby(["branch", "generation"])["model_name"]
        .agg(lambda x: "|".join(sorted({str(v) for v in x.tolist()})))
        .reset_index()
        .rename(columns={"model_name": "generation_model_name"})
    )
    grouped = grouped.merge(model_map, on=["branch", "generation"], how="left", validate="one_to_one")

    if "pred_parse_success" in all_eval.columns:
        parse_df = all_eval.copy()
        parse_df["pred_parse_success"] = parse_df["pred_parse_success"].fillna(False).astype(bool)
        parse_stats = (
            parse_df.groupby(["branch", "generation"], as_index=False)
            .agg(parse_failure_pred_count=("pred_parse_success", lambda s: int((~s).sum())))
            .sort_values(["branch", "generation"])
        )
        grouped = grouped.merge(parse_stats, on=["branch", "generation"], how="left", validate="one_to_one")
        grouped["parse_failure_pred_rate"] = (
            grouped["parse_failure_pred_count"] / grouped["sample_count"].replace({0: pd.NA})
        ).fillna(0.0)
    else:
        grouped["parse_failure_pred_count"] = 0
        grouped["parse_failure_pred_rate"] = 0.0

    grouped["base_model_name"] = base_model_name
    grouped["evaluation_mode"] = FEASIBILITY_MODE

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_feasibility_summary.csv"
    parquet_path = out_dir / "training_feasibility_summary.parquet"
    grouped.to_csv(csv_path, index=False)
    grouped.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path, grouped


def _export_feasibility_qualitative(*, all_eval: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    required = {"generation", "branch", "example_id", "is_correct", "overall_pedagogical_score", "is_silent_error"}
    missing = required.difference(all_eval.columns)
    if missing:
        raise TrainingFeasibilityError(
            f"Cannot build train-feasibility qualitative export: missing columns {sorted(missing)}"
        )
    out = all_eval[
        (all_eval["is_correct"] == True)  # noqa: E712
        & (all_eval["overall_pedagogical_score"] <= 2)
        & (all_eval["is_silent_error"] == True)  # noqa: E712
    ].copy()
    out["evaluation_mode"] = FEASIBILITY_MODE
    out = out.sort_values(["generation", "branch", "overall_pedagogical_score", "example_id"])

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_feasibility_qualitative_candidates.csv"
    parquet_path = out_dir / "training_feasibility_qualitative_candidates.parquet"
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def _export_training_records(
    *,
    records: list[TrainingRecord],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "branch": r.branch,
            "generation_from": r.generation_from,
            "generation_to": r.generation_to,
            "base_model_name": r.base_model_name,
            "target_model_name": r.target_model_name,
            "trained_model_name": r.trained_model_name,
            "backend": r.backend,
            "training_data_path": str(r.training_data_path),
            "metadata_path": str(r.metadata_path),
            "is_stub": r.is_stub,
            "evaluation_mode": FEASIBILITY_MODE,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    csv_path = out_dir / "training_feasibility_records.csv"
    parquet_path = out_dir / "training_feasibility_records.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def _resolve_resume_cfg(*, cfg: AppConfig, run_dir: Path) -> tuple[AppConfig, Path]:
    snapshot_path = run_dir / "run_config.snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run snapshot for resume: {snapshot_path}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_cfg = AppConfig.model_validate(snapshot["config"])
    data_root = Path(snapshot_cfg.paths.data_root)
    merged_cfg = _override_judge_config_for_resume(snapshot_cfg=snapshot_cfg, requested_cfg=cfg)
    raw = merged_cfg.model_dump(mode="python")
    raw["training"] = cfg.training.model_dump(mode="python")
    raw["experiment"]["mode"] = cfg.experiment.mode
    out_cfg = AppConfig.model_validate(raw)
    return out_cfg, data_root


def run_training_recycling_feasibility(
    *,
    cfg: AppConfig,
    sample_size: int,
    run_dir: Path | None = None,
) -> TrainingFeasibilitySummary:
    _ensure_real_judge_provider(cfg)
    _preflight_real_judge_auth(cfg)
    _ensure_mode(cfg)
    backend = _build_backend(cfg)

    if run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_data_root = cfg.paths.output_root / "train_feasibility_inputs" / ts
        exp_data_root.mkdir(parents=True, exist_ok=True)
        used_sample_size = prepare_first_experiment_splits(
            cfg=cfg,
            sample_size=sample_size,
            experiment_data_root=exp_data_root,
            seed=cfg.project.seed,
        )
        exp_cfg = build_first_experiment_config(cfg=cfg, data_root=exp_data_root)
        runner = ExperimentRunner(exp_cfg)
        runner.save_run_metadata()
    else:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        exp_cfg, exp_data_root = _resolve_resume_cfg(cfg=cfg, run_dir=run_dir)
        runner = ExperimentRunner(exp_cfg, run_dir=run_dir)
        used_sample_size = int(len(pd.read_parquet(exp_data_root / "splits" / "heldout_test.parquet")))

    model_name = exp_cfg.models.local_models[0].name
    branches = [b.name for b in exp_cfg.experiment.branches]
    if not branches:
        raise TrainingFeasibilityError("No branches configured for train feasibility run.")

    runner.run_stage("data_prep")

    plan_path = runner.ctx.run_dir / TRAINING_PLAN_FILE
    plan = _load_or_init_plan(plan_path)
    plan["evaluation_mode"] = FEASIBILITY_MODE
    plan["base_model_name"] = model_name
    plan["seed"] = exp_cfg.project.seed

    training_records: list[TrainingRecord] = []
    for branch in branches:
        # Gen-0: current stable context path (inference + synthetic + anchoring).
        runner.resume_from_checkpoint(
            model_name=model_name,
            branch=branch,
            generation=GEN0,
            seed=exp_cfg.project.seed,
            force=exp_cfg.runtime.force_recompute,
        )

        gen0_dir = runner.ctx.run_dir / model_name.replace(":", "_") / branch / f"gen_{GEN0}"
        train_source_path = gen0_dir / "synthetic_train_next.parquet"
        if not train_source_path.exists():
            raise TrainingFeasibilityError(f"Missing training source dataset: {train_source_path}")

        train_df = pd.read_parquet(train_source_path)
        train_df = _limit_training_rows(
            df=train_df,
            max_train_rows=exp_cfg.training.max_train_rows,
            seed=exp_cfg.project.seed,
        )
        prepared_train_path = gen0_dir / "synthetic_train_for_feasibility.parquet"
        train_df.to_parquet(prepared_train_path, index=False)

        target_model_name = _build_target_model_name(
            template=exp_cfg.training.output_model_name_template,
            base_model_name=model_name,
            branch=branch,
            seed=exp_cfg.project.seed,
            generation_from=GEN0,
            generation_to=GEN1,
        )
        training_out_dir = gen0_dir / "training_stage"

        existing = _plan_record_by_branch(plan, branch)
        if existing is not None:
            existing_meta_path = Path(str(existing.get("metadata_path", "")))
            if existing_meta_path.exists():
                trained_model_name = str(existing.get("trained_model_name", "")).strip()
                if not trained_model_name:
                    raise TrainingFeasibilityError(
                        f"Training plan record has empty trained_model_name for branch={branch}"
                    )
                training_result = TrainStageResult(
                    backend=str(existing.get("backend", "unknown")),
                    trained_model_name=trained_model_name,
                    metadata_path=existing_meta_path,
                    is_stub=bool(existing.get("is_stub", False)),
                )
            else:
                existing = None

        if existing is None:
            request = TrainStageRequest(
                run_dir=runner.ctx.run_dir,
                branch=branch,
                seed=exp_cfg.project.seed,
                generation_from=GEN0,
                generation_to=GEN1,
                base_model_name=model_name,
                target_model_name=target_model_name,
                training_data_path=prepared_train_path,
                output_dir=training_out_dir,
            )
            training_result = backend.run(request)
            if training_result.is_stub and not exp_cfg.training.allow_stub_for_smoke:
                raise TrainingFeasibilityError(
                    "Stub train-stage output detected while allow_stub_for_smoke=false. "
                    "This mode is non-scientific."
                )
            _upsert_plan_record(
                plan,
                {
                    "branch": branch,
                    "generation_from": GEN0,
                    "generation_to": GEN1,
                    "base_model_name": model_name,
                    "target_model_name": target_model_name,
                    "trained_model_name": training_result.trained_model_name,
                    "backend": training_result.backend,
                    "metadata_path": str(training_result.metadata_path),
                    "training_data_path": str(prepared_train_path),
                    "is_stub": training_result.is_stub,
                    "created_at": datetime.now().isoformat(),
                },
            )
            _save_plan(plan_path, plan)

        trained_model_name = training_result.trained_model_name
        for stage in ("generation", "answer_extraction", "accuracy", "judge"):
            runner.run_stage(
                stage,  # type: ignore[arg-type]
                model_name=trained_model_name,
                branch=branch,
                generation=GEN1,
                seed=exp_cfg.project.seed,
                force=exp_cfg.runtime.force_recompute,
            )
        if exp_cfg.training.run_gen1_recycling_stages:
            for stage in ("synthetic_build", "anchoring"):
                runner.run_stage(
                    stage,  # type: ignore[arg-type]
                    model_name=trained_model_name,
                    branch=branch,
                    generation=GEN1,
                    seed=exp_cfg.project.seed,
                    force=exp_cfg.runtime.force_recompute,
                )

        training_records.append(
            TrainingRecord(
                branch=branch,
                generation_from=GEN0,
                generation_to=GEN1,
                base_model_name=model_name,
                target_model_name=target_model_name,
                trained_model_name=trained_model_name,
                backend=training_result.backend,
                training_data_path=prepared_train_path,
                metadata_path=training_result.metadata_path,
                is_stub=training_result.is_stub,
            )
        )

    # Force recompute to include newly completed contexts after resume-style increments.
    runner.run_stage("aggregate", force=True)
    runner.run_stage("plotting", force=True)

    all_eval = pd.read_parquet(runner.ctx.run_dir / "all_eval_merged.parquet")
    summary_csv, summary_parquet, _ = _export_feasibility_summary(
        all_eval=all_eval,
        out_dir=runner.ctx.run_dir / "tables",
        base_model_name=model_name,
    )
    qual_csv, qual_parquet = _export_feasibility_qualitative(
        all_eval=all_eval,
        out_dir=runner.ctx.run_dir / "tables",
    )
    records_csv, records_parquet = _export_training_records(
        records=training_records,
        out_dir=runner.ctx.run_dir / "tables",
    )

    return TrainingFeasibilitySummary(
        run_dir=runner.ctx.run_dir,
        data_root_used=exp_data_root,
        base_model_name=model_name,
        branches=branches,
        sample_size_requested=sample_size,
        sample_size_used=used_sample_size,
        summary_table_path_csv=summary_csv,
        summary_table_path_parquet=summary_parquet,
        qualitative_path_csv=qual_csv,
        qualitative_path_parquet=qual_parquet,
        training_records_csv=records_csv,
        training_records_parquet=records_parquet,
        evaluation_mode=FEASIBILITY_MODE,
    )
