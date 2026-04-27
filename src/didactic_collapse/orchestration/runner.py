from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from didactic_collapse.analysis.aggregate import aggregate_metrics
from didactic_collapse.analysis.plots import plot_metric_by_generation
from didactic_collapse.clients.base import JudgeClient
from didactic_collapse.clients.judge_client import (
    MockJudgeClient,
    OpenAICompatibleJudgeClient,
    build_cerebras_judge_client,
    build_gemini_judge_client,
)
from didactic_collapse.clients.ollama_client import OllamaClient
from didactic_collapse.config.settings import AppConfig
from didactic_collapse.judging.accuracy import evaluate_accuracy
from didactic_collapse.pipeline.extract_answer import extract_final_answer_result
from didactic_collapse.pipeline.generate_outputs import run_generation
from didactic_collapse.pipeline.judge_outputs import run_judging
from didactic_collapse.prompts.prompt_registry import load_judge_prompt
from didactic_collapse.recycling.anchoring import (
    AnchorPolicy,
    AnchorSelectionContext,
    save_anchoring_artifacts,
    select_human_anchors,
)
from didactic_collapse.recycling.pedagogical_verification_filter import (
    PVFPolicy,
    apply_pedagogical_verification_filter,
    save_pvf_artifacts,
)
from didactic_collapse.utils.io_utils import save_tabular

logger = logging.getLogger(__name__)

StageName = Literal[
    "data_prep",
    "generation",
    "answer_extraction",
    "accuracy",
    "judge",
    "synthetic_build",
    "anchoring",
    "aggregate",
    "plotting",
]

RUN_STAGES: tuple[StageName, ...] = ("data_prep", "aggregate", "plotting")
CONTEXT_STAGES: tuple[StageName, ...] = (
    "generation",
    "answer_extraction",
    "accuracy",
    "judge",
    "synthetic_build",
    "anchoring",
)
ALL_STAGES: tuple[StageName, ...] = RUN_STAGES + CONTEXT_STAGES


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OrchestrationError(RuntimeError):
    """Raised when checkpoint/manifest/artifact invariants are violated."""


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_name: StageName
    status: StageStatus
    timestamp_start: str | None
    timestamp_end: str | None
    model_name: str | None
    generation: int | None
    branch: str | None
    seed: int
    config_hash: str
    input_artifacts: list[str]
    output_artifacts: list[str]
    row_count: int | None
    error_message: str | None


class StageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    run_id: str
    run_dir: str
    scope: Literal["run", "context"]
    model_name: str | None
    generation: int | None
    branch: str | None
    seed: int
    config_hash: str
    stages: dict[str, StageRecord]


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path


@dataclass(frozen=True)
class StageContext:
    stage_name: StageName
    run_dir: Path
    model_name: str | None
    branch: str | None
    generation: int | None
    seed: int
    config_hash: str
    step_dir: Path
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class StageExecutionResult:
    row_count: int | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_config_hash(cfg: AppConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_readable_artifact(path: Path, must_be_non_empty: bool = True) -> int | None:
    if not path.exists():
        raise OrchestrationError(f"Missing artifact file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if must_be_non_empty and len(df) == 0:
            raise OrchestrationError(f"Artifact has zero rows: {path}")
        return int(len(df))
    if suffix == ".csv":
        df = pd.read_csv(path)
        if must_be_non_empty and len(df) == 0:
            raise OrchestrationError(f"Artifact has zero rows: {path}")
        return int(len(df))
    if suffix == ".json":
        _ = json.loads(path.read_text(encoding="utf-8"))
        return None
    return None


class ExperimentRunner:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        run_dir: Path | None = None,
        stage_executors: dict[StageName, Callable[[StageContext], StageExecutionResult]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.config_hash = _compute_config_hash(cfg)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_run_dir = run_dir or (cfg.paths.output_root / "runs" / f"{cfg.project.run_tag}_{ts}")
        self.ctx = RunContext(
            run_id=f"{cfg.project.name}_{cfg.project.run_tag}_{ts}",
            run_dir=resolved_run_dir,
        )

        self._stage_executors = stage_executors or {}
        self._gen_client: OllamaClient | None = None
        self._judge_client: JudgeClient | None = None
        self._judge_prompt: str | None = None
        self._heldout_df: pd.DataFrame | None = None
        self._anchor_pool_df: pd.DataFrame | None = None

    def save_run_metadata(self) -> None:
        self.ctx.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.ctx.run_id,
            "created_at": datetime.now().isoformat(),
            "config_hash": self.config_hash,
            "config": self.cfg.model_dump(mode="json"),
        }
        (self.ctx.run_dir / "run_config.snapshot.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _get_gen_client(self) -> OllamaClient:
        if self._gen_client is None:
            self._gen_client = OllamaClient()
        return self._gen_client

    def _get_judge_client(self) -> JudgeClient:
        if self._judge_client is None:
            provider = self.cfg.judge.provider.strip().lower()
            if provider in {"mock", "stub", "mock_judge"}:
                self._judge_client = MockJudgeClient()
            elif provider == "cerebras":
                self._judge_client = build_cerebras_judge_client(
                    model_name=self.cfg.judge.model_name,
                    base_url=self.cfg.judge.base_url,
                    api_key_env=self.cfg.judge.api_key_env,
                    timeout_sec=self.cfg.judge.timeout_sec,
                    max_retries=self.cfg.judge.max_retries,
                )
            elif provider in {"gemini", "gemini_sdk", "gemini_openai_compatible"}:
                # Even for legacy provider names, use official Gemini SDK auth path only.
                self._judge_client = build_gemini_judge_client(
                    model_name=self.cfg.judge.model_name,
                    api_key_env=self.cfg.judge.api_key_env,
                )
            else:
                self._judge_client = OpenAICompatibleJudgeClient(
                    base_url=self.cfg.judge.base_url,
                    model_name=self.cfg.judge.model_name,
                    api_key_env=self.cfg.judge.api_key_env,
                    timeout_sec=self.cfg.judge.timeout_sec,
                    max_retries=self.cfg.judge.max_retries,
                )
        return self._judge_client

    def _get_judge_prompt(self) -> str:
        if self._judge_prompt is None:
            self._judge_prompt = load_judge_prompt(self.cfg.paths.prompt_dir)
        return self._judge_prompt

    def _get_heldout(self) -> pd.DataFrame:
        if self._heldout_df is None:
            self._heldout_df = pd.read_parquet(self.cfg.paths.data_root / "splits" / "heldout_test.parquet")
        return self._heldout_df

    def _get_anchor_pool(self) -> pd.DataFrame:
        if self._anchor_pool_df is None:
            self._anchor_pool_df = pd.read_parquet(self.cfg.paths.data_root / "splits" / "anchor_pool.parquet")
        return self._anchor_pool_df

    def _step_dir(self, *, model_name: str, branch: str, generation: int) -> Path:
        return self.ctx.run_dir / model_name.replace(":", "_") / branch / f"gen_{generation}"

    def _context_artifacts(self, step_dir: Path) -> dict[str, Path]:
        return {
            "model_outputs": step_dir / "model_outputs.parquet",
            "generation_partial": step_dir / "generation_partial.parquet",
            "generation_failures": step_dir / "generation_failures.parquet",
            "generation_metadata": step_dir / "generation_progress.json",
            "answer_extraction": step_dir / "answer_extraction.parquet",
            "accuracy_table": step_dir / "accuracy_table.parquet",
            "judge_outputs": step_dir / "judge_outputs.parquet",
            "judge_partial": step_dir / "judge_partial.parquet",
            "judge_failures": step_dir / "judge_failures.parquet",
            "judge_metadata": step_dir / "judge_progress.json",
            "eval_merged": step_dir / "eval_merged.parquet",
            "synthetic_base": step_dir / "synthetic_base.parquet",
            "synthetic_train_next": step_dir / "synthetic_train_next.parquet",
            "pvf_filtered_training": step_dir / "filtered_training_dataset.parquet",
            "pvf_rejected_examples": step_dir / "rejected_examples.parquet",
            "pvf_report": step_dir / "pvf_filter_report.json",
            "anchor_metadata": step_dir / "anchor_selection_manifest.json",
            "used_anchor_ids": step_dir / "used_anchor_ids.json",
            "anchor_quality_diagnostics": step_dir / "anchor_quality_diagnostics.parquet",
        }

    def _run_artifacts(self) -> dict[str, Path]:
        return {
            "base_train": self.cfg.paths.data_root / "splits" / "base_train.parquet",
            "anchor_pool": self.cfg.paths.data_root / "splits" / "anchor_pool.parquet",
            "heldout_test": self.cfg.paths.data_root / "splits" / "heldout_test.parquet",
            "split_metadata": self.cfg.paths.data_root / "splits" / "split_metadata.json",
            "all_eval_merged_parquet": self.ctx.run_dir / "all_eval_merged.parquet",
            "all_eval_merged_csv": self.ctx.run_dir / "all_eval_merged.csv",
            "metrics_csv": self.ctx.run_dir / "tables" / "metrics_by_generation.csv",
            "accuracy_plot": self.ctx.run_dir / "figures" / "accuracy_vs_generation.png",
            "pedagogical_plot": self.ctx.run_dir / "figures" / "pedagogical_vs_generation.png",
            "silent_error_plot": self.ctx.run_dir / "figures" / "silent_error_vs_generation.png",
        }

    def _stage_expected_outputs(self, stage_name: StageName, artifacts: dict[str, Path]) -> list[Path]:
        if stage_name == "data_prep":
            return [
                artifacts["base_train"],
                artifacts["anchor_pool"],
                artifacts["heldout_test"],
                artifacts["split_metadata"],
            ]
        if stage_name == "generation":
            return [
                artifacts["model_outputs"],
                artifacts["generation_partial"],
                artifacts["generation_failures"],
                artifacts["generation_metadata"],
            ]
        if stage_name == "answer_extraction":
            return [artifacts["answer_extraction"]]
        if stage_name == "accuracy":
            return [artifacts["accuracy_table"]]
        if stage_name == "judge":
            return [
                artifacts["judge_outputs"],
                artifacts["judge_partial"],
                artifacts["judge_failures"],
                artifacts["judge_metadata"],
                artifacts["eval_merged"],
            ]
        if stage_name == "synthetic_build":
            return [artifacts["synthetic_base"]]
        if stage_name == "anchoring":
            return [
                artifacts["synthetic_train_next"],
                artifacts["anchor_metadata"],
                artifacts["used_anchor_ids"],
                artifacts["pvf_filtered_training"],
                artifacts["pvf_rejected_examples"],
                artifacts["pvf_report"],
            ]
        if stage_name == "aggregate":
            return [artifacts["all_eval_merged_parquet"], artifacts["metrics_csv"]]
        if stage_name == "plotting":
            return [artifacts["accuracy_plot"], artifacts["pedagogical_plot"], artifacts["silent_error_plot"]]
        raise OrchestrationError(f"Unknown stage: {stage_name}")

    def _stage_expected_inputs(self, stage_name: StageName, artifacts: dict[str, Path]) -> list[Path]:
        if stage_name == "data_prep":
            return []
        if stage_name == "generation":
            return [artifacts["heldout_test"]]
        if stage_name == "answer_extraction":
            return [artifacts["model_outputs"]]
        if stage_name == "accuracy":
            return [artifacts["answer_extraction"], artifacts["heldout_test"]]
        if stage_name == "judge":
            return [artifacts["model_outputs"], artifacts["heldout_test"], artifacts["accuracy_table"]]
        if stage_name == "synthetic_build":
            return [artifacts["answer_extraction"], artifacts["heldout_test"]]
        if stage_name == "anchoring":
            return [
                artifacts["synthetic_base"],
                artifacts["accuracy_table"],
                artifacts["judge_outputs"],
                artifacts["anchor_pool"],
                artifacts["base_train"],
                artifacts["heldout_test"],
            ]
        if stage_name == "aggregate":
            return []
        if stage_name == "plotting":
            return [artifacts["metrics_csv"]]
        raise OrchestrationError(f"Unknown stage: {stage_name}")

    def _manifest_path(self, scope: Literal["run", "context"], step_dir: Path | None) -> Path:
        if scope == "run":
            return self.ctx.run_dir / "run_stage_manifest.json"
        if step_dir is None:
            raise OrchestrationError("Context manifest requires step_dir")
        return step_dir / "stage_manifest.json"

    def _build_manifest(
        self,
        *,
        scope: Literal["run", "context"],
        model_name: str | None,
        branch: str | None,
        generation: int | None,
        seed: int,
    ) -> StageManifest:
        stages_for_scope = RUN_STAGES if scope == "run" else CONTEXT_STAGES
        records: dict[str, StageRecord] = {}
        for stage in stages_for_scope:
            records[stage] = StageRecord(
                stage_name=stage,
                status=StageStatus.PENDING,
                timestamp_start=None,
                timestamp_end=None,
                model_name=model_name,
                generation=generation,
                branch=branch,
                seed=seed,
                config_hash=self.config_hash,
                input_artifacts=[],
                output_artifacts=[],
                row_count=None,
                error_message=None,
            )
        return StageManifest(
            schema_version=1,
            run_id=self.ctx.run_id,
            run_dir=str(self.ctx.run_dir),
            scope=scope,
            model_name=model_name,
            generation=generation,
            branch=branch,
            seed=seed,
            config_hash=self.config_hash,
            stages=records,
        )

    def _save_manifest(self, path: Path, manifest: StageManifest) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    def _validate_manifest_lineage(
        self,
        *,
        manifest: StageManifest,
        scope: Literal["run", "context"],
        model_name: str | None,
        branch: str | None,
        generation: int | None,
        seed: int,
    ) -> None:
        if manifest.scope != scope:
            raise OrchestrationError(f"Manifest scope mismatch: expected {scope}, got {manifest.scope}")
        if manifest.config_hash != self.config_hash:
            raise OrchestrationError("Manifest config_hash mismatch")
        if manifest.model_name != model_name:
            raise OrchestrationError("Manifest model_name mismatch")
        if manifest.branch != branch:
            raise OrchestrationError("Manifest branch mismatch")
        if manifest.generation != generation:
            raise OrchestrationError("Manifest generation mismatch")
        if manifest.seed != seed:
            raise OrchestrationError("Manifest seed mismatch")

        expected_stages = RUN_STAGES if scope == "run" else CONTEXT_STAGES
        for stage in expected_stages:
            if stage not in manifest.stages:
                raise OrchestrationError(f"Manifest missing stage record: {stage}")

    def _load_or_init_manifest(
        self,
        *,
        scope: Literal["run", "context"],
        manifest_path: Path,
        model_name: str | None,
        branch: str | None,
        generation: int | None,
        seed: int,
    ) -> StageManifest:
        if not manifest_path.exists():
            manifest = self._build_manifest(
                scope=scope,
                model_name=model_name,
                branch=branch,
                generation=generation,
                seed=seed,
            )
            self._save_manifest(manifest_path, manifest)
            return manifest

        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = StageManifest.model_validate(loaded)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise OrchestrationError(f"Corrupted manifest at {manifest_path}: {exc}") from exc

        self._validate_manifest_lineage(
            manifest=manifest,
            scope=scope,
            model_name=model_name,
            branch=branch,
            generation=generation,
            seed=seed,
        )
        return manifest

    def _build_stage_context(
        self,
        *,
        stage_name: StageName,
        model_name: str | None,
        branch: str | None,
        generation: int | None,
        seed: int,
    ) -> StageContext:
        if stage_name in CONTEXT_STAGES:
            if model_name is None or branch is None or generation is None:
                raise OrchestrationError(
                    f"Stage '{stage_name}' requires model_name, branch, and generation"
                )
            step_dir = self._step_dir(model_name=model_name, branch=branch, generation=generation)
            artifacts = self._context_artifacts(step_dir)
            artifacts["anchor_pool"] = self.cfg.paths.data_root / "splits" / "anchor_pool.parquet"
            artifacts["heldout_test"] = self.cfg.paths.data_root / "splits" / "heldout_test.parquet"
            artifacts["base_train"] = self.cfg.paths.data_root / "splits" / "base_train.parquet"
        else:
            step_dir = self.ctx.run_dir
            artifacts = self._run_artifacts()

        return StageContext(
            stage_name=stage_name,
            run_dir=self.ctx.run_dir,
            model_name=model_name,
            branch=branch,
            generation=generation,
            seed=seed,
            config_hash=self.config_hash,
            step_dir=step_dir,
            artifacts=artifacts,
        )

    def _artifact_must_be_non_empty(self, path: Path) -> bool:
        # Failure artifacts may legitimately be empty while still being valid outputs.
        if "failures" in path.name:
            return False
        if "rejected_examples" in path.name:
            return False
        return True

    def _artifact_is_optional(self, path: Path) -> bool:
        # Backward-compatible with runs created before row-level sidecar artifacts existed.
        optional_names = {
            "generation_partial.parquet",
            "generation_failures.parquet",
            "generation_progress.json",
            "judge_partial.parquet",
            "judge_failures.parquet",
            "judge_progress.json",
            # Present only for pvf branches / newer runs.
            "filtered_training_dataset.parquet",
            "rejected_examples.parquet",
            "pvf_filter_report.json",
        }
        return path.name in optional_names

    def _stage_supports_row_level_resume(self, stage_name: StageName) -> bool:
        return stage_name in {"judge", "generation"}

    def _validate_stage_artifacts(self, output_paths: list[Path]) -> int | None:
        row_count: int | None = None
        for path in output_paths:
            if not path.exists() and self._artifact_is_optional(path):
                continue
            must_non_empty = self._artifact_must_be_non_empty(path)
            maybe_rows = _validate_readable_artifact(path, must_be_non_empty=must_non_empty)
            if maybe_rows is not None and must_non_empty:
                row_count = maybe_rows if row_count is None else min(row_count, maybe_rows)
        return row_count

    def _assert_no_partial_outputs(self, stage_name: StageName, output_paths: list[Path]) -> None:
        existing = [p for p in output_paths if p.exists()]
        if existing:
            raise OrchestrationError(
                f"Partial artifacts detected for incomplete stage '{stage_name}': "
                + ", ".join(str(p) for p in existing)
            )

    def _load_previously_used_anchor_ids(self, *, model_name: str, branch: str, generation: int) -> set[str]:
        used_ids: set[str] = set()
        branch_dir = self.ctx.run_dir / model_name.replace(":", "_") / branch
        if not branch_dir.exists():
            return used_ids

        for manifest_path in branch_dir.glob("gen_*/used_anchor_ids.json"):
            gen_name = manifest_path.parent.name
            if not gen_name.startswith("gen_"):
                continue
            try:
                gen_num = int(gen_name.split("_", maxsplit=1)[1])
            except ValueError:
                continue
            if gen_num >= generation:
                continue
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise OrchestrationError(f"Invalid used_anchor_ids artifact: {manifest_path}")
            for item in loaded:
                if not isinstance(item, str):
                    raise OrchestrationError(f"Invalid used_anchor_ids entry in {manifest_path}")
                used_ids.add(item)
        return used_ids

    def _resolve_branch_type(self, branch_cfg: Any) -> str:
        declared = getattr(branch_cfg, "branch_type", None)
        if declared is not None:
            return str(declared)
        name = str(getattr(branch_cfg, "name", ""))
        if name.startswith("pvf_"):
            return "pvf_medium"
        anchor_ratio = float(getattr(branch_cfg, "anchor_ratio", 0.0))
        if anchor_ratio > 0.0:
            return "human_anchoring"
        return "pure_recycling"

    def run_stage(
        self,
        stage_name: StageName,
        *,
        model_name: str | None = None,
        branch: str | None = None,
        generation: int | None = None,
        seed: int | None = None,
        force: bool = False,
    ) -> StageStatus:
        if stage_name not in ALL_STAGES:
            raise OrchestrationError(f"Unknown stage: {stage_name}")

        seed_value = self.cfg.project.seed if seed is None else seed
        context = self._build_stage_context(
            stage_name=stage_name,
            model_name=model_name,
            branch=branch,
            generation=generation,
            seed=seed_value,
        )

        scope: Literal["run", "context"] = "context" if stage_name in CONTEXT_STAGES else "run"
        manifest_path = self._manifest_path(scope, context.step_dir if scope == "context" else None)
        manifest = self._load_or_init_manifest(
            scope=scope,
            manifest_path=manifest_path,
            model_name=model_name,
            branch=branch,
            generation=generation,
            seed=seed_value,
        )

        record = manifest.stages[stage_name]
        output_paths = self._stage_expected_outputs(stage_name, context.artifacts)
        input_paths = self._stage_expected_inputs(stage_name, context.artifacts)

        if record.config_hash != self.config_hash:
            raise OrchestrationError(f"Lineage mismatch for stage '{stage_name}': config_hash")
        if record.model_name != model_name or record.branch != branch or record.generation != generation:
            raise OrchestrationError(f"Lineage mismatch for stage '{stage_name}': model/branch/generation")
        if record.seed != seed_value:
            raise OrchestrationError(f"Lineage mismatch for stage '{stage_name}': seed")

        if record.status == StageStatus.COMPLETED and not force:
            if record.output_artifacts and sorted(record.output_artifacts) != sorted(str(p) for p in output_paths):
                raise OrchestrationError(
                    f"Lineage mismatch for stage '{stage_name}': output artifacts differ from manifest"
                )
            self._validate_stage_artifacts(output_paths)
            logger.info("Skipping completed stage %s", stage_name)
            return StageStatus.COMPLETED

        if (
            stage_name != "data_prep"
            and record.status in (StageStatus.PENDING, StageStatus.RUNNING, StageStatus.FAILED)
            and not force
            and not self._stage_supports_row_level_resume(stage_name)
        ):
            self._assert_no_partial_outputs(stage_name, output_paths)

        for input_path in input_paths:
            _validate_readable_artifact(input_path, must_be_non_empty=True)

        record.status = StageStatus.RUNNING
        record.timestamp_start = _now_iso()
        record.timestamp_end = None
        record.error_message = None
        record.input_artifacts = [str(p) for p in input_paths]
        record.output_artifacts = [str(p) for p in output_paths]
        record.row_count = None
        manifest.stages[stage_name] = record
        self._save_manifest(manifest_path, manifest)

        try:
            result = self._execute_stage(stage_name, context)
            validated_rows = self._validate_stage_artifacts(output_paths)
            record.status = StageStatus.COMPLETED
            record.timestamp_end = _now_iso()
            record.row_count = result.row_count if result.row_count is not None else validated_rows
            record.error_message = None
        except Exception as exc:
            record.status = StageStatus.FAILED
            record.timestamp_end = _now_iso()
            record.error_message = str(exc)
            manifest.stages[stage_name] = record
            self._save_manifest(manifest_path, manifest)
            raise

        manifest.stages[stage_name] = record
        self._save_manifest(manifest_path, manifest)
        return StageStatus.COMPLETED

    def resume_from_checkpoint(
        self,
        *,
        model_name: str,
        branch: str,
        generation: int,
        seed: int | None = None,
        force: bool = False,
    ) -> None:
        seed_value = self.cfg.project.seed if seed is None else seed
        step_dir = self._step_dir(model_name=model_name, branch=branch, generation=generation)
        manifest_path = self._manifest_path("context", step_dir)

        manifest = self._load_or_init_manifest(
            scope="context",
            manifest_path=manifest_path,
            model_name=model_name,
            branch=branch,
            generation=generation,
            seed=seed_value,
        )

        start_index: int | None = None
        artifacts = self._context_artifacts(step_dir)
        artifacts["anchor_pool"] = self.cfg.paths.data_root / "splits" / "anchor_pool.parquet"
        artifacts["heldout_test"] = self.cfg.paths.data_root / "splits" / "heldout_test.parquet"
        artifacts["base_train"] = self.cfg.paths.data_root / "splits" / "base_train.parquet"

        for idx, stage_name in enumerate(CONTEXT_STAGES):
            record = manifest.stages[stage_name]
            expected_outputs = self._stage_expected_outputs(stage_name, artifacts)

            if record.status == StageStatus.COMPLETED:
                self._validate_stage_artifacts(expected_outputs)
                continue

            if record.status in (StageStatus.PENDING, StageStatus.RUNNING, StageStatus.FAILED):
                if not force and not self._stage_supports_row_level_resume(stage_name):
                    self._assert_no_partial_outputs(stage_name, expected_outputs)
                start_index = idx
                break

            raise OrchestrationError(f"Unknown stage status in manifest: {record.status}")

        if start_index is None:
            logger.info(
                "All context stages already completed for model=%s branch=%s gen=%d",
                model_name,
                branch,
                generation,
            )
            return

        for stage_name in CONTEXT_STAGES[start_index:]:
            self.run_stage(
                stage_name,
                model_name=model_name,
                branch=branch,
                generation=generation,
                seed=seed_value,
                force=force,
            )

    def _execute_stage(self, stage_name: StageName, context: StageContext) -> StageExecutionResult:
        if stage_name in self._stage_executors:
            return self._stage_executors[stage_name](context)

        if stage_name == "data_prep":
            _ = _validate_readable_artifact(context.artifacts["base_train"], must_be_non_empty=True)
            _ = _validate_readable_artifact(context.artifacts["anchor_pool"], must_be_non_empty=True)
            _ = _validate_readable_artifact(context.artifacts["heldout_test"], must_be_non_empty=True)
            _ = _validate_readable_artifact(context.artifacts["split_metadata"], must_be_non_empty=False)
            return StageExecutionResult(row_count=None)

        if stage_name == "generation":
            if context.model_name is None or context.branch is None or context.generation is None:
                raise OrchestrationError("Missing context fields for generation stage")
            heldout = self._get_heldout()
            out_df = run_generation(
                client=self._get_gen_client(),
                examples_df=heldout,
                model_name=context.model_name,
                branch=context.branch,
                generation=context.generation,
                run_id=self.ctx.run_id,
                prompt_version=self.cfg.runtime.generation_prompt_version,
                temperature=self.cfg.sampling.temperature,
                top_p=self.cfg.sampling.top_p,
                max_tokens=self.cfg.sampling.max_tokens,
                out_path=context.artifacts["model_outputs"],
                partial_path=context.artifacts["generation_partial"],
                failures_path=context.artifacts["generation_failures"],
                metadata_path=context.artifacts["generation_metadata"],
                partial_save_every_n=self.cfg.runtime.partial_save_every_n,
                max_row_failures=self.cfg.runtime.max_row_failures,
                continue_on_row_error=self.cfg.runtime.continue_on_row_error,
            )
            return StageExecutionResult(row_count=int(len(out_df)))

        if stage_name == "answer_extraction":
            outputs_df = pd.read_parquet(context.artifacts["model_outputs"])
            extracted_rows: list[dict[str, Any]] = []
            for rec in outputs_df.to_dict(orient="records"):
                extraction = extract_final_answer_result(str(rec.get("raw_response", "")))
                row = dict(rec)
                row["parsed_final_answer"] = extraction.extracted_answer
                row["normalized_predicted"] = extraction.normalized_answer
                row["pred_parse_success"] = extraction.parse_success
                row["pred_parse_strategy"] = extraction.parse_strategy
                row["pred_parse_failure_reason"] = extraction.parse_failure_reason
                extracted_rows.append(row)
            out_df = pd.DataFrame(extracted_rows)
            out_df.to_parquet(context.artifacts["answer_extraction"], index=False)
            return StageExecutionResult(row_count=int(len(out_df)))

        if stage_name == "accuracy":
            extracted_df = pd.read_parquet(context.artifacts["answer_extraction"])
            heldout = self._get_heldout()
            out_df = evaluate_accuracy(
                outputs_df=extracted_df,
                gold_df=heldout,
                out_path=context.artifacts["accuracy_table"],
            )
            return StageExecutionResult(row_count=int(len(out_df)))

        if stage_name == "judge":
            outputs_df = pd.read_parquet(context.artifacts["model_outputs"])
            heldout = self._get_heldout()
            judge_df = run_judging(
                client=self._get_judge_client(),
                generations_df=outputs_df,
                questions_df=heldout,
                judge_provider=self.cfg.judge.provider,
                judge_model=self.cfg.judge.model_name,
                rubric_prompt=self._get_judge_prompt(),
                out_path=context.artifacts["judge_outputs"],
                request_delay_sec=self.cfg.judge.request_delay_sec,
                partial_path=context.artifacts["judge_partial"],
                failures_path=context.artifacts["judge_failures"],
                metadata_path=context.artifacts["judge_metadata"],
                partial_save_every_n=self.cfg.runtime.partial_save_every_n,
                max_row_failures=self.cfg.runtime.max_row_failures,
                continue_on_row_error=self.cfg.runtime.continue_on_row_error,
            )

            accuracy_df = pd.read_parquet(context.artifacts["accuracy_table"])
            eval_df = accuracy_df.merge(
                judge_df[["example_id", "overall_pedagogical_score", "is_silent_error"]],
                on="example_id",
                how="left",
                validate="one_to_one",
            )
            missing_judge_mask = eval_df["overall_pedagogical_score"].isna() | eval_df["is_silent_error"].isna()
            if missing_judge_mask.any():
                missing_ids = eval_df.loc[missing_judge_mask, "example_id"].astype(str).head(5).tolist()
                missing_count = int(missing_judge_mask.sum())
                if (not self.cfg.runtime.continue_on_row_error) or (
                    missing_count > self.cfg.runtime.max_row_failures
                ):
                    raise OrchestrationError(
                        "Judge stage produced incomplete evaluation merge beyond allowed threshold. "
                        f"missing_count={missing_count}, max_row_failures={self.cfg.runtime.max_row_failures}, "
                        f"sample_example_ids={missing_ids}"
                    )
                logger.warning(
                    "judge_eval_merge_dropping_failed_rows missing_count=%d sample_example_ids=%s",
                    missing_count,
                    missing_ids,
                )
                eval_df = eval_df.loc[~missing_judge_mask].copy()
                if eval_df.empty:
                    raise OrchestrationError(
                        "Judge stage dropped all rows due to missing judge outputs after fault-tolerant processing."
                    )
            eval_df.to_parquet(context.artifacts["eval_merged"], index=False)
            return StageExecutionResult(row_count=int(len(eval_df)))

        if stage_name == "synthetic_build":
            extracted_df = pd.read_parquet(context.artifacts["answer_extraction"])
            heldout = self._get_heldout()[["example_id", "question"]].rename(columns={"question": "question_gold"})
            try:
                merged = extracted_df.merge(heldout, on="example_id", how="left", validate="one_to_one")
            except pd.errors.MergeError as exc:
                dup_extracted = (
                    extracted_df.loc[extracted_df["example_id"].duplicated(), "example_id"]
                    .astype(str)
                    .head(5)
                    .tolist()
                )
                dup_heldout = (
                    heldout.loc[heldout["example_id"].duplicated(), "example_id"]
                    .astype(str)
                    .head(5)
                    .tolist()
                )
                raise OrchestrationError(
                    "Synthetic build merge cardinality violation on example_id (expected one_to_one). "
                    f"sample_duplicate_extracted_ids={dup_extracted}, "
                    f"sample_duplicate_heldout_ids={dup_heldout}"
                ) from exc
            missing_q_mask = merged["question_gold"].isna()
            if missing_q_mask.any():
                missing_ids = merged.loc[missing_q_mask, "example_id"].astype(str).head(5).tolist()
                raise OrchestrationError(
                    "Synthetic build merge missing heldout question text. "
                    f"missing_count={int(missing_q_mask.sum())}, sample_example_ids={missing_ids}"
                )
            synthetic_df = pd.DataFrame(
                {
                    "example_id": merged["example_id"],
                    "question": merged["question_gold"],
                    "answer_for_training": merged["raw_response"],
                    "source": "synthetic",
                }
            )
            synthetic_df.to_parquet(context.artifacts["synthetic_base"], index=False)
            return StageExecutionResult(row_count=int(len(synthetic_df)))

        if stage_name == "anchoring":
            if context.model_name is None or context.branch is None or context.generation is None:
                raise OrchestrationError("Missing context fields for anchoring stage")

            synthetic_base = pd.read_parquet(context.artifacts["synthetic_base"])
            accuracy_df = pd.read_parquet(context.artifacts["accuracy_table"])
            judge_df = pd.read_parquet(context.artifacts["judge_outputs"])

            branch_cfg = next((b for b in self.cfg.experiment.branches if b.name == context.branch), None)
            if branch_cfg is None:
                raise OrchestrationError(f"Unknown branch in config: {context.branch}")

            branch_type = self._resolve_branch_type(branch_cfg)

            if branch_type == "pvf_medium":
                pvf_policy = PVFPolicy(
                    threshold_score=int(getattr(branch_cfg, "pvf_threshold_score", 5)),
                    min_keep_ratio=float(getattr(branch_cfg, "pvf_min_keep_ratio", 0.0)),
                )
                pvf_result = apply_pedagogical_verification_filter(
                    synthetic_df=synthetic_base,
                    accuracy_df=accuracy_df,
                    judge_df=judge_df,
                    model_name=context.model_name,
                    branch=context.branch,
                    generation=context.generation,
                    seed=context.seed,
                    policy=pvf_policy,
                    allow_partial_inputs=bool(self.cfg.runtime.continue_on_row_error),
                )
                save_pvf_artifacts(
                    result=pvf_result,
                    filtered_path=context.artifacts["pvf_filtered_training"],
                    rejected_path=context.artifacts["pvf_rejected_examples"],
                    report_path=context.artifacts["pvf_report"],
                )
                pvf_result.filtered_training_df.to_parquet(context.artifacts["synthetic_train_next"], index=False)
                pvf_meta = {
                    "method": "pvf_medium",
                    "model_name": context.model_name,
                    "branch": context.branch,
                    "generation": context.generation,
                    "seed": context.seed,
                    "threshold_score": pvf_policy.threshold_score,
                    "min_keep_ratio": pvf_policy.min_keep_ratio,
                    "total_candidates": pvf_result.report.total_candidates,
                    "kept_count": pvf_result.report.kept_count,
                    "rejected_count": pvf_result.report.rejected_count,
                    "keep_rate": pvf_result.report.keep_rate,
                }
                context.artifacts["anchor_metadata"].write_text(
                    json.dumps(pvf_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                context.artifacts["used_anchor_ids"].write_text("[]", encoding="utf-8")
                # Keep diagnostics contract explicit even for PVF branch.
                if not context.artifacts["anchor_quality_diagnostics"].exists():
                    pd.DataFrame(columns=["pairing_kind"]).to_parquet(
                        context.artifacts["anchor_quality_diagnostics"], index=False
                    )
                return StageExecutionResult(row_count=int(len(pvf_result.filtered_training_df)))

            anchor_pool = self._get_anchor_pool()
            base_train = pd.read_parquet(context.artifacts["base_train"])
            heldout_test = self._get_heldout()
            previously_used_anchor_ids = self._load_previously_used_anchor_ids(
                model_name=context.model_name,
                branch=context.branch,
                generation=context.generation,
            )
            policy = AnchorPolicy(
                anchor_ratio=branch_cfg.anchor_ratio,
                allow_reuse=False,
                mixing_mode=branch_cfg.mixing_mode,
            )
            selection_context = AnchorSelectionContext(
                model_name=context.model_name,
                branch=context.branch,
                generation=context.generation,
                seed=context.seed,
            )
            anchoring_result = select_human_anchors(
                anchor_pool_df=anchor_pool,
                synthetic_df=synthetic_base,
                base_train_df=base_train,
                heldout_test_df=heldout_test,
                previously_used_anchor_ids=previously_used_anchor_ids,
                policy=policy,
                context=selection_context,
            )
            save_anchoring_artifacts(
                result=anchoring_result,
                mixed_dataset_path=context.artifacts["synthetic_train_next"],
                metadata_path=context.artifacts["anchor_metadata"],
                used_anchor_ids_path=context.artifacts["used_anchor_ids"],
                diagnostics_path=context.artifacts["anchor_quality_diagnostics"],
            )
            return StageExecutionResult(row_count=int(len(anchoring_result.mixed_training_df)))

        if stage_name == "aggregate":
            eval_paths = list(self.ctx.run_dir.glob("**/eval_merged.parquet"))
            if not eval_paths:
                raise OrchestrationError("No eval_merged artifacts found for aggregate stage")
            frames = [pd.read_parquet(p) for p in eval_paths]
            combined = pd.concat(frames, ignore_index=True)
            required_cols = {"model_name", "branch", "generation", "example_id"}
            missing = required_cols.difference(combined.columns)
            if missing:
                raise OrchestrationError(f"Aggregate input missing required columns: {sorted(missing)}")
            dup_mask = combined.duplicated(subset=["model_name", "branch", "generation", "example_id"], keep=False)
            if dup_mask.any():
                sample = (
                    combined.loc[dup_mask, ["model_name", "branch", "generation", "example_id"]]
                    .astype(str)
                    .head(5)
                    .to_dict(orient="records")
                )
                raise OrchestrationError(
                    "Aggregate stage detected duplicate eval keys "
                    "(model_name, branch, generation, example_id). "
                    f"duplicate_count={int(dup_mask.sum())}, sample={sample}"
                )
            save_tabular(
                combined,
                self.ctx.run_dir / "all_eval_merged",
                save_csv=self.cfg.runtime.save_csv,
                save_parquet=self.cfg.runtime.save_parquet,
            )
            _ = aggregate_metrics(combined, self.ctx.run_dir / "tables" / "metrics_by_generation.csv")
            return StageExecutionResult(row_count=int(len(combined)))

        if stage_name == "plotting":
            metrics_path = self.ctx.run_dir / "tables" / "metrics_by_generation.csv"
            agg = pd.read_csv(metrics_path)
            if agg.empty:
                raise OrchestrationError("Plotting stage received empty metrics table")
            plot_metric_by_generation(agg, "accuracy", self.ctx.run_dir / "figures" / "accuracy_vs_generation.png")
            plot_metric_by_generation(
                agg,
                "pedagogical_score_mean",
                self.ctx.run_dir / "figures" / "pedagogical_vs_generation.png",
            )
            plot_metric_by_generation(
                agg,
                "silent_error_rate",
                self.ctx.run_dir / "figures" / "silent_error_vs_generation.png",
            )
            return StageExecutionResult(row_count=int(len(agg)))

        raise OrchestrationError(f"No executor for stage: {stage_name}")

    def run_full(self) -> None:
        self.save_run_metadata()
        self.run_stage("data_prep")

        for model in self.cfg.models.local_models:
            for branch in self.cfg.experiment.branches:
                for gen in range(1, self.cfg.experiment.generations + 1):
                    for stage_name in CONTEXT_STAGES:
                        self.run_stage(
                            stage_name,
                            model_name=model.name,
                            branch=branch.name,
                            generation=gen,
                            seed=self.cfg.project.seed,
                            force=self.cfg.runtime.force_recompute,
                        )
                    logger.info("Finished model=%s branch=%s gen=%d", model.name, branch.name, gen)

        self.run_stage("aggregate", force=self.cfg.runtime.force_recompute)
        self.run_stage("plotting", force=self.cfg.runtime.force_recompute)
