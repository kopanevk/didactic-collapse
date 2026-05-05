from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str
    seed: int = 42
    run_tag: str = "default"


class PathsConfig(BaseModel):
    data_root: Path
    output_root: Path
    prompt_dir: Path


class LocalModelSpec(BaseModel):
    name: str
    role: str = "subject"


class ModelsConfig(BaseModel):
    local_models: list[LocalModelSpec]


class JudgeConfig(BaseModel):
    provider: str
    model_name: str
    base_url: str
    api_key_env: str
    timeout_sec: int = 60
    max_retries: int = 3
    request_delay_sec: float = 0.25
    max_completion_tokens: int = 180
    comment_max_chars: int = 120
    cache_enabled: bool = True
    cache_path: Path = Path("outputs/cache/judge_cache.sqlite")
    cerebras_min_request_interval_sec: float = 0.8
    cerebras_max_retry_after_sec: float = 60.0
    cerebras_max_429_retries: int = 6
    cerebras_jitter_sec: float = 0.75


class RepairConfig(BaseModel):
    provider: str = "cerebras"
    model_name: str | None = None
    base_url: str | None = None
    api_key_env: str = "CEREBRAS_API_KEY"
    timeout_sec: int = 120
    max_retries: int = 3
    request_delay_sec: float = 0.5
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 700


class SamplingConfig(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512


class BranchConfig(BaseModel):
    name: str
    anchor_ratio: float = Field(ge=0.0, le=1.0)
    mixing_mode: Literal["append", "replace"] = "append"
    branch_type: Literal[
        "pure_recycling",
        "human_anchoring",
        "pvf_medium",
        "soft_pvf_medium",
        "dbr_medium",
        "pvr_repair_medium",
        "pair_lite_medium",
        "csr_medium",
    ] | None = None
    soft_pvf_policy_name: Literal[
        "soft_pvf_medium",
        "soft_pvf_lenient",
        "soft_pvf_noisy_keep",
        "soft_pvf_silent_only",
    ] | None = None
    pvf_threshold_score: int = Field(default=5, ge=0, le=8)
    pvf_min_keep_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    soft_pvf_high_quality_threshold: int = Field(default=6, ge=0, le=8)
    soft_pvf_medium_quality_threshold: int = Field(default=4, ge=0, le=8)
    soft_pvf_weight_high: float = Field(default=1.0, ge=0.0, le=1.0)
    soft_pvf_weight_medium: float = Field(default=0.5, ge=0.0, le=1.0)
    soft_pvf_weight_low_correct: float = Field(default=0.25, ge=0.0, le=1.0)
    soft_pvf_weight_incorrect: float = Field(default=0.1, ge=0.0, le=1.0)
    soft_pvf_min_keep_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    pvr_threshold_score: int = Field(default=6, ge=0, le=8)
    pvr_min_keep_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    pair_lite_threshold_score: int = Field(default=6, ge=0, le=8)
    pair_lite_min_keep_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    dbr_target_size_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
    dbr_min_selection_rate: float = Field(default=0.80, ge=0.0, le=1.0)
    dbr_budget_parse_failure: float = Field(default=0.00, ge=0.0, le=1.0)
    dbr_budget_silent_error: float = Field(default=0.10, ge=0.0, le=1.0)
    dbr_budget_incorrect_answer: float = Field(default=0.30, ge=0.0, le=1.0)
    dbr_budget_low_reasoning: float = Field(default=0.25, ge=0.0, le=1.0)
    dbr_budget_low_structure: float = Field(default=0.30, ge=0.0, le=1.0)
    dbr_allow_parse_failure_fallback: bool = False
    csr_num_candidates: int = Field(default=3, ge=1, le=8)
    csr_candidate_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    csr_min_pair_quality_gap: float = Field(default=2.0, ge=0.0, le=8.0)
    csr_require_best_correct: bool = True
    csr_require_best_non_silent: bool = True
    csr_allow_worst_incorrect: bool = True
    csr_allow_worst_silent: bool = True
    csr_max_no_pair_rate_warn: float = Field(default=0.60, ge=0.0, le=1.0)


class ExperimentConfig(BaseModel):
    mode: Literal["inference_recycling_only", "training_recycling_feasibility"] = (
        "inference_recycling_only"
    )
    generations: int = 3
    branches: list[BranchConfig]


class DatasetConfig(BaseModel):
    source: str = "gsm8k"
    base_train_size: int
    anchor_pool_size: int
    heldout_test_size: int


class RuntimeConfig(BaseModel):
    force_recompute: bool = False
    save_parquet: bool = True
    save_csv: bool = True
    generation_prompt_version: str = "v1"
    partial_save_every_n: int = 10
    max_row_failures: int = 5
    continue_on_row_error: bool = True


class TrainingConfig(BaseModel):
    backend: Literal["command", "stub"] = "command"
    command_template: str | None = None
    command_timeout_sec: int = 3600
    output_model_name_template: str = "{base_model}__{branch}__ft_g{generation_to}_s{seed}"
    result_filename: str = "training_result.json"
    run_gen1_recycling_stages: bool = False
    allow_stub_for_smoke: bool = False
    max_train_rows: int | None = 200


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    models: ModelsConfig
    judge: JudgeConfig
    repair: RepairConfig = Field(default_factory=RepairConfig)
    sampling: SamplingConfig
    experiment: ExperimentConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
