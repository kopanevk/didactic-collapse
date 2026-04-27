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


class SamplingConfig(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512


class BranchConfig(BaseModel):
    name: str
    anchor_ratio: float = Field(ge=0.0, le=1.0)
    mixing_mode: Literal["append", "replace"] = "append"
    branch_type: Literal["pure_recycling", "human_anchoring", "pvf_medium"] | None = None
    pvf_threshold_score: int = Field(default=5, ge=0, le=8)
    pvf_min_keep_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


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
    sampling: SamplingConfig
    experiment: ExperimentConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
