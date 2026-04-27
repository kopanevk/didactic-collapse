from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.training_feasibility import (
    StubTrainStageBackend,
    TrainStageRequest,
    TrainingFeasibilityError,
    _build_backend,
    _build_target_model_name,
    _ensure_mode,
)


def _mk_test_dir(prefix: str) -> Path:
    base = Path("outputs") / ".tmp" / "unit_training_feasibility"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{prefix}_{uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _base_cfg_dict(root: Path) -> dict:
    return {
        "project": {"name": "dc", "seed": 7, "run_tag": "tf"},
        "paths": {"data_root": str(root / "data"), "output_root": str(root / "out"), "prompt_dir": "configs/prompts"},
        "models": {"local_models": [{"name": "qwen2.5:0.5b", "role": "subject"}]},
        "judge": {
            "provider": "cerebras",
            "model_name": "llama-3.1-8b",
            "base_url": "https://api.cerebras.ai/v1",
            "api_key_env": "CEREBRAS_API_KEY",
        },
        "sampling": {},
        "experiment": {
            "mode": "training_recycling_feasibility",
            "generations": 2,
            "branches": [
                {"name": "pure_recycling", "anchor_ratio": 0.0},
                {"name": "anchor_20_append", "anchor_ratio": 0.2},
            ],
        },
        "dataset": {"base_train_size": 10, "anchor_pool_size": 10, "heldout_test_size": 10},
        "runtime": {},
        "training": {
            "backend": "command",
            "command_template": "echo ok",
            "allow_stub_for_smoke": False,
        },
    }


def test_training_mode_is_parsed() -> None:
    cfg = AppConfig.model_validate(
        {
            "project": {"name": "dc"},
            "paths": {"data_root": "data", "output_root": "outputs", "prompt_dir": "configs/prompts"},
            "models": {"local_models": [{"name": "qwen2.5:0.5b"}]},
            "judge": {
                "provider": "cerebras",
                "model_name": "llama-3.1-8b",
                "base_url": "https://api.cerebras.ai/v1",
                "api_key_env": "CEREBRAS_API_KEY",
            },
            "sampling": {},
            "experiment": {"mode": "training_recycling_feasibility", "generations": 2, "branches": [{"name": "pure_recycling", "anchor_ratio": 0.0}]},
            "dataset": {"base_train_size": 10, "anchor_pool_size": 10, "heldout_test_size": 10},
            "runtime": {},
        }
    )
    assert cfg.experiment.mode == "training_recycling_feasibility"


def test_ensure_mode_rejects_inference_mode() -> None:
    raw = _base_cfg_dict(_mk_test_dir("mode_reject"))
    raw["experiment"]["mode"] = "inference_recycling_only"
    cfg = AppConfig.model_validate(raw)
    with pytest.raises(TrainingFeasibilityError, match="not in train feasibility mode"):
        _ensure_mode(cfg)


def test_build_backend_requires_command_template() -> None:
    raw = _base_cfg_dict(_mk_test_dir("backend_template"))
    raw["training"]["command_template"] = "   "
    cfg = AppConfig.model_validate(raw)
    with pytest.raises(TrainingFeasibilityError, match="requires non-empty"):
        _build_backend(cfg)


def test_stub_backend_writes_contract_file() -> None:
    root = _mk_test_dir("stub_contract")
    req = TrainStageRequest(
        run_dir=root / "run",
        branch="anchor_20_append",
        seed=7,
        generation_from=0,
        generation_to=1,
        base_model_name="qwen2.5:0.5b",
        target_model_name="qwen2.5:0.5b__anchor_20_append__ft_g1_s7",
        training_data_path=root / "train.parquet",
        output_dir=root / "trainer",
    )
    backend = StubTrainStageBackend(result_filename="training_result.json")
    result = backend.run(req)
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert result.is_stub is True
    assert payload["trained_model_name"] == req.target_model_name


def test_target_model_name_template_has_stable_fields() -> None:
    model_name = _build_target_model_name(
        template="{base_model}__{branch}__ft_g{generation_to}_s{seed}",
        base_model_name="qwen2.5:0.5b",
        branch="anchor_20_append",
        seed=54,
        generation_from=0,
        generation_to=1,
    )
    assert model_name == "qwen2.5:0.5b__anchor_20_append__ft_g1_s54"
