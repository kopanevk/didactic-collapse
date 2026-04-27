from __future__ import annotations

from didactic_collapse.config.settings import AppConfig


def test_branch_mixing_mode_defaults_to_append() -> None:
    cfg = AppConfig.model_validate(
        {
            "project": {"name": "dc", "seed": 1, "run_tag": "t"},
            "paths": {"data_root": "data", "output_root": "outputs", "prompt_dir": "configs/prompts"},
            "models": {"local_models": [{"name": "qwen2.5:0.5b"}]},
            "judge": {
                "provider": "cerebras",
                "model_name": "llama-3.1-8b",
                "base_url": "https://api.cerebras.ai/v1",
                "api_key_env": "CEREBRAS_API_KEY",
            },
            "sampling": {},
            "experiment": {"generations": 2, "branches": [{"name": "anchor_10", "anchor_ratio": 0.1}]},
            "dataset": {"base_train_size": 10, "anchor_pool_size": 10, "heldout_test_size": 10},
            "runtime": {},
        }
    )
    assert cfg.experiment.branches[0].mixing_mode == "append"


def test_branch_mixing_mode_accepts_replace() -> None:
    cfg = AppConfig.model_validate(
        {
            "project": {"name": "dc", "seed": 1, "run_tag": "t"},
            "paths": {"data_root": "data", "output_root": "outputs", "prompt_dir": "configs/prompts"},
            "models": {"local_models": [{"name": "qwen2.5:0.5b"}]},
            "judge": {
                "provider": "cerebras",
                "model_name": "llama-3.1-8b",
                "base_url": "https://api.cerebras.ai/v1",
                "api_key_env": "CEREBRAS_API_KEY",
            },
            "sampling": {},
            "experiment": {
                "generations": 2,
                "branches": [{"name": "anchor_10_replace", "anchor_ratio": 0.1, "mixing_mode": "replace"}],
            },
            "dataset": {"base_train_size": 10, "anchor_pool_size": 10, "heldout_test_size": 10},
            "runtime": {},
        }
    )
    assert cfg.experiment.branches[0].mixing_mode == "replace"
