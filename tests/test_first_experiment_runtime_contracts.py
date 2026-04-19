from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.config.settings import AppConfig
from didactic_collapse.orchestration.first_experiment import (
    _preflight_real_judge_auth,
    verify_first_experiment_artifacts,
)


def _cfg_for_cerebras() -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"name": "dc", "seed": 42, "run_tag": "first"},
            "paths": {"data_root": "data", "output_root": "outputs", "prompt_dir": "configs/prompts"},
            "models": {"local_models": [{"name": "qwen2.5:0.5b", "role": "subject"}]},
            "judge": {
                "provider": "cerebras",
                "model_name": "llama-3.1-8b",
                "base_url": "https://api.cerebras.ai/v1",
                "api_key_env": "CEREBRAS_API_KEY",
                "timeout_sec": 60,
                "max_retries": 3,
            },
            "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 128},
            "experiment": {
                "generations": 2,
                "branches": [
                    {"name": "pure_recycling", "anchor_ratio": 0.0},
                    {"name": "anchor_10", "anchor_ratio": 0.1},
                ],
            },
            "dataset": {
                "source": "gsm8k",
                "base_train_size": 100,
                "anchor_pool_size": 200,
                "heldout_test_size": 100,
            },
            "runtime": {"force_recompute": False, "save_parquet": True, "save_csv": True},
        }
    )


def _make_run_dir() -> Path:
    run_dir = Path("outputs/.tmp") / f"first_contract_{uuid.uuid4().hex}"
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def test_preflight_real_judge_auth_missing_cerebras_key_fails_fast() -> None:
    cfg = _cfg_for_cerebras()
    old = os.environ.pop("CEREBRAS_API_KEY", None)
    try:
        with pytest.raises(Exception, match="Missing Cerebras API key"):
            _preflight_real_judge_auth(cfg)
    finally:
        if old is not None:
            os.environ["CEREBRAS_API_KEY"] = old


def test_verify_first_experiment_artifacts_contract() -> None:
    run_dir = _make_run_dir()
    try:
        summary = pd.DataFrame(
            [
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "pure_recycling",
                    "generation": 0,
                    "sample_count": 10,
                    "accuracy_mean": 0.5,
                    "pedagogical_score_mean": 4.0,
                    "silent_error_rate": 0.1,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "anchor_10",
                    "generation": 1,
                    "sample_count": 10,
                    "accuracy_mean": 0.55,
                    "pedagogical_score_mean": 4.2,
                    "silent_error_rate": 0.08,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "pure_recycling",
                    "generation": 1,
                    "sample_count": 10,
                    "accuracy_mean": 0.45,
                    "pedagogical_score_mean": 3.9,
                    "silent_error_rate": 0.12,
                },
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": "anchor_10",
                    "generation": 0,
                    "sample_count": 10,
                    "accuracy_mean": 0.53,
                    "pedagogical_score_mean": 4.1,
                    "silent_error_rate": 0.09,
                },
            ]
        )
        summary_csv = run_dir / "tables" / "first_experiment_summary.csv"
        summary_pq = run_dir / "tables" / "first_experiment_summary.parquet"
        summary.to_csv(summary_csv, index=False)
        summary.to_parquet(summary_pq, index=False)

        qualitative = pd.DataFrame(columns=["example_id", "generation", "branch"])
        qual_csv = run_dir / "tables" / "qualitative_silent_error_candidates.csv"
        qual_pq = run_dir / "tables" / "qualitative_silent_error_candidates.parquet"
        qualitative.to_csv(qual_csv, index=False)
        qualitative.to_parquet(qual_pq, index=False)
        qual_meta = run_dir / "tables" / "qualitative_silent_error_candidates.meta.json"
        qual_meta.write_text(json.dumps({"is_empty": True, "row_count": 0}), encoding="utf-8")

        pd.DataFrame([{"x": 1}]).to_parquet(run_dir / "all_eval_merged.parquet", index=False)
        pd.DataFrame([{"x": 1}]).to_csv(run_dir / "tables" / "metrics_by_generation.csv", index=False)
        for plot_name in (
            "accuracy_vs_generation.png",
            "pedagogical_vs_generation.png",
            "silent_error_vs_generation.png",
        ):
            (run_dir / "figures" / plot_name).write_bytes(b"png")

        verify_first_experiment_artifacts(
            run_dir=run_dir,
            summary_csv=summary_csv,
            summary_parquet=summary_pq,
            qualitative_csv=qual_csv,
            qualitative_parquet=qual_pq,
            qualitative_meta_path=qual_meta,
            branches=["pure_recycling", "anchor_10"],
            generations=[0, 1],
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
