from __future__ import annotations

import json
from pathlib import Path

from didactic_collapse.orchestration.dbr_confirmatory import _is_run_manifest_completed


def _write_run_manifest(run_dir: Path) -> None:
    payload = {
        "stages": {
            "data_prep": {"status": "completed"},
            "aggregate": {"status": "completed"},
            "plotting": {"status": "completed"},
        }
    }
    (run_dir / "run_stage_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_context_manifest(context_dir: Path, *, judge_status: str = "completed") -> None:
    payload = {
        "stages": {
            "generation": {"status": "completed"},
            "answer_extraction": {"status": "completed"},
            "accuracy": {"status": "completed"},
            "judge": {"status": judge_status},
            "synthetic_build": {"status": "completed"},
            "anchoring": {"status": "completed"},
        }
    }
    (context_dir / "stage_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def test_run_manifest_not_completed_when_any_context_stage_incomplete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ctx_ok = run_dir / "qwen2.5_0.5b" / "pure_recycling" / "gen_0"
    ctx_bad = run_dir / "qwen2.5_0.5b" / "soft_pvf_noisy_keep" / "gen_0"
    ctx_ok.mkdir(parents=True, exist_ok=True)
    ctx_bad.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(run_dir)
    _write_context_manifest(ctx_ok, judge_status="completed")
    _write_context_manifest(ctx_bad, judge_status="running")

    assert _is_run_manifest_completed(run_dir) is False


def test_run_manifest_completed_when_run_and_context_stages_completed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ctx_a = run_dir / "qwen2.5_0.5b" / "pure_recycling" / "gen_0"
    ctx_b = run_dir / "qwen2.5_0.5b" / "dbr_medium" / "gen_1"
    ctx_a.mkdir(parents=True, exist_ok=True)
    ctx_b.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(run_dir)
    _write_context_manifest(ctx_a, judge_status="completed")
    _write_context_manifest(ctx_b, judge_status="completed")

    assert _is_run_manifest_completed(run_dir) is True

