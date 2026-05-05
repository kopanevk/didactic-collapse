from __future__ import annotations

import json
from pathlib import Path
import shutil

from didactic_collapse.analysis.artifact_integrity_audit import (
    AuditFinding,
    _analysis_source_checks,
    _collect_manifest_status,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def test_collect_manifest_status_allows_terminal_pending_stages() -> None:
    root = Path("test_workdirs") / "audit_terminal_pending_allowed"
    _reset_dir(root)
    run_dir = root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_config.snapshot.json",
        {"config": {"experiment": {"generations": 2}}},
    )
    _write_json(
        run_dir / "run_stage_manifest.json",
        {"stages": {"data_prep": {"status": "completed"}, "aggregate": {"status": "completed"}, "plotting": {"status": "completed"}}},
    )
    _write_json(
        run_dir / "m" / "b" / "gen_1" / "stage_manifest.json",
        {
            "stages": {
                "generation": {"status": "completed"},
                "answer_extraction": {"status": "completed"},
                "accuracy": {"status": "completed"},
                "judge": {"status": "completed"},
                "synthetic_build": {"status": "pending"},
                "anchoring": {"status": "pending"},
            }
        },
    )

    findings: list[AuditFinding] = []
    df = _collect_manifest_status(run_dir, findings)
    assert not [f for f in findings if f.message == "Context stage not completed"]
    context_pending = df[(df["scope"] == "context") & (df["status"] == "pending")]
    assert len(context_pending) == 2
    assert context_pending["expected_pending_terminal_stage"].fillna(False).all()


def test_collect_manifest_status_flags_non_terminal_pending_stages() -> None:
    root = Path("test_workdirs") / "audit_nonterminal_pending_flagged"
    _reset_dir(root)
    run_dir = root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "run_config.snapshot.json",
        {"config": {"experiment": {"generations": 3}}},
    )
    _write_json(
        run_dir / "run_stage_manifest.json",
        {"stages": {"data_prep": {"status": "completed"}, "aggregate": {"status": "completed"}, "plotting": {"status": "completed"}}},
    )
    _write_json(
        run_dir / "m" / "b" / "gen_0" / "stage_manifest.json",
        {"stages": {"synthetic_build": {"status": "pending"}}},
    )

    findings: list[AuditFinding] = []
    _ = _collect_manifest_status(run_dir, findings)
    assert any(f.message == "Context stage not completed" for f in findings)


def test_analysis_source_checks_uses_explicit_analysis_dirs() -> None:
    root = Path("test_workdirs") / "audit_analysis_dirs"
    _reset_dir(root)
    analysis_dir = root / "pvf_confirmatory" / "series_x"
    tables_dir = analysis_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        tables_dir / "pvf_confirmatory_metadata.json",
        {"run_dirs": ["outputs/runs/pvf_confirmatory_seed91_x", "outputs/runs/wrong_family_seed99_x"]},
    )

    findings: list[AuditFinding] = []
    checks = _analysis_source_checks(root, findings, analysis_dirs=[analysis_dir])
    mismatches = checks[checks["status"] == "mismatch"]
    assert not mismatches.empty
    assert any(f.category == "analysis_sources" for f in findings)
