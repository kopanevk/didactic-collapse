from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pandas as pd

from didactic_collapse.analysis.anchoring_ablation import export_anchoring_ablation_analysis


def _mk_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "config": {
            "project": {"seed": 123},
            "paths": {"data_root": str(tmp_path / "data")},
            "experiment": {
                "branches": [
                    {"name": "pure_recycling", "anchor_ratio": 0.0, "mixing_mode": "append"},
                    {"name": "anchor_10_replace", "anchor_ratio": 0.1, "mixing_mode": "replace"},
                    {"name": "anchor_10_append", "anchor_ratio": 0.1, "mixing_mode": "append"},
                    {"name": "anchor_20_append", "anchor_ratio": 0.2, "mixing_mode": "append"},
                ]
            },
        }
    }
    (run_dir / "run_config.snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    return run_dir


def _write_context(
    run_dir: Path,
    *,
    branch: str,
    generation: int,
    accuracy: float,
    pedagogical: float,
    silent: float,
) -> None:
    step = run_dir / "qwen2.5_0.5b" / branch / f"gen_{generation}"
    step.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(3):
        rows.append(
            {
                "model_name": "qwen2.5:0.5b",
                "branch": branch,
                "generation": generation,
                "example_id": f"{branch}_{generation}_{i}",
                "pred_parse_success": True,
            }
        )
    pd.DataFrame(rows).to_parquet(step / "accuracy_table.parquet", index=False)

    eval_rows = []
    for i in range(3):
        eval_rows.append(
            {
                "model_name": "qwen2.5:0.5b",
                "branch": branch,
                "generation": generation,
                "example_id": f"{branch}_{generation}_{i}",
                "is_correct": i == 0 if accuracy < 0.5 else i != 2,
                "overall_pedagogical_score": pedagogical,
                "is_silent_error": silent > 0.2,
            }
        )
    pd.DataFrame(eval_rows).to_parquet(step / "eval_merged.parquet", index=False)

    pd.DataFrame(
        [
            {
                "model_name": "qwen2.5:0.5b",
                "branch": branch,
                "generation": generation,
                "anchor_example_id": f"a_{branch}_{generation}",
                "synthetic_example_id": f"{branch}_{generation}_0",
                "pairing_kind": "replaced_synthetic" if "replace" in branch else "sampled_synthetic_reference",
                "mixing_mode": "replace" if "replace" in branch else "append",
                "anchor_question_length": 20,
                "synthetic_question_length": 25,
                "anchor_answer_length": 5,
                "synthetic_answer_length": 50,
                "delta_answer_length_anchor_minus_synth": -45,
                "synthetic_has_final_answer_marker": True,
            }
        ]
    ).to_parquet(step / "anchor_quality_diagnostics.parquet", index=False)


def test_export_anchoring_ablation_analysis() -> None:
    base = Path("outputs/.tmp") / f"anchoring_ablation_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = _mk_run_dir(base)
        (base / "data" / "splits").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"example_id": "x", "question": "q", "answer_gold": "1"}]
        ).to_parquet(base / "data" / "splits" / "heldout_test.parquet", index=False)

        contexts = [
            ("pure_recycling", 0, 0.4, 5.2, 0.3),
            ("pure_recycling", 1, 0.5, 5.1, 0.2),
            ("anchor_10_replace", 0, 0.3, 4.8, 0.4),
            ("anchor_10_replace", 1, 0.4, 4.7, 0.3),
            ("anchor_10_append", 0, 0.45, 5.0, 0.3),
            ("anchor_10_append", 1, 0.5, 5.1, 0.2),
            ("anchor_20_append", 0, 0.35, 4.9, 0.35),
            ("anchor_20_append", 1, 0.45, 5.0, 0.25),
        ]
        for branch, gen, acc, ped, silent in contexts:
            _write_context(run_dir, branch=branch, generation=gen, accuracy=acc, pedagogical=ped, silent=silent)

        summary = pd.DataFrame(
            [
                {
                    "model_name": "qwen2.5:0.5b",
                    "branch": branch,
                    "generation": gen,
                    "sample_count": 3,
                    "accuracy_mean": acc,
                    "pedagogical_score_mean": ped,
                    "silent_error_rate": silent,
                }
                for branch, gen, acc, ped, silent in contexts
            ]
        )
        tables = run_dir / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        summary.to_csv(tables / "first_experiment_summary.csv", index=False)

        artifacts = export_anchoring_ablation_analysis(run_dir=run_dir, out_dir=run_dir / "tables")
        run_level = pd.read_csv(artifacts.run_level_csv)
        mode_deltas = pd.read_csv(artifacts.mode_deltas_csv)
        ratio_deltas = pd.read_csv(artifacts.ratio_deltas_csv)
        quality = pd.read_csv(artifacts.anchor_quality_summary_csv)

        assert not run_level.empty
        assert "mixing_mode" in run_level.columns
        assert "anchor_ratio" in run_level.columns
        assert (run_level["evaluation_mode"] == "inference_recycling_only").all()
        assert not mode_deltas.empty
        assert not ratio_deltas.empty
        assert not quality.empty
    finally:
        shutil.rmtree(base, ignore_errors=True)
