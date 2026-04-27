from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.baseline_series import EVAL_MODE, build_qualitative_candidates
from didactic_collapse.analysis.compare_runs import build_first_experiment_run_metrics


@dataclass(frozen=True)
class AnchoringAblationArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    mode_deltas_csv: Path
    mode_deltas_parquet: Path
    ratio_deltas_csv: Path
    ratio_deltas_parquet: Path
    anchor_quality_pairs_csv: Path
    anchor_quality_pairs_parquet: Path
    anchor_quality_summary_csv: Path
    anchor_quality_summary_parquet: Path
    qualitative_csv: Path
    qualitative_parquet: Path


def _load_run_snapshot(run_dir: Path) -> dict:
    snapshot_path = run_dir / "run_config.snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run snapshot: {snapshot_path}")
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _extract_seed(snapshot: dict) -> int:
    return int(snapshot["config"]["project"]["seed"])


def _branch_specs_from_snapshot(snapshot: dict) -> dict[str, dict[str, object]]:
    raw = snapshot["config"]["experiment"]["branches"]
    specs: dict[str, dict[str, object]] = {}
    for item in raw:
        name = str(item["name"])
        specs[name] = {
            "anchor_ratio": float(item.get("anchor_ratio", 0.0)),
            "mixing_mode": str(item.get("mixing_mode", "append")),
        }
    return specs


def _load_anchor_quality_pairs(run_dir: Path) -> pd.DataFrame:
    paths = sorted(run_dir.glob("*/*/gen_*/anchor_quality_diagnostics.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    out["evaluation_mode"] = EVAL_MODE
    return out


def _load_eval_context(run_dir: Path) -> pd.DataFrame:
    paths = sorted(run_dir.glob("*/*/gen_*/eval_merged.parquet"))
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p)
        required = {"model_name", "branch", "generation", "example_id", "overall_pedagogical_score"}
        missing = required.difference(df.columns)
        if missing:
            continue
        frames.append(df[list(required)].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_branch_deltas(run_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, generation), group in run_level.groupby(["model_name", "generation"], as_index=False):
        pure = group[group["branch"] == "pure_recycling"]
        if pure.empty:
            continue
        pure_row = pure.iloc[0]
        for _, row in group[group["branch"] != "pure_recycling"].iterrows():
            rows.append(
                {
                    "model_name": model_name,
                    "generation": int(generation),
                    "anchor_branch": str(row["branch"]),
                    "anchor_ratio": float(row["anchor_ratio"]),
                    "mixing_mode": str(row["mixing_mode"]),
                    "delta_accuracy_mean": float(row["accuracy_mean"]) - float(pure_row["accuracy_mean"]),
                    "delta_pedagogical_score_mean": float(row["pedagogical_score_mean"])
                    - float(pure_row["pedagogical_score_mean"]),
                    "delta_silent_error_rate": float(row["silent_error_rate"]) - float(pure_row["silent_error_rate"]),
                    "delta_parse_failure_pred_rate": float(row["parse_failure_pred_rate"])
                    - float(pure_row["parse_failure_pred_rate"]),
                    "evaluation_mode": EVAL_MODE,
                }
            )
    return pd.DataFrame(rows).sort_values(["generation", "mixing_mode", "anchor_ratio"]).reset_index(drop=True)


def _build_mode_deltas(run_level: pd.DataFrame) -> pd.DataFrame:
    anchor_only = run_level[run_level["branch"] != "pure_recycling"].copy()
    rows: list[dict[str, object]] = []
    for (model_name, generation, anchor_ratio), group in anchor_only.groupby(
        ["model_name", "generation", "anchor_ratio"],
        as_index=False,
    ):
        append = group[group["mixing_mode"] == "append"]
        replace = group[group["mixing_mode"] == "replace"]
        if append.empty or replace.empty:
            continue
        a = append.iloc[0]
        r = replace.iloc[0]
        rows.append(
            {
                "model_name": model_name,
                "generation": int(generation),
                "anchor_ratio": float(anchor_ratio),
                "append_branch": str(a["branch"]),
                "replace_branch": str(r["branch"]),
                "delta_accuracy_append_minus_replace": float(a["accuracy_mean"]) - float(r["accuracy_mean"]),
                "delta_pedagogical_append_minus_replace": float(a["pedagogical_score_mean"])
                - float(r["pedagogical_score_mean"]),
                "delta_silent_error_append_minus_replace": float(a["silent_error_rate"])
                - float(r["silent_error_rate"]),
                "delta_parse_failure_append_minus_replace": float(a["parse_failure_pred_rate"])
                - float(r["parse_failure_pred_rate"]),
                "evaluation_mode": EVAL_MODE,
            }
        )
    return pd.DataFrame(rows).sort_values(["generation", "anchor_ratio"]).reset_index(drop=True)


def _build_ratio_deltas(run_level: pd.DataFrame) -> pd.DataFrame:
    anchor_only = run_level[run_level["branch"] != "pure_recycling"].copy()
    rows: list[dict[str, object]] = []
    for (model_name, generation, mixing_mode), group in anchor_only.groupby(
        ["model_name", "generation", "mixing_mode"],
        as_index=False,
    ):
        group = group.sort_values("anchor_ratio")
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                low = group.iloc[i]
                high = group.iloc[j]
                rows.append(
                    {
                        "model_name": model_name,
                        "generation": int(generation),
                        "mixing_mode": str(mixing_mode),
                        "ratio_high": float(high["anchor_ratio"]),
                        "ratio_low": float(low["anchor_ratio"]),
                        "delta_accuracy_high_minus_low": float(high["accuracy_mean"]) - float(low["accuracy_mean"]),
                        "delta_pedagogical_high_minus_low": float(high["pedagogical_score_mean"])
                        - float(low["pedagogical_score_mean"]),
                        "delta_silent_error_high_minus_low": float(high["silent_error_rate"])
                        - float(low["silent_error_rate"]),
                        "delta_parse_failure_high_minus_low": float(high["parse_failure_pred_rate"])
                        - float(low["parse_failure_pred_rate"]),
                        "high_branch": str(high["branch"]),
                        "low_branch": str(low["branch"]),
                        "evaluation_mode": EVAL_MODE,
                    }
                )
    return pd.DataFrame(rows).sort_values(["generation", "mixing_mode", "ratio_low", "ratio_high"]).reset_index(
        drop=True
    )


def export_anchoring_ablation_analysis(*, run_dir: Path, out_dir: Path) -> AnchoringAblationArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _load_run_snapshot(run_dir)
    seed = _extract_seed(snapshot)
    branch_specs = _branch_specs_from_snapshot(snapshot)

    run_level = build_first_experiment_run_metrics(run_dir)
    run_level["run_id"] = run_dir.name
    run_level["seed"] = seed
    run_level["evaluation_mode"] = EVAL_MODE
    run_level["anchor_ratio"] = run_level["branch"].map(lambda b: float(branch_specs.get(str(b), {}).get("anchor_ratio", 0.0)))
    run_level["mixing_mode"] = run_level["branch"].map(lambda b: str(branch_specs.get(str(b), {}).get("mixing_mode", "append")))
    run_level = run_level.sort_values(["model_name", "generation", "branch"]).reset_index(drop=True)

    branch_deltas = _build_branch_deltas(run_level)
    mode_deltas = _build_mode_deltas(run_level)
    ratio_deltas = _build_ratio_deltas(run_level)

    anchor_pairs = _load_anchor_quality_pairs(run_dir)
    if not anchor_pairs.empty:
        eval_ctx = _load_eval_context(run_dir)
        if not eval_ctx.empty:
            anchor_pairs = anchor_pairs.merge(
                eval_ctx.rename(
                    columns={
                        "example_id": "synthetic_example_id",
                        "overall_pedagogical_score": "synthetic_pedagogical_score",
                    }
                )[["model_name", "branch", "generation", "synthetic_example_id", "synthetic_pedagogical_score"]],
                on=["model_name", "branch", "generation", "synthetic_example_id"],
                how="left",
                validate="many_to_one",
            )
    anchor_summary = (
        anchor_pairs.groupby(["model_name", "branch", "generation", "mixing_mode", "pairing_kind"], as_index=False)
        .agg(
            pair_count=("anchor_example_id", "count"),
            anchor_question_len_mean=("anchor_question_length", "mean"),
            synthetic_question_len_mean=("synthetic_question_length", "mean"),
            anchor_answer_len_mean=("anchor_answer_length", "mean"),
            synthetic_answer_len_mean=("synthetic_answer_length", "mean"),
            delta_answer_len_mean=("delta_answer_length_anchor_minus_synth", "mean"),
            synthetic_final_answer_marker_rate=("synthetic_has_final_answer_marker", "mean"),
            synthetic_pedagogical_mean=("synthetic_pedagogical_score", "mean"),
        )
        if not anchor_pairs.empty
        else pd.DataFrame(
            columns=[
                "model_name",
                "branch",
                "generation",
                "mixing_mode",
                "pairing_kind",
                "pair_count",
                "anchor_question_len_mean",
                "synthetic_question_len_mean",
                "anchor_answer_len_mean",
                "synthetic_answer_len_mean",
                "delta_answer_len_mean",
                "synthetic_final_answer_marker_rate",
                "synthetic_pedagogical_mean",
            ]
        )
    )
    if not anchor_summary.empty:
        anchor_summary["evaluation_mode"] = EVAL_MODE

    try:
        qualitative = build_qualitative_candidates([run_dir], max_total=12)
    except ValueError:
        qualitative = pd.DataFrame(
            columns=[
                "category",
                "run_id",
                "seed",
                "model_name",
                "branch",
                "generation",
                "example_id",
                "question",
                "answer_gold",
                "is_correct",
                "overall_pedagogical_score",
                "is_silent_error",
                "raw_response",
                "evaluation_mode",
            ]
        )

    run_level_csv = out_dir / "anchoring_ablation_run_level.csv"
    run_level_parquet = out_dir / "anchoring_ablation_run_level.parquet"
    branch_deltas_csv = out_dir / "anchoring_ablation_branch_deltas_vs_pure.csv"
    branch_deltas_parquet = out_dir / "anchoring_ablation_branch_deltas_vs_pure.parquet"
    mode_deltas_csv = out_dir / "anchoring_ablation_mode_deltas.csv"
    mode_deltas_parquet = out_dir / "anchoring_ablation_mode_deltas.parquet"
    ratio_deltas_csv = out_dir / "anchoring_ablation_ratio_deltas.csv"
    ratio_deltas_parquet = out_dir / "anchoring_ablation_ratio_deltas.parquet"
    anchor_pairs_csv = out_dir / "anchor_quality_pairs.csv"
    anchor_pairs_parquet = out_dir / "anchor_quality_pairs.parquet"
    anchor_summary_csv = out_dir / "anchor_quality_summary.csv"
    anchor_summary_parquet = out_dir / "anchor_quality_summary.parquet"
    qualitative_csv = out_dir / "anchoring_ablation_qualitative_candidates.csv"
    qualitative_parquet = out_dir / "anchoring_ablation_qualitative_candidates.parquet"

    run_level.to_csv(run_level_csv, index=False)
    run_level.to_parquet(run_level_parquet, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    branch_deltas.to_parquet(branch_deltas_parquet, index=False)
    mode_deltas.to_csv(mode_deltas_csv, index=False)
    mode_deltas.to_parquet(mode_deltas_parquet, index=False)
    ratio_deltas.to_csv(ratio_deltas_csv, index=False)
    ratio_deltas.to_parquet(ratio_deltas_parquet, index=False)
    anchor_pairs.to_csv(anchor_pairs_csv, index=False)
    anchor_pairs.to_parquet(anchor_pairs_parquet, index=False)
    anchor_summary.to_csv(anchor_summary_csv, index=False)
    anchor_summary.to_parquet(anchor_summary_parquet, index=False)
    qualitative.to_csv(qualitative_csv, index=False)
    qualitative.to_parquet(qualitative_parquet, index=False)

    return AnchoringAblationArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        mode_deltas_csv=mode_deltas_csv,
        mode_deltas_parquet=mode_deltas_parquet,
        ratio_deltas_csv=ratio_deltas_csv,
        ratio_deltas_parquet=ratio_deltas_parquet,
        anchor_quality_pairs_csv=anchor_pairs_csv,
        anchor_quality_pairs_parquet=anchor_pairs_parquet,
        anchor_quality_summary_csv=anchor_summary_csv,
        anchor_quality_summary_parquet=anchor_summary_parquet,
        qualitative_csv=qualitative_csv,
        qualitative_parquet=qualitative_parquet,
    )
