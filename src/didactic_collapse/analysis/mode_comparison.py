from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from didactic_collapse.analysis.compare_runs import build_first_experiment_run_metrics

INFERENCE_MODE = "inference_recycling_only"
TRAINING_MODE = "training_recycling_feasibility"
_METRIC_COLS = [
    "accuracy_mean",
    "pedagogical_score_mean",
    "silent_error_rate",
    "parse_failure_pred_rate",
]


def _empty_generation_deltas() -> pd.DataFrame:
    cols = [
        "run_id",
        "seed",
        "evaluation_mode",
        "branch",
        "generation_start",
        "generation_end",
        "delta_generation",
    ] + [f"delta_{metric}" for metric in _METRIC_COLS]
    return pd.DataFrame(columns=cols)


def _empty_branch_deltas() -> pd.DataFrame:
    cols = [
        "run_id",
        "seed",
        "evaluation_mode",
        "generation",
        "anchor_branch",
        "comparison",
    ] + [f"delta_{metric}" for metric in _METRIC_COLS]
    return pd.DataFrame(columns=cols)


def _empty_mode_deltas() -> pd.DataFrame:
    cols = ["seed", "branch", "generation", "comparison"] + [
        f"delta_{metric}" for metric in _METRIC_COLS
    ]
    return pd.DataFrame(columns=cols)


@dataclass(frozen=True)
class ModeComparisonArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    seed_stats_csv: Path
    seed_stats_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    mode_deltas_csv: Path
    mode_deltas_parquet: Path
    qualitative_csv: Path
    qualitative_parquet: Path
    accuracy_plot: Path
    pedagogical_plot: Path
    silent_error_plot: Path


def _load_snapshot(run_dir: Path) -> dict:
    path = run_dir / "run_config.snapshot.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run snapshot: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_run(run_dir: Path) -> int:
    payload = _load_snapshot(run_dir)
    try:
        return int(payload["config"]["project"]["seed"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Cannot extract seed from run snapshot: {run_dir}") from exc


def _mode_from_run(run_dir: Path) -> str:
    payload = _load_snapshot(run_dir)
    mode = str(payload["config"].get("experiment", {}).get("mode", INFERENCE_MODE)).strip()
    if mode not in {INFERENCE_MODE, TRAINING_MODE}:
        raise RuntimeError(f"Unsupported experiment mode in run snapshot ({run_dir}): {mode}")
    return mode


def _base_model_from_run(run_dir: Path) -> str:
    payload = _load_snapshot(run_dir)
    models = payload["config"].get("models", {}).get("local_models", [])
    if not models:
        raise RuntimeError(f"Run snapshot missing local_models: {run_dir}")
    return str(models[0]["name"])


def _heldout_for_run(run_dir: Path) -> pd.DataFrame:
    payload = _load_snapshot(run_dir)
    data_root = Path(payload["config"]["paths"]["data_root"])
    heldout_path = data_root / "splits" / "heldout_test.parquet"
    if not heldout_path.exists():
        raise FileNotFoundError(f"Missing heldout split: {heldout_path}")
    heldout = pd.read_parquet(heldout_path)
    needed = {"example_id", "question", "answer_gold"}
    missing = needed.difference(heldout.columns)
    if missing:
        raise RuntimeError(f"Heldout split missing columns {sorted(missing)} ({heldout_path})")
    return heldout[list(needed)].copy()


def _collect_inference_metrics(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        metrics = build_first_experiment_run_metrics(run_dir)
        metrics["run_id"] = run_dir.name
        metrics["run_dir"] = str(run_dir)
        metrics["seed"] = _seed_from_run(run_dir)
        metrics["evaluation_mode"] = _mode_from_run(run_dir)
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _collect_training_metrics(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        summary_csv = run_dir / "tables" / "training_feasibility_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing training feasibility summary: {summary_csv}")
        df = pd.read_csv(summary_csv)
        required = {
            "branch",
            "generation",
            "sample_count",
            "accuracy_mean",
            "pedagogical_score_mean",
            "silent_error_rate",
        }
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(
                f"Training summary missing required columns: {sorted(missing)} ({summary_csv})"
            )
        if "parse_failure_pred_rate" not in df.columns:
            df["parse_failure_pred_rate"] = math.nan
        if "parse_failure_pred_count" not in df.columns:
            df["parse_failure_pred_count"] = 0
        if "sample_count_from_accuracy" not in df.columns:
            df["sample_count_from_accuracy"] = df["sample_count"]
        if "accuracy_source" not in df.columns:
            df["accuracy_source"] = "training_feasibility_summary"

        df["model_name"] = _base_model_from_run(run_dir)
        df["run_id"] = run_dir.name
        df["run_dir"] = str(run_dir)
        df["seed"] = _seed_from_run(run_dir)
        df["evaluation_mode"] = _mode_from_run(run_dir)

        rows.append(
            df[
                [
                    "run_id",
                    "run_dir",
                    "seed",
                    "model_name",
                    "branch",
                    "generation",
                    "sample_count",
                    "accuracy_mean",
                    "pedagogical_score_mean",
                    "silent_error_rate",
                    "parse_failure_pred_count",
                    "sample_count_from_accuracy",
                    "parse_failure_pred_rate",
                    "accuracy_source",
                    "evaluation_mode",
                ]
            ].copy()
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def collect_mode_run_metrics(*, inference_run_dirs: Sequence[Path], training_run_dirs: Sequence[Path]) -> pd.DataFrame:
    inf = _collect_inference_metrics(inference_run_dirs)
    trn = _collect_training_metrics(training_run_dirs)
    if inf.empty or trn.empty:
        raise ValueError("Both inference_run_dirs and training_run_dirs must be non-empty")
    out = pd.concat([inf, trn], ignore_index=True)
    return out.sort_values(["evaluation_mode", "seed", "branch", "generation"]).reset_index(drop=True)


def _bootstrap_ci_mean(
    values: Iterable[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> tuple[float, float]:
    arr = [float(v) for v in values if pd.notna(v)]
    if not arr:
        return (math.nan, math.nan)
    if len(arr) == 1:
        return (arr[0], arr[0])

    rng = random.Random(rng_seed)
    means: list[float] = []
    n = len(arr)
    for _ in range(max(100, n_boot)):
        sample = [arr[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int((alpha / 2.0) * (len(means) - 1))
    high_idx = int((1.0 - alpha / 2.0) * (len(means) - 1))
    return means[low_idx], means[high_idx]


def build_seed_stats(run_level_df: pd.DataFrame, *, bootstrap_seed: int = 42) -> pd.DataFrame:
    required = {"evaluation_mode", "seed", "branch", "generation", *(_METRIC_COLS)}
    missing = required.difference(run_level_df.columns)
    if missing:
        raise ValueError(f"run_level_df missing columns for seed stats: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    for _, group in run_level_df.groupby(["evaluation_mode", "branch", "generation"], as_index=False):
        row: dict[str, float | int | str] = {
            "evaluation_mode": str(group["evaluation_mode"].iloc[0]),
            "branch": str(group["branch"].iloc[0]),
            "generation": int(group["generation"].iloc[0]),
            "seed_count": int(group["seed"].nunique()),
            "run_count": int(len(group)),
        }
        for metric in _METRIC_COLS:
            vals = [float(v) for v in group[metric].tolist()]
            row[f"{metric}_mean"] = float(pd.Series(vals).mean())
            row[f"{metric}_std"] = float(pd.Series(vals).std(ddof=1)) if len(vals) > 1 else 0.0
            ci_low, ci_high = _bootstrap_ci_mean(
                vals,
                rng_seed=bootstrap_seed + abs(hash((row["evaluation_mode"], row["branch"], row["generation"], metric))) % 10_000,
            )
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evaluation_mode", "branch", "generation"]).reset_index(drop=True)


def build_generation_deltas(run_level_df: pd.DataFrame) -> pd.DataFrame:
    required = {"run_id", "seed", "evaluation_mode", "branch", "generation", *(_METRIC_COLS)}
    missing = required.difference(run_level_df.columns)
    if missing:
        raise ValueError(f"run_level_df missing columns for generation deltas: {sorted(missing)}")
    rows: list[dict[str, float | int | str]] = []
    grouped = run_level_df.groupby(["run_id", "seed", "evaluation_mode", "branch"], as_index=False)
    for _, group in grouped:
        gens = sorted(set(int(x) for x in group["generation"].tolist()))
        if len(gens) < 2:
            continue
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                g0, g1 = gens[i], gens[j]
                start = group[group["generation"] == g0].iloc[0]
                end = group[group["generation"] == g1].iloc[0]
                row: dict[str, float | int | str] = {
                    "run_id": str(start["run_id"]),
                    "seed": int(start["seed"]),
                    "evaluation_mode": str(start["evaluation_mode"]),
                    "branch": str(start["branch"]),
                    "generation_start": g0,
                    "generation_end": g1,
                    "delta_generation": f"gen{g1}_minus_gen{g0}",
                }
                for metric in _METRIC_COLS:
                    row[f"delta_{metric}"] = float(end[metric]) - float(start[metric])
                rows.append(row)
    if not rows:
        return _empty_generation_deltas()
    return pd.DataFrame(rows).sort_values(
        ["evaluation_mode", "seed", "branch", "generation_start", "generation_end"]
    ).reset_index(drop=True)


def build_branch_deltas(run_level_df: pd.DataFrame) -> pd.DataFrame:
    required = {"run_id", "seed", "evaluation_mode", "branch", "generation", *(_METRIC_COLS)}
    missing = required.difference(run_level_df.columns)
    if missing:
        raise ValueError(f"run_level_df missing columns for branch deltas: {sorted(missing)}")
    rows: list[dict[str, float | int | str]] = []
    grouped = run_level_df.groupby(["run_id", "seed", "evaluation_mode", "generation"], as_index=False)
    for _, group in grouped:
        pure = group[group["branch"] == "pure_recycling"]
        if pure.empty:
            continue
        pure_row = pure.iloc[0]
        for _, anchor_row in group[group["branch"] != "pure_recycling"].iterrows():
            row: dict[str, float | int | str] = {
                "run_id": str(anchor_row["run_id"]),
                "seed": int(anchor_row["seed"]),
                "evaluation_mode": str(anchor_row["evaluation_mode"]),
                "generation": int(anchor_row["generation"]),
                "anchor_branch": str(anchor_row["branch"]),
                "comparison": f"{anchor_row['branch']}_minus_pure_recycling",
            }
            for metric in _METRIC_COLS:
                row[f"delta_{metric}"] = float(anchor_row[metric]) - float(pure_row[metric])
            rows.append(row)
    if not rows:
        return _empty_branch_deltas()
    return pd.DataFrame(rows).sort_values(
        ["evaluation_mode", "seed", "generation", "anchor_branch"]
    ).reset_index(drop=True)


def build_mode_deltas(run_level_df: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "evaluation_mode", "branch", "generation", *(_METRIC_COLS)}
    missing = required.difference(run_level_df.columns)
    if missing:
        raise ValueError(f"run_level_df missing columns for mode deltas: {sorted(missing)}")

    inf = run_level_df[run_level_df["evaluation_mode"] == INFERENCE_MODE].copy()
    trn = run_level_df[run_level_df["evaluation_mode"] == TRAINING_MODE].copy()
    if inf.empty or trn.empty:
        raise ValueError("Need both inference and training rows to compute mode deltas")

    inf_seeds = set(int(x) for x in inf["seed"].tolist())
    trn_seeds = set(int(x) for x in trn["seed"].tolist())
    if inf_seeds != trn_seeds:
        raise ValueError(
            "Inference/training seed sets differ for controlled comparison. "
            f"inference={sorted(inf_seeds)}, training={sorted(trn_seeds)}"
        )

    key_cols = ["seed", "branch", "generation"]
    merged = trn.merge(
        inf,
        on=key_cols,
        how="inner",
        suffixes=("_training", "_inference"),
        validate="one_to_one",
    )
    out_rows: list[dict[str, float | int | str]] = []
    for _, rec in merged.iterrows():
        row: dict[str, float | int | str] = {
            "seed": int(rec["seed"]),
            "branch": str(rec["branch"]),
            "generation": int(rec["generation"]),
            "comparison": "training_minus_inference",
        }
        for metric in _METRIC_COLS:
            row[f"delta_{metric}"] = float(rec[f"{metric}_training"]) - float(rec[f"{metric}_inference"])
        out_rows.append(row)
    if not out_rows:
        return _empty_mode_deltas()
    return pd.DataFrame(out_rows).sort_values(["seed", "branch", "generation"]).reset_index(drop=True)


def _load_eval_rows(run_dir: Path, *, mode: str, seed: int) -> pd.DataFrame:
    eval_paths = sorted(run_dir.glob("*/*/gen_*/eval_merged.parquet"))
    if not eval_paths:
        return pd.DataFrame()
    try:
        heldout = _heldout_for_run(run_dir)
    except (FileNotFoundError, RuntimeError):
        # Qualitative examples are optional; if heldout context is unavailable we
        # keep main numeric comparison artifacts and skip qualitative extraction.
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for p in eval_paths:
        df = pd.read_parquet(p)
        cols = ["example_id", "model_name", "branch", "generation", "is_correct", "overall_pedagogical_score", "is_silent_error"]
        missing = set(cols).difference(df.columns)
        if missing:
            continue
        out = df[cols].copy()
        out = out.merge(heldout, on="example_id", how="left", validate="many_to_one")
        out["run_id"] = run_dir.name
        out["run_dir"] = str(run_dir)
        out["seed"] = seed
        out["evaluation_mode"] = mode
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_qualitative_mode_examples(
    *,
    inference_run_dirs: Sequence[Path],
    training_run_dirs: Sequence[Path],
    max_total: int = 12,
) -> pd.DataFrame:
    inf_rows: list[pd.DataFrame] = []
    trn_rows: list[pd.DataFrame] = []
    for run_dir in inference_run_dirs:
        inf_rows.append(_load_eval_rows(run_dir, mode=INFERENCE_MODE, seed=_seed_from_run(run_dir)))
    for run_dir in training_run_dirs:
        trn_rows.append(_load_eval_rows(run_dir, mode=TRAINING_MODE, seed=_seed_from_run(run_dir)))

    inf_nonempty = [x for x in inf_rows if not x.empty]
    trn_nonempty = [x for x in trn_rows if not x.empty]
    inf = pd.concat(inf_nonempty, ignore_index=True) if inf_nonempty else pd.DataFrame()
    trn = pd.concat(trn_nonempty, ignore_index=True) if trn_nonempty else pd.DataFrame()
    if inf.empty or trn.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "seed",
                "branch",
                "generation",
                "example_id",
                "question",
                "answer_gold",
                "is_correct_training",
                "is_correct_inference",
                "overall_pedagogical_score_training",
                "overall_pedagogical_score_inference",
                "is_silent_error_training",
                "is_silent_error_inference",
            ]
        )

    merged = trn.merge(
        inf,
        on=["seed", "branch", "generation", "example_id", "question", "answer_gold"],
        how="inner",
        suffixes=("_training", "_inference"),
    )
    if merged.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    better_trn = merged[
        (merged["overall_pedagogical_score_training"] - merged["overall_pedagogical_score_inference"] >= 1)
        | ((merged["is_correct_training"] == True) & (merged["is_correct_inference"] == False))  # noqa: E712
        | ((merged["is_silent_error_training"] == False) & (merged["is_silent_error_inference"] == True))  # noqa: E712
    ].copy()
    better_trn["category"] = "training_better_than_inference"
    better_trn = better_trn.sort_values(
        ["generation", "overall_pedagogical_score_training"], ascending=[False, False]
    ).head(max_total // 3 + 1)
    frames.append(better_trn)

    better_inf = merged[
        (merged["overall_pedagogical_score_inference"] - merged["overall_pedagogical_score_training"] >= 1)
        | ((merged["is_correct_inference"] == True) & (merged["is_correct_training"] == False))  # noqa: E712
    ].copy()
    better_inf["category"] = "inference_better_than_training"
    better_inf = better_inf.sort_values(
        ["generation", "overall_pedagogical_score_inference"], ascending=[False, False]
    ).head(max_total // 3 + 1)
    frames.append(better_inf)

    silent_flip = merged[
        merged["is_silent_error_training"].astype(bool) != merged["is_silent_error_inference"].astype(bool)
    ].copy()
    silent_flip["category"] = "silent_error_mode_disagreement"
    silent_flip = silent_flip.sort_values(["generation"]).head(max_total // 3 + 1)
    frames.append(silent_flip)

    out = pd.concat([x for x in frames if not x.empty], ignore_index=True)
    if out.empty:
        return out
    keep = [
        "category",
        "seed",
        "branch",
        "generation",
        "example_id",
        "question",
        "answer_gold",
        "is_correct_training",
        "is_correct_inference",
        "overall_pedagogical_score_training",
        "overall_pedagogical_score_inference",
        "is_silent_error_training",
        "is_silent_error_inference",
    ]
    return out[keep].head(max_total).reset_index(drop=True)


def _plot_metric(seed_stats: pd.DataFrame, *, metric: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_ci_low"
    high_col = f"{metric}_ci_high"
    required = {"evaluation_mode", "branch", "generation", mean_col, low_col, high_col}
    missing = required.difference(seed_stats.columns)
    if missing:
        raise ValueError(f"seed_stats missing columns for plotting: {sorted(missing)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    for (mode, branch), sub in seed_stats.groupby(["evaluation_mode", "branch"]):
        sub = sub.sort_values("generation")
        x = sub["generation"].to_numpy()
        y = sub[mean_col].to_numpy()
        lo = sub[low_col].to_numpy()
        hi = sub[high_col].to_numpy()
        plt.plot(x, y, marker="o", label=f"{mode}|{branch}")
        plt.fill_between(x, lo, hi, alpha=0.15)
    plt.xlabel("Generation")
    plt.ylabel(metric)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_mode_comparison_analysis(
    *,
    inference_run_dirs: Sequence[Path],
    training_run_dirs: Sequence[Path],
    out_dir: Path,
    bootstrap_seed: int = 42,
) -> ModeComparisonArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_level = collect_mode_run_metrics(
        inference_run_dirs=inference_run_dirs,
        training_run_dirs=training_run_dirs,
    )
    seed_stats = build_seed_stats(run_level, bootstrap_seed=bootstrap_seed)
    generation_deltas = build_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)
    mode_deltas = build_mode_deltas(run_level)
    qualitative = build_qualitative_mode_examples(
        inference_run_dirs=inference_run_dirs,
        training_run_dirs=training_run_dirs,
        max_total=12,
    )

    run_level_csv = out_dir / "mode_comparison_run_level.csv"
    run_level_parquet = out_dir / "mode_comparison_run_level.parquet"
    seed_stats_csv = out_dir / "mode_comparison_seed_stats.csv"
    seed_stats_parquet = out_dir / "mode_comparison_seed_stats.parquet"
    generation_deltas_csv = out_dir / "mode_comparison_generation_deltas.csv"
    generation_deltas_parquet = out_dir / "mode_comparison_generation_deltas.parquet"
    branch_deltas_csv = out_dir / "mode_comparison_branch_deltas.csv"
    branch_deltas_parquet = out_dir / "mode_comparison_branch_deltas.parquet"
    mode_deltas_csv = out_dir / "mode_comparison_mode_deltas.csv"
    mode_deltas_parquet = out_dir / "mode_comparison_mode_deltas.parquet"
    qualitative_csv = out_dir / "mode_comparison_qualitative_candidates.csv"
    qualitative_parquet = out_dir / "mode_comparison_qualitative_candidates.parquet"

    run_level.to_csv(run_level_csv, index=False)
    run_level.to_parquet(run_level_parquet, index=False)
    seed_stats.to_csv(seed_stats_csv, index=False)
    seed_stats.to_parquet(seed_stats_parquet, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    generation_deltas.to_parquet(generation_deltas_parquet, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    branch_deltas.to_parquet(branch_deltas_parquet, index=False)
    mode_deltas.to_csv(mode_deltas_csv, index=False)
    mode_deltas.to_parquet(mode_deltas_parquet, index=False)
    qualitative.to_csv(qualitative_csv, index=False)
    qualitative.to_parquet(qualitative_parquet, index=False)

    figures_dir = out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot = figures_dir / "mode_comparison_accuracy_by_branch_generation.png"
    pedagogical_plot = figures_dir / "mode_comparison_pedagogical_by_branch_generation.png"
    silent_error_plot = figures_dir / "mode_comparison_silent_error_by_branch_generation.png"
    _plot_metric(seed_stats, metric="accuracy_mean", out_path=accuracy_plot)
    _plot_metric(seed_stats, metric="pedagogical_score_mean", out_path=pedagogical_plot)
    _plot_metric(seed_stats, metric="silent_error_rate", out_path=silent_error_plot)

    return ModeComparisonArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        seed_stats_csv=seed_stats_csv,
        seed_stats_parquet=seed_stats_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        mode_deltas_csv=mode_deltas_csv,
        mode_deltas_parquet=mode_deltas_parquet,
        qualitative_csv=qualitative_csv,
        qualitative_parquet=qualitative_parquet,
        accuracy_plot=accuracy_plot,
        pedagogical_plot=pedagogical_plot,
        silent_error_plot=silent_error_plot,
    )
