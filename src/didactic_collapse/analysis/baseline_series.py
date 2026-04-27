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

EVAL_MODE = "inference_recycling_only"
_KEY_COLS = ["run_id", "seed", "model_name", "branch", "generation"]
_METRIC_COLS = [
    "accuracy_mean",
    "pedagogical_score_mean",
    "silent_error_rate",
    "parse_failure_pred_rate",
]


@dataclass(frozen=True)
class BaselineSeriesArtifacts:
    run_level_csv: Path
    run_level_parquet: Path
    seed_stats_csv: Path
    seed_stats_parquet: Path
    generation_deltas_csv: Path
    generation_deltas_parquet: Path
    branch_deltas_csv: Path
    branch_deltas_parquet: Path
    qualitative_csv: Path
    qualitative_parquet: Path
    accuracy_plot: Path
    pedagogical_plot: Path
    silent_error_plot: Path


def _load_run_snapshot(run_dir: Path) -> dict:
    snapshot_path = run_dir / "run_config.snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run snapshot: {snapshot_path}")
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _seed_from_run(run_dir: Path) -> int:
    payload = _load_run_snapshot(run_dir)
    try:
        return int(payload["config"]["project"]["seed"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Cannot extract seed from run snapshot: {run_dir}") from exc


def _heldout_from_run(run_dir: Path) -> pd.DataFrame:
    payload = _load_run_snapshot(run_dir)
    data_root = Path(payload["config"]["paths"]["data_root"])
    heldout_path = data_root / "splits" / "heldout_test.parquet"
    if not heldout_path.exists():
        raise FileNotFoundError(f"Missing heldout split for run: {heldout_path}")
    heldout = pd.read_parquet(heldout_path)
    required = {"example_id", "question", "answer_gold"}
    missing = required.difference(heldout.columns)
    if missing:
        raise RuntimeError(
            f"Heldout split missing required columns for qualitative export: {sorted(missing)} ({heldout_path})"
        )
    return heldout[list(required)].copy()


def collect_baseline_run_metrics(run_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        metrics = build_first_experiment_run_metrics(run_dir)
        metrics["run_id"] = run_dir.name
        metrics["run_dir"] = str(run_dir)
        metrics["seed"] = _seed_from_run(run_dir)
        metrics["evaluation_mode"] = EVAL_MODE
        rows.append(metrics)

    if not rows:
        raise ValueError("run_dirs cannot be empty")

    out = pd.concat(rows, ignore_index=True)
    out = out[
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
    return out.sort_values(["seed", "model_name", "branch", "generation"]).reset_index(drop=True)


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


def build_seed_level_summary(
    run_level_df: pd.DataFrame,
    *,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    required = {"model_name", "branch", "generation", "seed", *(_METRIC_COLS)}
    missing = required.difference(run_level_df.columns)
    if missing:
        raise ValueError(f"run_level_df missing columns for seed summary: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    grouped = run_level_df.groupby(["model_name", "branch", "generation"], as_index=False)
    for _, group in grouped:
        row: dict[str, float | int | str] = {
            "model_name": str(group["model_name"].iloc[0]),
            "branch": str(group["branch"].iloc[0]),
            "generation": int(group["generation"].iloc[0]),
            "seed_count": int(group["seed"].nunique()),
            "run_count": int(len(group)),
            "evaluation_mode": EVAL_MODE,
        }
        for metric in _METRIC_COLS:
            vals = [float(v) for v in group[metric].tolist()]
            mean_val = float(pd.Series(vals).mean())
            std_val = float(pd.Series(vals).std(ddof=1)) if len(vals) > 1 else 0.0
            ci_low, ci_high = _bootstrap_ci_mean(
                vals,
                rng_seed=bootstrap_seed + abs(hash((row["branch"], row["generation"], metric))) % 10_000,
            )
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["model_name", "branch", "generation"]).reset_index(drop=True)


def build_generation_deltas(run_level_df: pd.DataFrame) -> pd.DataFrame:
    required = {"run_id", "seed", "model_name", "branch", "generation", *(_METRIC_COLS)}
    missing = required.difference(run_level_df.columns)
    if missing:
        raise ValueError(f"run_level_df missing columns for generation deltas: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    grouped = run_level_df.groupby(["run_id", "seed", "model_name", "branch"], as_index=False)
    for _, group in grouped:
        generations = sorted(set(int(x) for x in group["generation"].tolist()))
        if len(generations) < 2:
            continue
        pair_set: set[tuple[int, int]] = set()
        for i in range(len(generations)):
            for j in range(i + 1, len(generations)):
                pair_set.add((generations[i], generations[j]))

        for start_gen, end_gen in sorted(pair_set):
            start_row = group[group["generation"] == start_gen].iloc[0]
            end_row = group[group["generation"] == end_gen].iloc[0]
            row: dict[str, float | int | str] = {
                "run_id": str(start_row["run_id"]),
                "seed": int(start_row["seed"]),
                "model_name": str(start_row["model_name"]),
                "branch": str(start_row["branch"]),
                "generation_start": int(start_gen),
                "generation_end": int(end_gen),
                "delta_generation": f"gen{end_gen}_minus_gen{start_gen}",
                "evaluation_mode": EVAL_MODE,
            }
            for metric in _METRIC_COLS:
                row[f"delta_{metric}"] = float(end_row[metric]) - float(start_row[metric])
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["seed", "model_name", "branch", "generation_start", "generation_end"])
        .reset_index(drop=True)
    )


def build_branch_deltas(run_level_df: pd.DataFrame) -> pd.DataFrame:
    required = {"run_id", "seed", "model_name", "branch", "generation", *(_METRIC_COLS)}
    missing = required.difference(run_level_df.columns)
    if missing:
        raise ValueError(f"run_level_df missing columns for branch deltas: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    grouped = run_level_df.groupby(["run_id", "seed", "model_name", "generation"], as_index=False)
    for _, group in grouped:
        pure = group[group["branch"] == "pure_recycling"]
        if pure.empty:
            continue
        pure_row = pure.iloc[0]
        for _, anchor_row in group[group["branch"] != "pure_recycling"].iterrows():
            row: dict[str, float | int | str] = {
                "run_id": str(anchor_row["run_id"]),
                "seed": int(anchor_row["seed"]),
                "model_name": str(anchor_row["model_name"]),
                "generation": int(anchor_row["generation"]),
                "comparison": f"{anchor_row['branch']}_minus_pure_recycling",
                "anchor_branch": str(anchor_row["branch"]),
                "evaluation_mode": EVAL_MODE,
            }
            for metric in _METRIC_COLS:
                row[f"delta_{metric}"] = float(anchor_row[metric]) - float(pure_row[metric])
            rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["seed", "model_name", "generation", "anchor_branch"])
        .reset_index(drop=True)
    )


def _load_row_level_eval(run_dir: Path, *, seed: int) -> pd.DataFrame:
    heldout = _heldout_from_run(run_dir)
    rows: list[pd.DataFrame] = []
    for eval_path in sorted(run_dir.glob("*/*/gen_*/eval_merged.parquet")):
        eval_df = pd.read_parquet(eval_path)
        model_outputs_path = eval_path.with_name("model_outputs.parquet")
        if not model_outputs_path.exists():
            continue
        outputs_df = pd.read_parquet(model_outputs_path)[["example_id", "raw_response"]]
        merged = eval_df.merge(outputs_df, on="example_id", how="left", validate="one_to_one")
        merged = merged.merge(heldout, on="example_id", how="left", validate="one_to_one")
        merged["run_id"] = run_dir.name
        merged["run_dir"] = str(run_dir)
        merged["seed"] = seed
        merged["evaluation_mode"] = EVAL_MODE
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_qualitative_candidates(
    run_dirs: Sequence[Path],
    *,
    max_total: int = 10,
) -> pd.DataFrame:
    row_level: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        seed = _seed_from_run(run_dir)
        row_level.append(_load_row_level_eval(run_dir, seed=seed))
    df = pd.concat([x for x in row_level if not x.empty], ignore_index=True) if row_level else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(
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

    keep_cols = [
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
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None

    # 1) Silent-error candidates in later generations with correct factual answer.
    silent = df[
        (df["is_correct"] == True)  # noqa: E712
        & (df["is_silent_error"] == True)  # noqa: E712
        & (df["generation"] >= 1)
    ].copy()
    silent["category"] = "silent_error_candidate"
    silent = silent.sort_values(["generation", "overall_pedagogical_score", "branch"], ascending=[False, True, True]).head(4)

    # 2) Strong monotonic pedagogical decline trajectory Gen-0 -> Gen-1 -> Gen-2.
    g0 = df[df["generation"] == 0].copy()
    g1 = df[df["generation"] == 1].copy()
    g2 = df[df["generation"] == 2].copy()
    decline = pd.DataFrame()
    if (not g0.empty) and (not g1.empty) and (not g2.empty):
        g01 = g0.merge(
            g1,
            on=["run_id", "seed", "model_name", "branch", "example_id", "question", "answer_gold"],
            suffixes=("_g0", "_g1"),
            how="inner",
            validate="one_to_one",
        )
        g012 = g01.merge(
            g2,
            on=["run_id", "seed", "model_name", "branch", "example_id", "question", "answer_gold"],
            how="inner",
            validate="one_to_one",
        )
        g012["pedagogical_delta_gen2_minus_gen0"] = (
            g012["overall_pedagogical_score"] - g012["overall_pedagogical_score_g0"]
        )
        g012 = g012[
            (g012["overall_pedagogical_score_g1"] < g012["overall_pedagogical_score_g0"])
            & (g012["overall_pedagogical_score"] < g012["overall_pedagogical_score_g1"])
        ].copy()
        decline = g012.sort_values("pedagogical_delta_gen2_minus_gen0").head(3)
    if not decline.empty:
        decline = decline.rename(
            columns={
                "generation": "generation",
                "is_correct": "is_correct",
                "overall_pedagogical_score": "overall_pedagogical_score",
                "is_silent_error": "is_silent_error",
                "raw_response": "raw_response",
                "evaluation_mode": "evaluation_mode",
            }
        )
        decline["category"] = "pedagogical_decline_gen0_to_gen2"
        decline = decline[["category", *keep_cols]]

    # 3) Anchor branch improves relative to pure_recycling on the same example.
    pure = df[df["branch"] == "pure_recycling"].copy()
    anchors = df[df["branch"] != "pure_recycling"].copy()
    anchor_gain = anchors.merge(
        pure,
        on=["run_id", "seed", "model_name", "generation", "example_id", "question", "answer_gold"],
        suffixes=("_anchor", "_pure"),
        how="inner",
        # Multiple anchor branches can map to the same pure example context.
        validate="many_to_one",
    )
    anchor_gain["pedagogical_gain"] = (
        anchor_gain["overall_pedagogical_score_anchor"] - anchor_gain["overall_pedagogical_score_pure"]
    )
    anchor_gain["is_better_anchor"] = (
        (anchor_gain["pedagogical_gain"] >= 1)
        | (
            (anchor_gain["is_correct_anchor"] == True)  # noqa: E712
            & (anchor_gain["is_correct_pure"] == False)  # noqa: E712
        )
        | (
            (anchor_gain["is_silent_error_anchor"] == False)  # noqa: E712
            & (anchor_gain["is_silent_error_pure"] == True)  # noqa: E712
        )
    )
    anchor_gain = anchor_gain[anchor_gain["is_better_anchor"] == True].copy()  # noqa: E712
    anchor_gain = anchor_gain.sort_values(["pedagogical_gain", "generation"], ascending=[False, True]).head(3)
    if not anchor_gain.empty:
        anchor_gain = anchor_gain.rename(
            columns={
                "branch_anchor": "branch",
                "is_correct_anchor": "is_correct",
                "overall_pedagogical_score_anchor": "overall_pedagogical_score",
                "is_silent_error_anchor": "is_silent_error",
                "raw_response_anchor": "raw_response",
                "evaluation_mode_anchor": "evaluation_mode",
            }
        )
        anchor_gain["category"] = "anchor_better_than_pure"
        anchor_gain = anchor_gain[["category", *keep_cols]]

    # 4) Pure path degrades stronger than anchor from Gen-0 to Gen-2 on same example.
    pure_decline = pd.DataFrame()
    if not pure.empty:
        pure0 = pure[pure["generation"] == 0].copy()
        pure2 = pure[pure["generation"] == 2].copy()
        anchor0 = anchors[anchors["generation"] == 0].copy()
        anchor2 = anchors[anchors["generation"] == 2].copy()
        if not pure0.empty and not pure2.empty and not anchor0.empty and not anchor2.empty:
            pure_pair = pure0.merge(
                pure2,
                on=["run_id", "seed", "model_name", "example_id", "question", "answer_gold"],
                suffixes=("_pure_g0", "_pure_g2"),
                how="inner",
                validate="one_to_one",
            )
            anchor_pair = anchor0.merge(
                anchor2,
                on=["run_id", "seed", "model_name", "branch", "example_id", "question", "answer_gold"],
                suffixes=("_anchor_g0", "_anchor_g2"),
                how="inner",
                validate="one_to_one",
            )
            pure_anchor = anchor_pair.merge(
                pure_pair,
                on=["run_id", "seed", "model_name", "example_id", "question", "answer_gold"],
                how="inner",
                validate="many_to_one",
            )
            pure_anchor["pure_decline"] = (
                pure_anchor["overall_pedagogical_score_pure_g2"] - pure_anchor["overall_pedagogical_score_pure_g0"]
            )
            pure_anchor["anchor_decline"] = (
                pure_anchor["overall_pedagogical_score_anchor_g2"] - pure_anchor["overall_pedagogical_score_anchor_g0"]
            )
            pure_anchor["decline_gap"] = pure_anchor["pure_decline"] - pure_anchor["anchor_decline"]
            pure_decline = pure_anchor[pure_anchor["decline_gap"] <= -1].copy()
            pure_decline = pure_decline.sort_values("decline_gap").head(3)
    if not pure_decline.empty:
        pure_decline = pure_decline.rename(
            columns={
                "branch": "branch",
                "generation_anchor_g2": "generation",
                "is_correct_anchor_g2": "is_correct",
                "overall_pedagogical_score_anchor_g2": "overall_pedagogical_score",
                "is_silent_error_anchor_g2": "is_silent_error",
                "raw_response_anchor_g2": "raw_response",
                "evaluation_mode_anchor_g2": "evaluation_mode",
            }
        )
        pure_decline["generation"] = 2
        pure_decline["category"] = "pure_degrades_more_than_anchor_gen0_to_gen2"
        pure_decline = pure_decline[["category", *keep_cols]]

    frames = [silent[["category", *keep_cols]]]
    if isinstance(decline, pd.DataFrame) and not decline.empty:
        frames.append(decline)
    if isinstance(anchor_gain, pd.DataFrame) and not anchor_gain.empty:
        frames.append(anchor_gain)
    if isinstance(pure_decline, pd.DataFrame) and not pure_decline.empty:
        frames.append(pure_decline)

    out = pd.concat(frames, ignore_index=True).head(max_total)
    return out.reset_index(drop=True)


def _plot_seed_summary(
    seed_stats_df: pd.DataFrame,
    *,
    metric: str,
    out_path: Path,
) -> None:
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_ci_low"
    high_col = f"{metric}_ci_high"
    required = {"generation", "branch", mean_col, low_col, high_col}
    missing = required.difference(seed_stats_df.columns)
    if missing:
        raise ValueError(f"seed_stats_df missing plot columns: {sorted(missing)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for branch, sub in seed_stats_df.groupby("branch"):
        sub = sub.sort_values("generation")
        x = sub["generation"].to_numpy()
        y = sub[mean_col].to_numpy()
        y_low = sub[low_col].to_numpy()
        y_high = sub[high_col].to_numpy()
        plt.plot(x, y, marker="o", label=str(branch))
        plt.fill_between(x, y_low, y_high, alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def export_baseline_series_analysis(
    *,
    run_dirs: Sequence[Path],
    out_dir: Path,
    bootstrap_seed: int = 42,
) -> BaselineSeriesArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_level = collect_baseline_run_metrics(run_dirs)
    seed_stats = build_seed_level_summary(run_level, bootstrap_seed=bootstrap_seed)
    generation_deltas = build_generation_deltas(run_level)
    branch_deltas = build_branch_deltas(run_level)
    qualitative = build_qualitative_candidates(run_dirs, max_total=10)

    run_level_csv = out_dir / "baseline_series_run_level.csv"
    run_level_parquet = out_dir / "baseline_series_run_level.parquet"
    seed_stats_csv = out_dir / "baseline_series_seed_stats.csv"
    seed_stats_parquet = out_dir / "baseline_series_seed_stats.parquet"
    generation_deltas_csv = out_dir / "baseline_series_generation_deltas.csv"
    generation_deltas_parquet = out_dir / "baseline_series_generation_deltas.parquet"
    branch_deltas_csv = out_dir / "baseline_series_branch_deltas.csv"
    branch_deltas_parquet = out_dir / "baseline_series_branch_deltas.parquet"
    qualitative_csv = out_dir / "baseline_series_qualitative_candidates.csv"
    qualitative_parquet = out_dir / "baseline_series_qualitative_candidates.parquet"

    run_level.to_csv(run_level_csv, index=False)
    run_level.to_parquet(run_level_parquet, index=False)
    seed_stats.to_csv(seed_stats_csv, index=False)
    seed_stats.to_parquet(seed_stats_parquet, index=False)
    generation_deltas.to_csv(generation_deltas_csv, index=False)
    generation_deltas.to_parquet(generation_deltas_parquet, index=False)
    branch_deltas.to_csv(branch_deltas_csv, index=False)
    branch_deltas.to_parquet(branch_deltas_parquet, index=False)
    qualitative.to_csv(qualitative_csv, index=False)
    qualitative.to_parquet(qualitative_parquet, index=False)

    figures_dir = out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    accuracy_plot = figures_dir / "baseline_series_accuracy_by_branch_generation.png"
    pedagogical_plot = figures_dir / "baseline_series_pedagogical_by_branch_generation.png"
    silent_error_plot = figures_dir / "baseline_series_silent_error_by_branch_generation.png"

    _plot_seed_summary(seed_stats, metric="accuracy_mean", out_path=accuracy_plot)
    _plot_seed_summary(seed_stats, metric="pedagogical_score_mean", out_path=pedagogical_plot)
    _plot_seed_summary(seed_stats, metric="silent_error_rate", out_path=silent_error_plot)

    return BaselineSeriesArtifacts(
        run_level_csv=run_level_csv,
        run_level_parquet=run_level_parquet,
        seed_stats_csv=seed_stats_csv,
        seed_stats_parquet=seed_stats_parquet,
        generation_deltas_csv=generation_deltas_csv,
        generation_deltas_parquet=generation_deltas_parquet,
        branch_deltas_csv=branch_deltas_csv,
        branch_deltas_parquet=branch_deltas_parquet,
        qualitative_csv=qualitative_csv,
        qualitative_parquet=qualitative_parquet,
        accuracy_plot=accuracy_plot,
        pedagogical_plot=pedagogical_plot,
        silent_error_plot=silent_error_plot,
    )
