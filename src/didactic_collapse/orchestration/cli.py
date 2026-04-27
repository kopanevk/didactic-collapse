from __future__ import annotations

from pathlib import Path

import typer

from didactic_collapse.analysis.anchoring_ablation import export_anchoring_ablation_analysis
from didactic_collapse.analysis.baseline_series import export_baseline_series_analysis
from didactic_collapse.analysis.compare_runs import compare_first_experiment_runs
from didactic_collapse.analysis.judge_sensitivity import run_qwen_judge_sensitivity
from didactic_collapse.analysis.pairwise_judge_sensitivity import run_pairwise_judge_sensitivity
from didactic_collapse.analysis.mode_comparison import export_mode_comparison_analysis
from didactic_collapse.analysis.pvf_confirmatory import export_pvf_confirmatory_analysis
from didactic_collapse.analysis.pvf_stress import export_pvf_stress_analysis
from didactic_collapse.clients.judge_client import (
    cerebras_judge_auth_smoke_check,
    cerebras_judge_rubric_format_check,
    gemini_judge_auth_smoke_check,
)
from didactic_collapse.config.settings import load_config
from didactic_collapse.orchestration.baseline_series import parse_seed_list, run_baseline_series
from didactic_collapse.orchestration.first_experiment import run_first_experiment, resume_first_experiment
from didactic_collapse.orchestration.pilot import run_pilot
from didactic_collapse.orchestration.runner import ExperimentRunner
from didactic_collapse.orchestration.training_feasibility import (
    run_training_recycling_feasibility,
)
from didactic_collapse.orchestration.training_feasibility_series import run_training_feasibility_series
from didactic_collapse.orchestration.pvf_confirmatory import run_pvf_confirmatory_series
from didactic_collapse.utils.logging_utils import setup_logging

app = typer.Typer(help="Didactic collapse experiment orchestration CLI")


@app.command()
def full(config: str = "configs/experiment.yaml") -> None:
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "latest.log"
    setup_logging(log_file)
    runner = ExperimentRunner(cfg)
    runner.run_full()


@app.command()
def stage(
    stage_name: str,
    config: str = "configs/experiment.yaml",
    model_name: str | None = None,
    branch: str | None = None,
    generation: int | None = None,
    seed: int | None = None,
    force: bool = False,
) -> None:
    """Run a single stage with optional context and force-rerun."""
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "latest.log"
    setup_logging(log_file)
    runner = ExperimentRunner(cfg)
    runner.run_stage(
        stage_name=stage_name,  # type: ignore[arg-type]
        model_name=model_name,
        branch=branch,
        generation=generation,
        seed=seed,
        force=force,
    )


@app.command()
def pilot(
    config: str = "configs/pilot.yaml",
    sample_size: int = 30,
    mock_judge: bool = True,
    dry_run: bool = False,
) -> None:
    """Run a fast end-to-end pilot smoke test.

    mock_judge=True is a non-scientific smoke mode.
    """
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "pilot_latest.log"
    setup_logging(log_file)
    summary = run_pilot(cfg=cfg, sample_size=sample_size, mock_judge=mock_judge, dry_run=dry_run)

    typer.echo("Pilot run finished")
    typer.echo(f"Run dir: {summary.run_dir}")
    typer.echo(f"Pilot data root: {summary.pilot_data_root}")
    typer.echo(f"Model: {summary.model_name}")
    typer.echo(f"Branches: {', '.join(summary.branches)}")
    typer.echo(f"Generations: {summary.generations}")
    typer.echo(f"Sample size requested/used: {summary.sample_size_requested}/{summary.sample_size_used}")
    typer.echo(f"Examples scored: {summary.total_examples_scored}")
    typer.echo(f"Completed stages: {summary.completed_stages}")
    typer.echo(f"Skip/resume events: {summary.skip_or_resume_events}")
    typer.echo(f"Artifacts valid: {summary.artifacts_valid}")
    typer.echo(f"Dry run mode: {dry_run}")
    typer.echo(f"Mock judge mode: {mock_judge or dry_run}")
    if summary.missing_artifacts:
        typer.echo("Missing/invalid artifacts:")
        for item in summary.missing_artifacts:
            typer.echo(f" - {item}")


@app.command()
def first_experiment(
    config: str = "configs/first_experiment.yaml",
    sample_size: int = 50,
) -> None:
    """Run first real small-scale experiment (non-mock)."""
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "first_experiment_latest.log"
    setup_logging(log_file)

    summary = run_first_experiment(cfg=cfg, sample_size=sample_size)

    typer.echo("First real experiment finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Run dir: {summary.run_dir}")
    typer.echo(f"Data root used: {summary.data_root_used}")
    typer.echo(f"Model: {summary.model_name}")
    typer.echo(f"Branches: {', '.join(summary.branches)}")
    typer.echo(f"Generations: {', '.join(str(g) for g in summary.generations)}")
    typer.echo(f"Sample size requested/used: {summary.sample_size_requested}/{summary.sample_size_used}")
    typer.echo(f"Total examples scored: {summary.total_examples_scored}")
    typer.echo(f"Summary CSV: {summary.summary_table_path_csv}")
    typer.echo(f"Summary Parquet: {summary.summary_table_path_parquet}")
    typer.echo(f"Qualitative CSV: {summary.qualitative_path_csv}")
    typer.echo(f"Qualitative Parquet: {summary.qualitative_path_parquet}")


@app.command()
def first_experiment_resume(
    config: str = "configs/first_experiment.yaml",
    run_dir: str = "",
) -> None:
    """Resume first experiment from existing run dir via checkpoint manifests."""
    if not run_dir:
        raise typer.BadParameter("run_dir is required")

    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "first_experiment_latest.log"
    setup_logging(log_file)
    summary = resume_first_experiment(cfg=cfg, run_dir=Path(run_dir))

    typer.echo("First real experiment resumed")
    typer.echo(f"Run dir: {summary.run_dir}")
    typer.echo(f"Summary CSV: {summary.summary_table_path_csv}")
    typer.echo(f"Qualitative CSV: {summary.qualitative_path_csv}")


@app.command()
def pvf_stress_test(
    config: str = "configs/pvf_stress_test.yaml",
    sample_size: int = 50,
) -> None:
    """Run PVF stress-test preset and export didactic collapse stress tables."""
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "pvf_stress_latest.log"
    setup_logging(log_file)

    summary = run_first_experiment(cfg=cfg, sample_size=sample_size)
    artifacts = export_pvf_stress_analysis(
        run_dir=summary.run_dir,
        out_dir=summary.run_dir / "tables",
    )

    typer.echo("PVF stress test finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Run dir: {summary.run_dir}")
    typer.echo(f"Model: {summary.model_name}")
    typer.echo(f"Branches: {', '.join(summary.branches)}")
    typer.echo(f"Generations: {', '.join(str(g) for g in summary.generations)}")
    typer.echo(f"Sample size requested/used: {summary.sample_size_requested}/{summary.sample_size_used}")
    typer.echo(f"First summary CSV: {summary.summary_table_path_csv}")
    typer.echo(f"Stress summary CSV: {artifacts.stress_summary_csv}")
    typer.echo(f"Generation deltas CSV: {artifacts.generation_deltas_csv}")
    typer.echo(f"PVF reject reasons CSV: {artifacts.reject_reasons_csv}")
    typer.echo(f"PVF keep-vs-rejected CSV: {artifacts.keep_reject_quality_csv}")
    typer.echo("Evaluation mode: inference_recycling_only (not full retraining)")


@app.command()
def train_feasibility(
    config: str = "configs/train_feasibility.yaml",
    sample_size: int = 50,
    run_dir: str = "",
) -> None:
    """Run or resume Gen-0 -> train -> Gen-1 feasibility experiment."""
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "train_feasibility_latest.log"
    setup_logging(log_file)

    summary = run_training_recycling_feasibility(
        cfg=cfg,
        sample_size=sample_size,
        run_dir=Path(run_dir) if run_dir else None,
    )

    typer.echo("Train feasibility experiment finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Run dir: {summary.run_dir}")
    typer.echo(f"Data root used: {summary.data_root_used}")
    typer.echo(f"Base model: {summary.base_model_name}")
    typer.echo(f"Branches: {', '.join(summary.branches)}")
    typer.echo(f"Sample size requested/used: {summary.sample_size_requested}/{summary.sample_size_used}")
    typer.echo(f"Summary CSV: {summary.summary_table_path_csv}")
    typer.echo(f"Summary Parquet: {summary.summary_table_path_parquet}")
    typer.echo(f"Qualitative CSV: {summary.qualitative_path_csv}")
    typer.echo(f"Training records CSV: {summary.training_records_csv}")
    typer.echo(f"Evaluation mode: {summary.evaluation_mode}")


@app.command()
def first_experiment_compare(
    old_run_dir: str,
    new_run_dir: str,
    out_dir: str = "",
) -> None:
    """Compare first-experiment runs on corrected accuracy and parse-failure metrics."""
    artifacts = compare_first_experiment_runs(
        old_run_dir=Path(old_run_dir),
        new_run_dir=Path(new_run_dir),
        out_dir=Path(out_dir) if out_dir else None,
    )
    typer.echo("First experiment comparison export finished")
    typer.echo(f"CSV: {artifacts.csv_path}")
    typer.echo(f"Parquet: {artifacts.parquet_path}")


@app.command()
def baseline_series(
    config: str = "configs/baseline_series.yaml",
    sample_size: int = 100,
    seeds: str = "42,43",
) -> None:
    """Run multi-seed baseline series in inference_recycling_only mode."""
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "baseline_series_latest.log"
    setup_logging(log_file)

    summary = run_baseline_series(
        cfg=cfg,
        sample_size=sample_size,
        seeds=parse_seed_list(seeds),
    )

    typer.echo("Baseline series finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Seeds: {', '.join(str(s) for s in summary.seeds)}")
    typer.echo(f"Sample size: {summary.sample_size}")
    for seed, item in zip(summary.seeds, summary.runs):
        typer.echo(f"Run (seed={seed})")
        typer.echo(f" - run_dir: {item.run_dir}")
        typer.echo(f" - branches: {', '.join(item.branches)}")
        typer.echo(f" - generations: {', '.join(str(g) for g in item.generations)}")
        typer.echo(f" - summary: {item.summary_table_path_csv}")
    typer.echo(f"Series analysis dir: {summary.analysis_dir}")
    typer.echo(f"Run-level CSV: {summary.artifacts.run_level_csv}")
    typer.echo(f"Seed stats CSV: {summary.artifacts.seed_stats_csv}")
    typer.echo(f"Generation deltas CSV: {summary.artifacts.generation_deltas_csv}")
    typer.echo(f"Branch deltas CSV: {summary.artifacts.branch_deltas_csv}")
    typer.echo(f"Qualitative CSV: {summary.artifacts.qualitative_csv}")
    typer.echo(f"Accuracy plot: {summary.artifacts.accuracy_plot}")
    typer.echo(f"Pedagogical plot: {summary.artifacts.pedagogical_plot}")
    typer.echo("Evaluation mode: inference_recycling_only (not retraining)")


@app.command()
def baseline_series_analyze(
    run_dirs: list[str],
    out_dir: str = "",
) -> None:
    """Analyze already finished first-experiment runs as one baseline series."""
    if not run_dirs:
        raise typer.BadParameter("Provide at least one run_dir")
    out_path = Path(out_dir) if out_dir else Path(run_dirs[-1]) / "tables"
    artifacts = export_baseline_series_analysis(
        run_dirs=[Path(x) for x in run_dirs],
        out_dir=out_path,
    )
    typer.echo("Baseline series analysis export finished")
    typer.echo(f"Run-level CSV: {artifacts.run_level_csv}")
    typer.echo(f"Seed stats CSV: {artifacts.seed_stats_csv}")
    typer.echo(f"Generation deltas CSV: {artifacts.generation_deltas_csv}")
    typer.echo(f"Branch deltas CSV: {artifacts.branch_deltas_csv}")
    typer.echo(f"Qualitative CSV: {artifacts.qualitative_csv}")
    typer.echo(f"Accuracy plot: {artifacts.accuracy_plot}")
    typer.echo(f"Pedagogical plot: {artifacts.pedagogical_plot}")
    typer.echo("Evaluation mode: inference_recycling_only (not retraining)")


@app.command()
def pvf_confirmatory(
    config: str = "configs/pvf_confirmatory.yaml",
    sample_size: int = 50,
    seeds: str = "91,92,93",
) -> None:
    """Run multi-seed confirmatory PVF stress series with per-seed resume."""
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "pvf_confirmatory_latest.log"
    setup_logging(log_file)

    summary = run_pvf_confirmatory_series(
        cfg=cfg,
        sample_size=sample_size,
        seeds=parse_seed_list(seeds),
    )

    typer.echo("PVF confirmatory series finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Seeds: {', '.join(str(s) for s in summary.seeds)}")
    typer.echo(f"Sample size: {summary.sample_size}")
    for seed, item in zip(summary.seeds, summary.runs):
        typer.echo(f"Run (seed={seed})")
        typer.echo(f" - run_dir: {item.run_dir}")
        typer.echo(f" - branches: {', '.join(item.branches)}")
        typer.echo(f" - generations: {', '.join(str(g) for g in item.generations)}")
        typer.echo(f" - summary: {item.summary_table_path_csv}")
    typer.echo(f"Analysis dir: {summary.analysis_dir}")
    typer.echo(f"Run-level CSV: {summary.artifacts.run_level_csv}")
    typer.echo(f"Seed stats CSV: {summary.artifacts.seed_stats_csv}")
    typer.echo(f"Generation deltas CSV: {summary.artifacts.generation_deltas_csv}")
    typer.echo(f"Branch deltas CSV: {summary.artifacts.branch_deltas_csv}")
    typer.echo(f"Reject reasons CSV: {summary.artifacts.reject_reasons_csv}")
    typer.echo(f"Keep-vs-rejected CSV: {summary.artifacts.keep_reject_quality_csv}")
    typer.echo(f"Stress summary CSV: {summary.artifacts.stress_summary_csv}")
    typer.echo(f"Accuracy plot: {summary.artifacts.accuracy_plot}")
    typer.echo(f"Pedagogical plot: {summary.artifacts.pedagogical_plot}")
    typer.echo(f"Silent-error plot: {summary.artifacts.silent_error_plot}")
    typer.echo(f"Keep-rate plot: {summary.artifacts.keep_rate_plot}")
    typer.echo("Evaluation mode: inference_recycling_only (not full retraining)")


@app.command()
def pvf_confirmatory_analyze(
    run_dirs: list[str],
    out_dir: str = "",
) -> None:
    """Analyze finished runs as one PVF confirmatory series."""
    if not run_dirs:
        raise typer.BadParameter("Provide at least one run_dir")
    target_out = Path(out_dir) if out_dir else Path(run_dirs[-1]) / "tables"
    artifacts = export_pvf_confirmatory_analysis(
        run_dirs=[Path(x) for x in run_dirs],
        out_dir=target_out,
    )
    typer.echo("PVF confirmatory analysis export finished")
    typer.echo(f"Run-level CSV: {artifacts.run_level_csv}")
    typer.echo(f"Seed stats CSV: {artifacts.seed_stats_csv}")
    typer.echo(f"Generation deltas CSV: {artifacts.generation_deltas_csv}")
    typer.echo(f"Branch deltas CSV: {artifacts.branch_deltas_csv}")
    typer.echo(f"Reject reasons CSV: {artifacts.reject_reasons_csv}")
    typer.echo(f"Keep-vs-rejected CSV: {artifacts.keep_reject_quality_csv}")
    typer.echo(f"Stress summary CSV: {artifacts.stress_summary_csv}")
    typer.echo(f"Accuracy plot: {artifacts.accuracy_plot}")
    typer.echo(f"Pedagogical plot: {artifacts.pedagogical_plot}")
    typer.echo(f"Silent-error plot: {artifacts.silent_error_plot}")
    typer.echo(f"Keep-rate plot: {artifacts.keep_rate_plot}")


@app.command()
def train_feasibility_series(
    config: str = "configs/mode_compare_training.yaml",
    sample_size: int = 50,
    seeds: str = "61,62",
) -> None:
    """Run multi-seed training-feasibility series."""
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "train_feasibility_series_latest.log"
    setup_logging(log_file)

    from didactic_collapse.orchestration.baseline_series import parse_seed_list

    summary = run_training_feasibility_series(
        cfg=cfg,
        sample_size=sample_size,
        seeds=parse_seed_list(seeds),
    )
    typer.echo("Train-feasibility series finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Seeds: {', '.join(str(s) for s in summary.seeds)}")
    typer.echo(f"Sample size: {summary.sample_size}")
    for seed, item in zip(summary.seeds, summary.runs):
        typer.echo(f"Run (seed={seed})")
        typer.echo(f" - run_dir: {item.run_dir}")
        typer.echo(f" - branches: {', '.join(item.branches)}")
        typer.echo(f" - summary: {item.summary_table_path_csv}")
        typer.echo(f" - records: {item.training_records_csv}")
    typer.echo("Evaluation mode: training_recycling_feasibility (minimal backend, not full fine-tuning)")


@app.command()
def mode_compare_analyze(
    inference_run_dirs: str = typer.Option(
        ...,
        "--inference-run-dirs",
        help="Comma-separated inference run directories.",
    ),
    training_run_dirs: str = typer.Option(
        ...,
        "--training-run-dirs",
        help="Comma-separated training-feasibility run directories.",
    ),
    out_dir: str = "",
) -> None:
    """Compare inference_recycling_only vs training_recycling_feasibility runs."""
    inf_dirs = [x.strip() for x in inference_run_dirs.split(",") if x.strip()]
    trn_dirs = [x.strip() for x in training_run_dirs.split(",") if x.strip()]
    if not inf_dirs:
        raise typer.BadParameter("Provide at least one inference run_dir")
    if not trn_dirs:
        raise typer.BadParameter("Provide at least one training run_dir")

    target_out = Path(out_dir) if out_dir else Path(trn_dirs[-1]) / "tables"
    artifacts = export_mode_comparison_analysis(
        inference_run_dirs=[Path(x) for x in inf_dirs],
        training_run_dirs=[Path(x) for x in trn_dirs],
        out_dir=target_out,
    )
    typer.echo("Mode comparison analysis export finished")
    typer.echo(f"Run-level CSV: {artifacts.run_level_csv}")
    typer.echo(f"Seed stats CSV: {artifacts.seed_stats_csv}")
    typer.echo(f"Generation deltas CSV: {artifacts.generation_deltas_csv}")
    typer.echo(f"Branch deltas CSV: {artifacts.branch_deltas_csv}")
    typer.echo(f"Mode deltas CSV: {artifacts.mode_deltas_csv}")
    typer.echo(f"Qualitative CSV: {artifacts.qualitative_csv}")
    typer.echo(f"Accuracy plot: {artifacts.accuracy_plot}")
    typer.echo(f"Pedagogical plot: {artifacts.pedagogical_plot}")
    typer.echo("Modes: inference_recycling_only vs training_recycling_feasibility")


@app.command()
def anchoring_ablation_analyze(
    run_dir: str,
    out_dir: str = "",
) -> None:
    """Analyze one anchoring-ablation run (ratio + mixing mode deltas)."""
    run_path = Path(run_dir)
    target_out = Path(out_dir) if out_dir else (run_path / "tables")
    artifacts = export_anchoring_ablation_analysis(run_dir=run_path, out_dir=target_out)
    typer.echo("Anchoring ablation analysis export finished")
    typer.echo(f"Run-level CSV: {artifacts.run_level_csv}")
    typer.echo(f"Branch deltas CSV: {artifacts.branch_deltas_csv}")
    typer.echo(f"Mode deltas CSV: {artifacts.mode_deltas_csv}")
    typer.echo(f"Ratio deltas CSV: {artifacts.ratio_deltas_csv}")
    typer.echo(f"Anchor quality pairs CSV: {artifacts.anchor_quality_pairs_csv}")
    typer.echo(f"Anchor quality summary CSV: {artifacts.anchor_quality_summary_csv}")
    typer.echo(f"Qualitative CSV: {artifacts.qualitative_csv}")
    typer.echo("Evaluation mode: inference_recycling_only (not retraining)")


@app.command()
def judge_sensitivity_qwen(
    config: str = "configs/judge_sensitivity_qwen.yaml",
    run_dirs: str = "",
    sample_size: int = 48,
    sample_seed: int = 42,
    out_dir: str = "",
    skip_preflight: bool = False,
) -> None:
    """Re-judge a small balanced sample with Qwen3-235B on Cerebras."""
    if not run_dirs.strip():
        raise typer.BadParameter("run_dirs is required (comma-separated)")
    dirs = [x.strip() for x in run_dirs.split(",") if x.strip()]
    if not dirs:
        raise typer.BadParameter("No valid run_dirs parsed")

    cfg = load_config(config)
    provider = cfg.judge.provider.strip().lower()
    if provider != "cerebras":
        raise typer.BadParameter("judge_sensitivity_qwen currently requires provider=cerebras")

    if not skip_preflight:
        # Preflight checks via the same path used in pipeline/rejudge calls.
        _ = cerebras_judge_auth_smoke_check(
            model_name=cfg.judge.model_name,
            base_url=cfg.judge.base_url,
            api_key_env=cfg.judge.api_key_env,
            timeout_sec=cfg.judge.timeout_sec,
        )
        _ = cerebras_judge_rubric_format_check(
            model_name=cfg.judge.model_name,
            base_url=cfg.judge.base_url,
            api_key_env=cfg.judge.api_key_env,
            timeout_sec=cfg.judge.timeout_sec,
        )

    artifacts = run_qwen_judge_sensitivity(
        cfg=cfg,
        run_dirs=[Path(x) for x in dirs],
        sample_size=sample_size,
        sample_seed=sample_seed,
        out_dir=Path(out_dir) if out_dir else None,
    )
    typer.echo("Qwen judge sensitivity check finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Skip preflight: {skip_preflight}")
    typer.echo(f"Source run_dirs: {', '.join(dirs)}")
    typer.echo(f"Selected sample CSV: {artifacts.selected_sample_csv}")
    typer.echo(f"Qwen re-judge CSV: {artifacts.qwen_rejudge_results_csv}")
    typer.echo(f"Qwen failures CSV: {artifacts.qwen_rejudge_failures_csv}")
    typer.echo(f"Comparison CSV: {artifacts.comparison_csv}")
    typer.echo(f"Summary CSV: {artifacts.summary_csv}")
    typer.echo(f"Branch summary CSV: {artifacts.branch_summary_csv}")
    typer.echo(f"Seed-branch summary CSV: {artifacts.seed_branch_summary_csv}")
    typer.echo(f"Scatter plot: {artifacts.scatter_plot}")
    typer.echo(f"Delta histogram: {artifacts.delta_hist_plot}")
    typer.echo(f"Branch plot: {artifacts.branch_bar_plot}")


@app.command()
def judge_sensitivity_qwen_confirmatory(
    config: str = "configs/judge_sensitivity_qwen.yaml",
    run_dirs: str = "",
    sample_size: int = 48,
    sample_seed: int = 4242,
    focus_generation: int = 1,
    out_dir: str = "",
    skip_preflight: bool = False,
) -> None:
    """Focused Qwen3 sensitivity check for confirmatory training runs (seed×branch balanced, Gen1-first)."""
    if not run_dirs.strip():
        raise typer.BadParameter("run_dirs is required (comma-separated)")
    dirs = [x.strip() for x in run_dirs.split(",") if x.strip()]
    if not dirs:
        raise typer.BadParameter("No valid run_dirs parsed")

    cfg = load_config(config)
    provider = cfg.judge.provider.strip().lower()
    if provider != "cerebras":
        raise typer.BadParameter("judge_sensitivity_qwen_confirmatory requires provider=cerebras")

    if not skip_preflight:
        _ = cerebras_judge_auth_smoke_check(
            model_name=cfg.judge.model_name,
            base_url=cfg.judge.base_url,
            api_key_env=cfg.judge.api_key_env,
            timeout_sec=cfg.judge.timeout_sec,
        )
        _ = cerebras_judge_rubric_format_check(
            model_name=cfg.judge.model_name,
            base_url=cfg.judge.base_url,
            api_key_env=cfg.judge.api_key_env,
            timeout_sec=cfg.judge.timeout_sec,
        )

    artifacts = run_qwen_judge_sensitivity(
        cfg=cfg,
        run_dirs=[Path(x) for x in dirs],
        sample_size=sample_size,
        sample_seed=sample_seed,
        sampling_strategy="confirmatory_gen1_seed_branch",
        focus_generation=focus_generation,
        out_dir=Path(out_dir) if out_dir else None,
    )
    typer.echo("Qwen confirmatory judge sensitivity check finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Skip preflight: {skip_preflight}")
    typer.echo(f"Source run_dirs: {', '.join(dirs)}")
    typer.echo(f"Focus generation: {focus_generation}")
    typer.echo(f"Selected sample CSV: {artifacts.selected_sample_csv}")
    typer.echo(f"Qwen re-judge CSV: {artifacts.qwen_rejudge_results_csv}")
    typer.echo(f"Qwen failures CSV: {artifacts.qwen_rejudge_failures_csv}")
    typer.echo(f"Comparison CSV: {artifacts.comparison_csv}")
    typer.echo(f"Summary CSV: {artifacts.summary_csv}")
    typer.echo(f"Branch summary CSV: {artifacts.branch_summary_csv}")
    typer.echo(f"Seed-branch summary CSV: {artifacts.seed_branch_summary_csv}")
    typer.echo(f"Scatter plot: {artifacts.scatter_plot}")
    typer.echo(f"Delta histogram: {artifacts.delta_hist_plot}")
    typer.echo(f"Branch plot: {artifacts.branch_bar_plot}")


@app.command()
def judge_auth_check(config: str = "configs/first_experiment.yaml") -> None:
    """Preflight judge auth using the same client path as pipeline."""
    cfg = load_config(config)
    provider = cfg.judge.provider.strip().lower()
    if provider in {"gemini", "gemini_sdk", "gemini_openai_compatible"}:
        text = gemini_judge_auth_smoke_check(
            model_name=cfg.judge.model_name,
            api_key_env=cfg.judge.api_key_env,
        )
        typer.echo("Gemini judge auth check succeeded.")
        typer.echo(f"Response preview: {text[:120].replace(chr(10), ' ')}")
        return

    if provider == "cerebras":
        text = cerebras_judge_auth_smoke_check(
            model_name=cfg.judge.model_name,
            base_url=cfg.judge.base_url,
            api_key_env=cfg.judge.api_key_env,
            timeout_sec=cfg.judge.timeout_sec,
        )
        typer.echo("Cerebras judge auth check succeeded.")
        typer.echo(f"Response preview: {text[:120].replace(chr(10), ' ')}")
        return

    if provider in {"mock", "stub", "mock_judge"}:
        raise typer.BadParameter(
            "judge_auth_check requires a real provider and does not support mock/stub."
        )
    raise typer.BadParameter(
        "judge_auth_check currently supports providers: cerebras, gemini, gemini_sdk."
    )


@app.command()
def judge_rubric_check(config: str = "configs/first_experiment.yaml") -> None:
    """Validate rubric-format readiness using the same parser as judge stage."""
    cfg = load_config(config)
    provider = cfg.judge.provider.strip().lower()
    if provider != "cerebras":
        raise typer.BadParameter(
            "judge_rubric_check currently supports provider=cerebras only."
        )
    score = cerebras_judge_rubric_format_check(
        model_name=cfg.judge.model_name,
        base_url=cfg.judge.base_url,
        api_key_env=cfg.judge.api_key_env,
        timeout_sec=cfg.judge.timeout_sec,
    )
    typer.echo("Cerebras judge rubric-format check succeeded.")
    typer.echo(f"Rubric sample: {score}")


@app.command()
def judge_pairwise_sensitivity(
    config: str = "configs/judge_pairwise_sensitivity.yaml",
    run_dirs: str = "",
    sample_size: int = 48,
    sample_seed: int = 4242,
    generation: int = 1,
    llama_model_name: str = "llama-3.1-8b",
    qwen_model_name: str = "qwen-3-235b-a22b-instruct-2507",
    out_dir: str = "",
    skip_preflight: bool = False,
) -> None:
    """Blinded pairwise judge sensitivity check (pure vs anchor_20_append) on existing run artifacts."""
    if not run_dirs.strip():
        raise typer.BadParameter("run_dirs is required (comma-separated)")
    dirs = [x.strip() for x in run_dirs.split(",") if x.strip()]
    if not dirs:
        raise typer.BadParameter("No valid run_dirs parsed")

    cfg = load_config(config)
    provider = cfg.judge.provider.strip().lower()
    if provider != "cerebras":
        raise typer.BadParameter("judge_pairwise_sensitivity currently requires provider=cerebras")

    if not skip_preflight:
        _ = cerebras_judge_auth_smoke_check(
            model_name=llama_model_name,
            base_url=cfg.judge.base_url,
            api_key_env=cfg.judge.api_key_env,
            timeout_sec=cfg.judge.timeout_sec,
        )
        _ = cerebras_judge_auth_smoke_check(
            model_name=qwen_model_name,
            base_url=cfg.judge.base_url,
            api_key_env=cfg.judge.api_key_env,
            timeout_sec=cfg.judge.timeout_sec,
        )

    artifacts = run_pairwise_judge_sensitivity(
        cfg=cfg,
        run_dirs=[Path(x) for x in dirs],
        sample_size=sample_size,
        sample_seed=sample_seed,
        generation=generation,
        llama_model_name=llama_model_name,
        qwen_model_name=qwen_model_name,
        out_dir=Path(out_dir) if out_dir else None,
    )
    typer.echo("Pairwise judge sensitivity check finished")
    typer.echo(f"Config: {config}")
    typer.echo(f"Skip preflight: {skip_preflight}")
    typer.echo(f"Source run_dirs: {', '.join(dirs)}")
    typer.echo(f"Generation: {generation}")
    typer.echo(f"Selected pairs CSV: {artifacts.selected_pairs_csv}")
    typer.echo(f"Hidden key CSV: {artifacts.hidden_key_csv}")
    typer.echo(f"Manual audit CSV: {artifacts.manual_audit_template_csv}")
    typer.echo(f"Manual audit XLSX: {artifacts.manual_audit_template_xlsx}")
    typer.echo(f"Llama results CSV: {artifacts.llama_results_csv}")
    typer.echo(f"Qwen results CSV: {artifacts.qwen_results_csv}")
    typer.echo(f"Comparison CSV: {artifacts.comparison_csv}")
    typer.echo(f"Summary CSV: {artifacts.summary_csv}")
    typer.echo(f"Seed summary CSV: {artifacts.seed_branch_summary_csv}")


if __name__ == "__main__":
    app()
