from __future__ import annotations

from pathlib import Path

import typer

from didactic_collapse.clients.judge_client import gemini_judge_auth_smoke_check
from didactic_collapse.config.settings import load_config
from didactic_collapse.orchestration.first_experiment import run_first_experiment, resume_first_experiment
from didactic_collapse.orchestration.pilot import run_pilot
from didactic_collapse.orchestration.runner import ExperimentRunner
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
def judge_auth_check(config: str = "configs/first_experiment.yaml") -> None:
    """Preflight Gemini judge auth using the same SDK auth path as pipeline."""
    cfg = load_config(config)
    provider = cfg.judge.provider.strip().lower()
    if provider not in {"gemini", "gemini_sdk", "gemini_openai_compatible"}:
        raise typer.BadParameter(
            f"judge_auth_check supports Gemini providers only. Current provider: {cfg.judge.provider}"
        )

    text = gemini_judge_auth_smoke_check(
        model_name=cfg.judge.model_name,
        api_key_env=cfg.judge.api_key_env,
    )
    typer.echo("Gemini judge auth check succeeded.")
    typer.echo(f"Response preview: {text[:120].replace(chr(10), ' ')}")


if __name__ == "__main__":
    app()
