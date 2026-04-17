from __future__ import annotations

from pathlib import Path

import typer

from didactic_collapse.config.settings import load_config
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
def stage(stage_name: str, config: str = "configs/experiment.yaml") -> None:
    """Run a single stage (placeholder for incremental execution)."""
    cfg = load_config(config)
    log_file = Path(cfg.paths.output_root) / "runs" / "latest.log"
    setup_logging(log_file)
    _ = ExperimentRunner(cfg)
    raise NotImplementedError(f"Stage runner is not implemented yet: {stage_name}")


if __name__ == "__main__":
    app()
