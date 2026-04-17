from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from didactic_collapse.analysis.aggregate import aggregate_metrics
from didactic_collapse.analysis.plots import plot_metric_by_generation
from didactic_collapse.clients.judge_client import OpenAICompatibleJudgeClient
from didactic_collapse.clients.ollama_client import OllamaClient
from didactic_collapse.config.settings import AppConfig
from didactic_collapse.eval.compute_metrics import compute_eval_table
from didactic_collapse.pipeline.build_synthetic_dataset import build_next_generation_train_set
from didactic_collapse.pipeline.generate_outputs import run_generation
from didactic_collapse.pipeline.judge_outputs import run_judging
from didactic_collapse.prompts.prompt_registry import load_judge_prompt
from didactic_collapse.utils.io_utils import save_tabular

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path


class ExperimentRunner:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ctx = RunContext(
            run_id=f"{cfg.project.name}_{cfg.project.run_tag}_{ts}",
            run_dir=cfg.paths.output_root / "runs" / f"{cfg.project.run_tag}_{ts}",
        )

    def save_run_metadata(self) -> None:
        self.ctx.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.ctx.run_id,
            "created_at": datetime.now().isoformat(),
            "config": self.cfg.model_dump(mode="json"),
        }
        (self.ctx.run_dir / "run_config.snapshot.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run_full(self) -> None:
        self.save_run_metadata()
        judge_prompt = load_judge_prompt(self.cfg.paths.prompt_dir)

        gen_client = OllamaClient()
        judge_client = OpenAICompatibleJudgeClient(
            base_url=self.cfg.judge.base_url,
            model_name=self.cfg.judge.model_name,
            api_key_env=self.cfg.judge.api_key_env,
            timeout_sec=self.cfg.judge.timeout_sec,
        )

        heldout = pd.read_parquet(self.cfg.paths.data_root / "splits" / "heldout_test.parquet")
        anchor_pool = pd.read_parquet(self.cfg.paths.data_root / "splits" / "anchor_pool.parquet")

        all_eval_rows: list[pd.DataFrame] = []
        for model in self.cfg.models.local_models:
            for branch in self.cfg.experiment.branches:
                for gen in range(1, self.cfg.experiment.generations + 1):
                    step_dir = self.ctx.run_dir / model.name.replace(":", "_") / branch.name / f"gen_{gen}"
                    step_dir.mkdir(parents=True, exist_ok=True)

                    outputs_path = step_dir / "model_outputs.parquet"
                    judge_path = step_dir / "judge_outputs.parquet"
                    eval_path = step_dir / "eval_merged.parquet"
                    synth_path = step_dir / "synthetic_train_next.parquet"

                    if outputs_path.exists() and not self.cfg.runtime.force_recompute:
                        out_df = pd.read_parquet(outputs_path)
                    else:
                        out_df = run_generation(
                            client=gen_client,
                            examples_df=heldout,
                            model_name=model.name,
                            branch=branch.name,
                            generation=gen,
                            run_id=self.ctx.run_id,
                            prompt_version="v1",
                            temperature=self.cfg.sampling.temperature,
                            top_p=self.cfg.sampling.top_p,
                            max_tokens=self.cfg.sampling.max_tokens,
                            out_path=outputs_path,
                        )

                    if judge_path.exists() and not self.cfg.runtime.force_recompute:
                        judge_df = pd.read_parquet(judge_path)
                    else:
                        judge_df = run_judging(
                            client=judge_client,
                            generations_df=out_df,
                            questions_df=heldout,
                            judge_provider=self.cfg.judge.provider,
                            judge_model=self.cfg.judge.model_name,
                            rubric_prompt=judge_prompt,
                            out_path=judge_path,
                        )

                    eval_df = compute_eval_table(
                        outputs_df=out_df,
                        judge_df=judge_df,
                        gold_df=heldout,
                        out_path=eval_path,
                    )
                    all_eval_rows.append(eval_df)

                    out_for_training = out_df.merge(heldout[["example_id", "question"]], on="example_id", how="left")
                    _ = build_next_generation_train_set(
                        synthetic_outputs_df=out_for_training,
                        anchor_pool_df=anchor_pool,
                        anchor_ratio=branch.anchor_ratio,
                        seed=self.cfg.project.seed + gen,
                        out_path=synth_path,
                    )

                    logger.info("Finished model=%s branch=%s gen=%d", model.name, branch.name, gen)

        combined = pd.concat(all_eval_rows, ignore_index=True)
        save_tabular(
            combined,
            self.ctx.run_dir / "all_eval_merged",
            save_csv=self.cfg.runtime.save_csv,
            save_parquet=self.cfg.runtime.save_parquet,
        )

        agg = aggregate_metrics(combined, self.ctx.run_dir / "tables" / "metrics_by_generation.csv")
        plot_metric_by_generation(agg, "accuracy", self.ctx.run_dir / "figures" / "accuracy_vs_generation.png")
        plot_metric_by_generation(
            agg,
            "pedagogical_score_mean",
            self.ctx.run_dir / "figures" / "pedagogical_vs_generation.png",
        )
        plot_metric_by_generation(
            agg,
            "silent_error_rate",
            self.ctx.run_dir / "figures" / "silent_error_vs_generation.png",
        )

    # TODO: add run_stage(stage_name: str), resume_from_checkpoint(), and per-step locks.
