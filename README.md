# Didactic Collapse

Research pipeline for studying degradation of pedagogical quality under iterative training on synthetic data, with Human Anchoring interventions.

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

```bash
dc-run full --config configs/experiment.yaml
```

```bash
dc-run pilot --config configs/pilot.yaml --sample-size 30 --dry-run true
```

```bash
dc-run first-experiment --config configs/first_experiment.yaml --sample-size 50
```

```bash
dc-run judge-auth-check --config configs/first_experiment.yaml
dc-run first-experiment-resume --config configs/first_experiment.yaml --run-dir outputs/runs/first_real_experiment_YYYYMMDD_HHMMSS
```

### Cerebras judge auth

```bash
# Windows PowerShell
$env:CEREBRAS_API_KEY="your_key_here"
```

Judge auth smoke-check (uses the same client path as pipeline):

```bash
dc-run judge-auth-check --config configs/first_experiment.yaml
dc-run judge-rubric-check --config configs/first_experiment.yaml
```

Run or resume first experiment:

```bash
dc-run first-experiment --config configs/first_experiment.yaml --sample-size 50
dc-run first-experiment-resume --config configs/first_experiment.yaml --run-dir outputs/runs/first_real_experiment_YYYYMMDD_HHMMSS
```

Improved-generation small real run (strict final answer format):

```bash
dc-run first-experiment --config configs/first_experiment_v2.yaml --sample-size 30
```

Second real run preset (more informative signal, larger sample):

```bash
dc-run first-experiment --config configs/second_real_experiment.yaml --sample-size 100
```

Baseline presets for Human Anchoring sweeps:

```bash
dc-run first-experiment --config configs/baseline_anchor10.yaml --sample-size 100
dc-run first-experiment --config configs/baseline_anchor20.yaml --sample-size 100
```

Multi-seed baseline series (pure_recycling + anchor_10 + anchor_20):

```bash
dc-run baseline-series --config configs/baseline_series.yaml --sample-size 100 --seeds 42,43
```

Gen-2 baseline trajectory mode is enabled in baseline presets (`generations: 3`, i.e. Gen-0/1/2).

Re-analyze an already finished set of runs:

```bash
dc-run baseline-series-analyze outputs/runs/<run_1> outputs/runs/<run_2> --out-dir outputs/baseline_series/<tag>/tables
```

Train-stage feasibility (Gen-0 -> train -> Gen-1) with explicit mode:

```bash
dc-run train-feasibility --config configs/train_feasibility.yaml --sample-size 50
dc-run train-feasibility --config configs/train_feasibility.yaml --sample-size 50 --run-dir outputs/runs/train_feasibility_YYYYMMDD_HHMMSS
```

`configs/train_feasibility.yaml` is set to `experiment.mode: training_recycling_feasibility`.
By default it uses `training.backend: command` and calls `scripts/train_feasibility_adapter.py`.
That adapter is scaffolded and fails fast until you wire a real fine-tune routine.
For non-scientific smoke checks only, you can pass `--stub` in adapter command and set `training.allow_stub_for_smoke: true`.

Old vs new run comparison export:

```bash
dc-run first-experiment-compare \
  outputs/runs/first_real_experiment_20260417_191914 \
  outputs/runs/second_real_experiment_YYYYMMDD_HHMMSS
```

Expected first-experiment outputs:

- `outputs/runs/<run_id>/tables/first_experiment_summary.csv`
- `outputs/runs/<run_id>/tables/first_experiment_summary.parquet`
- `outputs/runs/<run_id>/tables/qualitative_silent_error_candidates.csv`
- `outputs/runs/<run_id>/tables/qualitative_silent_error_candidates.parquet`
- `outputs/runs/<run_id>/tables/qualitative_silent_error_candidates.meta.json`
- `outputs/runs/<run_id>/tables/metrics_by_generation.csv`
- `outputs/runs/<run_id>/figures/accuracy_vs_generation.png`
- `outputs/runs/<run_id>/figures/pedagogical_vs_generation.png`
- `outputs/runs/<run_id>/figures/silent_error_vs_generation.png`

Note: for real first-experiment runs, judge auth is validated before expensive stages. If `CEREBRAS_API_KEY` is missing, run fails fast before generation.
Note: baseline-series outputs are explicitly marked as `inference_recycling_only` and are not equivalent to full retraining dynamics.

## Core ideas

- Branches: `pure_recycling`, `anchor_5`, `anchor_10`
- Generations: `gen0 -> gen1 -> gen2 -> gen3`
- Metrics: accuracy, pedagogical score, silent error rate
- Backends: local generation via Ollama, judge via pluggable API client
