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

## Core ideas

- Branches: `pure_recycling`, `anchor_5`, `anchor_10`
- Generations: `gen0 -> gen1 -> gen2 -> gen3`
- Metrics: accuracy, pedagogical score, silent error rate
- Backends: local generation via Ollama, judge via pluggable API client
