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

## Core ideas

- Branches: `pure_recycling`, `anchor_5`, `anchor_10`
- Generations: `gen0 -> gen1 -> gen2 -> gen3`
- Metrics: accuracy, pedagogical score, silent error rate
- Backends: local generation via Ollama, judge via pluggable API client
