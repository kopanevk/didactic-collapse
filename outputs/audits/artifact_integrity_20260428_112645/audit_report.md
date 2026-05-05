# Artifact Integrity Audit Report

- Generated at: 2026-04-28T11:26:45.825577
- Scope: audit only (no rerun of generation/judge/training)

## Severity Summary
- CRITICAL: 0
- HIGH: 2
- MEDIUM: 0
- LOW: 0

## Table Stats
- run manifest rows: 780
- row-count rows: 123
- summary checks rows: 444
- analysis-source checks rows: 9
- pairwise checks rows: 10
- pvf/soft checks rows: 45

## Findings
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\anchoring_confirmatory_append_seed53_20260423_221619\qwen2.5_0.5b\pure_recycling\gen_0\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "synthetic_build", "status": "pending", "model_dir": "qwen2.5_0.5b", "branch": "pure_recycling", "generation": 0}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\anchoring_confirmatory_append_seed53_20260423_221619\qwen2.5_0.5b\pure_recycling\gen_0\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "anchoring", "status": "pending", "model_dir": "qwen2.5_0.5b", "branch": "pure_recycling", "generation": 0}

## Integrity Verdict
- Key claims are NOT fully reliable until CRITICAL/HIGH issues are resolved or excluded.
- Evaluation mode should remain explicitly marked as inference_recycling_only / feasibility where applicable.
