# Artifact Integrity Audit Report

- Generated at: 2026-04-28T11:23:48.034889
- Scope: audit only (no rerun of generation/judge/training)

## Severity Summary
- CRITICAL: 0
- HIGH: 14
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
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed71_20260425_141959\qwen2.5_0.5b__anchor_20_append__ft_g1_s71\anchor_20_append\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "synthetic_build", "status": "pending", "model_dir": "qwen2.5_0.5b__anchor_20_append__ft_g1_s71", "branch": "anchor_20_append", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed71_20260425_141959\qwen2.5_0.5b__anchor_20_append__ft_g1_s71\anchor_20_append\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "anchoring", "status": "pending", "model_dir": "qwen2.5_0.5b__anchor_20_append__ft_g1_s71", "branch": "anchor_20_append", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed71_20260425_141959\qwen2.5_0.5b__pure_recycling__ft_g1_s71\pure_recycling\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "synthetic_build", "status": "pending", "model_dir": "qwen2.5_0.5b__pure_recycling__ft_g1_s71", "branch": "pure_recycling", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed71_20260425_141959\qwen2.5_0.5b__pure_recycling__ft_g1_s71\pure_recycling\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "anchoring", "status": "pending", "model_dir": "qwen2.5_0.5b__pure_recycling__ft_g1_s71", "branch": "pure_recycling", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed72_20260426_101349\qwen2.5_0.5b__anchor_20_append__ft_g1_s72\anchor_20_append\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "synthetic_build", "status": "pending", "model_dir": "qwen2.5_0.5b__anchor_20_append__ft_g1_s72", "branch": "anchor_20_append", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed72_20260426_101349\qwen2.5_0.5b__anchor_20_append__ft_g1_s72\anchor_20_append\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "anchoring", "status": "pending", "model_dir": "qwen2.5_0.5b__anchor_20_append__ft_g1_s72", "branch": "anchor_20_append", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed72_20260426_101349\qwen2.5_0.5b__pure_recycling__ft_g1_s72\pure_recycling\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "synthetic_build", "status": "pending", "model_dir": "qwen2.5_0.5b__pure_recycling__ft_g1_s72", "branch": "pure_recycling", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed72_20260426_101349\qwen2.5_0.5b__pure_recycling__ft_g1_s72\pure_recycling\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "anchoring", "status": "pending", "model_dir": "qwen2.5_0.5b__pure_recycling__ft_g1_s72", "branch": "pure_recycling", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed73_20260426_103916\qwen2.5_0.5b__anchor_20_append__ft_g1_s73\anchor_20_append\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "synthetic_build", "status": "pending", "model_dir": "qwen2.5_0.5b__anchor_20_append__ft_g1_s73", "branch": "anchor_20_append", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed73_20260426_103916\qwen2.5_0.5b__anchor_20_append__ft_g1_s73\anchor_20_append\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "anchoring", "status": "pending", "model_dir": "qwen2.5_0.5b__anchor_20_append__ft_g1_s73", "branch": "anchor_20_append", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed73_20260426_103916\qwen2.5_0.5b__pure_recycling__ft_g1_s73\pure_recycling\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "synthetic_build", "status": "pending", "model_dir": "qwen2.5_0.5b__pure_recycling__ft_g1_s73", "branch": "pure_recycling", "generation": 1}
- [HIGH] manifest :: C:\Users\kiris\Documents\New project\outputs\runs\training_confirmatory_anchor20_seed73_20260426_103916\qwen2.5_0.5b__pure_recycling__ft_g1_s73\pure_recycling\gen_1\stage_manifest.json :: Context stage not completed
  - details: {"stage_name": "anchoring", "status": "pending", "model_dir": "qwen2.5_0.5b__pure_recycling__ft_g1_s73", "branch": "pure_recycling", "generation": 1}

## Integrity Verdict
- Key claims are NOT fully reliable until CRITICAL/HIGH issues are resolved or excluded.
- Evaluation mode should remain explicitly marked as inference_recycling_only / feasibility where applicable.
