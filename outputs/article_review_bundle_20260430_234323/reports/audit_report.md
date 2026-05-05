# Artifact Integrity Audit Report

- Generated at: 2026-04-30T14:40:15.761695
- Scope: audit only (no rerun of generation/judge/training)

## Severity Summary
- CRITICAL: 2
- HIGH: 0
- MEDIUM: 0
- LOW: 0

## Table Stats
- run manifest rows: 171
- row-count rows: 27
- summary checks rows: 108
- analysis-source checks rows: 0
- pairwise checks rows: 10
- pvf/soft checks rows: 45

## Findings
- [CRITICAL] summary :: C:\Users\kiris\Documents\New project\outputs\runs\dbr_confirmatory_seed211_20260428_222918 :: Summary metric mismatch: pedagogical_score_mean
  - details: {"run_dir": "C:\\Users\\kiris\\Documents\\New project\\outputs\\runs\\dbr_confirmatory_seed211_20260428_222918", "model_name": "qwen2.5:0.5b", "branch": "dbr_medium", "generation": 0.0, "summary_value": 5.44, "recomputed_value": 5.38, "abs_diff": 0.0600000000000005}
- [CRITICAL] summary :: C:\Users\kiris\Documents\New project\outputs\runs\dbr_confirmatory_seed211_20260428_222918 :: Summary metric mismatch: silent_error_rate
  - details: {"run_dir": "C:\\Users\\kiris\\Documents\\New project\\outputs\\runs\\dbr_confirmatory_seed211_20260428_222918", "model_name": "qwen2.5:0.5b", "branch": "dbr_medium", "generation": 0.0, "summary_value": 0.3, "recomputed_value": 0.28, "abs_diff": 0.019999999999999962}

## Integrity Verdict
- Key claims are NOT fully reliable until CRITICAL/HIGH issues are resolved or excluded.
- Evaluation mode should remain explicitly marked as inference_recycling_only / feasibility where applicable.
