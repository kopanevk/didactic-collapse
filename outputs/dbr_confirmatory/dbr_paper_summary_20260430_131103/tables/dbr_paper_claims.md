# DBR Confirmatory Paper-Ready Summary

Evaluation mode: `inference_recycling_only` (not full retraining).

## Runs
- seed 211: `outputs\runs\dbr_confirmatory_seed211_20260428_222918`
- seed 212: `outputs\runs\dbr_confirmatory_seed212_20260428_230709`
- seed 213: `outputs\runs\dbr_confirmatory_seed213_20260430_115105`

## Gen2 Pure vs DBR by Seed
- seed 211: accuracy 0.5200 -> 0.4600 (? -0.0600); pedagogy 5.1400 -> 4.9800 (? -0.1600); silent 0.3000 -> 0.2800 (? -0.0200).
- seed 212: accuracy 0.3125 -> 0.4000 (? +0.0875); pedagogy 5.0417 -> 4.7000 (? -0.3417); silent 0.2708 -> 0.2200 (? -0.0508).
- seed 213: accuracy 0.4200 -> 0.5800 (? +0.1600); pedagogy 4.4800 -> 5.4200 (? +0.9400); silent 0.3000 -> 0.1800 (? -0.1200).

## Delta(dbr - pure), Gen2, n=3 seeds
- accuracy: mean +0.0625, std 0.1121, 95% CI [-0.2160, +0.3410]
- pedagogical_score: mean +0.1461, std 0.6935, 95% CI [-1.5766, +1.8689]
- silent_error_rate: mean -0.0636, std 0.0512, 95% CI [-0.1908, +0.0636]

## Matched Gen2 comparison (common example_id)
- seed 211 matched=50 / pure=50 / dbr=50: ?acc -0.0600, ?ped -0.1600, ?silent -0.0200.
- seed 212 matched=48 / pure=48 / dbr=50: ?acc +0.0833, ?ped -0.3125, ?silent -0.0417.
- seed 213 matched=50 / pure=50 / dbr=50: ?acc +0.1600, ?ped +0.9400, ?silent -0.1200.

## DBR Selection Rate by Generation
- Gen0: selection_rate mean 0.9800 (std 0.0000, 95% CI [0.9800, 0.9800]), selected_count_mean 49.00/50.00.
- Gen1: selection_rate mean 0.9733 (std 0.0115, 95% CI [0.9446, 1.0020]), selected_count_mean 48.67/50.00.
- Gen2: selection_rate mean 0.9733 (std 0.0115, 95% CI [0.9446, 1.0020]), selected_count_mean 48.67/50.00.

## Defect Rates Before/After + Budgets
- `parse_failure` stayed at zero after selection in all generations (mean violation_count=0).
- Budget relaxations were used in all generations: relax_low_structure -> relax_low_reasoning -> relax_incorrect_answer -> relax_silent_error.
- Persistent violations remain mostly in `incorrect_answer`, `low_reasoning`, and `silent_error` when filling target size.

## Comparison vs soft_pvf_noisy_keep (Gen2)
- soft_pvf_noisy_keep branch available in all three seeds.
- DBR - soft (mean across seeds): ?acc +0.0533, ?ped +0.1933, ?silent -0.0733.

## Final Verdict
- Verdict label: `mixed_result_secondary_baseline`.
- Interpretation: DBR shows consistent silent-error reduction and small positive accuracy shift on average, but pedagogical gains are not stable across seeds (1/3 positive only).
- Recommendation for paper framing: **secondary baseline / mixed result**, not main method yet.

## Claim Checklist
- Pedagogy >= pure on Gen2 in >=2/3 seeds: 1/3 (not met).
- Silent_error <= pure on Gen2 in >=2/3 seeds: 3/3 (met).
- Accuracy >= pure - 0.05 on Gen2 in >=2/3 seeds: 2/3 (met).