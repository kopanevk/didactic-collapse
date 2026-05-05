# Article Numbers Cheatsheet

1. DBR-only Gen2 delta (dbr - pure) accuracy mean: 0.0333
2. DBR-only Gen2 delta (dbr - pure) pedagogy mean: 0.0200
3. DBR-only Gen2 delta (dbr - pure) silent mean: 0.0267
4. DBR mean selection_rate (all families): 0.9706
5. DBR parse_failure after-rate mean: 0.0000
6. Qwen DBR pairwise win rate: 0.3333
7. Qwen pure pairwise win rate: 0.2051
8. Qwen tie rate: 0.4615
9. CSR k=3 mean pair_construction_rate: 0.5111
10. CSR k=5 mean pair_construction_rate: 0.5889
11. CSR k=3 mean quality_gap: 5.3968
12. CSR k=5 mean quality_gap: 5.8029
13. Legacy caveat: artifact_integrity latest reports 2 CRITICAL summary mismatches in seed211/gen0/dbr.
14. Focused DBR recompute audit reports 0 CRITICAL / 0 HIGH findings.
15. Recommended framing: DBR = strongest practical baseline; effect mixed but reproducible with explicit caveats.

## Abstract-ready candidates
- Report DBR Gen2 deltas with CI from recompute-validated tables.
- Report selection_rate and parse-failure suppression as mechanism evidence.
- Report Qwen pairwise as sensitivity appendix (partial support, mixed by seed).

## Required caveats
- Inference recycling only (no full retraining claim).
- Seed sensitivity and wide CI.
- Legacy summary mismatch excluded in favor of recompute-validated numbers.
