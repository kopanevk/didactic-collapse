from __future__ import annotations

import argparse
from pathlib import Path

from didactic_collapse.analysis.dbr_focused_verification import (
    recompute_gen2_deltas_by_seed,
    recompute_gen2_metrics,
    run_dbr_recompute_audit,
    run_qwen_dbr_pairwise_sensitivity,
)
from didactic_collapse.config.settings import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused DBR verification: recompute audit + optional Qwen3 pairwise sensitivity."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/judge_sensitivity_qwen.yaml",
        help="Config with cerebras/qwen judge settings for pairwise sensitivity.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="DBR confirmatory run dir. Provide 3 times for seeds 211/212/213.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=48,
        help="Pairwise sample size (default: 48).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=4242,
        help="Deterministic sampling seed for pair selection/blinding.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Outputs root path (default: outputs).",
    )
    parser.add_argument(
        "--skip-pairwise",
        action="store_true",
        help="Run only Part A audit and skip Qwen pairwise Part B.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dirs = [Path(x).resolve() for x in args.run_dir]
    outputs_root = args.outputs_root.resolve()

    print("Running Part A: DBR recompute audit...")
    audit = run_dbr_recompute_audit(run_dirs=run_dirs, outputs_root=outputs_root)
    print(f"Audit out: {audit.out_dir}")
    print(f"Audit report: {audit.report_md}")
    print(f"Audit findings: {audit.findings_json}")
    print(f"Audit table comparison: {audit.table_comparison_csv}")

    if args.skip_pairwise:
        print("Skipping Part B by --skip-pairwise.")
        return

    if audit.has_blocking_findings:
        print("Part A found CRITICAL/HIGH issues. Stopping before Part B.")
        return

    cfg = load_config(args.config)
    deltas = recompute_gen2_deltas_by_seed(recompute_gen2_metrics(run_dirs))
    llama_agg_ped_delta = float(deltas["delta_pedagogy_dbr_minus_pure"].mean())

    print("Running Part B: Qwen DBR pairwise sensitivity...")
    pairwise = run_qwen_dbr_pairwise_sensitivity(
        cfg=cfg,
        run_dirs=run_dirs,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        out_dir=None,
        llama_aggregate_pedagogy_delta=llama_agg_ped_delta,
    )
    print(f"Pairwise out: {pairwise.out_dir}")
    print(f"Selected pairs: {pairwise.selected_pairs_csv}")
    print(f"Hidden key: {pairwise.hidden_key_csv}")
    print(f"Pairwise results: {pairwise.pairwise_results_csv}")
    print(f"Pairwise comparison: {pairwise.pairwise_comparison_csv}")
    print(f"Pairwise summary: {pairwise.pairwise_summary_csv}")
    print(f"Seed summary: {pairwise.seed_summary_csv}")


if __name__ == "__main__":
    main()

