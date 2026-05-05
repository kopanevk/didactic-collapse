from __future__ import annotations

import argparse
from pathlib import Path

from didactic_collapse.analysis.dbr_article_evidence import export_dbr_article_evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export article-ready DBR evidence tables from completed run artifacts."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Completed DBR confirmatory run directory. Pass once per seed.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory (default: outputs/dbr_confirmatory/dbr_article_evidence_<timestamp>).",
    )
    parser.add_argument(
        "--manual-sample-size",
        type=int,
        default=36,
        help="Blinded manual audit pair sample size (default: 36).",
    )
    parser.add_argument(
        "--manual-sample-seed",
        type=int,
        default=4242,
        help="Sampling seed for manual audit pairs (default: 4242).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dirs = [Path(x).resolve() for x in args.run_dir]
    artifacts = export_dbr_article_evidence(
        run_dirs=run_dirs,
        out_dir=args.out_dir.resolve() if args.out_dir else None,
        manual_sample_size=int(args.manual_sample_size),
        manual_sample_seed=int(args.manual_sample_seed),
    )
    print("DBR article evidence export finished.")
    print(f"Out dir: {artifacts.out_dir}")
    print(f"Collapse by seed: {artifacts.collapse_by_seed_csv}")
    print(f"Collapse summary: {artifacts.collapse_summary_csv}")
    print(f"Generation curves: {artifacts.generation_curves_csv}")
    print(f"Mechanism defects: {artifacts.mechanism_defect_before_after_csv}")
    print(f"Mechanism budget violations: {artifacts.mechanism_budget_violations_csv}")
    print(f"Mechanism selection rate: {artifacts.mechanism_selection_rate_csv}")
    print(f"Mechanism bucket coverage: {artifacts.mechanism_bucket_coverage_csv}")
    print(f"Mechanism severity distribution: {artifacts.mechanism_severity_distribution_csv}")
    print(f"Manual audit CSV: {artifacts.manual_audit_template_csv}")
    print(f"Manual audit XLSX: {artifacts.manual_audit_template_xlsx}")
    print(f"Manual hidden key: {artifacts.manual_audit_hidden_key_csv}")
    print(f"Integrity report: {artifacts.integrity_report_json}")


if __name__ == "__main__":
    main()

