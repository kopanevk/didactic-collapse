from __future__ import annotations

import argparse
from pathlib import Path

from didactic_collapse.analysis.artifact_integrity_audit import (
    discover_default_targets,
    run_artifact_integrity_audit,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run integrity audit over existing didactic_collapse run/analysis artifacts."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Outputs root directory (default: outputs).",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Optional explicit run directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--analysis-dir",
        action="append",
        default=[],
        help="Optional explicit analysis directory. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs_root = args.outputs_root.resolve()
    run_dirs = [Path(p).resolve() for p in args.run_dir] if args.run_dir else None
    analysis_dirs = [Path(p).resolve() for p in args.analysis_dir] if args.analysis_dir else None

    if run_dirs is None or analysis_dirs is None:
        defaults = discover_default_targets(outputs_root)
        print("Discovered default targets:")
        for run_dir in defaults["run_dirs"]:
            print(f"  run: {run_dir}")
        for analysis_dir in defaults["analysis_dirs"]:
            print(f"  analysis: {analysis_dir}")

    artifacts = run_artifact_integrity_audit(
        outputs_root=outputs_root,
        run_dirs=run_dirs,
        analysis_dirs=analysis_dirs,
    )
    print(f"Audit output directory: {artifacts.out_dir}")
    print(f"Report: {artifacts.audit_report_md}")
    print(f"Findings JSON: {artifacts.audit_findings_json}")


if __name__ == "__main__":
    main()

