from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from didactic_collapse.judging.accuracy import score_prediction


def _context_from_path(path: Path) -> tuple[str, int]:
    branch = path.parent.parent.name
    generation = int(path.parent.name.split("_", maxsplit=1)[1])
    return branch, generation


def _response_fingerprint(df: pd.DataFrame) -> str:
    payload = "\n".join(df.sort_values("example_id")["raw_response"].astype(str).tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def diagnose_run(run_dir: Path) -> dict[str, Path]:
    out_dir = run_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    accuracy_paths = sorted(run_dir.glob("qwen2.5_0.5b/*/gen_*/accuracy_table.parquet"))
    model_output_paths = sorted(run_dir.glob("qwen2.5_0.5b/*/gen_*/model_outputs.parquet"))
    if not accuracy_paths:
        raise RuntimeError(f"No accuracy tables found under {run_dir}")

    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    response_fingerprints: dict[tuple[str, int], str] = {}

    for mp in model_output_paths:
        branch, gen = _context_from_path(mp)
        md = pd.read_parquet(mp)
        response_fingerprints[(branch, gen)] = _response_fingerprint(md)

    for ap in accuracy_paths:
        branch, gen = _context_from_path(ap)
        df = pd.read_parquet(ap).copy()

        merge_issue = False
        merge_issue_detail = ""
        if df["example_id"].duplicated().any():
            merge_issue = True
            merge_issue_detail = "duplicate_example_id_in_accuracy_table"

        for row in df.to_dict(orient="records"):
            new_res = score_prediction(
                model_output=str(row.get("raw_response", "")),
                gold_answer=str(row.get("answer_gold", "")),
                parsed_final_answer=row.get("parsed_final_answer"),
            )
            old_is_correct = bool(row.get("is_correct", False))

            if merge_issue:
                category = "merge_data_contract_issue"
            elif not new_res.gold_parse_success:
                category = "parse_failure_gold"
            elif not new_res.pred_parse_success:
                category = "parse_failure_pred"
            elif new_res.is_correct and not old_is_correct:
                category = "normalization_mismatch"
            elif new_res.is_correct:
                category = "correct"
            else:
                category = "model_is_wrong"

            records.append(
                {
                    "model_name": row.get("model_name"),
                    "branch": branch,
                    "generation": gen,
                    "example_id": row.get("example_id"),
                    "question": row.get("prompt_text", ""),
                    "answer_gold": row.get("answer_gold"),
                    "raw_response": row.get("raw_response"),
                    "parsed_final_answer": row.get("parsed_final_answer"),
                    "normalized_predicted_old": row.get("normalized_predicted"),
                    "normalized_gold_old": row.get("normalized_gold"),
                    "accuracy_label_old": row.get("accuracy_label"),
                    "is_correct_old": old_is_correct,
                    "pred_parse_success_old": row.get("pred_parse_success"),
                    "gold_parse_success_old": row.get("gold_parse_success"),
                    "accuracy_parse_failure_reason_old": row.get("accuracy_parse_failure_reason"),
                    "normalized_predicted_new": new_res.normalized_predicted,
                    "normalized_gold_new": new_res.normalized_gold,
                    "accuracy_label_new": new_res.accuracy_label,
                    "is_correct_new": new_res.is_correct,
                    "pred_parse_success_new": new_res.pred_parse_success,
                    "gold_parse_success_new": new_res.gold_parse_success,
                    "accuracy_parse_failure_reason_new": new_res.parse_failure_reason,
                    "diagnostic_category": category,
                    "merge_issue_detail": merge_issue_detail,
                }
            )

        ctx_rows = [r for r in records if r["branch"] == branch and r["generation"] == gen]
        old_acc = sum(1 for r in ctx_rows if r["is_correct_old"]) / max(1, len(ctx_rows))
        new_acc = sum(1 for r in ctx_rows if r["is_correct_new"]) / max(1, len(ctx_rows))
        summary_rows.append(
            {
                "branch": branch,
                "generation": gen,
                "row_count": len(ctx_rows),
                "accuracy_old": old_acc,
                "accuracy_new": new_acc,
                "response_fingerprint": response_fingerprints.get((branch, gen), ""),
                "merge_issue": merge_issue,
                "merge_issue_detail": merge_issue_detail,
            }
        )

    diag_df = pd.DataFrame(records)
    buckets = (
        diag_df.groupby(["branch", "generation", "diagnostic_category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["branch", "generation", "diagnostic_category"])
    )

    sample_parts: list[pd.DataFrame] = []
    for category in ["normalization_mismatch", "parse_failure_pred", "parse_failure_gold", "model_is_wrong"]:
        part = (
            diag_df[diag_df["diagnostic_category"] == category]
            .sort_values(["branch", "generation", "example_id"])
            .head(15)
        )
        if not part.empty:
            sample_parts.append(part)
    sample_mismatches = pd.concat(sample_parts, ignore_index=True) if sample_parts else diag_df.head(0).copy()

    # suspicious pattern: same outputs across contexts
    summary_df = pd.DataFrame(summary_rows).sort_values(["branch", "generation"])
    fp_counts = summary_df.groupby("response_fingerprint").size().reset_index(name="contexts_with_same_outputs")
    summary_df = summary_df.merge(fp_counts, on="response_fingerprint", how="left")

    report_path = out_dir / "accuracy_diagnostic_report.csv"
    row_level_path = out_dir / "accuracy_row_level_audit.csv"
    buckets_path = out_dir / "accuracy_failure_buckets.csv"
    mismatches_path = out_dir / "sample_mismatches.csv"
    corrected_summary_path = out_dir / "first_experiment_summary_corrected_accuracy.csv"

    summary_df.to_csv(report_path, index=False)
    diag_df.to_csv(row_level_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    sample_mismatches.to_csv(mismatches_path, index=False)

    original_summary_path = out_dir / "first_experiment_summary.csv"
    if original_summary_path.exists():
        original_summary = pd.read_csv(original_summary_path)
        corrected = original_summary.merge(
            summary_df[["branch", "generation", "accuracy_new"]].rename(
                columns={"accuracy_new": "accuracy_mean_corrected"}
            ),
            on=["branch", "generation"],
            how="left",
            validate="many_to_one",
        )
        corrected.to_csv(corrected_summary_path, index=False)

    return {
        "report": report_path,
        "row_level": row_level_path,
        "buckets": buckets_path,
        "mismatches": mismatches_path,
        "corrected_summary": corrected_summary_path if corrected_summary_path.exists() else original_summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    paths = diagnose_run(Path(args.run_dir))
    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
