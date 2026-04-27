from __future__ import annotations

import json
import logging
import math
import random
import re
import time
import zipfile
from xml.sax.saxutils import escape
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from didactic_collapse.clients.judge_client import (
    _classify_openai_compatible_exception,
    build_cerebras_judge_client,
)
from didactic_collapse.config.settings import AppConfig

logger = logging.getLogger(__name__)

_TARGET_BRANCHES = ("pure_recycling", "anchor_20_append")
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_INT_LIKE_RE = re.compile(r"^[+-]?\d+$")

_PAIRWISE_SYSTEM_PROMPT = (
    "You are an evaluation engine for pedagogical quality.\n"
    "Given question, gold answer, explanation A, explanation B, select which explanation is better pedagogically.\n"
    "Criteria: clarity, step structure, terminology, reasoning soundness, and absence of misleading explanation.\n"
    "Return exactly one JSON object and nothing else.\n"
    "No markdown. No prose. No code fences.\n"
    "Allowed JSON schema only:\n"
    "{\n"
    '  "winner": "A" | "B" | "Tie",\n'
    '  "confidence": 0 | 1 | 2,\n'
    '  "reason": "short"\n'
    "}\n"
)

_PAIRWISE_REPAIR_PROMPT = (
    "Your previous answer was not parseable by schema.\n"
    "Return exactly one minified JSON object with keys winner, confidence, reason.\n"
    "winner must be A, B, or Tie. confidence must be 0,1,2.\n"
    "No markdown, no extra keys, no prose."
)


class PairwiseJudgeValidationError(ValueError):
    """Raised when pairwise judge response cannot be parsed/validated."""


class PairwiseJudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    winner: Literal["A", "B", "Tie"]
    confidence: int
    reason: str

    @model_validator(mode="after")
    def validate_ranges(self) -> "PairwiseJudgeDecision":
        if not (0 <= self.confidence <= 2):
            raise ValueError(f"confidence must be in [0, 2], got {self.confidence}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        return self


@dataclass(frozen=True)
class PairwiseParseResult:
    decision: PairwiseJudgeDecision
    repair_applied: bool
    repair_actions: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseJudgeSensitivityArtifacts:
    out_dir: Path
    selected_pairs_csv: Path
    selected_pairs_parquet: Path
    hidden_key_csv: Path
    hidden_key_parquet: Path
    manual_audit_template_csv: Path
    manual_audit_template_xlsx: Path
    llama_results_csv: Path
    llama_results_parquet: Path
    qwen_results_csv: Path
    qwen_results_parquet: Path
    comparison_csv: Path
    comparison_parquet: Path
    summary_csv: Path
    summary_parquet: Path
    seed_branch_summary_csv: Path
    seed_branch_summary_parquet: Path
    metadata_json: Path


def _xlsx_column_name(col_idx_1_based: int) -> str:
    letters: list[str] = []
    n = col_idx_1_based
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def _write_simple_xlsx(df: pd.DataFrame, out_path: Path) -> None:
    """Write a minimal XLSX workbook without external deps.

    This fallback keeps manual audit export available in environments without
    openpyxl/xlsxwriter.
    """
    rows_xml: list[str] = []
    headers = [str(c) for c in df.columns.tolist()]
    all_rows = [headers] + df.astype(object).where(pd.notna(df), "").values.tolist()
    for r_idx, row in enumerate(all_rows, start=1):
        cells_xml: list[str] = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_xlsx_column_name(c_idx)}{r_idx}"
            if value is None or value == "":
                cells_xml.append(f'<c r="{ref}" t="inlineStr"><is><t></t></is></c>')
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    text_val = ""
                    cells_xml.append(f'<c r="{ref}" t="inlineStr"><is><t>{text_val}</t></is></c>')
                else:
                    cells_xml.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text_val = escape(str(value))
                cells_xml.append(f'<c r="{ref}" t="inlineStr"><is><t>{text_val}</t></is></c>')
        rows_xml.append(f'<row r="{r_idx}">{"".join(cells_xml)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        f'{"".join(rows_xml)}'
        "</sheetData>"
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        '<sheet name="audit" sheetId="1" r:id="rId1"/>'
        "</sheets>"
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _read_snapshot(run_dir: Path) -> dict[str, Any]:
    snap = run_dir / "run_config.snapshot.json"
    if not snap.exists():
        raise FileNotFoundError(f"Missing run snapshot: {snap}")
    return json.loads(snap.read_text(encoding="utf-8"))


def _seed_from_payload(payload: dict[str, Any], run_dir: Path) -> int:
    try:
        return int(payload["config"]["project"]["seed"])
    except Exception:  # noqa: BLE001
        match = re.search(r"seed(\d+)", run_dir.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"Cannot determine seed for run_dir={run_dir}")


def _load_rows_from_run(run_dir: Path, *, generation: int) -> pd.DataFrame:
    payload = _read_snapshot(run_dir)
    source_seed = _seed_from_payload(payload, run_dir)

    data_root = Path(payload["config"]["paths"]["data_root"])
    heldout_path = data_root / "splits" / "heldout_test.parquet"
    if not heldout_path.exists():
        raise FileNotFoundError(f"Missing heldout split: {heldout_path}")
    heldout = pd.read_parquet(heldout_path)
    heldout_cols = {"example_id", "question", "answer_gold"}
    missing = heldout_cols.difference(heldout.columns)
    if missing:
        raise ValueError(f"Heldout split missing columns {sorted(missing)} in {heldout_path}")
    heldout = heldout[list(heldout_cols)].copy()
    heldout["example_id"] = heldout["example_id"].astype(str)

    frames: list[pd.DataFrame] = []
    for gen_dir in sorted(run_dir.glob(f"*/*/gen_{generation}")):
        outputs_path = gen_dir / "model_outputs.parquet"
        judge_path = gen_dir / "judge_outputs.parquet"
        if not outputs_path.exists() or not judge_path.exists():
            continue

        outputs = pd.read_parquet(outputs_path)
        judge = pd.read_parquet(judge_path)
        need_outputs = {"run_id", "branch", "generation", "model_name", "example_id", "raw_response"}
        need_judge = {"example_id", "overall_pedagogical_score", "is_silent_error", "comment", "judge_model"}
        miss_outputs = need_outputs.difference(outputs.columns)
        miss_judge = need_judge.difference(judge.columns)
        if miss_outputs:
            raise ValueError(f"model_outputs missing columns {sorted(miss_outputs)} in {outputs_path}")
        if miss_judge:
            raise ValueError(f"judge_outputs missing columns {sorted(miss_judge)} in {judge_path}")

        outputs = outputs[list(need_outputs)].copy()
        judge = judge[list(need_judge)].copy()
        outputs["example_id"] = outputs["example_id"].astype(str)
        judge["example_id"] = judge["example_id"].astype(str)

        merged = outputs.merge(
            judge,
            on="example_id",
            how="left",
            validate="one_to_one",
        ).merge(
            heldout,
            on="example_id",
            how="left",
            validate="one_to_one",
        )
        missing_qa = merged["question"].isna() | merged["answer_gold"].isna()
        if missing_qa.any():
            sample_ids = merged.loc[missing_qa, "example_id"].head(5).tolist()
            raise ValueError(
                f"Missing question/answer_gold after merge for {run_dir}; sample_example_ids={sample_ids}"
            )

        merged["source_run_dir"] = str(run_dir)
        merged["source_seed"] = int(source_seed)
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["branch"].isin(_TARGET_BRANCHES)].copy()
    return out.reset_index(drop=True)


def load_pairwise_source_rows(run_dirs: Sequence[Path], *, generation: int = 1) -> pd.DataFrame:
    if not run_dirs:
        raise ValueError("run_dirs must be non-empty")
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Source run_dir does not exist: {run_path}")
        rows = _load_rows_from_run(run_path, generation=generation)
        if not rows.empty:
            frames.append(rows)
    if not frames:
        raise ValueError("No source rows found for pairwise sensitivity")
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def build_pair_candidates(source_rows_df: pd.DataFrame, *, generation: int = 1) -> pd.DataFrame:
    required = {
        "source_run_dir",
        "source_seed",
        "run_id",
        "generation",
        "model_name",
        "branch",
        "example_id",
        "question",
        "answer_gold",
        "raw_response",
        "overall_pedagogical_score",
        "is_silent_error",
        "comment",
        "judge_model",
    }
    missing = required.difference(source_rows_df.columns)
    if missing:
        raise ValueError(f"source_rows_df missing required columns: {sorted(missing)}")

    work = source_rows_df.copy()
    work["generation"] = pd.to_numeric(work["generation"], errors="coerce").astype("Int64")
    work = work[work["generation"] == int(generation)].copy()
    if work.empty:
        raise ValueError(f"No rows for generation={generation}")

    key_cols = ["source_run_dir", "source_seed", "generation", "example_id", "question", "answer_gold"]
    pure = work[work["branch"] == "pure_recycling"].copy()
    anchor = work[work["branch"] == "anchor_20_append"].copy()
    if pure.empty or anchor.empty:
        raise ValueError("Need both pure_recycling and anchor_20_append rows for pairing")

    pure = pure.rename(
        columns={
            "raw_response": "response_pure",
            "overall_pedagogical_score": "old_llama_score_pure",
            "is_silent_error": "old_llama_silent_pure",
            "comment": "old_llama_comment_pure",
            "judge_model": "old_llama_model_pure",
        }
    )
    anchor = anchor.rename(
        columns={
            "raw_response": "response_anchor",
            "overall_pedagogical_score": "old_llama_score_anchor",
            "is_silent_error": "old_llama_silent_anchor",
            "comment": "old_llama_comment_anchor",
            "judge_model": "old_llama_model_anchor",
        }
    )

    keep_pure = key_cols + [
        "run_id",
        "model_name",
        "response_pure",
        "old_llama_score_pure",
        "old_llama_silent_pure",
        "old_llama_comment_pure",
        "old_llama_model_pure",
    ]
    keep_anchor = key_cols + [
        "run_id",
        "model_name",
        "response_anchor",
        "old_llama_score_anchor",
        "old_llama_silent_anchor",
        "old_llama_comment_anchor",
        "old_llama_model_anchor",
    ]

    merged = pure[keep_pure].merge(
        anchor[keep_anchor],
        on=key_cols,
        how="inner",
        validate="many_to_many",
        suffixes=("_pure_meta", "_anchor_meta"),
    )
    if merged.empty:
        raise ValueError("No paired examples with both branches for same example_id")
    merged["run_id"] = (
        merged.get("run_id_pure_meta")
        .fillna(merged.get("run_id_anchor_meta"))
        .astype(str)
    )
    merged["model_name"] = (
        merged.get("model_name_pure_meta")
        .fillna(merged.get("model_name_anchor_meta"))
        .astype(str)
    )
    merged = merged.drop_duplicates(subset=["source_run_dir", "example_id"], keep="first").reset_index(drop=True)
    return merged


def select_balanced_pair_sample(candidates_df: pd.DataFrame, *, total_n: int = 48, seed: int = 42) -> pd.DataFrame:
    if total_n <= 0:
        raise ValueError("total_n must be > 0")
    required = {"source_seed", "source_run_dir", "example_id", "response_pure", "response_anchor"}
    missing = required.difference(candidates_df.columns)
    if missing:
        raise ValueError(f"candidates_df missing required columns: {sorted(missing)}")

    work = candidates_df.copy().sample(frac=1.0, random_state=seed).reset_index(drop=True)
    seeds = sorted(int(s) for s in work["source_seed"].dropna().astype(int).unique().tolist())
    if not seeds:
        raise ValueError("No source_seed values for balanced sampling")

    if total_n < len(seeds):
        raise ValueError(f"total_n={total_n} too small for number of seeds={len(seeds)}")

    base = total_n // len(seeds)
    extra = total_n % len(seeds)
    selected_parts: list[pd.DataFrame] = []
    for idx, s in enumerate(seeds):
        target = base + (1 if idx < extra else 0)
        seed_rows = work[work["source_seed"].astype(int) == s].copy()
        if len(seed_rows) < target:
            raise ValueError(
                f"Insufficient rows for seed={s}: available={len(seed_rows)} target={target}"
            )
        selected_parts.append(seed_rows.head(target))

    selected = pd.concat(selected_parts, ignore_index=True).reset_index(drop=True)
    if len(selected) != total_n:
        raise ValueError(
            f"Selected pair count mismatch: selected={len(selected)} expected={total_n}"
        )
    return selected


def build_blinded_pair_tables(selected_pairs_df: pd.DataFrame, *, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "source_run_dir",
        "source_seed",
        "run_id",
        "generation",
        "model_name",
        "example_id",
        "question",
        "answer_gold",
        "response_pure",
        "response_anchor",
        "old_llama_score_pure",
        "old_llama_score_anchor",
        "old_llama_silent_pure",
        "old_llama_silent_anchor",
    }
    missing = required.difference(selected_pairs_df.columns)
    if missing:
        raise ValueError(f"selected_pairs_df missing required columns: {sorted(missing)}")

    rng = random.Random(seed)
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(selected_pairs_df.to_dict(orient="records"), start=1):
        pair_id = f"pair_{idx:04d}"
        anchor_is_a = rng.random() < 0.5
        response_a = rec["response_anchor"] if anchor_is_a else rec["response_pure"]
        response_b = rec["response_pure"] if anchor_is_a else rec["response_anchor"]
        branch_a = "anchor_20_append" if anchor_is_a else "pure_recycling"
        branch_b = "pure_recycling" if anchor_is_a else "anchor_20_append"

        public_rows.append(
            {
                "pair_id": pair_id,
                "source_run_dir": rec["source_run_dir"],
                "source_seed": int(rec["source_seed"]),
                "run_id": rec["run_id"],
                "generation": int(rec["generation"]),
                "model_name": rec["model_name"],
                "example_id": rec["example_id"],
                "question": rec["question"],
                "answer_gold": rec["answer_gold"],
                "response_A": response_a,
                "response_B": response_b,
            }
        )
        hidden_rows.append(
            {
                "pair_id": pair_id,
                "source_run_dir": rec["source_run_dir"],
                "source_seed": int(rec["source_seed"]),
                "run_id": rec["run_id"],
                "generation": int(rec["generation"]),
                "model_name": rec["model_name"],
                "example_id": rec["example_id"],
                "A_branch": branch_a,
                "B_branch": branch_b,
                "old_llama_score_pure": rec["old_llama_score_pure"],
                "old_llama_score_anchor": rec["old_llama_score_anchor"],
                "old_llama_silent_pure": bool(rec["old_llama_silent_pure"]),
                "old_llama_silent_anchor": bool(rec["old_llama_silent_anchor"]),
                "old_llama_delta_anchor_minus_pure": float(rec["old_llama_score_anchor"]) - float(rec["old_llama_score_pure"]),
            }
        )

    public_df = pd.DataFrame(public_rows)
    hidden_df = pd.DataFrame(hidden_rows)
    manual_df = public_df[["pair_id", "question", "answer_gold", "response_A", "response_B"]].copy()
    manual_df["human_winner"] = pd.NA
    manual_df["human_confidence"] = pd.NA
    manual_df["human_notes"] = pd.NA
    return public_df, hidden_df, manual_df


def _extract_first_json_object(raw_text: str) -> tuple[str, bool, str]:
    stripped = raw_text.strip()
    if not stripped:
        raise PairwiseJudgeValidationError("Pairwise judge returned empty response")
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, False, "direct_json"

    match = _FENCED_JSON_RE.search(stripped)
    if match:
        return match.group(1).strip(), True, "extracted_fenced_json"

    in_string = False
    escaped = False
    depth = 0
    start: int | None = None
    for idx, ch in enumerate(stripped):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return stripped[start : idx + 1], True, "extracted_json_from_prose"
    raise PairwiseJudgeValidationError("Could not extract JSON object from pairwise response")


def _safe_int(value: Any, field_name: str) -> tuple[int, bool]:
    if isinstance(value, bool):
        raise PairwiseJudgeValidationError(f"{field_name} must be int, got bool")
    if isinstance(value, int):
        return value, False
    if isinstance(value, float) and value.is_integer():
        return int(value), True
    if isinstance(value, str) and _INT_LIKE_RE.match(value.strip()):
        return int(value.strip()), True
    raise PairwiseJudgeValidationError(f"{field_name} must be integer or int-like")


def _safe_winner(value: Any) -> tuple[str, bool]:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"A", "B", "Tie"}:
            return normalized, False
        low = normalized.lower()
        if low == "a":
            return "A", True
        if low == "b":
            return "B", True
        if low == "tie":
            return "Tie", True
    raise PairwiseJudgeValidationError("winner must be one of A/B/Tie")


def parse_pairwise_judge_response(raw_text: str) -> PairwiseParseResult:
    json_text, extracted, extraction_action = _extract_first_json_object(raw_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise PairwiseJudgeValidationError(f"Invalid JSON from pairwise judge: {exc}") from exc
    if not isinstance(payload, dict):
        raise PairwiseJudgeValidationError("Pairwise JSON must be an object")

    required_fields = {"winner", "confidence", "reason"}
    payload_fields = set(payload.keys())
    missing = required_fields.difference(payload_fields)
    extra = payload_fields.difference(required_fields)
    if missing:
        raise PairwiseJudgeValidationError(f"Missing required fields: {sorted(missing)}")
    if extra:
        raise PairwiseJudgeValidationError(f"Unexpected extra fields: {sorted(extra)}")

    repaired = dict(payload)
    actions: list[str] = []
    winner, winner_changed = _safe_winner(repaired["winner"])
    repaired["winner"] = winner
    if winner_changed:
        actions.append("normalized_winner_case")

    confidence, conf_changed = _safe_int(repaired["confidence"], "confidence")
    repaired["confidence"] = confidence
    if conf_changed:
        actions.append("coerced_confidence_to_int")

    if not isinstance(repaired["reason"], str):
        raise PairwiseJudgeValidationError("reason must be a string")

    try:
        decision = PairwiseJudgeDecision.model_validate(repaired)
    except ValidationError as exc:
        raise PairwiseJudgeValidationError(f"Pairwise schema validation failed: {exc}") from exc

    if extracted:
        actions.insert(0, extraction_action)
    return PairwiseParseResult(
        decision=decision,
        repair_applied=bool(actions),
        repair_actions=tuple(actions),
    )


def _build_pairwise_user_prompt(*, question: str, gold_answer: str, response_a: str, response_b: str) -> str:
    return (
        "Evaluate pedagogical quality only.\n"
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
        f"Explanation A: {response_a}\n"
        f"Explanation B: {response_b}\n"
        "Return only JSON winner/confidence/reason."
    )


def _build_pairwise_repair_prompt(*, previous_response: str) -> str:
    return (
        f"{_PAIRWISE_REPAIR_PROMPT}\n"
        "Previous response:\n"
        f"{previous_response}\n"
        "Now return JSON only."
    )


def run_pairwise_judge_for_model(
    *,
    selected_pairs_df: pd.DataFrame,
    cfg: AppConfig,
    judge_model_name: str,
    judge_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"pair_id", "question", "answer_gold", "response_A", "response_B"}
    missing = required.difference(selected_pairs_df.columns)
    if missing:
        raise ValueError(f"selected_pairs_df missing required columns: {sorted(missing)}")
    if cfg.judge.provider.strip().lower() != "cerebras":
        raise ValueError("Pairwise judge sensitivity currently supports provider=cerebras only")

    client = build_cerebras_judge_client(
        model_name=judge_model_name,
        base_url=cfg.judge.base_url,
        api_key_env=cfg.judge.api_key_env,
        timeout_sec=cfg.judge.timeout_sec,
        max_retries=cfg.judge.max_retries,
    )
    request_delay_sec = max(0.0, float(cfg.judge.request_delay_sec))
    max_attempts = max(1, int(cfg.judge.max_retries) + 1)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for rec in selected_pairs_df.to_dict(orient="records"):
        payload = {
            "model": judge_model_name,
            "messages": [
                {"role": "system", "content": _PAIRWISE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_pairwise_user_prompt(
                        question=str(rec["question"]),
                        gold_answer=str(rec["answer_gold"]),
                        response_a=str(rec["response_A"]),
                        response_b=str(rec["response_B"]),
                    ),
                },
            ],
            "temperature": 0,
            "top_p": 1,
            "response_format": {"type": "json_object"},
        }

        attempt = 1
        raw_content = ""
        while True:
            try:
                raw_content = client._request_raw_content(payload=payload)  # noqa: SLF001
                parsed = parse_pairwise_judge_response(raw_content)
                results.append(
                    {
                        "pair_id": rec["pair_id"],
                        "judge_label": judge_label,
                        "judge_provider": cfg.judge.provider,
                        "judge_model": judge_model_name,
                        "winner": parsed.decision.winner,
                        "confidence": parsed.decision.confidence,
                        "reason": parsed.decision.reason,
                        "repair_applied": bool(parsed.repair_applied),
                        "repair_actions": "|".join(parsed.repair_actions),
                    }
                )
                if request_delay_sec > 0:
                    time.sleep(request_delay_sec)
                break
            except PairwiseJudgeValidationError as exc:
                # One bounded provider re-emit attempt for formatting/schema drift.
                try:
                    repair_payload = dict(payload)
                    repair_payload["messages"] = [
                        {"role": "system", "content": _PAIRWISE_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_pairwise_repair_prompt(previous_response=raw_content)},
                    ]
                    repair_raw = client._request_raw_content(payload=repair_payload)  # noqa: SLF001
                    parsed = parse_pairwise_judge_response(repair_raw)
                    actions = list(parsed.repair_actions)
                    actions.append("repair_reemit_attempt")
                    results.append(
                        {
                            "pair_id": rec["pair_id"],
                            "judge_label": judge_label,
                            "judge_provider": cfg.judge.provider,
                            "judge_model": judge_model_name,
                            "winner": parsed.decision.winner,
                            "confidence": parsed.decision.confidence,
                            "reason": parsed.decision.reason,
                            "repair_applied": True,
                            "repair_actions": "|".join(actions),
                        }
                    )
                except Exception as reemit_exc:  # noqa: BLE001
                    failures.append(
                        {
                            "pair_id": rec["pair_id"],
                            "judge_label": judge_label,
                            "judge_model": judge_model_name,
                            "error_category": reemit_exc.__class__.__name__,
                            "error_message": str(reemit_exc)[:500],
                        }
                    )
                    logger.warning(
                        "pairwise_validation_failure judge=%s model=%s pair_id=%s detail=%s",
                        judge_label,
                        judge_model_name,
                        rec["pair_id"],
                        str(exc),
                    )
                if request_delay_sec > 0:
                    time.sleep(request_delay_sec)
                break
            except Exception as exc:  # noqa: BLE001
                category, retryable, retry_after = _classify_openai_compatible_exception(exc)
                if (not retryable) or attempt >= max_attempts:
                    failures.append(
                        {
                            "pair_id": rec["pair_id"],
                            "judge_label": judge_label,
                            "judge_model": judge_model_name,
                            "error_category": category,
                            "error_message": str(exc)[:500],
                        }
                    )
                    break
                sleep_sec = client._compute_retry_sleep(  # noqa: SLF001
                    attempt_number=attempt,
                    retry_after_sec=retry_after,
                )
                logger.warning(
                    "pairwise_retry judge=%s model=%s category=%s attempt=%d/%d sleep_sec=%.2f",
                    judge_label,
                    judge_model_name,
                    category,
                    attempt,
                    max_attempts,
                    sleep_sec,
                )
                time.sleep(sleep_sec)
                attempt += 1

    results_df = pd.DataFrame(
        results,
        columns=[
            "pair_id",
            "judge_label",
            "judge_provider",
            "judge_model",
            "winner",
            "confidence",
            "reason",
            "repair_applied",
            "repair_actions",
        ],
    )
    failures_df = pd.DataFrame(
        failures,
        columns=["pair_id", "judge_label", "judge_model", "error_category", "error_message"],
    )
    if results_df.empty:
        raise RuntimeError(
            f"Pairwise judging produced zero successful rows for judge={judge_label}, "
            f"failures={len(failures_df)}"
        )
    return results_df, failures_df


def decode_winner_branch(results_df: pd.DataFrame, hidden_key_df: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    required_results = {"pair_id", "winner", "confidence", "reason", "judge_model", "judge_label"}
    required_key = {"pair_id", "A_branch", "B_branch", "source_seed"}
    miss_r = required_results.difference(results_df.columns)
    miss_k = required_key.difference(hidden_key_df.columns)
    if miss_r:
        raise ValueError(f"results_df missing required columns: {sorted(miss_r)}")
    if miss_k:
        raise ValueError(f"hidden_key_df missing required columns: {sorted(miss_k)}")

    merged = hidden_key_df.merge(results_df, on="pair_id", how="inner", validate="one_to_one")

    def _winner_branch(row: pd.Series) -> str:
        w = row["winner"]
        if w == "A":
            return str(row["A_branch"])
        if w == "B":
            return str(row["B_branch"])
        return "Tie"

    merged[f"{prefix}_winner_branch"] = merged.apply(_winner_branch, axis=1)
    merged[f"{prefix}_anchor_win"] = merged[f"{prefix}_winner_branch"] == "anchor_20_append"
    merged[f"{prefix}_pure_win"] = merged[f"{prefix}_winner_branch"] == "pure_recycling"
    merged[f"{prefix}_tie"] = merged[f"{prefix}_winner_branch"] == "Tie"
    merged = merged.rename(
        columns={
            "winner": f"{prefix}_winner",
            "confidence": f"{prefix}_confidence",
            "reason": f"{prefix}_reason",
            "judge_model": f"{prefix}_judge_model",
            "judge_label": f"{prefix}_judge_label",
            "repair_applied": f"{prefix}_repair_applied",
            "repair_actions": f"{prefix}_repair_actions",
        }
    )
    keep_cols = [
        "pair_id",
        "source_seed",
        "source_run_dir",
        "run_id",
        "generation",
        "model_name",
        "example_id",
        "A_branch",
        "B_branch",
        f"{prefix}_winner",
        f"{prefix}_confidence",
        f"{prefix}_reason",
        f"{prefix}_judge_model",
        f"{prefix}_judge_label",
        f"{prefix}_repair_applied",
        f"{prefix}_repair_actions",
        f"{prefix}_winner_branch",
        f"{prefix}_anchor_win",
        f"{prefix}_pure_win",
        f"{prefix}_tie",
    ]
    return merged[keep_cols].copy()


def build_pairwise_comparison(llama_decoded_df: pd.DataFrame, qwen_decoded_df: pd.DataFrame) -> pd.DataFrame:
    merged = llama_decoded_df.merge(
        qwen_decoded_df,
        on=[
            "pair_id",
            "source_seed",
            "source_run_dir",
            "run_id",
            "generation",
            "model_name",
            "example_id",
            "A_branch",
            "B_branch",
        ],
        how="inner",
        validate="one_to_one",
    )
    merged["judges_agree_winner_branch"] = (
        merged["llama_winner_branch"].astype(str) == merged["qwen_winner_branch"].astype(str)
    )
    merged["judges_agree_winner_raw"] = merged["llama_winner"].astype(str) == merged["qwen_winner"].astype(str)
    return merged


def build_pairwise_summary(
    comparison_df: pd.DataFrame,
    *,
    llama_model_name: str,
    qwen_model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")
    n = len(comparison_df)
    summary = {
        "n_pairs": int(n),
        "llama_model_name": llama_model_name,
        "qwen_model_name": qwen_model_name,
        "llama_anchor_win_rate": float(comparison_df["llama_anchor_win"].mean()),
        "llama_pure_win_rate": float(comparison_df["llama_pure_win"].mean()),
        "llama_tie_rate": float(comparison_df["llama_tie"].mean()),
        "qwen_anchor_win_rate": float(comparison_df["qwen_anchor_win"].mean()),
        "qwen_pure_win_rate": float(comparison_df["qwen_pure_win"].mean()),
        "qwen_tie_rate": float(comparison_df["qwen_tie"].mean()),
        "llama_qwen_agreement_winner_branch_rate": float(comparison_df["judges_agree_winner_branch"].mean()),
        "llama_qwen_agreement_winner_raw_rate": float(comparison_df["judges_agree_winner_raw"].mean()),
        "llama_anchor_minus_pure_win_rate": float(comparison_df["llama_anchor_win"].mean() - comparison_df["llama_pure_win"].mean()),
        "qwen_anchor_minus_pure_win_rate": float(comparison_df["qwen_anchor_win"].mean() - comparison_df["qwen_pure_win"].mean()),
        "branch_conclusion_changed": bool(
            (
                (comparison_df["llama_anchor_win"].mean() - comparison_df["llama_pure_win"].mean()) >= 0
                and (comparison_df["qwen_anchor_win"].mean() - comparison_df["qwen_pure_win"].mean()) < 0
            )
            or (
                (comparison_df["llama_anchor_win"].mean() - comparison_df["llama_pure_win"].mean()) < 0
                and (comparison_df["qwen_anchor_win"].mean() - comparison_df["qwen_pure_win"].mean()) >= 0
            )
        ),
    }

    seed_rows: list[dict[str, Any]] = []
    for source_seed, grp in comparison_df.groupby("source_seed", as_index=False):
        llama_delta = float(grp["llama_anchor_win"].mean() - grp["llama_pure_win"].mean())
        qwen_delta = float(grp["qwen_anchor_win"].mean() - grp["qwen_pure_win"].mean())
        seed_rows.append(
            {
                "source_seed": int(source_seed),
                "n_pairs": int(len(grp)),
                "llama_anchor_win_rate": float(grp["llama_anchor_win"].mean()),
                "llama_pure_win_rate": float(grp["llama_pure_win"].mean()),
                "llama_tie_rate": float(grp["llama_tie"].mean()),
                "qwen_anchor_win_rate": float(grp["qwen_anchor_win"].mean()),
                "qwen_pure_win_rate": float(grp["qwen_pure_win"].mean()),
                "qwen_tie_rate": float(grp["qwen_tie"].mean()),
                "llama_anchor_minus_pure_win_rate": llama_delta,
                "qwen_anchor_minus_pure_win_rate": qwen_delta,
                "judges_agree_winner_branch_rate": float(grp["judges_agree_winner_branch"].mean()),
                "seed_conclusion_changed": bool(
                    (llama_delta >= 0 and qwen_delta < 0) or (llama_delta < 0 and qwen_delta >= 0)
                ),
            }
        )

    summary_df = pd.DataFrame([summary])
    seed_df = pd.DataFrame(seed_rows).sort_values("source_seed").reset_index(drop=True)
    return summary_df, seed_df


def run_pairwise_judge_sensitivity(
    *,
    cfg: AppConfig,
    run_dirs: Sequence[Path],
    sample_size: int = 48,
    sample_seed: int = 4242,
    generation: int = 1,
    llama_model_name: str = "llama-3.1-8b",
    qwen_model_name: str = "qwen-3-235b-a22b-instruct-2507",
    out_dir: Path | None = None,
) -> PairwiseJudgeSensitivityArtifacts:
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    run_dirs = [Path(p) for p in run_dirs]
    if not run_dirs:
        raise ValueError("run_dirs must be non-empty")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tables_dir = Path(out_dir or (cfg.paths.output_root / "judge_sensitivity" / f"pairwise_confirmatory_{ts}" / "tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_pairwise_source_rows(run_dirs, generation=generation)
    candidates = build_pair_candidates(source_rows, generation=generation)
    selected_candidates = select_balanced_pair_sample(candidates, total_n=sample_size, seed=sample_seed)
    selected_pairs, hidden_key, manual_template = build_blinded_pair_tables(selected_candidates, seed=sample_seed)

    llama_results, llama_failures = run_pairwise_judge_for_model(
        selected_pairs_df=selected_pairs,
        cfg=cfg,
        judge_model_name=llama_model_name,
        judge_label="llama_pairwise",
    )
    qwen_results, qwen_failures = run_pairwise_judge_for_model(
        selected_pairs_df=selected_pairs,
        cfg=cfg,
        judge_model_name=qwen_model_name,
        judge_label="qwen_pairwise",
    )

    llama_decoded = decode_winner_branch(llama_results, hidden_key, prefix="llama")
    qwen_decoded = decode_winner_branch(qwen_results, hidden_key, prefix="qwen")
    comparison = build_pairwise_comparison(llama_decoded, qwen_decoded)
    summary, seed_summary = build_pairwise_summary(
        comparison,
        llama_model_name=llama_model_name,
        qwen_model_name=qwen_model_name,
    )

    selected_pairs_csv = tables_dir / "pairwise_selected_pairs.csv"
    selected_pairs_parquet = tables_dir / "pairwise_selected_pairs.parquet"
    hidden_key_csv = tables_dir / "pairwise_hidden_key.csv"
    hidden_key_parquet = tables_dir / "pairwise_hidden_key.parquet"
    manual_audit_template_csv = tables_dir / "pairwise_manual_audit_template.csv"
    manual_audit_template_xlsx = tables_dir / "pairwise_manual_audit_template.xlsx"
    llama_results_csv = tables_dir / "pairwise_llama_results.csv"
    llama_results_parquet = tables_dir / "pairwise_llama_results.parquet"
    qwen_results_csv = tables_dir / "pairwise_qwen_results.csv"
    qwen_results_parquet = tables_dir / "pairwise_qwen_results.parquet"
    comparison_csv = tables_dir / "pairwise_judge_comparison.csv"
    comparison_parquet = tables_dir / "pairwise_judge_comparison.parquet"
    summary_csv = tables_dir / "pairwise_summary.csv"
    summary_parquet = tables_dir / "pairwise_summary.parquet"
    seed_branch_summary_csv = tables_dir / "pairwise_seed_branch_summary.csv"
    seed_branch_summary_parquet = tables_dir / "pairwise_seed_branch_summary.parquet"
    metadata_json = tables_dir / "pairwise_metadata.json"

    selected_pairs.to_csv(selected_pairs_csv, index=False)
    selected_pairs.to_parquet(selected_pairs_parquet, index=False)
    hidden_key.to_csv(hidden_key_csv, index=False)
    hidden_key.to_parquet(hidden_key_parquet, index=False)
    manual_template.to_csv(manual_audit_template_csv, index=False)
    try:
        manual_template.to_excel(manual_audit_template_xlsx, index=False)
    except Exception:  # noqa: BLE001
        _write_simple_xlsx(manual_template, manual_audit_template_xlsx)
        logger.warning(
            "Saved XLSX manual template via dependency-free fallback writer at %s",
            manual_audit_template_xlsx,
        )

    llama_full = llama_decoded.merge(
        llama_failures.rename(columns={"judge_model": "llama_judge_model_failure"}),
        on="pair_id",
        how="left",
    )
    qwen_full = qwen_decoded.merge(
        qwen_failures.rename(columns={"judge_model": "qwen_judge_model_failure"}),
        on="pair_id",
        how="left",
    )
    llama_full.to_csv(llama_results_csv, index=False)
    llama_full.to_parquet(llama_results_parquet, index=False)
    qwen_full.to_csv(qwen_results_csv, index=False)
    qwen_full.to_parquet(qwen_results_parquet, index=False)
    comparison.to_csv(comparison_csv, index=False)
    comparison.to_parquet(comparison_parquet, index=False)
    summary.to_csv(summary_csv, index=False)
    summary.to_parquet(summary_parquet, index=False)
    seed_summary.to_csv(seed_branch_summary_csv, index=False)
    seed_summary.to_parquet(seed_branch_summary_parquet, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "judge_provider": cfg.judge.provider,
        "generation": int(generation),
        "sample_size_requested": int(sample_size),
        "sample_size_selected": int(len(selected_pairs)),
        "sample_seed": int(sample_seed),
        "source_run_dirs": [str(p) for p in run_dirs],
        "llama_model_name": llama_model_name,
        "qwen_model_name": qwen_model_name,
        "llama_success_count": int(len(llama_results)),
        "llama_failure_count": int(len(llama_failures)),
        "qwen_success_count": int(len(qwen_results)),
        "qwen_failure_count": int(len(qwen_failures)),
        "evaluation_mode": "pairwise_blinded_robustness_check_only",
        "note": "No generation/training rerun; pairwise judging on existing outputs only.",
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return PairwiseJudgeSensitivityArtifacts(
        out_dir=tables_dir.parent,
        selected_pairs_csv=selected_pairs_csv,
        selected_pairs_parquet=selected_pairs_parquet,
        hidden_key_csv=hidden_key_csv,
        hidden_key_parquet=hidden_key_parquet,
        manual_audit_template_csv=manual_audit_template_csv,
        manual_audit_template_xlsx=manual_audit_template_xlsx,
        llama_results_csv=llama_results_csv,
        llama_results_parquet=llama_results_parquet,
        qwen_results_csv=qwen_results_csv,
        qwen_results_parquet=qwen_results_parquet,
        comparison_csv=comparison_csv,
        comparison_parquet=comparison_parquet,
        summary_csv=summary_csv,
        summary_parquet=summary_parquet,
        seed_branch_summary_csv=seed_branch_summary_csv,
        seed_branch_summary_parquet=seed_branch_summary_parquet,
        metadata_json=metadata_json,
    )
