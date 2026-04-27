from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class TrainAdapterError(RuntimeError):
    """Raised when train adapter contract fails."""


@dataclass(frozen=True)
class TrainAdapterRequest:
    input_path: Path
    output_dir: Path
    base_model: str
    target_model: str
    seed: int
    base_url: str = "http://localhost:11434"
    timeout_sec: int = 600
    max_examples: int = 32
    max_chars: int = 5000


def _require_columns(df: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise TrainAdapterError(f"{label} missing required columns: {sorted(missing)}")


def build_training_system_prompt(
    *,
    train_df: pd.DataFrame,
    max_examples: int,
    max_chars: int,
) -> str:
    _require_columns(
        train_df,
        {"question", "answer_for_training"},
        label="train_df",
    )
    if max_examples <= 0:
        raise TrainAdapterError("max_examples must be > 0")
    if max_chars <= 512:
        raise TrainAdapterError("max_chars must be > 512")

    df = train_df.copy()
    if "source" not in df.columns:
        df["source"] = "synthetic"

    selected = df.head(max_examples)
    blocks: list[str] = [
        "You are a concise math tutor.",
        "Always provide short reasoning and end exactly with: Final answer: <number>",
        "Do not append any text after final answer line.",
        "",
        "Few-shot style references:",
    ]

    for idx, rec in enumerate(selected.to_dict(orient="records"), start=1):
        q = str(rec.get("question", "")).strip().replace("\n", " ")
        a = str(rec.get("answer_for_training", "")).strip().replace("\n", " ")
        src = str(rec.get("source", "")).strip()
        block = f"[Example {idx} | source={src}] Q: {q}\nA: {a}"
        blocks.append(block)

    text = "\n".join(blocks)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n[truncated_for_limit]"
    return text


def _create_model(
    *,
    client: httpx.Client,
    base_url: str,
    base_model: str,
    target_model: str,
    system_prompt: str,
) -> None:
    payload = {
        "model": target_model,
        "from": base_model,
        "system": system_prompt,
        "stream": False,
    }
    resp = client.post(f"{base_url.rstrip('/')}/api/create", json=payload)
    resp.raise_for_status()
    data = resp.json()
    # Ollama create response shape varies by version; fail only on explicit error field.
    err = str(data.get("error", "")).strip()
    if err:
        raise TrainAdapterError(f"Ollama create returned error: {err}")


def validate_trained_model(
    *,
    client: httpx.Client,
    base_url: str,
    model_name: str,
) -> None:
    # 1) Existence check.
    show_resp = client.post(f"{base_url.rstrip('/')}/api/show", json={"name": model_name})
    show_resp.raise_for_status()
    show_data = show_resp.json()
    err = str(show_data.get("error", "")).strip()
    if err:
        raise TrainAdapterError(f"Ollama show error for model '{model_name}': {err}")

    # 2) Usability check via generation endpoint.
    gen_resp = client.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model_name,
            "prompt": "1+1?",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
        },
    )
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()
    out = str(gen_data.get("response", "")).strip()
    if not out:
        raise TrainAdapterError(f"Trained model usability check returned empty response: {model_name}")


def run_ollama_train_adapter(request: TrainAdapterRequest) -> Path:
    if not request.input_path.exists():
        raise TrainAdapterError(f"Training input dataset not found: {request.input_path}")
    request.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "train_adapter_start input=%s output_dir=%s base_model=%s target_model=%s base_url=%s",
        request.input_path,
        request.output_dir,
        request.base_model,
        request.target_model,
        request.base_url,
    )

    train_df = pd.read_parquet(request.input_path)
    if train_df.empty:
        raise TrainAdapterError(f"Training dataset is empty: {request.input_path}")

    system_prompt = build_training_system_prompt(
        train_df=train_df,
        max_examples=request.max_examples,
        max_chars=request.max_chars,
    )
    (request.output_dir / "training_system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    with httpx.Client(timeout=request.timeout_sec) as client:
        _create_model(
            client=client,
            base_url=request.base_url,
            base_model=request.base_model,
            target_model=request.target_model,
            system_prompt=system_prompt,
        )
        validate_trained_model(
            client=client,
            base_url=request.base_url,
            model_name=request.target_model,
        )

    result_path = request.output_dir / "training_result.json"
    payload = {
        "created_at": datetime.now().isoformat(),
        "is_stub": False,
        "trained_model_name": request.target_model,
        "base_model_name": request.base_model,
        "seed": request.seed,
        "input_path": str(request.input_path),
        "base_url": request.base_url,
        "max_examples": request.max_examples,
        "max_chars": request.max_chars,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "train_adapter_done trained_model_name=%s validation=ok result=%s",
        request.target_model,
        result_path,
    )
    return result_path

