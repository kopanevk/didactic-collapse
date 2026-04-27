from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from didactic_collapse.training.ollama_train_backend import (
    TrainAdapterRequest,
    run_ollama_train_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapter contract for train-stage feasibility runs. "
            "Integrate your real fine-tune here and write training_result.json."
        )
    )
    parser.add_argument("--input", required=True, help="Path to training dataset parquet")
    parser.add_argument("--output-dir", required=True, help="Directory where training_result.json is written")
    parser.add_argument("--base-model", required=True, help="Base local model name")
    parser.add_argument("--target-model", required=True, help="Target model name for Gen-1 inference")
    parser.add_argument("--seed", required=True, type=int, help="Experiment seed")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama API base url for create/show/generate validation",
    )
    parser.add_argument(
        "--timeout-sec",
        default=600,
        type=int,
        help="HTTP timeout for model creation and validation",
    )
    parser.add_argument(
        "--max-examples",
        default=32,
        type=int,
        help="Max number of train examples to include in adaptation prompt",
    )
    parser.add_argument(
        "--max-chars",
        default=5000,
        type=int,
        help="Max characters for adaptation system prompt",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Non-scientific smoke mode: do not train, only emit a stub training_result.json. "
            "Do not use for real experimental conclusions."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "training_result.json"

    if args.stub:
        payload = {
            "created_at": datetime.now().isoformat(),
            "is_stub": True,
            "trained_model_name": args.target_model,
            "base_model_name": args.base_model,
            "seed": args.seed,
            "input_path": str(Path(args.input)),
            "note": "stub_mode_non_scientific",
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    run_ollama_train_adapter(
        TrainAdapterRequest(
            input_path=Path(args.input),
            output_dir=out_dir,
            base_model=str(args.base_model),
            target_model=str(args.target_model),
            seed=int(args.seed),
            base_url=str(args.base_url),
            timeout_sec=int(args.timeout_sec),
            max_examples=int(args.max_examples),
            max_chars=int(args.max_chars),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
