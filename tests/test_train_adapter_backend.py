from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from didactic_collapse.orchestration.training_feasibility import (
    CommandTrainStageBackend,
    TrainStageRequest,
    TrainingFeasibilityError,
)
from didactic_collapse.training.ollama_train_backend import (
    TrainAdapterError,
    TrainAdapterRequest,
    run_ollama_train_adapter,
)


def _mk_dir(prefix: str) -> Path:
    base = Path("outputs") / ".tmp" / "unit_train_adapter"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{prefix}_{uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, *, usable: bool = True) -> None:
        self.usable = usable

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def post(self, url: str, json: dict) -> _FakeResponse:  # type: ignore[override]
        if url.endswith("/api/create"):
            return _FakeResponse(200, {"status": "success"})
        if url.endswith("/api/show"):
            return _FakeResponse(200, {"details": {"family": "qwen"}})
        if url.endswith("/api/generate"):
            if self.usable:
                return _FakeResponse(200, {"response": "Final answer: 2"})
            return _FakeResponse(200, {"response": ""})
        return _FakeResponse(404, {"error": "not found"})


def test_train_adapter_writes_valid_result_json(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _mk_dir("adapter_valid")
    train_path = root / "train.parquet"
    pd.DataFrame(
        [
            {"example_id": "e1", "question": "1+1", "answer_for_training": "Final answer: 2", "source": "synthetic"}
        ]
    ).to_parquet(train_path, index=False)

    import didactic_collapse.training.ollama_train_backend as backend_mod

    monkeypatch.setattr(backend_mod.httpx, "Client", lambda timeout: _FakeClient(usable=True))
    result_path = run_ollama_train_adapter(
        TrainAdapterRequest(
            input_path=train_path,
            output_dir=root / "out",
            base_model="qwen2.5:0.5b",
            target_model="qwen2.5:0.5b__ft_test",
            seed=1,
            base_url="http://localhost:11434",
            timeout_sec=60,
        )
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["trained_model_name"] == "qwen2.5:0.5b__ft_test"
    assert payload["is_stub"] is False


def test_train_adapter_fails_on_unusable_trained_model(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _mk_dir("adapter_unusable")
    train_path = root / "train.parquet"
    pd.DataFrame(
        [
            {"example_id": "e1", "question": "1+1", "answer_for_training": "Final answer: 2", "source": "synthetic"}
        ]
    ).to_parquet(train_path, index=False)

    import didactic_collapse.training.ollama_train_backend as backend_mod

    monkeypatch.setattr(backend_mod.httpx, "Client", lambda timeout: _FakeClient(usable=False))
    with pytest.raises(TrainAdapterError, match="usability check returned empty response"):
        run_ollama_train_adapter(
            TrainAdapterRequest(
                input_path=train_path,
                output_dir=root / "out",
                base_model="qwen2.5:0.5b",
                target_model="qwen2.5:0.5b__ft_bad",
                seed=1,
                base_url="http://localhost:11434",
                timeout_sec=60,
            )
        )


def test_command_backend_contract_success_via_stub_adapter() -> None:
    root = _mk_dir("command_backend_ok")
    req = TrainStageRequest(
        run_dir=root,
        branch="pure_recycling",
        seed=1,
        generation_from=0,
        generation_to=1,
        base_model_name="qwen2.5:0.5b",
        target_model_name="qwen2.5:0.5b__ft_cmd",
        training_data_path=root / "train.parquet",
        output_dir=root / "trainer_out",
    )
    backend = CommandTrainStageBackend(
        command_template=(
            "python scripts/train_feasibility_adapter.py "
            '--input "{train_path}" --output-dir "{output_dir}" '
            '--base-model "{base_model}" --target-model "{target_model}" --seed {seed} --stub'
        ),
        timeout_sec=60,
        result_filename="training_result.json",
    )
    result = backend.run(req)
    assert result.trained_model_name == req.target_model_name
    assert result.metadata_path.exists()


def test_command_backend_missing_trained_model_name_fails() -> None:
    root = _mk_dir("command_backend_missing_name")
    req = TrainStageRequest(
        run_dir=root,
        branch="pure_recycling",
        seed=1,
        generation_from=0,
        generation_to=1,
        base_model_name="qwen2.5:0.5b",
        target_model_name="qwen2.5:0.5b__ft_cmd",
        training_data_path=root / "train.parquet",
        output_dir=root / "trainer_out",
    )
    backend = CommandTrainStageBackend(
        command_template=(
            "python -c \"import json, pathlib; "
            "p=pathlib.Path(r'{output_dir}')/'training_result.json'; "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text(json.dumps({{}}))\""
        ),
        timeout_sec=60,
        result_filename="training_result.json",
    )
    with pytest.raises(TrainingFeasibilityError, match="missing non-empty trained_model_name"):
        backend.run(req)
