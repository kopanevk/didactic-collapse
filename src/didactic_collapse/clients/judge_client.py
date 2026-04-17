from __future__ import annotations

import json
import os

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from didactic_collapse.clients.base import JudgeClient


class OpenAICompatibleJudgeClient(JudgeClient):
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key_env: str,
        timeout_sec: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env: {api_key_env}")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = httpx.Client(timeout=timeout_sec, headers=self._headers)

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        user_prompt = (
            "Evaluate educational quality and silent error risk. "
            "Return strict JSON only.\n\n"
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Model output: {model_output}\n"
        )
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": rubric_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        resp = self._http.post(f"{self._base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
