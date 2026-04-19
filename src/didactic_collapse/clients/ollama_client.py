from __future__ import annotations

import socket

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from didactic_collapse.clients.base import GenerationClient


def _should_retry_ollama(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or (500 <= status_code <= 599)
    return False


class OllamaClient(GenerationClient):
    def __init__(self, base_url: str = "http://localhost:11434", timeout_sec: int = 60) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout_sec)

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_should_retry_ollama),
        reraise=True,
    )
    def generate(self, *, model_name: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        resp = self._http.post(f"{self._base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", ""))
