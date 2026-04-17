from __future__ import annotations

from dataclasses import dataclass

from didactic_collapse.clients.base import GenerationClient


@dataclass(slots=True)
class GenerationRequest:
    model_name: str
    prompt: str
    temperature: float
    top_p: float
    max_tokens: int


class GenerationService:
    """Thin orchestration layer over generation client.

    Keeps request shape stable and centralizes generation-time policies.
    """

    def __init__(self, client: GenerationClient) -> None:
        self.client = client

    def generate_one(self, req: GenerationRequest) -> str:
        """Generate one response for a single prompt.

        TODO:
        - Add structured tracing of latency and token usage
        - Add optional deterministic mode presets
        """
        return self.client.generate(
            model_name=req.model_name,
            prompt=req.prompt,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
        )
