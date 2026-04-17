from __future__ import annotations

from abc import ABC, abstractmethod


class GenerationClient(ABC):
    @abstractmethod
    def generate(self, *, model_name: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        raise NotImplementedError


class JudgeClient(ABC):
    @abstractmethod
    def score(self, *, question: str, gold_answer: str, model_output: str, rubric_prompt: str) -> dict:
        raise NotImplementedError
