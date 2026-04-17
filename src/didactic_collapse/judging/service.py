from __future__ import annotations

from dataclasses import dataclass

from didactic_collapse.clients.base import JudgeClient


@dataclass(slots=True)
class RubricScore:
    clarity: int
    structure: int
    terminology: int
    reasoning_soundness: int
    overall_pedagogical_score: int
    is_silent_error: bool
    comment: str


class JudgeService:
    """Adapter-friendly service for pedagogical scoring."""

    def __init__(self, client: JudgeClient, rubric_prompt: str) -> None:
        self.client = client
        self.rubric_prompt = rubric_prompt

    def score_one(self, *, question: str, gold_answer: str, model_output: str) -> RubricScore:
        payload = self.client.score(
            question=question,
            gold_answer=gold_answer,
            model_output=model_output,
            rubric_prompt=self.rubric_prompt,
        )
        # TODO: add strict pydantic validation and repair strategy for malformed JSON.
        return RubricScore(**payload)
