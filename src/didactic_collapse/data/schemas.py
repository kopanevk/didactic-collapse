from __future__ import annotations

from pydantic import BaseModel, Field


class RawExample(BaseModel):
    example_id: str
    split: str
    question: str
    answer_gold: str
    dataset_name: str = "gsm8k"


class ModelOutput(BaseModel):
    run_id: str
    branch: str
    generation: int
    model_name: str
    example_id: str
    prompt_version: str
    prompt_text: str
    raw_response: str
    parsed_final_answer: str | None = None


class JudgeOutput(BaseModel):
    run_id: str
    branch: str
    generation: int
    model_name: str
    example_id: str
    judge_provider: str
    judge_model: str
    clarity: int = Field(ge=0, le=2)
    structure: int = Field(ge=0, le=2)
    terminology: int = Field(ge=0, le=2)
    reasoning_soundness: int = Field(ge=0, le=2)
    overall_pedagogical_score: int = Field(ge=0, le=8)
    is_silent_error: bool
    comment: str


class EvalRow(BaseModel):
    run_id: str
    branch: str
    generation: int
    model_name: str
    example_id: str
    answer_gold: str
    parsed_final_answer: str | None
    is_correct: bool
    overall_pedagogical_score: int
    is_silent_error: bool


class SyntheticTrainRow(BaseModel):
    run_id: str
    branch: str
    generation: int
    source: str
    example_id: str
    question: str
    answer_for_training: str
