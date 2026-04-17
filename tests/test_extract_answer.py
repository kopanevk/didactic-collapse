from didactic_collapse.pipeline.extract_answer import extract_final_answer


def test_extract_final_answer_fallback() -> None:
    text = "Some chain of thought\nFinal answer: 42"
    assert extract_final_answer(text) == "42"
