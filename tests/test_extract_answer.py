from didactic_collapse.pipeline.extract_answer import extract_final_answer_result


def test_extract_final_answer_marker() -> None:
    res = extract_final_answer_result("Reasoning...\nFinal answer: 42")
    assert res.parse_success is True
    assert res.normalized_answer == "42"


def test_extract_answer_marker() -> None:
    res = extract_final_answer_result("Answer: 42")
    assert res.parse_success is True
    assert res.normalized_answer == "42"


def test_extract_boxed() -> None:
    res = extract_final_answer_result("We get \\boxed{42}.")
    assert res.parse_success is True
    assert res.normalized_answer == "42"


def test_extract_trailing_numeric() -> None:
    res = extract_final_answer_result("steps...\ntherefore result is 17")
    assert res.parse_success is True
    assert res.normalized_answer == "17"


def test_extract_negative_integer() -> None:
    res = extract_final_answer_result("Final answer: -8")
    assert res.parse_success is True
    assert res.normalized_answer == "-8"


def test_extract_decimal() -> None:
    res = extract_final_answer_result("The answer is 3.50")
    assert res.parse_success is True
    assert res.normalized_answer == "3.50"


def test_extract_fraction() -> None:
    res = extract_final_answer_result("Answer: 3/4")
    assert res.parse_success is True
    assert res.normalized_answer == "3/4"


def test_extract_whitespace_and_punctuation() -> None:
    res = extract_final_answer_result("Final answer: ( 42 ).")
    assert res.parse_success is True
    assert res.normalized_answer == "42"


def test_extract_no_parse() -> None:
    res = extract_final_answer_result("This is only explanation without final number.")
    assert res.parse_success is False
    assert res.parse_failure_reason == "no_numeric_candidate"


def test_extract_multiple_candidate_answers() -> None:
    res = extract_final_answer_result("Final answer: 41\nActually, Final answer: 42")
    assert res.parse_success is False
    assert res.parse_failure_reason == "multiple_candidate_answers"
