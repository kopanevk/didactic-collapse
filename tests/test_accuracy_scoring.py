from didactic_collapse.judging.accuracy import score_prediction


def test_gold_pred_normalized_equivalence_integer_decimal() -> None:
    res = score_prediction(model_output="Final answer: 1.0", gold_answer="1")
    assert res.gold_parse_success is True
    assert res.pred_parse_success is True
    assert res.is_correct is True


def test_gold_pred_normalized_equivalence_fraction_decimal() -> None:
    res = score_prediction(model_output="Answer: 0.75", gold_answer="3/4")
    assert res.is_correct is True


def test_accuracy_mismatch_case() -> None:
    res = score_prediction(model_output="Final answer: 5", gold_answer="6")
    assert res.pred_parse_success is True
    assert res.gold_parse_success is True
    assert res.is_correct is False
    assert res.accuracy_label == "incorrect"


def test_prediction_parse_failure_case() -> None:
    res = score_prediction(model_output="I cannot tell", gold_answer="6")
    assert res.pred_parse_success is False
    assert res.is_correct is False
    assert res.accuracy_label == "prediction_parse_failure"


def test_gold_parse_failure_case() -> None:
    res = score_prediction(model_output="Final answer: 6", gold_answer="six")
    assert res.gold_parse_success is False
    assert res.is_correct is False
    assert res.accuracy_label == "gold_parse_failure"
