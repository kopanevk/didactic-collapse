from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.judging.accuracy import evaluate_accuracy, normalize_gold_answer, score_prediction


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


def test_gsm8k_gold_normalization_uses_final_marker() -> None:
    gold = "The bags weigh 16 + 30 = 46. Remove 4 => 42. #### 42"
    assert normalize_gold_answer(gold) == "42"


def test_gsm8k_gold_final_marker_improves_accuracy_match() -> None:
    gold = "Step 1 gives 16, step 2 gives 46, final is #### 42"
    res = score_prediction(model_output="Final answer: 42", gold_answer=gold)
    assert res.gold_parse_success is True
    assert res.is_correct is True


def test_nan_parsed_answer_falls_back_to_raw_output_extraction() -> None:
    res = score_prediction(
        model_output="Reasoning... Final answer: 6",
        gold_answer="#### 6",
        parsed_final_answer=float("nan"),
    )
    assert res.pred_parse_success is True
    assert res.is_correct is True


def test_evaluate_accuracy_fails_on_duplicate_example_ids() -> None:
    outputs = pd.DataFrame(
        [
            {"example_id": "ex_1", "raw_response": "Final answer: 1"},
            {"example_id": "ex_1", "raw_response": "Final answer: 1"},
        ]
    )
    gold = pd.DataFrame([{"example_id": "ex_1", "answer_gold": "1"}])

    with pytest.raises(ValueError, match="cardinality violation"):
        evaluate_accuracy(outputs_df=outputs, gold_df=gold, out_path=Path("outputs/.tmp/unused.parquet"))


def test_evaluate_accuracy_fails_when_gold_is_missing() -> None:
    outputs = pd.DataFrame([{"example_id": "ex_missing", "raw_response": "Final answer: 1"}])
    gold = pd.DataFrame([{"example_id": "ex_1", "answer_gold": "1"}])

    with pytest.raises(ValueError, match="without gold answers"):
        evaluate_accuracy(outputs_df=outputs, gold_df=gold, out_path=Path("outputs/.tmp/unused.parquet"))
