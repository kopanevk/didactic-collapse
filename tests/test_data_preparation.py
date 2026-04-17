from __future__ import annotations

import pandas as pd
import pytest

from didactic_collapse.data.loaders import generate_stable_example_id, validate_split_integrity
from didactic_collapse.data.splitter import create_splits


def _make_df(n: int) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for i in range(n):
        rows.append(
            {
                "example_id": f"ex_{i:04d}",
                "question": f"What is {i}+{i}?",
                "answer_gold": str(i + i),
            }
        )
    return pd.DataFrame(rows)


def test_create_splits_is_deterministic_for_same_seed(tmp_path) -> None:
    df = _make_df(50).sample(frac=1.0, random_state=123).reset_index(drop=True)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    a1 = create_splits(df, out1, seed=42, base_train_size=20, anchor_pool_size=10, heldout_test_size=10)
    a2 = create_splits(df, out2, seed=42, base_train_size=20, anchor_pool_size=10, heldout_test_size=10)

    b1 = pd.read_parquet(a1.base_train_path)["example_id"].tolist()
    b2 = pd.read_parquet(a2.base_train_path)["example_id"].tolist()
    p1 = pd.read_parquet(a1.anchor_pool_path)["example_id"].tolist()
    p2 = pd.read_parquet(a2.anchor_pool_path)["example_id"].tolist()
    h1 = pd.read_parquet(a1.heldout_test_path)["example_id"].tolist()
    h2 = pd.read_parquet(a2.heldout_test_path)["example_id"].tolist()

    assert b1 == b2
    assert p1 == p2
    assert h1 == h2


def test_validate_split_integrity_detects_overlap() -> None:
    base = pd.DataFrame(
        [
            {"example_id": "ex_1", "question": "q1", "answer_gold": "1"},
            {"example_id": "ex_2", "question": "q2", "answer_gold": "2"},
        ]
    )
    anchor = pd.DataFrame(
        [
            {"example_id": "ex_2", "question": "q2", "answer_gold": "2"},
        ]
    )
    heldout = pd.DataFrame(
        [
            {"example_id": "ex_3", "question": "q3", "answer_gold": "3"},
        ]
    )

    with pytest.raises(ValueError, match="overlap"):
        validate_split_integrity(base_train_df=base, anchor_pool_df=anchor, heldout_test_df=heldout)


def test_generate_stable_example_id_is_repeatable() -> None:
    first = generate_stable_example_id(
        question="If John has 2 apples and buys 3 more, how many apples?",
        answer_gold="5",
        dataset_name="gsm8k",
    )
    second = generate_stable_example_id(
        question="If John has 2 apples and buys 3 more, how many apples?",
        answer_gold="5",
        dataset_name="gsm8k",
    )
    third = generate_stable_example_id(
        question="If John has 2 apples and buys 4 more, how many apples?",
        answer_gold="6",
        dataset_name="gsm8k",
    )

    assert first == second
    assert first != third
