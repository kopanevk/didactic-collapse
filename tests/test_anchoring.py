from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from didactic_collapse.recycling.anchoring import (
    AnchorPolicy,
    AnchorSelectionContext,
    AnchoringError,
    save_anchoring_artifacts,
    select_human_anchors,
)


def _anchor_pool_df(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [f"a_{i:03d}" for i in range(n)],
            "question": [f"anchor q{i}" for i in range(n)],
            "answer_gold": [str(i) for i in range(n)],
        }
    )


def _synthetic_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [f"s_{i:03d}" for i in range(n)],
            "question": [f"synthetic q{i}" for i in range(n)],
            "answer_for_training": [f"ans{i}" for i in range(n)],
            "source": ["synthetic" for _ in range(n)],
        }
    )


def _base_df() -> pd.DataFrame:
    return pd.DataFrame({"example_id": ["b_001", "b_002"]})


def _heldout_df() -> pd.DataFrame:
    return pd.DataFrame({"example_id": ["h_001", "h_002"]})


def test_deterministic_anchor_selection() -> None:
    pool = _anchor_pool_df(30)
    synth = _synthetic_df(10)
    context = AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=1, seed=42)
    policy = AnchorPolicy(anchor_ratio=0.1, allow_reuse=False)

    first = select_human_anchors(
        anchor_pool_df=pool,
        synthetic_df=synth,
        base_train_df=_base_df(),
        heldout_test_df=_heldout_df(),
        previously_used_anchor_ids=set(),
        policy=policy,
        context=context,
    )
    second = select_human_anchors(
        anchor_pool_df=pool,
        synthetic_df=synth,
        base_train_df=_base_df(),
        heldout_test_df=_heldout_df(),
        previously_used_anchor_ids=set(),
        policy=policy,
        context=context,
    )

    assert first.metadata.selected_anchor_ids == second.metadata.selected_anchor_ids


def test_no_reuse_across_generations() -> None:
    pool = _anchor_pool_df(30)
    synth = _synthetic_df(10)
    policy = AnchorPolicy(anchor_ratio=0.2, allow_reuse=False)

    gen1 = select_human_anchors(
        anchor_pool_df=pool,
        synthetic_df=synth,
        base_train_df=_base_df(),
        heldout_test_df=_heldout_df(),
        previously_used_anchor_ids=set(),
        policy=policy,
        context=AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=1, seed=42),
    )

    gen2 = select_human_anchors(
        anchor_pool_df=pool,
        synthetic_df=synth,
        base_train_df=_base_df(),
        heldout_test_df=_heldout_df(),
        previously_used_anchor_ids=set(gen1.metadata.selected_anchor_ids),
        policy=policy,
        context=AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=2, seed=42),
    )

    assert set(gen1.metadata.selected_anchor_ids).isdisjoint(set(gen2.metadata.selected_anchor_ids))


def test_overlap_with_forbidden_pools_causes_failure() -> None:
    pool = _anchor_pool_df(10)
    synth = _synthetic_df(10)
    # Force overlap with base_train
    pool.loc[0, "example_id"] = "b_001"

    with pytest.raises(AnchoringError, match="overlap"):
        select_human_anchors(
            anchor_pool_df=pool,
            synthetic_df=synth,
            base_train_df=_base_df(),
            heldout_test_df=_heldout_df(),
            previously_used_anchor_ids=set(),
            policy=AnchorPolicy(anchor_ratio=0.1, allow_reuse=False),
            context=AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=1, seed=1),
        )


def test_insufficient_anchor_pool_handled_explicitly() -> None:
    pool = _anchor_pool_df(2)
    synth = _synthetic_df(20)

    with pytest.raises(AnchoringError, match="Insufficient anchors"):
        select_human_anchors(
            anchor_pool_df=pool,
            synthetic_df=synth,
            base_train_df=_base_df(),
            heldout_test_df=_heldout_df(),
            previously_used_anchor_ids=set(),
            policy=AnchorPolicy(anchor_ratio=0.5, allow_reuse=False),
            context=AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=1, seed=1),
        )


def test_anchor_metadata_written_correctly(tmp_path: Path) -> None:
    pool = _anchor_pool_df(20)
    synth = _synthetic_df(10)

    result = select_human_anchors(
        anchor_pool_df=pool,
        synthetic_df=synth,
        base_train_df=_base_df(),
        heldout_test_df=_heldout_df(),
        previously_used_anchor_ids=set(),
        policy=AnchorPolicy(anchor_ratio=0.2, allow_reuse=False),
        context=AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=1, seed=99),
    )

    mixed_path = tmp_path / "mixed.parquet"
    meta_path = tmp_path / "anchor_manifest.json"
    ids_path = tmp_path / "used_anchor_ids.json"
    save_anchoring_artifacts(
        result=result,
        mixed_dataset_path=mixed_path,
        metadata_path=meta_path,
        used_anchor_ids_path=ids_path,
    )

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    ids = json.loads(ids_path.read_text(encoding="utf-8"))

    assert payload["model_name"] == "qwen2.5:0.5b"
    assert payload["generation"] == 1
    assert payload["anchor_count"] == 2
    assert payload["synthetic_count"] == 10
    assert len(ids) == 2


def test_realized_ratio_computed_correctly() -> None:
    pool = _anchor_pool_df(20)
    synth = _synthetic_df(10)

    result = select_human_anchors(
        anchor_pool_df=pool,
        synthetic_df=synth,
        base_train_df=_base_df(),
        heldout_test_df=_heldout_df(),
        previously_used_anchor_ids=set(),
        policy=AnchorPolicy(anchor_ratio=0.2, allow_reuse=False),
        context=AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=1, seed=99),
    )

    assert result.metadata.anchor_count == 2
    assert result.metadata.anchor_ratio_realized == 0.2


def test_ratio_rounds_to_zero_is_controlled() -> None:
    pool = _anchor_pool_df(20)
    synth = _synthetic_df(3)

    result = select_human_anchors(
        anchor_pool_df=pool,
        synthetic_df=synth,
        base_train_df=_base_df(),
        heldout_test_df=_heldout_df(),
        previously_used_anchor_ids=set(),
        policy=AnchorPolicy(anchor_ratio=0.01, allow_reuse=False),
        context=AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=1, seed=99),
    )
    assert result.metadata.anchor_count == 0
    assert result.metadata.anchor_ratio_realized == 0.0


def test_duplicate_anchor_pool_ids_fail() -> None:
    pool = _anchor_pool_df(20)
    pool.loc[1, "example_id"] = pool.loc[0, "example_id"]
    synth = _synthetic_df(10)

    with pytest.raises(AnchoringError, match="duplicate"):
        select_human_anchors(
            anchor_pool_df=pool,
            synthetic_df=synth,
            base_train_df=_base_df(),
            heldout_test_df=_heldout_df(),
            previously_used_anchor_ids=set(),
            policy=AnchorPolicy(anchor_ratio=0.2, allow_reuse=False),
            context=AnchorSelectionContext(model_name="qwen2.5:0.5b", branch="anchor_10", generation=1, seed=99),
        )
