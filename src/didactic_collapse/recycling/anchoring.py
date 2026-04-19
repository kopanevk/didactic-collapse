from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class AnchoringError(ValueError):
    """Raised when strict human anchoring constraints are violated."""


@dataclass(frozen=True)
class AnchorPolicy:
    """Human anchoring policy for one branch condition."""

    anchor_ratio: float
    allow_reuse: bool = False


@dataclass(frozen=True)
class AnchorSelectionContext:
    """Lineage context used for deterministic anchor selection."""

    model_name: str
    branch: str
    generation: int
    seed: int


class AnchorSelectionMetadata(BaseModel):
    """Auditable metadata for one anchoring operation."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    branch: str
    generation: int
    seed: int
    anchor_ratio_requested: float = Field(ge=0.0, le=1.0)
    anchor_ratio_realized: float = Field(ge=0.0)
    synthetic_count: int = Field(ge=0)
    anchor_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    remaining_anchor_pool_size: int = Field(ge=0)
    selected_anchor_ids: list[str]
    reused_anchor_ids: list[str]


@dataclass(frozen=True)
class AnchoringResult:
    mixed_training_df: pd.DataFrame
    selected_anchors_df: pd.DataFrame
    metadata: AnchorSelectionMetadata


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise AnchoringError(f"{name} missing required columns: {sorted(missing)}")


def _check_unique_example_ids(df: pd.DataFrame, name: str) -> None:
    if df["example_id"].duplicated().any():
        n_dup = int(df["example_id"].duplicated().sum())
        raise AnchoringError(f"{name} contains duplicate example_id values: {n_dup}")


def compute_target_anchor_count(*, synthetic_count: int, anchor_ratio: float) -> int:
    """Compute requested anchor count by ratio with deterministic rounding."""
    if synthetic_count < 0:
        raise AnchoringError("synthetic_count must be non-negative")
    if not (0.0 <= anchor_ratio <= 1.0):
        raise AnchoringError(f"anchor_ratio must be in [0,1], got {anchor_ratio}")
    return int(round(synthetic_count * anchor_ratio))


def _deterministic_random_state(*, context: AnchorSelectionContext, policy: AnchorPolicy) -> int:
    payload = (
        f"{context.seed}|{context.model_name}|{context.branch}|{context.generation}|"
        f"ratio={policy.anchor_ratio:.8f}|allow_reuse={policy.allow_reuse}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def validate_anchor_split_safety(
    *,
    selected_anchors_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    base_train_df: pd.DataFrame,
    heldout_test_df: pd.DataFrame,
) -> None:
    """Validate strict disjointness against forbidden pools."""
    selected_ids = set(selected_anchors_df["example_id"])
    synthetic_ids = set(synthetic_df["example_id"])
    base_ids = set(base_train_df["example_id"])
    heldout_ids = set(heldout_test_df["example_id"])

    if selected_ids.intersection(synthetic_ids):
        raise AnchoringError("Selected anchors overlap with synthetic dataset")
    if selected_ids.intersection(base_ids):
        raise AnchoringError("Selected anchors overlap with base_train")
    if selected_ids.intersection(heldout_ids):
        raise AnchoringError("Selected anchors overlap with heldout_test")


def _validate_anchor_pool_integrity(
    *,
    anchor_pool_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    base_train_df: pd.DataFrame,
    heldout_test_df: pd.DataFrame,
) -> None:
    """Fail fast when anchor_pool itself is contaminated with forbidden IDs."""
    pool_ids = set(anchor_pool_df["example_id"])
    synth_ids = set(synthetic_df["example_id"])
    base_ids = set(base_train_df["example_id"])
    heldout_ids = set(heldout_test_df["example_id"])

    overlap_pool_base = pool_ids.intersection(base_ids)
    overlap_pool_heldout = pool_ids.intersection(heldout_ids)
    overlap_pool_synth = pool_ids.intersection(synth_ids)

    if overlap_pool_base or overlap_pool_heldout or overlap_pool_synth:
        raise AnchoringError(
            "anchor_pool integrity violation: overlap with forbidden pools "
            f"(pool∩base={len(overlap_pool_base)}, "
            f"pool∩heldout={len(overlap_pool_heldout)}, "
            f"pool∩synthetic={len(overlap_pool_synth)})"
        )


def select_human_anchors(
    *,
    anchor_pool_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    base_train_df: pd.DataFrame,
    heldout_test_df: pd.DataFrame,
    previously_used_anchor_ids: set[str],
    policy: AnchorPolicy,
    context: AnchorSelectionContext,
) -> AnchoringResult:
    """Select anchors with deterministic no-reuse policy and full safety validation."""
    required_pool = {"example_id", "question", "answer_gold"}
    required_synth = {"example_id", "question", "answer_for_training", "source"}
    required_simple = {"example_id"}

    _require_columns(anchor_pool_df, required_pool, "anchor_pool_df")
    _require_columns(synthetic_df, required_synth, "synthetic_df")
    _require_columns(base_train_df, required_simple, "base_train_df")
    _require_columns(heldout_test_df, required_simple, "heldout_test_df")

    _check_unique_example_ids(anchor_pool_df, "anchor_pool_df")
    _check_unique_example_ids(synthetic_df, "synthetic_df")
    _check_unique_example_ids(base_train_df, "base_train_df")
    _check_unique_example_ids(heldout_test_df, "heldout_test_df")
    _validate_anchor_pool_integrity(
        anchor_pool_df=anchor_pool_df,
        synthetic_df=synthetic_df,
        base_train_df=base_train_df,
        heldout_test_df=heldout_test_df,
    )

    synthetic_count = len(synthetic_df)
    if synthetic_count == 0:
        raise AnchoringError("synthetic dataset is empty; cannot apply anchoring")

    target_anchor_count = compute_target_anchor_count(
        synthetic_count=synthetic_count,
        anchor_ratio=policy.anchor_ratio,
    )

    if target_anchor_count == 0:
        metadata = AnchorSelectionMetadata(
            model_name=context.model_name,
            branch=context.branch,
            generation=context.generation,
            seed=context.seed,
            anchor_ratio_requested=policy.anchor_ratio,
            anchor_ratio_realized=0.0,
            synthetic_count=synthetic_count,
            anchor_count=0,
            total_count=synthetic_count,
            remaining_anchor_pool_size=len(anchor_pool_df),
            selected_anchor_ids=[],
            reused_anchor_ids=[],
        )
        return AnchoringResult(
            mixed_training_df=synthetic_df.copy(),
            selected_anchors_df=anchor_pool_df.iloc[0:0].copy(),
            metadata=metadata,
        )

    candidates = anchor_pool_df.sort_values("example_id", kind="mergesort").copy()
    reused_in_candidates = sorted(set(candidates["example_id"]).intersection(previously_used_anchor_ids))

    if not policy.allow_reuse:
        candidates = candidates[~candidates["example_id"].isin(previously_used_anchor_ids)].copy()
    elif reused_in_candidates:
        # Explicitly allowed; accounted in metadata only.
        pass

    available = len(candidates)
    if available < target_anchor_count:
        raise AnchoringError(
            "Insufficient anchors under current no-reuse policy: "
            f"requested={target_anchor_count}, available={available}, "
            f"allow_reuse={policy.allow_reuse}"
        )

    random_state = _deterministic_random_state(context=context, policy=policy)
    selected = candidates.sample(n=target_anchor_count, random_state=random_state).copy()
    selected = selected.sort_values("example_id", kind="mergesort").reset_index(drop=True)

    validate_anchor_split_safety(
        selected_anchors_df=selected,
        synthetic_df=synthetic_df,
        base_train_df=base_train_df,
        heldout_test_df=heldout_test_df,
    )

    anchors_for_train = selected[["example_id", "question", "answer_gold"]].rename(
        columns={"answer_gold": "answer_for_training"}
    )
    anchors_for_train["source"] = "human_anchor"

    mixed = pd.concat([synthetic_df.copy(), anchors_for_train], ignore_index=True)

    selected_ids = selected["example_id"].astype(str).tolist()
    reused_selected = sorted(set(selected_ids).intersection(previously_used_anchor_ids))

    metadata = AnchorSelectionMetadata(
        model_name=context.model_name,
        branch=context.branch,
        generation=context.generation,
        seed=context.seed,
        anchor_ratio_requested=policy.anchor_ratio,
        anchor_ratio_realized=(len(selected_ids) / synthetic_count),
        synthetic_count=synthetic_count,
        anchor_count=len(selected_ids),
        total_count=len(mixed),
        remaining_anchor_pool_size=available - len(selected_ids),
        selected_anchor_ids=selected_ids,
        reused_anchor_ids=reused_selected,
    )

    return AnchoringResult(
        mixed_training_df=mixed,
        selected_anchors_df=selected,
        metadata=metadata,
    )


def save_anchoring_artifacts(
    *,
    result: AnchoringResult,
    mixed_dataset_path: Path,
    metadata_path: Path,
    used_anchor_ids_path: Path | None = None,
) -> None:
    """Persist mixed dataset + metadata + optional used IDs artifact."""
    mixed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    result.mixed_training_df.to_parquet(mixed_dataset_path, index=False)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(result.metadata.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if used_anchor_ids_path is not None:
        used_anchor_ids_path.parent.mkdir(parents=True, exist_ok=True)
        used_anchor_ids_path.write_text(
            json.dumps(sorted(result.metadata.selected_anchor_ids), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
