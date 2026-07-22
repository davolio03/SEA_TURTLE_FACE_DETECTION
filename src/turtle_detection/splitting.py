"""Leakage-safe train and validation split utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign_split_groups(similarity_groups: pd.DataFrame) -> pd.DataFrame:
    """Assign a stable unique group to every singleton image."""
    required = {"Image_ID", "similarity_group"}
    missing = required - set(similarity_groups.columns)
    if missing:
        raise ValueError(f"Missing group columns: {sorted(missing)}")
    if similarity_groups["Image_ID"].duplicated().any():
        raise ValueError("Image_ID values must be unique")

    result = similarity_groups.copy()
    related = result["similarity_group"].ge(0)
    related_count = int(result.loc[related, "similarity_group"].nunique())
    singleton_ids = sorted(result.loc[~related, "Image_ID"].astype(str))
    singleton_groups = {
        image_id: related_count + index
        for index, image_id in enumerate(singleton_ids)
    }
    result["group_id"] = result.apply(
        lambda row: (
            int(row["similarity_group"])
            if row["similarity_group"] >= 0
            else singleton_groups[str(row["Image_ID"])]
        ),
        axis=1,
    )
    return result


def create_group_shuffle_split(
    dataframe: pd.DataFrame,
    groups: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create one reproducible split while keeping every group together."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if dataframe["Image_ID"].duplicated().any():
        raise ValueError("Training Image_ID values must be unique")

    assignments = dataframe[["Image_ID"]].merge(
        groups[["Image_ID", "group_id"]],
        on="Image_ID",
        how="left",
        validate="one_to_one",
    )
    if assignments["group_id"].isna().any():
        missing = assignments.loc[assignments["group_id"].isna(), "Image_ID"].tolist()
        raise ValueError(f"Missing groups for training images: {missing[:10]}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )
    train_indices, validation_indices = next(
        splitter.split(assignments, groups=assignments["group_id"])
    )
    assignments["split"] = "train"
    assignments.loc[validation_indices, "split"] = "validation"
    assert_no_group_leakage(assignments)
    return assignments.sort_values("Image_ID", ignore_index=True)


def assert_no_group_leakage(assignments: pd.DataFrame) -> None:
    """Raise when any group appears in more than one split."""
    split_counts = assignments.groupby("group_id")["split"].nunique()
    leaking_groups = split_counts[split_counts > 1].index.tolist()
    if leaking_groups:
        raise AssertionError(f"Groups present in multiple splits: {leaking_groups}")


def save_split(assignments: pd.DataFrame, destination: Path) -> None:
    """Persist deterministic split assignments as CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(destination, index=False)

