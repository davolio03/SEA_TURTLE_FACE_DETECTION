"""Leakage-safe train and validation split utilities."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign_split_groups(similarity_groups: pd.DataFrame) -> pd.DataFrame:
    """Assign a stable unique group to every singleton image.

    Args:
        similarity_groups: Rows containing ``Image_ID`` and reviewed group IDs;
            negative IDs denote independent singleton images.

    Returns:
        Copy with non-overlapping integer ``group_id`` values for every image.
    """
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
    """Create one reproducible split while keeping every group together.

    Args:
        dataframe: Unique labeled image rows containing ``Image_ID``.
        groups: Complete image-to-group mapping with ``group_id`` values.
        validation_size: Approximate fraction of groups assigned to validation.
        random_state: Seed supplied to scikit-learn's group splitter.

    Returns:
        Sorted image IDs with group IDs and train/validation labels.
    """
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
    """Raise when any group appears in more than one split.

    Args:
        assignments: Rows containing ``group_id`` and ``split`` columns.
    """
    split_counts = assignments.groupby("group_id")["split"].nunique()
    leaking_groups = split_counts[split_counts > 1].index.tolist()
    if leaking_groups:
        raise AssertionError(f"Groups present in multiple splits: {leaking_groups}")


def save_split(assignments: pd.DataFrame, destination: Path) -> None:
    """Persist deterministic split assignments as CSV.

    Args:
        assignments: Image/group/split table to persist.
        destination: Local CSV path; parent directories are created if needed.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(destination, index=False)


def _standardized_mean_difference(
    train_values: np.ndarray,
    validation_values: np.ndarray,
) -> float:
    """Measure train-to-validation mean displacement in pooled SD units.

    Args:
        train_values: Finite numeric values from the training partition.
        validation_values: Finite numeric values from the validation partition.

    Returns:
        Signed standardized mean difference, or infinity for unequal constants.
    """
    pooled_variance = (train_values.var() + validation_values.var()) / 2
    if pooled_variance == 0:
        return 0.0 if train_values.mean() == validation_values.mean() else np.inf
    return float(
        (validation_values.mean() - train_values.mean())
        / np.sqrt(pooled_variance)
    )


def _wasserstein_distance(
    first_values: np.ndarray,
    second_values: np.ndarray,
) -> float:
    """Compute the exact one-dimensional empirical Wasserstein distance."""
    combined = np.sort(np.concatenate([first_values, second_values]))
    if len(combined) < 2:
        return 0.0
    deltas = np.diff(combined)
    first_cdf = np.searchsorted(
        np.sort(first_values), combined[:-1], side="right"
    ) / len(first_values)
    second_cdf = np.searchsorted(
        np.sort(second_values), combined[:-1], side="right"
    ) / len(second_values)
    return float(np.sum(np.abs(first_cdf - second_cdf) * deltas))


def summarize_continuous_split_balance(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    split_column: str = "split",
) -> pd.DataFrame:
    """Compare train and validation distributions with robust summaries."""
    required = set(columns) | {split_column}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing balance columns: {sorted(missing)}")

    labels = set(dataframe[split_column].dropna().unique())
    if labels != {"train", "validation"}:
        raise ValueError("split column must contain train and validation")

    rows = []
    for column in columns:
        train = dataframe.loc[dataframe[split_column].eq("train"), column]
        validation = dataframe.loc[
            dataframe[split_column].eq("validation"), column
        ]
        train_values = train.dropna().to_numpy(dtype=float)
        validation_values = validation.dropna().to_numpy(dtype=float)
        if not len(train_values) or not len(validation_values):
            raise ValueError(f"Column {column} has an empty split")
        if not (
            np.isfinite(train_values).all()
            and np.isfinite(validation_values).all()
        ):
            raise ValueError(f"Column {column} contains non-finite values")

        pooled = np.concatenate([train_values, validation_values])
        pooled_iqr = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
        normalization_scale = pooled_iqr if pooled_iqr > 0 else float(pooled.std())
        wasserstein = _wasserstein_distance(train_values, validation_values)
        normalized_wasserstein = (
            wasserstein / normalization_scale
            if normalization_scale > 0
            else (0.0 if wasserstein == 0 else np.inf)
        )
        rows.append(
            {
                "feature": column,
                "train_mean": train_values.mean(),
                "validation_mean": validation_values.mean(),
                "train_median": np.median(train_values),
                "validation_median": np.median(validation_values),
                "train_q05": np.quantile(train_values, 0.05),
                "validation_q05": np.quantile(validation_values, 0.05),
                "train_q95": np.quantile(train_values, 0.95),
                "validation_q95": np.quantile(validation_values, 0.95),
                "standardized_mean_difference": _standardized_mean_difference(
                    train_values, validation_values
                ),
                "normalized_wasserstein": normalized_wasserstein,
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def summarize_categorical_split_balance(
    dataframe: pd.DataFrame,
    column: str,
    split_column: str = "split",
) -> pd.DataFrame:
    """Compare category proportions between train and validation."""
    required = {column, split_column}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing categorical balance columns: {sorted(missing)}")
    labels = set(dataframe[split_column].dropna().unique())
    if labels != {"train", "validation"}:
        raise ValueError("split column must contain train and validation")

    proportions = pd.crosstab(
        dataframe[column],
        dataframe[split_column],
        normalize="columns",
    )
    for label in ("train", "validation"):
        if label not in proportions:
            proportions[label] = 0.0
    proportions = proportions[["train", "validation"]]
    proportions["absolute_difference"] = (
        proportions["validation"] - proportions["train"]
    ).abs()
    return proportions


def select_distribution_extremes(
    dataframe: pd.DataFrame,
    criteria: Mapping[str, tuple[str, str]],
    tail_fraction: float = 0.025,
) -> tuple[Mapping[str, pd.DataFrame], pd.DataFrame]:
    """Select global distribution tails using common thresholds for all splits."""
    if not 0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be between 0 and 0.5")

    selections: dict[str, pd.DataFrame] = {}
    threshold_rows = []
    for name, (column, direction) in criteria.items():
        if column not in dataframe:
            raise ValueError(f"Missing extreme-case column: {column}")
        if direction not in {"low", "high"}:
            raise ValueError("Extreme direction must be low or high")
        values = dataframe[column].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Column {column} must contain finite values")

        quantile = tail_fraction if direction == "low" else 1 - tail_fraction
        threshold = float(dataframe[column].quantile(quantile))
        mask = (
            dataframe[column].le(threshold)
            if direction == "low"
            else dataframe[column].ge(threshold)
        )
        selections[name] = dataframe.loc[mask].copy()
        threshold_rows.append(
            {
                "extreme_case": name,
                "feature": column,
                "direction": direction,
                "quantile": quantile,
                "threshold": threshold,
            }
        )
    return selections, pd.DataFrame(threshold_rows).set_index("extreme_case")
