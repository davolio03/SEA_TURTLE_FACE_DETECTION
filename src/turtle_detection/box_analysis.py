"""Feature engineering and extreme-case selection for normalized boxes."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

BOX_COLUMNS = ("x", "y", "w", "h")


def add_box_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized geometry features for each box.

    Args:
        dataframe: Rows with normalized top-left ``x, y, w, h`` columns.

    Returns:
        Copy with center, area, aspect ratio, border distance, and center distance.
    """
    missing = set(BOX_COLUMNS) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing box columns: {sorted(missing)}")

    result = dataframe.copy()
    result["center_x"] = result["x"] + result["w"] / 2
    result["center_y"] = result["y"] + result["h"] / 2
    result["relative_area"] = result["w"] * result["h"]
    result["box_aspect_ratio"] = result["w"] / result["h"]
    result["distance_to_border"] = pd.concat(
        [
            result["x"],
            result["y"],
            1 - (result["x"] + result["w"]),
            1 - (result["y"] + result["h"]),
        ],
        axis=1,
    ).min(axis=1)
    result["distance_to_center"] = np.hypot(
        result["center_x"] - 0.5,
        result["center_y"] - 0.5,
    )
    return result


def invalid_box_mask(dataframe: pd.DataFrame) -> pd.Series:
    """Return True for non-finite, empty, or out-of-bounds normalized boxes.

    Args:
        dataframe: Rows containing normalized top-left ``x, y, w, h`` columns.

    Returns:
        Boolean Series aligned to the input index.
    """
    boxes = dataframe.loc[:, BOX_COLUMNS]
    finite = np.isfinite(boxes.to_numpy()).all(axis=1)
    valid = (
        finite
        & dataframe["x"].ge(0)
        & dataframe["y"].ge(0)
        & dataframe["w"].gt(0)
        & dataframe["h"].gt(0)
        & dataframe["x"].add(dataframe["w"]).le(1)
        & dataframe["y"].add(dataframe["h"]).le(1)
    )
    return pd.Series(~valid, index=dataframe.index, name="invalid_box")


def select_extreme_boxes(
    dataframe: pd.DataFrame,
    count: int = 20,
) -> Mapping[str, pd.DataFrame]:
    """Select representative geometric extremes for visual review.

    Args:
        dataframe: Labeled rows containing valid normalized box columns.
        count: Maximum number of rows selected in each extreme category.

    Returns:
        Named samples for small/large, aspect-ratio, border, and center extremes.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    features = add_box_features(dataframe)
    return {
        "smallest": features.nsmallest(count, "relative_area"),
        "largest": features.nlargest(count, "relative_area"),
        "most_vertical": features.nsmallest(count, "box_aspect_ratio"),
        "most_horizontal": features.nlargest(count, "box_aspect_ratio"),
        "closest_to_border": features.nsmallest(count, "distance_to_border"),
        "furthest_from_center": features.nlargest(count, "distance_to_center"),
    }

