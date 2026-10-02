"""Image geometry and basic visual-quality measurements."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _laplacian_variance(grayscale: np.ndarray) -> float:
    """Return relative sharpness as the variance of a 4-neighbor Laplacian.

    Args:
        grayscale: Two-dimensional normalized luminance array at least 3x3.

    Returns:
        Relative sharpness score; it is not an absolute blur classification.
    """
    if grayscale.ndim != 2:
        raise ValueError("grayscale must be a two-dimensional array")
    if min(grayscale.shape) < 3:
        raise ValueError("images must be at least 3x3 pixels")

    padded = np.pad(grayscale, 1, mode="reflect")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4 * padded[1:-1, 1:-1]
    )
    return float(laplacian.var())


def measure_image_quality(image_path: Path) -> dict[str, object]:
    """Measure image geometry, luminance, contrast, and relative sharpness.

    Args:
        image_path: Readable local image file.

    Returns:
        ID, dimensions, aspect ratio, mean luminance, 5-95% contrast, and sharpness.
    """
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    height, width = rgb.shape[:2]
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    lower, upper = np.quantile(luminance, [0.05, 0.95])
    return {
        "Image_ID": image_path.stem,
        "image_width": width,
        "image_height": height,
        "image_aspect_ratio": width / height,
        "brightness": float(luminance.mean()),
        "contrast": float(upper - lower),
        "sharpness": _laplacian_variance(luminance),
    }


def analyze_image_quality(image_paths: Iterable[Path]) -> pd.DataFrame:
    """Return deterministic geometry and quality measurements for image paths.

    Args:
        image_paths: Paths to unique image files.

    Returns:
        ID-sorted measurements, or a typed empty table for an empty iterable.
    """
    rows = [measure_image_quality(Path(path)) for path in image_paths]
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "Image_ID",
                "image_width",
                "image_height",
                "image_aspect_ratio",
                "brightness",
                "contrast",
                "sharpness",
            ]
        )
    if result["Image_ID"].duplicated().any():
        duplicates = result.loc[result["Image_ID"].duplicated(), "Image_ID"].tolist()
        raise ValueError(f"Duplicate Image_ID values: {duplicates[:10]}")
    return result.sort_values("Image_ID", ignore_index=True)
