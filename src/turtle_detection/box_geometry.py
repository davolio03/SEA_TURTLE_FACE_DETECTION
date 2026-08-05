"""Coordinate conversion, clipping, validation, and IoU for bounding boxes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BoxArray = np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def _as_box_array(boxes: BoxArray) -> np.ndarray:
    result = np.asarray(boxes, dtype=np.float64)
    if result.ndim == 0 or result.shape[-1] != 4:
        raise ValueError("boxes must have shape (..., 4)")
    return result


def _validate_image_size(image_width: float, image_height: float) -> None:
    dimensions = np.asarray([image_width, image_height], dtype=np.float64)
    if not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("image dimensions must be finite and positive")


def normalized_xywh_to_pixel_xyxy(
    boxes: BoxArray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Convert normalized top-left xywh boxes to pixel xyxy coordinates."""
    _validate_image_size(image_width, image_height)
    source = _as_box_array(boxes)
    result = np.empty_like(source)
    result[..., 0] = source[..., 0] * image_width
    result[..., 1] = source[..., 1] * image_height
    result[..., 2] = (source[..., 0] + source[..., 2]) * image_width
    result[..., 3] = (source[..., 1] + source[..., 3]) * image_height
    return result


def pixel_xyxy_to_normalized_xywh(
    boxes: BoxArray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Convert pixel xyxy boxes to normalized top-left xywh coordinates."""
    _validate_image_size(image_width, image_height)
    source = _as_box_array(boxes)
    result = np.empty_like(source)
    result[..., 0] = source[..., 0] / image_width
    result[..., 1] = source[..., 1] / image_height
    result[..., 2] = (source[..., 2] - source[..., 0]) / image_width
    result[..., 3] = (source[..., 3] - source[..., 1]) / image_height
    return result


def clip_xyxy_to_image(
    boxes: BoxArray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Clip pixel xyxy coordinates to the closed image boundary."""
    _validate_image_size(image_width, image_height)
    result = _as_box_array(boxes).copy()
    result[..., (0, 2)] = np.clip(result[..., (0, 2)], 0, image_width)
    result[..., (1, 3)] = np.clip(result[..., (1, 3)], 0, image_height)
    return result


def valid_normalized_xywh_mask(boxes: BoxArray) -> np.ndarray:
    """Return validity for finite, positive, in-bounds normalized xywh boxes."""
    values = _as_box_array(boxes)
    finite = np.isfinite(values).all(axis=-1)
    return (
        finite
        & (values[..., 0] >= 0)
        & (values[..., 1] >= 0)
        & (values[..., 2] > 0)
        & (values[..., 3] > 0)
        & (values[..., 0] + values[..., 2] <= 1)
        & (values[..., 1] + values[..., 3] <= 1)
    )


def valid_xyxy_mask(
    boxes: BoxArray,
    image_width: float | None = None,
    image_height: float | None = None,
) -> np.ndarray:
    """Return validity for finite pixel xyxy boxes, optionally within an image."""
    if (image_width is None) != (image_height is None):
        raise ValueError("provide both image dimensions or neither")
    if image_width is not None and image_height is not None:
        _validate_image_size(image_width, image_height)

    values = _as_box_array(boxes)
    valid = (
        np.isfinite(values).all(axis=-1)
        & (values[..., 2] > values[..., 0])
        & (values[..., 3] > values[..., 1])
    )
    if image_width is not None and image_height is not None:
        valid &= (
            (values[..., 0] >= 0)
            & (values[..., 1] >= 0)
            & (values[..., 2] <= image_width)
            & (values[..., 3] <= image_height)
        )
    return valid


def validate_normalized_xywh(boxes: BoxArray) -> None:
    """Raise when any normalized xywh box is empty, non-finite, or out of bounds."""
    valid = valid_normalized_xywh_mask(boxes)
    if not np.asarray(valid).all():
        raise ValueError("normalized xywh boxes must be finite, positive, and in bounds")


def validate_xyxy(
    boxes: BoxArray,
    image_width: float | None = None,
    image_height: float | None = None,
) -> None:
    """Raise when any xyxy box is empty, non-finite, or outside given bounds."""
    valid = valid_xyxy_mask(boxes, image_width, image_height)
    if not np.asarray(valid).all():
        raise ValueError("xyxy boxes must be finite, positive, and within bounds")


def intersection_over_union_xyxy(
    first_boxes: BoxArray,
    second_boxes: BoxArray,
) -> np.ndarray:
    """Compute aligned IoU for broadcast-compatible valid xyxy boxes."""
    first = _as_box_array(first_boxes)
    second = _as_box_array(second_boxes)
    validate_xyxy(first)
    validate_xyxy(second)

    try:
        intersection_min = np.maximum(first[..., :2], second[..., :2])
        intersection_max = np.minimum(first[..., 2:], second[..., 2:])
    except ValueError as error:
        raise ValueError("box arrays must be broadcast-compatible") from error

    intersection_size = np.clip(intersection_max - intersection_min, 0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_area = (first[..., 2] - first[..., 0]) * (
        first[..., 3] - first[..., 1]
    )
    second_area = (second[..., 2] - second[..., 0]) * (
        second[..., 3] - second[..., 1]
    )
    union = first_area + second_area - intersection
    return intersection / union
