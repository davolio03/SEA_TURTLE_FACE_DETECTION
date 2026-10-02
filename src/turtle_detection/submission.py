"""Generate and validate single-box Zindi submission files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from .box_geometry import pixel_xyxy_to_normalized_xywh, valid_xyxy_mask
from .data_pipeline import DEFAULT_TARGET_SIZE, LetterboxTransform
from .detection import CLASS_ID_TURTLE_FACE, inverse_letterbox_xyxy


SUBMISSION_COLUMNS = ("Image_ID", "x", "y", "w", "h")
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_submission_inputs(
    sample_submission_path: str | Path,
    image_dir: str | Path,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load sample IDs in original order and resolve exactly one image per ID.

    Args:
        sample_submission_path: Local CSV with the required competition columns.
        image_dir: Directory of local test images keyed by image ID.

    Returns:
        Original sample table and one resolved path per row in the same order.
    """
    sample_path = Path(sample_submission_path)
    images_root = Path(image_dir)
    sample = pd.read_csv(sample_path, dtype={"Image_ID": "string"})
    if tuple(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"sample submission columns must be {list(SUBMISSION_COLUMNS)}"
        )
    if sample.empty:
        raise ValueError("sample submission contains no image IDs")
    if sample["Image_ID"].isna().any() or sample["Image_ID"].str.strip().eq("").any():
        raise ValueError("sample submission contains an empty image ID")
    if sample["Image_ID"].duplicated().any():
        raise ValueError("sample submission contains duplicate image IDs")
    if not images_root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {images_root}")

    image_index: dict[str, list[Path]] = {}
    for path in images_root.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            image_index.setdefault(path.stem.casefold(), []).append(path)

    resolved: list[Path] = []
    for image_id in sample["Image_ID"].astype(str):
        matches = image_index.get(image_id.casefold(), [])
        if len(matches) != 1:
            raise ValueError(
                f"image ID {image_id!r} resolves to {len(matches)} files; expected exactly one"
            )
        resolved.append(matches[0])
    return sample, resolved


def predict_faster_rcnn_submission(
    model: nn.Module,
    sample: pd.DataFrame,
    image_paths: list[str | Path],
    device: torch.device,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    batch_size: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict one top-confidence class-1 box per image in original-frame xywh.

    Args:
        model: Trained detector returning post-NMS boxes, scores, and labels.
        sample: Sample-submission rows in the intended output order.
        image_paths: Image path aligned to each sample row.
        device: Device used for batched inference.
        target_size: Letterbox canvas as ``(height, width)``.
        batch_size: Maximum images processed in one inference batch.

    Returns:
        Validated competition-format predictions and per-image diagnostics.
    """
    if tuple(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"sample columns must be {list(SUBMISSION_COLUMNS)}")
    if len(sample) != len(image_paths):
        raise ValueError("sample rows and image paths must have the same length")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sample["Image_ID"].isna().any() or sample["Image_ID"].duplicated().any():
        raise ValueError("sample image IDs must be present and unique")

    transform = LetterboxTransform(target_size=target_size, normalize=False)
    model = model.to(device)
    model.eval()
    predicted_boxes: list[np.ndarray] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for start in range(0, len(sample), batch_size):
        stop = min(start + batch_size, len(sample))
        tensors: list[torch.Tensor] = []
        original_sizes: list[tuple[int, int]] = []
        for path_value in image_paths[start:stop]:
            path = Path(path_value)
            with Image.open(path) as opened_image:
                image = opened_image.convert("RGB")
            original_width, original_height = image.size
            if original_width <= 0 or original_height <= 0:
                raise ValueError(f"image has invalid dimensions: {path}")
            # A full-image box lets the training transform create the same letterboxed pixels.
            tensor, _ = transform(
                image,
                np.asarray([0.0, 0.0, original_width, original_height]),
            )
            tensors.append(tensor)
            original_sizes.append((original_height, original_width))

        model_inputs = [tensor.to(device, non_blocking=True) for tensor in tensors]
        with torch.inference_mode():
            outputs = model(model_inputs)
        if len(outputs) != len(model_inputs):
            raise RuntimeError("detector returned a different number of outputs than inputs")

        for offset, (output, image_size) in enumerate(zip(outputs, original_sizes, strict=True)):
            image_id = str(sample.iloc[start + offset]["Image_ID"])
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            if not (len(boxes) == len(scores) == len(labels)):
                raise ValueError(f"detector output arrays have inconsistent lengths for {image_id}")

            class_one = np.flatnonzero(labels == CLASS_ID_TURTLE_FACE)
            selected_box: np.ndarray | None = None
            selected_score: float | None = None
            if len(class_one):
                selected_index = int(class_one[np.argmax(scores[class_one])])
                candidate = np.asarray(boxes[selected_index], dtype=np.float64)
                if valid_xyxy_mask(candidate).item() and np.isfinite(scores[selected_index]):
                    selected_box = candidate
                    selected_score = float(scores[selected_index])

            original_height, original_width = image_size
            original_box = (
                inverse_letterbox_xyxy(
                    selected_box,
                    original_height,
                    original_width,
                    target_size,
                )
                if selected_box is not None
                else None
            )
            if original_box is None:
                # Keep a missed detection in the submission and report its absence in coverage.
                normalized_xywh = np.zeros(4, dtype=np.float64)
                has_detection = False
            else:
                normalized_xywh = pixel_xyxy_to_normalized_xywh(
                    original_box,
                    original_width,
                    original_height,
                )
                has_detection = True

            predicted_boxes.append(normalized_xywh)
            diagnostic_rows.append(
                {
                    "Image_ID": image_id,
                    "confidence": selected_score,
                    "class_one_detections": int(len(class_one)),
                    "has_detection": has_detection,
                    "original_height": original_height,
                    "original_width": original_width,
                    "x": float(normalized_xywh[0]),
                    "y": float(normalized_xywh[1]),
                    "w": float(normalized_xywh[2]),
                    "h": float(normalized_xywh[3]),
                }
            )

    submission = sample.loc[:, ["Image_ID"]].copy().reset_index(drop=True)
    box_values = np.asarray(predicted_boxes, dtype=np.float64)
    for column_index, column in enumerate(("x", "y", "w", "h")):
        submission[column] = box_values[:, column_index]
    validate_submission(submission, expected_ids=sample["Image_ID"].astype(str).tolist())
    diagnostics = pd.DataFrame(diagnostic_rows)
    return submission, diagnostics


def validate_submission(
    submission: pd.DataFrame,
    expected_ids: list[str] | None = None,
) -> None:
    """Check the competition schema, row identity/order, finite values, and box bounds.

    Args:
        submission: Candidate table with one normalized xywh row per image.
        expected_ids: Optional reference order, normally from the sample CSV.
    """
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"submission columns must be {list(SUBMISSION_COLUMNS)}")
    if submission.empty:
        raise ValueError("submission contains no rows")
    if submission["Image_ID"].isna().any() or submission["Image_ID"].duplicated().any():
        raise ValueError("submission image IDs must be present and unique")
    if expected_ids is not None and submission["Image_ID"].astype(str).tolist() != expected_ids:
        raise ValueError("submission IDs or their order do not match the sample submission")

    values = submission.loc[:, ["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("submission box values must all be finite")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("submission box values must be within [0, 1]")
    if (values[:, 0] + values[:, 2] > 1 + 1e-6).any():
        raise ValueError("submission boxes exceed the right image boundary")
    if (values[:, 1] + values[:, 3] > 1 + 1e-6).any():
        raise ValueError("submission boxes exceed the bottom image boundary")
    zero_rows = np.all(values == 0, axis=1)
    partial_zero_rows = (values[:, 2:] == 0).any(axis=1) & ~zero_rows
    if partial_zero_rows.any():
        raise ValueError("a no-detection box must use four zeros; other boxes need positive size")


def write_submission_csv(
    submission: pd.DataFrame,
    output_path: str | Path,
    expected_ids: list[str] | None = None,
) -> Path:
    """Write a validated submission without an index, replacing the target atomically.

    Args:
        submission: Candidate competition-format table.
        output_path: Local destination CSV path.
        expected_ids: Optional required sample IDs and row order.

    Returns:
        Destination path after round-trip validation and replacement.
    """
    validate_submission(submission, expected_ids=expected_ids)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        submission.to_csv(temporary, index=False)
        written = pd.read_csv(temporary, dtype={"Image_ID": "string"})
        validate_submission(written, expected_ids=expected_ids)
        if len(written) != len(submission):
            raise RuntimeError("written submission row count changed during serialization")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output
