"""Windows-safe dataset, transforms, and DataLoader helpers for detection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .box_geometry import (
    clip_xyxy_to_image,
    normalized_xywh_to_pixel_xyxy,
    pixel_xyxy_to_normalized_xywh,
    validate_normalized_xywh,
    validate_xyxy,
)


DEFAULT_TARGET_SIZE = (384, 512)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LetterboxTransform:
    """Resize an image without distortion and update one xyxy box.

    ``target_size`` is expressed as ``(height, width)``. The transform uses
    black padding and can apply a horizontal flip before normalization.
    """

    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE
    horizontal_flip_probability: float = 0.0
    normalize: bool = True
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD

    def __post_init__(self) -> None:
        height, width = self.target_size
        if height <= 0 or width <= 0:
            raise ValueError("target_size dimensions must be positive")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must contain three channels")
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive")

    def __call__(
        self,
        image: Image.Image,
        box_xyxy: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        image = image.convert("RGB")
        original_width, original_height = image.size
        validate_xyxy(box_xyxy, original_width, original_height)

        target_height, target_width = self.target_size
        scale = min(target_width / original_width, target_height / original_height)
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        resized = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )

        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        canvas.paste(resized, (left, top))

        transformed_box = np.asarray(box_xyxy, dtype=np.float64).copy()
        transformed_box[[0, 2]] = transformed_box[[0, 2]] * scale + left
        transformed_box[[1, 3]] = transformed_box[[1, 3]] * scale + top

        if random.random() < self.horizontal_flip_probability:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x1, _, x2, _ = transformed_box
            transformed_box[0] = target_width - x2
            transformed_box[2] = target_width - x1

        transformed_box = clip_xyxy_to_image(
            transformed_box,
            target_width,
            target_height,
        )
        validate_xyxy(transformed_box, target_width, target_height)

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.normalize:
            mean = torch.tensor(self.mean, dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor(self.std, dtype=tensor.dtype).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor, transformed_box.astype(np.float32)


class SeaTurtleDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """Read one bounding-box annotation per image from the project dataset."""

    def __init__(
        self,
        annotations_path: str | Path,
        split_path: str | Path,
        image_dir: str | Path,
        split_name: str,
        transform: LetterboxTransform | None = None,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.split_path = Path(split_path)
        self.image_dir = Path(image_dir)
        self.split_name = split_name
        self.transform = transform or LetterboxTransform()
        self.records = self._load_records()

    def _load_records(self) -> pd.DataFrame:
        annotations = pd.read_csv(self.annotations_path)
        required_annotations = {"Image_ID", "x", "y", "w", "h"}
        missing_annotations = required_annotations.difference(annotations.columns)
        if missing_annotations:
            raise ValueError(f"annotation CSV missing columns: {sorted(missing_annotations)}")

        splits = pd.read_csv(self.split_path)
        required_splits = {"Image_ID", "split"}
        missing_splits = required_splits.difference(splits.columns)
        if missing_splits:
            raise ValueError(f"split CSV missing columns: {sorted(missing_splits)}")

        annotations = annotations.copy()
        splits = splits.copy()
        annotations["Image_ID"] = annotations["Image_ID"].astype(str)
        splits["Image_ID"] = splits["Image_ID"].astype(str)
        splits = splits.loc[splits["split"].eq(self.split_name), ["Image_ID", "split"]]
        records = splits.merge(annotations, on="Image_ID", how="left", validate="one_to_one")
        if records[["x", "y", "w", "h"]].isna().any().any():
            raise ValueError("split contains image IDs without annotations")
        normalized_boxes = records[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        validate_normalized_xywh(normalized_boxes)

        missing_images = [
            image_id
            for image_id in records["Image_ID"]
            if not self._image_path(image_id).is_file()
        ]
        if missing_images:
            preview = ", ".join(missing_images[:3])
            raise FileNotFoundError(
                f"missing image files for {len(missing_images)} records: {preview}"
            )
        return records.reset_index(drop=True)

    def _image_path(self, image_id: str) -> Path:
        for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg", ".png"):
            candidate = self.image_dir / f"{image_id}{suffix}"
            if candidate.is_file():
                return candidate
        return self.image_dir / f"{image_id}.JPG"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        row = self.records.iloc[index]
        image_id = str(row["Image_ID"])
        with Image.open(self._image_path(image_id)) as opened_image:
            image = opened_image.convert("RGB")
        original_width, original_height = image.size
        normalized_box = np.asarray([row["x"], row["y"], row["w"], row["h"]], dtype=np.float64)
        box_xyxy = normalized_xywh_to_pixel_xyxy(
            normalized_box,
            original_width,
            original_height,
        )
        image_tensor, transformed_box = self.transform(image, box_xyxy)
        target_height, target_width = self.transform.target_size
        transformed_normalized = pixel_xyxy_to_normalized_xywh(
            transformed_box,
            target_width,
            target_height,
        ).astype(np.float32)
        target: dict[str, Any] = {
            "boxes": torch.as_tensor(transformed_box, dtype=torch.float32).view(1, 4),
            "labels": torch.ones(1, dtype=torch.int64),
            "image_id": image_id,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.int64),
            "normalized_xywh": torch.as_tensor(transformed_normalized, dtype=torch.float32),
            "original_normalized_xywh": torch.as_tensor(normalized_box, dtype=torch.float32),
        }
        return image_tensor, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Keep variable-length detection targets as lists instead of stacking them."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_data_loaders(
    project_root: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Create Windows-safe train and validation loaders from project paths."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    root = Path(project_root)
    annotations_path = root / "data" / "raw" / "Train.csv"
    split_path = root / "data" / "splits" / "train_validation.csv"
    image_dir = root / "data" / "raw" / "IMAGES_512"
    train_dataset = SeaTurtleDataset(
        annotations_path,
        split_path,
        image_dir,
        split_name="train",
        transform=LetterboxTransform(target_size, horizontal_flip_probability=0.5),
    )
    validation_dataset = SeaTurtleDataset(
        annotations_path,
        split_path,
        image_dir,
        split_name="validation",
        transform=LetterboxTransform(target_size),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "collate_fn": detection_collate_fn,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    return train_loader, validation_loader
