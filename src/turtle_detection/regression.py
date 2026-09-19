"""Baseline bounding-box regression training and evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torchvision import models


@dataclass(frozen=True)
class RegressionMetrics:
    """Aggregated metrics and per-image values for one validation pass."""

    mean_loss: float
    mean_iou: float
    median_iou: float
    image_ids: tuple[str, ...]
    ious: tuple[float, ...]
    predictions: np.ndarray
    targets: np.ndarray


class BoxRegressor(nn.Module):
    """ResNet18 backbone with a four-value normalized xywh regression head."""

    def __init__(self, pretrained: bool = False, dropout: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        input_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_features, 4),
        )
        self.backbone = backbone

    def forward(self, images: Tensor) -> Tensor:
        """Return normalized xywh values bounded to the [0, 1] interval."""
        return torch.sigmoid(self.backbone(images))


def select_device() -> torch.device:
    """Select CUDA when available and otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stack_images(images: list[Tensor], device: torch.device) -> Tensor:
    if not images:
        raise ValueError("a batch must contain at least one image")
    return torch.stack(images).to(device, non_blocking=device.type == "cuda")


def _stack_targets(targets: list[dict[str, Any]], device: torch.device) -> Tensor:
    if not targets:
        raise ValueError("a batch must contain at least one target")
    return torch.stack([target["normalized_xywh"] for target in targets]).to(device)


def normalized_xywh_to_xyxy_tensor(boxes: Tensor) -> Tensor:
    """Convert normalized xywh tensors to clipped normalized xyxy tensors."""
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    if boxes.ndim < 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape (..., 4)")
    x1, y1, width, height = boxes.unbind(dim=-1)
    x2 = x1 + width
    y2 = y1 + height
    result = torch.stack((x1, y1, x2, y2), dim=-1).clone()
    result[..., (0, 2)] = result[..., (0, 2)].clamp(0.0, 1.0)
    result[..., (1, 3)] = result[..., (1, 3)].clamp(0.0, 1.0)
    return result


def intersection_over_union_tensor(first_boxes: Tensor, second_boxes: Tensor) -> Tensor:
    """Compute broadcast-compatible IoU for xyxy tensors."""
    if first_boxes.shape[-1] != 4 or second_boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape (..., 4)")
    first_min = first_boxes[..., :2]
    first_max = first_boxes[..., 2:]
    second_min = second_boxes[..., :2]
    second_max = second_boxes[..., 2:]
    intersection_min = torch.maximum(first_min, second_min)
    intersection_max = torch.minimum(first_max, second_max)
    intersection_size = (intersection_max - intersection_min).clamp_min(0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_size = (first_max - first_min).clamp_min(0.0)
    second_size = (second_max - second_min).clamp_min(0.0)
    first_area = first_size[..., 0] * first_size[..., 1]
    second_area = second_size[..., 0] * second_size[..., 1]
    union = first_area + second_area - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def regression_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute Smooth L1 loss in normalized xywh coordinates."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    return nn.functional.smooth_l1_loss(predictions, targets)


def _run_batch(
    model: nn.Module,
    batch: tuple[list[Tensor], list[dict[str, Any]]],
    device: torch.device,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    images, targets = batch
    stacked_images = _stack_images(images, device)
    stacked_targets = _stack_targets(targets, device)
    predictions = model(stacked_images)
    loss = regression_loss(predictions, stacked_targets)
    return loss, predictions, targets


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[list[Tensor], list[dict[str, Any]]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> float:
    """Train one epoch and return the image-weighted mean loss."""
    active_device = device or select_device()
    model.to(active_device)
    model.train()
    total_loss = 0.0
    total_images = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, _, targets = _run_batch(model, batch, active_device)
        loss.backward()
        optimizer.step()
        batch_size = len(targets)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_images += batch_size
    if total_images == 0:
        raise ValueError("the training loader produced no batches")
    return total_loss / total_images


@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    loader: Iterable[tuple[list[Tensor], list[dict[str, Any]]]],
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> RegressionMetrics:
    """Evaluate loss and IoU while retaining values for error analysis."""
    active_device = device or select_device()
    model.to(active_device)
    model.eval()
    losses: list[float] = []
    ious: list[float] = []
    image_ids: list[str] = []
    predictions_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        loss, predictions, targets = _run_batch(model, batch, active_device)
        target_tensor = _stack_targets(targets, active_device)
        prediction_xyxy = normalized_xywh_to_xyxy_tensor(predictions)
        target_xyxy = normalized_xywh_to_xyxy_tensor(target_tensor)
        batch_ious = intersection_over_union_tensor(prediction_xyxy, target_xyxy)
        losses.extend([float(loss.detach().cpu())] * len(targets))
        ious.extend(batch_ious.detach().cpu().tolist())
        image_ids.extend(str(target["image_id"]) for target in targets)
        predictions_out.extend(predictions.detach().cpu().numpy())
        targets_out.extend(target_tensor.detach().cpu().numpy())

    if not losses:
        raise ValueError("the validation loader produced no batches")
    return RegressionMetrics(
        mean_loss=float(np.mean(losses)),
        mean_iou=float(np.mean(ious)),
        median_iou=float(np.median(ious)),
        image_ids=tuple(image_ids),
        ious=tuple(float(value) for value in ious),
        predictions=np.asarray(predictions_out, dtype=np.float32),
        targets=np.asarray(targets_out, dtype=np.float32),
    )


def fit_regression_model(
    model: nn.Module,
    train_loader: Iterable[tuple[list[Tensor], list[dict[str, Any]]]],
    validation_loader: Iterable[tuple[list[Tensor], list[dict[str, Any]]]],
    epochs: int = 5,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> list[dict[str, float]]:
    """Fit a model and return one train/validation record per epoch."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    active_device = device or select_device()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            active_device,
            max_batches=max_train_batches,
        )
        validation = evaluate_regression(
            model,
            validation_loader,
            active_device,
            max_batches=max_validation_batches,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation.mean_loss,
                "validation_mean_iou": validation.mean_iou,
                "validation_median_iou": validation.median_iou,
            }
        )
    return history
