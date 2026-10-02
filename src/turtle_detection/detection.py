"""Faster R-CNN training, evaluation, and single-box inference helpers."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from .box_geometry import (
    intersection_over_union_xyxy,
    pixel_xyxy_to_normalized_xywh,
    valid_xyxy_mask,
)


CLASS_ID_TURTLE_FACE = 1
NUM_CLASSES_WITH_BACKGROUND = 2
DEFAULT_DETECTOR_SIZE = (384, 512)


def seed_experiment(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators.

    Args:
        seed: Integer seed shared across the supported random-number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_faster_rcnn_resnet50_fpn(
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = DEFAULT_DETECTOR_SIZE[0],
    max_size: int = DEFAULT_DETECTOR_SIZE[1],
    box_score_thresh: float = 0.0,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
) -> nn.Module:
    """Build TorchVision Faster R-CNN with a two-class ROI prediction head.

    Args:
        pretrained: Whether to initialize from the COCO detector weights.
        trainable_backbone_layers: Number of final ResNet stages to update.
        min_size: Short-side resize used by TorchVision's detector transform.
        max_size: Maximum long-side resize used by the detector transform.
        box_score_thresh: Minimum retained detection score before NMS output.
        box_nms_thresh: IoU threshold used for class-wise non-maximum suppression.
        box_detections_per_img: Maximum post-NMS detections per image.

    Returns:
        A detector whose predictor represents background and turtle face.
    """
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if not 0 <= trainable_backbone_layers <= 5:
        raise ValueError("trainable_backbone_layers must be between 0 and 5")
    if min_size <= 0 or max_size <= 0 or min_size > max_size:
        raise ValueError("min_size and max_size must be positive and ordered")
    if not 0.0 <= box_score_thresh <= 1.0:
        raise ValueError("box_score_thresh must be in [0, 1]")
    if not 0.0 <= box_nms_thresh <= 1.0:
        raise ValueError("box_nms_thresh must be in [0, 1]")
    if box_detections_per_img <= 0:
        raise ValueError("box_detections_per_img must be positive")

    # COCO weights provide transferable features; the scratch path must not trigger a hidden backbone download.
    weights = (
        FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    )
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # COCO has many classes, but this dataset needs only background and turtle face.
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES_WITH_BACKGROUND,
    )
    return model


def _to_model_targets(
    targets: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Move detector boxes and labels to the selected device.

    Args:
        targets: Detection dictionaries containing ``boxes`` and ``labels``.
        device: Destination device for both target tensors.

    Returns:
        New target dictionaries containing only device-local boxes and labels.
    """
    return [
        {
            "boxes": target["boxes"].to(device, non_blocking=True),
            "labels": target["labels"].to(device, non_blocking=True),
        }
        for target in targets
    ]


@contextmanager
def _loss_evaluation_state(model: nn.Module):
    """Enable detector loss heads while isolating module flags and random state.

    Args:
        model: Detector whose module modes and random streams must be restored.

    Yields:
        Control with training-mode loss heads, frozen normalization/dropout, and
        gradients disabled.
    """
    modules = list(model.modules())
    training_flags = [module.training for module in modules]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        # TorchVision computes its loss dictionary only when the generalized detector is training.
        model.train(True)
        # Training-only losses must not update running statistics or add dropout noise.
        for module in modules:
            if isinstance(module, (_BatchNorm, nn.modules.dropout._DropoutNd)):
                module.training = False
        with torch.no_grad():
            yield
    finally:
        for module, was_training in zip(modules, training_flags, strict=True):
            module.training = was_training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def evaluate_detector_loss(
    model: nn.Module,
    data_loader: Iterable[tuple[list[torch.Tensor], list[dict[str, Any]]]],
    device: torch.device,
) -> dict[str, float]:
    """Compute sample-weighted Faster R-CNN component and total validation losses.

    Inputs are letterboxed images and training-style box/label targets. TorchVision
    exposes its loss dictionary only in training mode, so normalization and dropout
    are held in evaluation mode while loss heads run. Region proposal sampling can
    remain stochastic; its random state and all module training flags are restored.

    Args:
        model: TorchVision-style detector returning a loss dictionary with targets.
        data_loader: Batches of letterboxed images and target boxes/labels.
        device: Device on which to evaluate the model.

    Returns:
        Sample-weighted component losses and their sum, with ``loss_`` prefixes.
    """
    totals: dict[str, torch.Tensor] = {}
    sample_count = 0

    with _loss_evaluation_state(model):
        for images, targets in data_loader:
            model_images = [image.to(device, non_blocking=True) for image in images]
            model_targets = _to_model_targets(targets, device)
            loss_components = model(model_images, model_targets)
            total_loss = sum(loss_components.values())
            if not torch.isfinite(total_loss).item():
                raise FloatingPointError("non-finite Faster R-CNN validation loss")

            batch_size = len(images)
            sample_count += batch_size
            totals["loss_total"] = totals.get(
                "loss_total", torch.zeros((), device=device)
            ) + total_loss.detach() * batch_size
            for name, value in loss_components.items():
                key = name if name.startswith("loss_") else f"loss_{name}"
                totals[key] = totals.get(key, torch.zeros((), device=device)) + (
                    value.detach() * batch_size
                )

    if sample_count == 0:
        raise ValueError("validation loss data loader produced no batches")
    return {name: float(value.item()) / sample_count for name, value in totals.items()}


def train_detector_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[list[torch.Tensor], list[dict[str, Any]]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train one epoch and return sample-weighted component losses.

    Args:
        model: Trainable detector returning component losses for each batch.
        data_loader: Training batches of images and detection targets.
        optimizer: Optimizer updated once for each batch.
        device: Device on which to run the training pass.

    Returns:
        Image-weighted loss components and their total.
    """
    model.train()
    totals: dict[str, torch.Tensor] = {}
    sample_count = 0

    for images, targets in data_loader:
        model_images = [image.to(device, non_blocking=True) for image in images]
        model_targets = _to_model_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        # TorchVision returns separate RPN and ROI losses; their sum is the training objective.
        loss_components = model(model_images, model_targets)
        total_loss = sum(loss_components.values())
        if not torch.isfinite(total_loss).item():
            raise FloatingPointError("non-finite Faster R-CNN training loss")
        total_loss.backward()
        optimizer.step()

        batch_size = len(images)
        sample_count += batch_size
        totals["loss_total"] = totals.get(
            "loss_total", torch.zeros((), device=device)
        ) + total_loss.detach() * batch_size
        for name, value in loss_components.items():
            key = name if name.startswith("loss_") else f"loss_{name}"
            totals[key] = totals.get(key, torch.zeros((), device=device)) + (
                value.detach() * batch_size
            )

    if sample_count == 0:
        raise ValueError("training data loader produced no batches")
    return {name: float(value.item()) / sample_count for name, value in totals.items()}


def inverse_letterbox_xyxy(
    box_xyxy: np.ndarray,
    original_height: int,
    original_width: int,
    target_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
) -> np.ndarray | None:
    """Map a detector-frame box back to the original image, clipping its edges.

    Args:
        box_xyxy: Candidate box in the letterboxed target canvas.
        original_height: Original image height in pixels.
        original_width: Original image width in pixels.
        target_size: Letterbox canvas as ``(height, width)``.

    Returns:
        Clipped original-image xyxy coordinates, or ``None`` for an invalid box.
    """
    target_height, target_width = target_size
    if min(original_height, original_width, target_height, target_width) <= 0:
        raise ValueError("image dimensions must be positive")

    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not valid_xyxy_mask(box).item():
        return None

    # Recreate letterboxing, including its integer rounding and centered padding, before inversion.
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2

    original_box = box.copy()
    original_box[[0, 2]] = (original_box[[0, 2]] - left) / scale
    original_box[[1, 3]] = (original_box[[1, 3]] - top) / scale
    original_box[[0, 2]] = np.clip(original_box[[0, 2]], 0, original_width)
    original_box[[1, 3]] = np.clip(original_box[[1, 3]], 0, original_height)
    if not valid_xyxy_mask(original_box).item():
        return None
    return original_box


def _best_class_one_prediction(
    output: dict[str, torch.Tensor],
) -> tuple[np.ndarray | None, float | None, int]:
    """Return the highest-confidence class-1 box and the count of such boxes.

    Args:
        output: One image's post-NMS detector output.

    Returns:
        Box, confidence, and class-1 detection count; box and score are ``None``
        when no turtle-face detection exists.
    """
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    class_one = np.flatnonzero(labels == CLASS_ID_TURTLE_FACE)
    if len(class_one) == 0:
        return None, None, 0
    # The evaluation protocol scores one face box, so retain the highest-scoring face after NMS.
    selected = int(class_one[np.argmax(scores[class_one])])
    return boxes[selected].astype(np.float64), float(scores[selected]), len(class_one)


def evaluate_faster_rcnn(
    model: nn.Module,
    data_loader: Iterable[tuple[list[torch.Tensor], list[dict[str, Any]]]],
    device: torch.device,
    target_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate top-confidence class-1 boxes against original-image annotations.

    Args:
        model: Detector returning post-NMS boxes, scores, and labels.
        data_loader: Validation batches with original image metadata.
        device: Device used for inference.
        target_size: Letterbox canvas as ``(height, width)``.

    Returns:
        Aggregate IoU/coverage metrics and one diagnostic row per image.
    """
    model.eval()
    per_image: list[dict[str, Any]] = []
    confidences: list[float] = []
    detection_counts: list[int] = []
    valid_predictions = 0

    with torch.inference_mode():
        for images, targets in data_loader:
            model_images = [image.to(device, non_blocking=True) for image in images]
            outputs = model(model_images)
            for output, target in zip(outputs, targets, strict=True):
                original_height, original_width = (
                    int(value) for value in target["original_size"].tolist()
                )
                transformed_box, confidence, detection_count = _best_class_one_prediction(output)
                prediction = (
                    inverse_letterbox_xyxy(
                        transformed_box,
                        original_height,
                        original_width,
                        target_size,
                    )
                    if transformed_box is not None
                    else None
                )
                ground_truth_xywh = target["original_normalized_xywh"].cpu().numpy().astype(np.float64)
                gt_xyxy = np.asarray(
                    [
                        ground_truth_xywh[0] * original_width,
                        ground_truth_xywh[1] * original_height,
                        (ground_truth_xywh[0] + ground_truth_xywh[2]) * original_width,
                        (ground_truth_xywh[1] + ground_truth_xywh[3]) * original_height,
                    ],
                    dtype=np.float64,
                )
                # A missing face is a miss rather than an omitted sample, so it contributes IoU zero.
                if prediction is None:
                    iou = 0.0
                    predicted_xywh = np.zeros(4, dtype=np.float64)
                    confidence_value = None
                else:
                    iou = float(intersection_over_union_xyxy(prediction, gt_xyxy))
                    predicted_xywh = pixel_xyxy_to_normalized_xywh(
                        prediction,
                        original_width,
                        original_height,
                    )
                    confidence_value = confidence
                    valid_predictions += 1
                    confidences.append(float(confidence))
                detection_counts.append(detection_count)
                per_image.append(
                    {
                        "image_id": str(target["image_id"]),
                        "gt_x": float(ground_truth_xywh[0]),
                        "gt_y": float(ground_truth_xywh[1]),
                        "gt_w": float(ground_truth_xywh[2]),
                        "gt_h": float(ground_truth_xywh[3]),
                        "pred_x": float(predicted_xywh[0]),
                        "pred_y": float(predicted_xywh[1]),
                        "pred_w": float(predicted_xywh[2]),
                        "pred_h": float(predicted_xywh[3]),
                        "confidence": confidence_value,
                        "iou": iou,
                        "class_one_detections": detection_count,
                        "has_valid_prediction": prediction is not None,
                    }
                )

    if not per_image:
        raise ValueError("validation data loader produced no images")
    ious = np.asarray([row["iou"] for row in per_image], dtype=np.float64)
    metrics = {
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "fraction_iou_ge_0_5": float(np.mean(ious >= 0.5)),
        "fraction_iou_ge_0_75": float(np.mean(ious >= 0.75)),
        "prediction_coverage": valid_predictions / len(per_image),
        "mean_selected_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "mean_class_one_detections": float(np.mean(detection_counts)),
        "image_count": float(len(per_image)),
    }
    return metrics, per_image


def fit_faster_rcnn(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    target_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Train, choose the checkpoint by validation mean IoU, and restore it.

    Args:
        model: Trainable detector.
        train_loader: Training batches with optional train-only augmentation.
        validation_loader: Fixed validation batches used for loss and IoU.
        optimizer: Optimizer for trainable detector parameters.
        scheduler: Optional epoch-level learning-rate scheduler.
        device: Execution device.
        epochs: Number of complete training epochs.
        checkpoint_path: Local path for the best-IoU model state.
        target_size: Letterbox canvas as ``(height, width)``.
        on_epoch_end: Optional callback receiving a copy of each metric row.

    Returns:
        Best-epoch metadata, final best-checkpoint metrics, predictions, and
        training history. Checkpoint selection remains based on validation IoU.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_iou = -math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_detector_epoch(model, train_loader, optimizer, device)
        validation_loss = evaluate_detector_loss(model, validation_loader, device)
        validation_metrics, _ = evaluate_faster_rcnn(
            model,
            validation_loader,
            device,
            target_size,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {"epoch": float(epoch), "learning_rate": learning_rate}
        row.update({f"train_{name}": value for name, value in train_metrics.items()})
        row.update({f"val_{name}": value for name, value in validation_loss.items()})
        row.update({f"val_{name}": value for name, value in validation_metrics.items()})
        history.append(row)

        # Select on the agreed validation metric and keep the prior checkpoint intact if saving fails.
        if validation_metrics["mean_iou"] > best_iou:
            best_iou = validation_metrics["mean_iou"]
            best_epoch = epoch
            best_metrics = validation_metrics
            temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": validation_metrics,
                },
                temporary_path,
            )
            os.replace(temporary_path, checkpoint_path)

        if on_epoch_end is not None:
            on_epoch_end(row.copy())
        if scheduler is not None:
            scheduler.step()

    # Return predictions from the best validation epoch, not automatically from the final epoch.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    final_metrics, best_per_image = evaluate_faster_rcnn(
        model,
        validation_loader,
        device,
        target_size,
    )
    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "validation_metrics": final_metrics,
        "per_image_metrics": best_per_image,
        "history": history,
        "checkpoint_path": checkpoint_path,
    }


class SingleBoxFasterRCNN(nn.Module):
    """Map letterboxed float images to normalized xywh and score per image.

    The output columns are ``x, y, width, height, confidence`` in the input
    letterbox canvas coordinate frame. The serving caller must undo letterbox
    padding when it needs original-image coordinates.
    """

    def __init__(
        self,
        detector: nn.Module,
        target_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
    ) -> None:
        """Wrap a detector with fixed letterbox dimensions for serving.

        Args:
            detector: Faster R-CNN model returning post-NMS detections.
            target_size: Input canvas as ``(height, width)``.
        """
        super().__init__()
        self.detector = detector
        self.target_height, self.target_width = target_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Select one class-1 box per image and normalize it to the canvas.

        Args:
            images: Batch tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Tensor rows of ``x, y, width, height, confidence``; misses are zeros.
        """
        outputs = self.detector(list(images.unbind(0)))
        rows = []
        for output in outputs:
            labels = output["labels"]
            matching = torch.where(labels == CLASS_ID_TURTLE_FACE)[0]
            if matching.numel() == 0:
                # Fixed-shape serving output uses an all-zero row as the no-detection sentinel.
                rows.append(images.new_zeros(5))
                continue
            # Keep serving aligned with validation: choose one top-confidence face after NMS.
            scores = output["scores"][matching]
            selected = matching[torch.argmax(scores)]
            x1, y1, x2, y2 = output["boxes"][selected]
            score = output["scores"][selected]
            rows.append(
                torch.stack(
                    (
                        x1 / self.target_width,
                        y1 / self.target_height,
                        (x2 - x1) / self.target_width,
                        (y2 - y1) / self.target_height,
                        score,
                    )
                )
            )
        return torch.stack(rows, dim=0)
