"""Tests for detector box selection and coordinate evaluation."""

import unittest
import tempfile
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from turtle_detection.detection import (
    SingleBoxFasterRCNN,
    evaluate_detector_loss,
    evaluate_faster_rcnn,
    fit_faster_rcnn,
    inverse_letterbox_xyxy,
)


class FixedOutputDetector(nn.Module):
    def __init__(self, output: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.output = output

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [self.output for _ in images]


class TinyTrainableDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            return {"loss_classifier": self.weight.square()}
        return [
            {
                "boxes": torch.tensor([[128.0, 128.0, 384.0, 256.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
            for _ in images
        ]


class LossStateDetector(nn.Module):
    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.normalization = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(p=0.9)
        self.fail = fail
        self.loss_modes = []

    def forward(self, images, targets=None):
        self.loss_modes.append((self.training, self.normalization.training, self.dropout.training))
        torch.rand(1)
        np.random.random()
        random.random()
        self.dropout(self.normalization(torch.ones((2, 1))))
        if self.fail:
            raise RuntimeError("synthetic model failure")
        assert targets is not None
        return {"loss_classifier": self.weight.square()}


class DetectionTests(unittest.TestCase):
    def test_validation_loss_uses_loss_mode_without_state_or_rng_side_effects(self) -> None:
        model = LossStateDetector()
        model.train()
        model.normalization.eval()
        flags_before = [module.training for module in model.modules()]
        running_mean_before = model.normalization.running_mean.clone()
        model.weight.grad = torch.tensor(7.0)
        grad_before = model.weight.grad.clone()
        torch.manual_seed(314)
        rng_before = torch.random.get_rng_state().clone()
        numpy_rng_before = np.random.get_state()
        python_rng_before = random.getstate()
        targets = [{"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "labels": torch.tensor([1])}]

        metrics = evaluate_detector_loss(
            model,
            [([torch.zeros(3, 8, 8)], targets)],
            torch.device("cpu"),
        )

        self.assertEqual(metrics["loss_classifier"], 4.0)
        self.assertEqual(metrics["loss_total"], 4.0)
        self.assertEqual(model.loss_modes, [(True, False, False)])
        self.assertEqual([module.training for module in model.modules()], flags_before)
        torch.testing.assert_close(model.normalization.running_mean, running_mean_before)
        torch.testing.assert_close(model.weight.grad, grad_before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertEqual(random.getstate(), python_rng_before)
        numpy_rng_after = np.random.get_state()
        self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
        np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
        self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])

    def test_validation_loss_restores_training_flags_when_model_raises(self) -> None:
        model = LossStateDetector(fail=True)
        model.train()
        model.dropout.eval()
        flags_before = [module.training for module in model.modules()]
        targets = [{"boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "labels": torch.tensor([1])}]

        with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
            evaluate_detector_loss(
                model,
                [([torch.zeros(3, 8, 8)], targets)],
                torch.device("cpu"),
            )

        self.assertEqual([module.training for module in model.modules()], flags_before)

    def test_inverse_letterbox_maps_and_clips_to_original_image(self) -> None:
        transformed = np.asarray([128.0, 128.0, 400.0, 260.0])

        restored = inverse_letterbox_xyxy(transformed, 128, 256, (384, 512))

        np.testing.assert_allclose(restored, [64.0, 32.0, 200.0, 98.0])

    def test_inverse_letterbox_rejects_box_outside_content(self) -> None:
        restored = inverse_letterbox_xyxy(
            np.asarray([10.0, 0.0, 20.0, 20.0]),
            original_height=128,
            original_width=256,
            target_size=(384, 512),
        )
        self.assertIsNone(restored)

    def test_evaluation_selects_top_confidence_class_one_and_uses_original_frame(self) -> None:
        output = {
            "boxes": torch.tensor(
                [[128.0, 128.0, 384.0, 256.0], [0.0, 0.0, 30.0, 30.0]]
            ),
            "scores": torch.tensor([0.9, 0.99]),
            "labels": torch.tensor([1, 2]),
        }
        model = FixedOutputDetector(output)
        images = [torch.zeros(3, 384, 512)]
        targets = [
            {
                "image_id": "turtle-1",
                "original_size": torch.tensor([128, 256]),
                "original_normalized_xywh": torch.tensor([0.25, 0.25, 0.5, 0.5]),
            }
        ]

        metrics, rows = evaluate_faster_rcnn(
            model,
            [(images, targets)],
            torch.device("cpu"),
            (384, 512),
        )

        self.assertEqual(metrics["mean_iou"], 1.0)
        self.assertEqual(metrics["prediction_coverage"], 1.0)
        self.assertEqual(rows[0]["class_one_detections"], 1)
        np.testing.assert_allclose(
            [rows[0]["pred_x"], rows[0]["pred_y"], rows[0]["pred_w"], rows[0]["pred_h"]],
            [0.25, 0.25, 0.5, 0.5],
        )

    def test_missing_detection_scores_zero_iou_and_zero_coverage(self) -> None:
        output = {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.int64),
        }
        metrics, rows = evaluate_faster_rcnn(
            FixedOutputDetector(output),
            [
                (
                    [torch.zeros(3, 384, 512)],
                    [
                        {
                            "image_id": "turtle-1",
                            "original_size": torch.tensor([128, 256]),
                            "original_normalized_xywh": torch.tensor(
                                [0.25, 0.25, 0.5, 0.5]
                            ),
                        }
                    ],
                )
            ],
            torch.device("cpu"),
        )
        self.assertEqual(metrics["mean_iou"], 0.0)
        self.assertEqual(metrics["prediction_coverage"], 0.0)
        self.assertFalse(rows[0]["has_valid_prediction"])

    def test_single_box_model_has_fixed_tensor_output(self) -> None:
        output = {
            "boxes": torch.tensor([[32.0, 48.0, 160.0, 176.0]]),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([1]),
        }
        model = SingleBoxFasterRCNN(FixedOutputDetector(output), (384, 512))

        result = model(torch.zeros(2, 3, 384, 512))

        self.assertEqual(tuple(result.shape), (2, 5))
        np.testing.assert_allclose(result[0].numpy(), [0.0625, 0.125, 0.25, 1 / 3, 0.8])

    def test_fit_saves_best_validation_iou_checkpoint(self) -> None:
        images = [torch.zeros(3, 384, 512)]
        targets = [
            {
                "image_id": "turtle-1",
                "boxes": torch.tensor([[128.0, 128.0, 384.0, 256.0]]),
                "labels": torch.tensor([1]),
                "original_size": torch.tensor([128, 256]),
                "original_normalized_xywh": torch.tensor([0.25, 0.25, 0.5, 0.5]),
            }
        ]
        batches = [(images, targets)]
        model = TinyTrainableDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            result = fit_faster_rcnn(
                model,
                batches,
                batches,
                optimizer,
                scheduler=None,
                device=torch.device("cpu"),
                epochs=2,
                checkpoint_path=checkpoint_path,
            )
            self.assertTrue(checkpoint_path.is_file())

        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["validation_metrics"]["mean_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
