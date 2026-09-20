"""Tests for the bounding-box regression baseline helpers."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from turtle_detection.regression import (
    evaluate_regression,
    fit_regression_model,
    intersection_over_union_tensor,
    load_regression_checkpoint,
    normalized_xywh_to_xyxy_tensor,
)


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 4),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(images))


def synthetic_loader() -> list[tuple[list[torch.Tensor], list[dict[str, object]]]]:
    images = [torch.zeros(3, 8, 8), torch.ones(3, 8, 8)]
    targets = [
        {
            "normalized_xywh": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "image_id": "A",
        },
        {
            "normalized_xywh": torch.tensor([0.2, 0.1, 0.4, 0.3]),
            "image_id": "B",
        },
    ]
    return [(images, targets)]


class RegressionTests(unittest.TestCase):
    def test_normalized_xywh_to_xyxy_is_clipped(self) -> None:
        result = normalized_xywh_to_xyxy_tensor(
            torch.tensor([[0.8, 0.7, 0.4, 0.5]])
        )

        torch.testing.assert_close(result, torch.tensor([[0.8, 0.7, 1.0, 1.0]]))

    def test_iou_supports_broadcasting(self) -> None:
        first = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
        second = torch.tensor([0.0, 0.0, 2.0, 2.0])

        result = intersection_over_union_tensor(first, second)

        torch.testing.assert_close(result, torch.tensor([1.0, 0.25]))

    def test_fit_returns_metrics_for_each_epoch(self) -> None:
        torch.manual_seed(7)
        model = TinyRegressor()
        loader = synthetic_loader()

        history = fit_regression_model(
            model,
            loader,
            loader,
            epochs=2,
            learning_rate=1e-2,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["epoch"], 2.0)
        self.assertGreaterEqual(history[-1]["validation_mean_iou"], 0.0)
        self.assertLessEqual(history[-1]["validation_mean_iou"], 1.0)

    def test_evaluation_preserves_image_ids_and_arrays(self) -> None:
        model = TinyRegressor()
        metrics = evaluate_regression(
            model,
            synthetic_loader(),
            device=torch.device("cpu"),
        )

        self.assertEqual(metrics.image_ids, ("A", "B"))
        self.assertEqual(metrics.predictions.shape, (2, 4))
        self.assertEqual(metrics.targets.shape, (2, 4))
        self.assertTrue(np.isfinite(metrics.ious).all())

    def test_fit_saves_and_loads_best_iou_checkpoint(self) -> None:
        torch.manual_seed(7)
        model = TinyRegressor()
        loader = synthetic_loader()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            history = fit_regression_model(
                model,
                loader,
                loader,
                epochs=2,
                learning_rate=1e-2,
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )

            self.assertTrue(checkpoint_path.is_file())
            payload = load_regression_checkpoint(
                TinyRegressor(),
                checkpoint_path,
                device=torch.device("cpu"),
            )
            self.assertEqual(payload["epoch"], max(history, key=lambda row: row["validation_mean_iou"])["epoch"])
            self.assertIn("optimizer_state_dict", payload)


if __name__ == "__main__":
    unittest.main()
