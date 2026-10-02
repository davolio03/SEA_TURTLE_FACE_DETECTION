"""Tests for the detection Dataset, transforms, and loader collation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from turtle_detection.data_pipeline import (
    LetterboxTransform,
    SeaTurtleDataset,
    create_data_loaders,
)


class DataPipelineTests(unittest.TestCase):
    def test_letterbox_adds_vertical_padding_and_scales_box(self) -> None:
        image = Image.new("RGB", (512, 288), (255, 255, 255))
        transform = LetterboxTransform((384, 512), normalize=False)

        tensor, box = transform(image, np.asarray([128, 72, 384, 216], dtype=float))

        self.assertEqual(tuple(tensor.shape), (3, 384, 512))
        np.testing.assert_allclose(box, [128, 120, 384, 264])

    def test_horizontal_flip_updates_box(self) -> None:
        image = Image.new("RGB", (100, 100), (255, 255, 255))
        transform = LetterboxTransform(
            (100, 100),
            horizontal_flip_probability=1.0,
            normalize=False,
        )

        _, box = transform(image, np.asarray([10, 20, 40, 60], dtype=float))

        np.testing.assert_allclose(box, [60, 20, 90, 60])

    def test_dataset_and_loader_return_detection_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (512, 384), (120, 120, 120)).save(image_dir / "A.JPG")
            Image.new("RGB", (512, 288), (80, 80, 80)).save(image_dir / "B.JPG")
            annotations = pd.DataFrame(
                {
                    "Image_ID": ["A", "B"],
                    "x": [0.1, 0.2],
                    "y": [0.2, 0.3],
                    "w": [0.4, 0.3],
                    "h": [0.3, 0.2],
                }
            )
            annotations.to_csv(root / "Train.csv", index=False)
            splits = pd.DataFrame(
                {"Image_ID": ["A", "B"], "group_id": [1, 2], "split": ["train", "validation"]}
            )
            splits.to_csv(root / "splits.csv", index=False)

            dataset = SeaTurtleDataset(
                root / "Train.csv",
                root / "splits.csv",
                image_dir,
                "train",
                LetterboxTransform((384, 512), normalize=False),
            )
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 384, 512))
            self.assertEqual(tuple(target["boxes"].shape), (1, 4))
            self.assertEqual(target["labels"].dtype, torch.int64)
            self.assertEqual(target["image_id"], "A")

            validation_dataset = SeaTurtleDataset(
                root / "Train.csv",
                root / "splits.csv",
                image_dir,
                "validation",
                LetterboxTransform((384, 512), normalize=False),
            )
            _, validation_target = validation_dataset[0]
            np.testing.assert_allclose(
                validation_target["normalized_xywh"].numpy(),
                [0.2, 0.35, 0.3, 0.15],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                validation_target["original_normalized_xywh"].numpy(),
                [0.2, 0.3, 0.3, 0.2],
                atol=1e-6,
            )

    def test_project_loaders_have_expected_split_sizes(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        train_loader, validation_loader = create_data_loaders(
            project_root,
            batch_size=4,
            num_workers=0,
            normalize=False,
        )

        self.assertEqual(len(train_loader.dataset), 1062)
        self.assertEqual(len(validation_loader.dataset), 264)
        images, targets = next(iter(validation_loader))
        self.assertEqual(len(images), 4)
        self.assertEqual(len(targets), 4)
        self.assertEqual(tuple(images[0].shape), (3, 384, 512))
        self.assertGreaterEqual(float(images[0].min()), 0.0)
        self.assertLessEqual(float(images[0].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
