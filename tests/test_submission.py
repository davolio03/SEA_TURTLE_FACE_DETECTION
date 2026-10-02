"""Tests for Zindi submission image resolution, predictions, and CSV validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from turtle_detection.submission import (
    load_submission_inputs,
    predict_faster_rcnn_submission,
    validate_submission,
    write_submission_csv,
)


class TwoOutputDetector(nn.Module):
    def forward(self, images):
        detection = {
            "boxes": torch.tensor([[128.0, 128.0, 384.0, 256.0], [0, 0, 20, 20]]),
            "scores": torch.tensor([0.9, 0.99]),
            "labels": torch.tensor([1, 2]),
        }
        no_detection = {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.int64),
        }
        return [detection if index == 0 else no_detection for index, _ in enumerate(images)]


class SubmissionTests(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        Image.new("RGB", (256, 128), color=(80, 120, 160)).save(path)

    def test_load_inputs_resolves_exactly_one_file_and_preserves_sample_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._write_image(image_dir / "second.JPG")
            self._write_image(image_dir / "first.jpeg")
            sample_path = root / "sample.csv"
            pd.DataFrame(
                {"Image_ID": ["second", "first"], "x": [0, 0], "y": [0, 0], "w": [0, 0], "h": [0, 0]}
            ).to_csv(sample_path, index=False)

            sample, image_paths = load_submission_inputs(sample_path, image_dir)

        self.assertEqual(sample["Image_ID"].astype(str).tolist(), ["second", "first"])
        self.assertEqual([path.name for path in image_paths], ["second.JPG", "first.jpeg"])

    def test_load_inputs_rejects_missing_or_ambiguous_image_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            self._write_image(image_dir / "turtle.jpg")
            self._write_image(image_dir / "turtle.png")
            sample_path = root / "sample.csv"
            pd.DataFrame(
                {"Image_ID": ["turtle"], "x": [0], "y": [0], "w": [0], "h": [0]}
            ).to_csv(sample_path, index=False)

            with self.assertRaisesRegex(ValueError, "resolves to 2 files"):
                load_submission_inputs(sample_path, image_dir)

            (image_dir / "turtle.png").unlink()
            (image_dir / "turtle.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "resolves to 0 files"):
                load_submission_inputs(sample_path, image_dir)

    def test_predicts_original_frame_box_and_zero_sentinel_for_missing_detection(self) -> None:
        sample = pd.DataFrame(
            {"Image_ID": ["turtle-a", "turtle-b"], "x": [0, 0], "y": [0, 0], "w": [0, 0], "h": [0, 0]}
        )
        with tempfile.TemporaryDirectory() as directory:
            image_paths = [Path(directory) / "a.jpg", Path(directory) / "b.jpg"]
            for path in image_paths:
                self._write_image(path)

            submission, diagnostics = predict_faster_rcnn_submission(
                TwoOutputDetector(),
                sample,
                image_paths,
                torch.device("cpu"),
                batch_size=2,
            )

        self.assertEqual(submission["Image_ID"].tolist(), ["turtle-a", "turtle-b"])
        np.testing.assert_allclose(submission.loc[0, ["x", "y", "w", "h"]].to_numpy(dtype=float), [0.25, 0.25, 0.5, 0.5])
        np.testing.assert_array_equal(submission.loc[1, ["x", "y", "w", "h"]].to_numpy(dtype=float), [0, 0, 0, 0])
        self.assertEqual(diagnostics["has_detection"].tolist(), [True, False])
        self.assertEqual(diagnostics.loc[0, "class_one_detections"], 1)

    def test_csv_writer_preserves_ids_and_omits_index(self) -> None:
        submission = pd.DataFrame(
            {"Image_ID": ["A", "B"], "x": [0.1, 0], "y": [0.2, 0], "w": [0.3, 0], "h": [0.4, 0]}
        )
        with tempfile.TemporaryDirectory() as directory:
            output = write_submission_csv(
                submission,
                Path(directory) / "submission.csv",
                expected_ids=["A", "B"],
            )
            text = output.read_text(encoding="utf-8").splitlines()
            loaded = pd.read_csv(output, dtype={"Image_ID": "string"})

        self.assertEqual(text[0], "Image_ID,x,y,w,h")
        self.assertEqual(loaded["Image_ID"].astype(str).tolist(), ["A", "B"])
        self.assertEqual(len(loaded), 2)

    def test_validator_rejects_out_of_bounds_boxes_and_order_drift(self) -> None:
        valid = pd.DataFrame(
            {"Image_ID": ["A"], "x": [0.8], "y": [0.2], "w": [0.3], "h": [0.4]}
        )
        with self.assertRaisesRegex(ValueError, "right image boundary"):
            validate_submission(valid)
        valid.loc[0, "x"] = 0.1
        with self.assertRaisesRegex(ValueError, "IDs or their order"):
            validate_submission(valid, expected_ids=["B"])


if __name__ == "__main__":
    unittest.main()
