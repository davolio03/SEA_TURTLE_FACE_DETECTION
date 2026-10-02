"""Tests for the compact CSV summary alongside detailed MLflow runs."""

import csv
import tempfile
import unittest
from pathlib import Path

from turtle_detection.mlflow_tracking import append_experiment_summary


class ExperimentSummaryTests(unittest.TestCase):
    def test_append_preserves_old_columns_and_adds_new_run_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "experiments.csv"
            registry.write_text("experiment_id,architecture\nbaseline,ResNet18\n", encoding="utf-8")

            result = append_experiment_summary(
                root,
                {
                    "experiment_id": "detector-1",
                    "architecture": "Faster R-CNN ResNet50-FPN",
                    "mlflow_run_id": "run-123",
                },
            )

            self.assertEqual(result, registry)
            with registry.open("r", newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0], {
                "experiment_id": "baseline",
                "architecture": "ResNet18",
                "mlflow_run_id": "",
            })
            self.assertEqual(rows[1]["experiment_id"], "detector-1")
            self.assertEqual(rows[1]["mlflow_run_id"], "run-123")


if __name__ == "__main__":
    unittest.main()
