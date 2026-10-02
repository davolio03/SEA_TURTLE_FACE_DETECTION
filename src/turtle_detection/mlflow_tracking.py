"""Local MLflow configuration and native Faster R-CNN model logging."""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .detection import DEFAULT_DETECTOR_SIZE, SingleBoxFasterRCNN


def configure_local_mlflow(
    project_root: str | Path,
    experiment_name: str = "sea-turtle-face-detection",
) -> dict[str, str]:
    """Configure a project-local SQLite backend and artifact directory.

    Args:
        project_root: Repository root under which local MLflow files are stored.
        experiment_name: Local MLflow experiment to create or select.

    Returns:
        Tracking URI, experiment identity, and local artifact directory.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from sqlalchemy.engine import URL

    root = Path(project_root).resolve()
    tracking_dir = root / "outputs" / "mlflow"
    artifact_dir = tracking_dir / "artifacts"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    database_path = tracking_dir / "tracking.db"
    # Normalize Windows separators so SQLAlchemy builds a portable SQLite file URI.
    tracking_uri = URL.create(
        "sqlite",
        database=str(database_path).replace("\\", "/"),
    ).render_as_string(hide_password=False)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Keep model artifacts beside the local run database instead of using a remote store.
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_dir.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "artifact_dir": str(artifact_dir),
    }


def get_git_metadata(project_root: str | Path) -> dict[str, Any]:
    """Read the source revision and whether the tracked worktree is dirty.

    Args:
        project_root: Repository root used as the Git command working directory.

    Returns:
        Commit and dirty-state metadata, using ``unknown`` when Git is unavailable.
    """
    root = Path(project_root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    # Missing Git metadata is reported as unknown rather than blocking experiment logging.
    return {
        "source_commit": revision.stdout.strip() if revision.returncode == 0 else "unknown",
        "source_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def append_experiment_summary(
    project_root: str | Path,
    row: dict[str, Any],
    filename: str = "experiments.csv",
) -> Path:
    """Append one compact run summary while preserving existing CSV columns.

    Args:
        project_root: Directory containing the local experiment registry.
        row: New run values; unseen keys extend the existing CSV schema.
        filename: Registry path relative to the project root.

    Returns:
        Registry path after an atomic replacement.
    """
    registry_path = Path(project_root).resolve() / filename
    existing_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if registry_path.is_file():
        with registry_path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    # Extend the existing registry schema without dropping columns from earlier experiments.
    for name in row:
        if name not in fieldnames:
            fieldnames.append(name)
    for existing in existing_rows:
        for name in fieldnames:
            existing.setdefault(name, "")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        # Write beside the destination so replacement is atomic on the same filesystem.
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=registry_path.parent,
            suffix=".tmp",
            prefix=f"{registry_path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
        os.replace(temporary_path, registry_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return registry_path


def log_native_faster_rcnn_model(
    model: torch.nn.Module,
    project_root: str | Path,
    target_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
    model_name: str = "faster_rcnn_single_box",
) -> Any:
    """Log a native MLflow PyTorch flavor with a fixed Tensor input/output API.

    Args:
        model: Trained Faster R-CNN detector.
        project_root: Repository root containing importable ``src`` code.
        target_size: Letterbox canvas as ``(height, width)``.
        model_name: MLflow artifact name for the logged model.

    Returns:
        MLflow model information for the logged local run.
    """
    import mlflow
    import torchvision
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    height, width = target_size
    # Expose a stable one-row-per-image API even though Faster R-CNN returns variable detections.
    wrapped_model = SingleBoxFasterRCNN(model, target_size).eval()
    signature = ModelSignature(
        inputs=Schema(
            [TensorSpec(np.dtype(np.float32), (-1, 3, height, width), "images")]
        ),
        outputs=Schema(
            [TensorSpec(np.dtype(np.float32), (-1, 5), "top_confidence_box")]
        ),
    )
    # Pin the runtime libraries so the native MLflow model records its inference environment.
    requirements = [
        f"mlflow=={mlflow.__version__}",
        f"numpy=={np.__version__}",
        f"torch=={torch.__version__}",
        f"torchvision=={torchvision.__version__}",
    ]
    if "+cu" in torch.__version__:
        requirements.append(
            "--extra-index-url https://download.pytorch.org/whl/cu128"
        )
    return mlflow.pytorch.log_model(
        pytorch_model=wrapped_model,
        name=model_name,
        signature=signature,
        code_paths=[str(Path(project_root).resolve() / "src")],
        pip_requirements=requirements,
        serialization_format="pickle",
    )
