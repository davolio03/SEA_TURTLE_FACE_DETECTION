This repository documents a computer-vision project based on the [Local Ocean Conservation Sea Turtle Face Detection competition](https://zindi.africa/competitions/local-ocean-conservation-sea-turtle-face-detection). The task is to locate a sea-turtle face with one bounding box per image. The project explores the annotations, prevents near-duplicate leakage in a grouped train/validation split, and compares a ResNet18 box-regression control with a pretrained Faster R-CNN detector. It is an object-detection task, not individual-turtle identification; performance is measured with intersection over union (IoU), not classification accuracy.

## Data and setup

Competition data and trained models are not distributed here. Download `Train.csv`, `SampleSubmission.csv`, and `IMAGES_512.zip` from the [official Zindi competition page](https://zindi.africa/competitions/local-ocean-conservation-sea-turtle-face-detection), then extract them to:

```text
data/raw/
|-- Train.csv
|-- SampleSubmission.csv
`-- IMAGES_512/
```

Use Python 3.11. From the repository root, create the environment and install the checked-in requirements and package:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --use-feature=truststore -r requirements.txt
python -m pip install -e ".[dev]"
```

The `src/turtle_detection` package is installed from the project configuration. Raw data, generated splits, checkpoints, submission files, MLflow databases, and MLflow artifacts stay local and are ignored by Git.

## Notebook workflow

Run notebooks in order from the repository root:

1. `notebooks/01_data_audit.ipynb` checks data and annotations, reviews duplicate candidates, and saves reviewed leakage groups and split diagnostics.
2. `notebooks/02_data_pipeline.ipynb` inspects letterboxed batches and target boxes.
3. `notebooks/03_regression_baseline.ipynb` trains the ResNet18 regression control and selects its best checkpoint by validation mean IoU.
4. `notebooks/04_faster_rcnn_experiment.ipynb` trains Faster R-CNN, records train and validation losses plus IoU metrics in local MLflow, and selects the best checkpoint by validation mean IoU.
5. `notebooks/05_zindi_submission.ipynb` loads a local checkpoint, predicts the supplied sample IDs, validates the output schema, and writes a local CSV. It does not upload a submission.

Read the train and validation loss curves as diagnostics, not as a strict like-for-like comparison. Training images may be randomly flipped horizontally; validation images are not, so the losses are computed on somewhat different inputs. Faster R-CNN also samples image regions when calculating its loss, which can make validation loss vary between evaluations. The curves are useful for broad trends, but validation mean IoU (the overlap between predicted and true boxes) is used to select the best checkpoint. Validation loss is recorded in new training runs; older runs predate that metric and only have training-loss history.

## Results and verification

The Faster R-CNN experiment reached validation mean IoU of approximately `0.9185` at epoch 8. The competition score of approximately `0.91` is user-reported and has not been independently verified here. These figures describe box localization only; the project does not claim individual-turtle identification.

Run the reproducible test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

To view local MLflow runs, start the server from the repository root:

```powershell
.\.venv\Scripts\mlflow.exe server --backend-store-uri sqlite:///outputs/mlflow/tracking.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The tracking database and artifacts are local; do not publish them because they can contain results derived from competition data.
