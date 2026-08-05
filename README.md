# Sea Turtle Face Detection

> **Work in progress** - the repository currently implements dataset auditing, bounding-box analysis, duplicate review, and a leakage-aware train/validation split. It does not yet contain a trained detection model or competition submission pipeline.

This project is based on the public Zindi competition [Local Ocean Conservation Sea Turtle Face Detection](https://zindi.africa/competitions/local-ocean-conservation-sea-turtle-face-detection). The task is to localize the facial scale region of a sea turtle by predicting one normalized bounding box for each image.

## Task definition

The competition is an object detection and localization problem with a single target region per image. Its annotations use normalized `xywh` coordinates:

```text
x, y, width, height
```

where every value is expressed as a fraction of the image dimensions.

This task must not be confused with two related problems:

| Task | Question | Implemented here? |
|---|---|---|
| Face detection / localization | Where is the turtle face? | Data preparation implemented; model not implemented |
| Individual identification | Which known turtle appears in the image? | No |
| Verification | Do two images show the same individual? | No |

The perceptual hashing code in this repository is used only to detect duplicate or near-duplicate captures before splitting the data. It is not an individual turtle recognition or verification system.

## Project goals

1. Build a reproducible pipeline from raw competition data to bounding-box predictions.
2. Validate image files and normalized bounding-box annotations before modeling.
3. Prevent leakage from duplicate or closely related captures.
4. Implement box conversions and Intersection over Union (IoU).
5. Train and evaluate a lightweight bounding-box regression baseline.
6. Compare the regression baseline with a standard object detector when justified.
7. Produce a valid Zindi submission without using the official test set for development decisions.

## Current status

### Implemented

- CSV schema, missing-value, duplicate-row, and image-correspondence checks.
- Image integrity, color mode, and resolution inspection.
- Validation of normalized `xywh` bounding boxes.
- Derived box features: center, relative area, aspect ratio, border distance, and center distance.
- Visual review of random annotations and geometric extreme cases.
- Exact duplicate detection with SHA-256.
- Near-duplicate candidate generation with perceptual hashing.
- Manual confirmation of related captures before grouping.
- Reproducible `GroupShuffleSplit` with unique groups for independent images.
- Explicit assertion that no group appears in both train and validation.
- Local persistence of the split at `data/splits/train_validation.csv`.
- Normalized `xywh` to pixel `xyxy` conversion and round-trip conversion.
- Explicit clipping and validation for normalized and pixel boxes.
- Broadcast-compatible IoU for valid `xyxy` boxes.
- Geometry tests covering conversions, clipping, validation, and IoU.

### Not implemented yet

- Model-level IoU evaluation and error analysis.
- PyTorch dataset, transforms, augmentations, or data loaders.
- Bounding-box regression or object detection models.
- Training, inference, checkpoints, or experiment tracking.
- Submission generation.
- Precision, recall, or mean Average Precision (mAP) evaluation.
- Individual turtle identification or verification.

No model scores, leaderboard results, or performance claims are reported because no model has been trained in this repository.

## Repository structure

```text
SEA_TURTLE_FACE_DETECTION/
|-- AGENTS.md
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- notebooks/
|   `-- 01_data_audit.ipynb
|-- src/
|   `-- turtle_detection/
|       |-- __init__.py
|       |-- box_analysis.py
|       |-- box_geometry.py
|       |-- image_quality.py
|       |-- image_similarity.py
|       `-- splitting.py
|-- tests/
|   `-- test_box_geometry.py
|-- data/                       # Local only; excluded from Git
|   |-- raw/
|   `-- splits/
`-- outputs/                    # Local only; excluded from Git
```

`AGENTS.md` contains repository working conventions.

## Installation

### Requirements

- Windows, Linux, or macOS
- Python 3.11
- No GPU is required for the current EDA and split workflow

### Create the environment

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The dependency versions are pinned to keep the current notebook environment reproducible.

## Obtain the data from Zindi

Competition data is **not distributed with this repository**.

1. Open the official [competition page](https://zindi.africa/competitions/local-ocean-conservation-sea-turtle-face-detection).
2. Sign in to Zindi, join the competition, and accept its current terms and rules.
3. Open the [competition data page](https://zindi.africa/competitions/local-ocean-conservation-sea-turtle-face-detection/data).
4. Download:
   - `Train.csv`
   - `SampleSubmission.csv`
   - `IMAGES_512.zip`
5. Extract the images without renaming the internal files.
6. Arrange the local files as follows:

```text
data/raw/
|-- Train.csv
|-- SampleSubmission.csv
`-- IMAGES_512/
    `-- <Image_ID>.JPG
```

The project starts with the 512-pixel image archive to reduce storage, loading time, and future GPU memory use. The 1024-pixel archive is intentionally deferred until experiments justify the additional cost.

## Evaluation strategy

The official competition metric is **Intersection over Union (IoU)** between the predicted and ground-truth bounding boxes. The repository now includes a tested IoU utility; it will become the primary local validation metric once prediction code exists.

Planned evaluation should include:

- IoU per image;
- mean and median IoU;
- IoU percentiles and low-IoU error cases;
- localization error by box size, position, and aspect ratio;
- visual comparison of predicted and ground-truth boxes.

Precision, recall, and mAP are standard object detection metrics when a model predicts detections with confidence scores, classes, and possibly multiple candidate boxes. They are not implemented or reported yet. For this competition's single-class, single-box output, IoU is the official and currently most relevant metric. Precision, recall, and mAP should be added only when the selected detector and evaluation protocol make them meaningful.

## Methodological risks

### Confirmed risks

- Duplicate and near-duplicate captures can leak information across partitions.
- The image archive contains more than one aspect ratio.
- The dataset is small enough that validation composition can materially affect IoU estimates.

### Risks still to audit

- repeated encounters or sessions not detected by pHash;
- validation balance by face size, position, orientation, and image difficulty;
- sensitivity to illumination, blur, occlusion, and extreme viewpoints;
- generalization from the 512-pixel archive to higher-resolution data;
- whether CNN embeddings are needed to identify additional related captures.

Class imbalance, samples per individual, and evaluation by individual are identification concerns. This dataset does not provide an implemented individual-identification target, so the repository does not claim that these analyses are available.

## Technologies currently used

- Python 3.11
- NumPy and pandas
- Pillow and Matplotlib
- ImageHash
- scikit-learn
- JupyterLab and ipykernel

PyTorch, OpenCV, object detection frameworks, transfer learning, and deep learning models are not currently implemented.

## Planned workflow

```text
raw data
-> integrity and annotation audit
-> duplicate-aware grouped split
-> box conversions and IoU tests
-> reusable dataset and transform pipeline
-> lightweight regression baseline
-> validation and error analysis
-> optional standard detector comparison
-> inference and Zindi submission
```
