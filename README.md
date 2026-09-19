# Sea Turtle Face Detection

> **Work in progress** - the repository currently implements dataset auditing, bounding-box analysis, duplicate review, a leakage-aware split, and a reusable PyTorch data pipeline. It does not yet contain a trained detection model or competition submission pipeline.

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
- A PyTorch Dataset that reads the accepted split and validates every annotation before loading.
- Aspect-ratio-preserving letterbox transforms with box updates and optional train-time horizontal flipping.
- Windows-safe train and validation DataLoaders with detection-style list targets.
- A visual inspection notebook for batches, transformed boxes, and the selected device.
- Data-pipeline tests covering letterbox geometry, flipping, split sizes, and target structure.
- A ResNet18 direct `xywh` regression baseline with reusable training and IoU evaluation helpers.
- A smoke-training notebook that runs on the available Windows CUDA device and visualizes predictions.

### Not implemented yet

- Model-level IoU evaluation and error analysis.
- Bounding-box regression or object detection models.
- Training, inference, checkpoints, or experiment tracking.
- Submission generation.
- Precision, recall, or mean Average Precision (mAP) evaluation.
- Individual turtle identification or verification.

The regression notebook includes only a short smoke run limited to a few batches. Its IoU is a pipeline check, not a benchmark or performance claim. No leaderboard result or full-training score is reported yet.

## Repository structure

```text
SEA_TURTLE_FACE_DETECTION/
|-- AGENTS.md
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- notebooks/
|   |-- 01_data_audit.ipynb
|   |-- 02_data_pipeline.ipynb
|   `-- 03_regression_baseline.ipynb
|-- src/
|   `-- turtle_detection/
|       |-- __init__.py
|       |-- box_analysis.py
|       |-- box_geometry.py
|       |-- data_pipeline.py
|       |-- image_quality.py
|       |-- image_similarity.py
|       |-- regression.py
|       `-- splitting.py
|-- tests/
|   |-- test_box_geometry.py
|   |-- test_data_pipeline.py
|   `-- test_regression.py
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
python -m pip install --use-feature=truststore -r requirements.txt
python -m pip install -e . --no-deps
```

The dependency versions are pinned to keep the EDA and training environment reproducible. The `truststore` option uses the Windows certificate store when the default Python certificate bundle cannot validate `download.pytorch.org`.

## Native Windows training feasibility

Native Windows is suitable for the next training stages. The official [PyTorch installation guide](https://docs.pytorch.org/get-started/locally/) supports Windows and Python 3.11, and provides CUDA wheels without requiring a source build. This project was verified locally with `torch==2.8.0+cu128`, `torchvision==0.23.0+cu128`, and `torch.cuda.is_available() == True` on an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM.

This hardware is enough for a lightweight regression baseline and small or medium pretrained detectors at the current 512-pixel image size. The practical limits are batch size, model size, input resolution, and experiment parallelism; 6 GB VRAM is not a good target for large detectors or high-resolution multi-model runs. The DataLoader defaults to `num_workers=0` because it is reliable in notebooks and avoids Windows multiprocessing-spawn issues. Worker processes can be increased later from a script entry point after measuring throughput.

WSL2 or a Linux machine is not required now. It becomes useful only if a future dependency is Linux-only, if distributed training is needed, or if a larger GPU/cloud workflow is introduced. The model code should remain device-agnostic so the same training module can move between native Windows, WSL2, and Linux.

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
- PyTorch and torchvision (training environment)

OpenCV, object detection frameworks, transfer learning models, and experiment tracking are not currently implemented.

## Planned workflow

```text
raw data
-> integrity and annotation audit
-> duplicate-aware grouped split
-> box conversions and IoU tests
-> reusable Dataset, transforms, and DataLoaders
-> lightweight regression baseline
-> validation and error analysis
-> optional standard detector comparison
-> inference and Zindi submission
```
