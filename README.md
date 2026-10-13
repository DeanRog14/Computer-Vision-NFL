# Computer-Vision-NFL

## Dataset exploration

[notebooks/dataset_exploration.ipynb](notebooks/dataset_exploration.ipynb)
explores helmet annotations, labeled impacts, video context, player tracking,
and data quality. It does not train models or modify the dataset.

## YOLO helmet detector

`data/processed/yolo/` contains independently copied images and YOLO labels,
`dataset.yaml`, a split manifest, provenance, validation reports, and
`annotation_preview.jpg`. All five helmet variants map to **0: helmet**.
Seed 42 assigns whole games to 80/10/10 splits, keeping plays and camera views
together. Ratios are approximate by image because games differ in size.

| Split | Images | Games | Helmet boxes |
|---|---:|---:|---:|
| Train | 7,938 | 508 | 154,383 |
| Validation | 978 | 64 | 19,026 |
| Test | 1,031 | 63 | 20,327 |

Game/play overlap is zero. Every copied image's header and SHA-256, label count,
class ID, normalized coordinate, and box edge passed validation. Nine converted
examples were visually checked. Raw files remain unchanged. Processed data and
model artifacts are gitignored; copy the prepared dataset with the repository
to the GPU machine, or regenerate it there. YAML paths are relative to its own
directory, so no machine-specific path edits are needed.

Use Python 3.11+ and run commands from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The current environment has CPU-only PyTorch and no usable CUDA device. On an
NVIDIA GPU machine with a CUDA 12.8-compatible driver, install the GPU
requirements instead (also replaces an existing CPU wheel):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For other GPU/driver combinations, use the [official PyTorch installer](https://pytorch.org/get-started/locally/).
The optional pinned pair follows [PyTorch's CUDA wheel commands](https://pytorch.org/get-started/previous-versions/).
GPU wheels were not installed on this CPU machine.

**Start GPU training later with this single command:**

```powershell
.\.venv\Scripts\python.exe -m src.detection.train_yolo --device 0 --require-gpu
```

`configs/helmet_yolo.yaml` selects `yolo11n.pt`, 960-pixel inputs, 50 epochs,
batch 8, early-stopping patience 10, checkpoints every 5 epochs, and standard
color/geometric/mosaic augmentation. CLI overrides include `--batch 4`,
`--imgsz 1280`, and `--epochs`. Automatic device selection prefers CUDA when
available. See [Ultralytics training settings](https://docs.ultralytics.com/modes/train/).
Only the explicit training command loads/downloads pretrained weights and
starts training. For offline training, set `model` to an existing weight file.
Checkpoints go to `runs/detect/helmet_yolo11n/weights/`; repeated runs may add a
numeric suffix. On Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

Revalidate preprocessing or check configuration without loading any model:

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare_yolo_data --validate-only
.\.venv\Scripts\python.exe -m src.detection.train_yolo --dry-run
```

To prepare a fresh copy, use `python -m src.data.prepare_yolo_data --output
data/processed/yolo_new`. Preparation uses `data/raw/` when its CSV exists,
otherwise `data/`; `--csv`, `--images-dir`, and `--seed` are configurable.
Invalid, missing, duplicate, or out-of-bounds annotations fail explicitly;
labels are never silently clipped or dropped. Existing destinations are never
erased. All four tools support `--help`, module invocation, and direct script
invocation; their paths resolve against the repository root.

After training, evaluate the test split and save precision, recall,
mAP@0.5, mAP@0.5:0.95, plots, and annotated prediction examples:

```powershell
.\.venv\Scripts\python.exe -m src.detection.evaluate_yolo --checkpoint runs/detect/helmet_yolo11n/weights/best.pt --device 0
```

Predict on an image, image directory, video, or one one-based CSV frame:

```powershell
.\.venv\Scripts\python.exe -m src.detection.predict_helmets --checkpoint runs/detect/helmet_yolo11n/weights/best.pt --source data/images --device 0
.\.venv\Scripts\python.exe -m src.detection.predict_helmets --checkpoint runs/detect/helmet_yolo11n/weights/best.pt --source data/videos/train/57583_000082_Endzone.mp4 --frame 38 --output runs/detect/frame_predictions --device 0
```

Omit `--frame` to process the complete video; results are streamed. Evaluation
and inference require an existing single-class helmet checkpoint, refuse to
overwrite output directories (choose a new `--output`), and offer `--dry-run`
for path checks without model loading. No trained checkpoint exists yet, so no
training, real evaluation, inference, or weight download was performed during
setup. CPU-only test results are retained in `data/processed/yolo/setup_validation.json`.
Impact classification is outside this phase.
