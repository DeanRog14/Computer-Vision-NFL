# NFL Helmet Impact Detection

A computer vision and deep learning project designed to automatically detect potential helmet impacts in American football video.

The system combines **YOLO11n for helmet detection** with a **3D ResNet-18 convolutional neural network for temporal impact classification**. Rather than relying solely on individual video frames, the project analyzes short sequences of frames to identify visual patterns associated with helmet impacts.

The end-to-end inference command accepts an NFL video, tracks helmets, and exports an annotated video and a report with potential impact timestamps and confidence scores.

This project was developed as a senior-level computer vision research project.

### Impact detection demo

https://github.com/user-attachments/assets/f4fd0f0c-ee99-438e-9271-f3a6a594b1ac

[Watch the annotated video](demo/impact-review.mp4) ·
[See alert timestamps and scores](demo/impact-alerts.csv)

## Project Overview

Helmet impacts are difficult to identify using traditional object detection because an impact is an event, not simply an object. Two helmets appearing close together or overlapping in an image does not necessarily indicate that a collision occurred.

To address this, the project separates the problem into two machine learning tasks:

1. **Helmet Detection:** Locate helmets in individual video frames using YOLO11n.
2. **Impact Classification:** Analyze nine consecutive frames surrounding a helmet to estimate whether an impact occurred using a 3D ResNet-18.

This approach combines spatial object detection with temporal video understanding.

### System Architecture

```text
                    NFL Game Video
                          |
                          v
                YOLO11n Helmet Detector
                          |
                          v
                 Helmet Bounding Boxes
                          |
                          v
                  Helmet Tracking
                          |
                          v
               Nine-Frame Clip Extraction
                          |
                          v
               3D ResNet-18 Classifier
                          |
                          v
                 Impact Probabilities
                          |
                          v
              Temporal Event Postprocessing
                          |
                          v
           Annotated Video + Impact Report
```

The individual models have been trained and evaluated. The video inference command runs both trained models in sequence.

## Models and Experiments

### 1. YOLO11n — Helmet Detection

The first model uses a pretrained YOLO11n object detector, fine-tuned on annotated NFL helmet images.

The detector identifies helmet locations and outputs bounding boxes that can subsequently be used for tracking and temporal impact classification.

**Training configuration:**

| Parameter | Configuration |
|---|---|
| Architecture | YOLO11n |
| Input resolution | 960 × 960 |
| Classes | 1 — Helmet |
| Training objective | Helmet localization |
| Maximum epochs | 50 |
| Batch size | 4 during initial GPU training |
| Data augmentation | Geometric and color transformations |
| Data splitting | Game-based, approximately 80/10/10 |

All original helmet annotation variants are mapped to a single `helmet` class.

**Recorded validation performance:**

| Metric | Result |
|---|---:|
| Precision | 93.34% |
| Recall | 84.29% |
| mAP@0.5 | 90.75% |
| mAP@0.5:0.95 | 54.70% |

These results demonstrated that individual helmets could be localized with relatively high accuracy, providing the spatial detection component for the larger system.

### 2. YOLO11n — Direct Impact Detection Baseline

An additional experiment explored whether YOLO could directly classify helmets as either normal helmets or impact helmets using individual images.

This model used two classes:

- `helmet`
- `impact_helmet`

**Impact-class performance:**

| Metric | Result |
|---|---:|
| Precision | 29.8% |
| Recall | 20.1% |
| mAP@0.5 | 19.6% |
| mAP@0.5:0.95 | 8.9% |

Although helmet detection remained relatively strong, direct impact classification performed poorly.

This experiment highlighted a central challenge: individual frames often do not contain enough information to reliably distinguish physical impacts from ordinary helmet proximity or overlap.

The results motivated the development of a temporal classifier.

### 3. 3D ResNet-18 — Temporal Impact Classification

The second major model uses a 3D convolutional neural network (`r3d_18`) implemented in PyTorch.

Unlike a conventional 2D CNN, a 3D CNN applies convolution across spatial and temporal dimensions, allowing it to learn how visual patterns change over successive frames.

Each classification sample contains:

- Nine consecutive video frames, centered on a candidate helmet.
- Four frames before and four frames after the center frame.
- A helmet-centered crop expanded to twice the helmet bounding-box dimensions.
- RGB images resized to 112 × 112 pixels.
- Channel normalization using Kinetics statistics.

The resulting input tensor has shape:

`[batch_size, 3, 9, 112, 112]`

The network produces a single impact classification logit, converted into a probability using the sigmoid function.

**Training configuration:**

| Parameter | Configuration |
|---|---|
| Architecture | 3D ResNet-18 |
| Framework | PyTorch / torchvision |
| Input | Nine-frame RGB clips |
| Resolution | 112 × 112 |
| Optimizer | AdamW |
| Loss | Weighted binary cross-entropy |
| Learning-rate scheduling | ReduceLROnPlateau |
| Maximum epochs | 30 |
| Early stopping | 5 epochs without F1 improvement |
| GPU training | Google Colab |
| Best checkpoint | Epoch 8 |

The implementation supports cached video clips, progress reporting, checkpoint saving, and training recovery.

**Best recorded epoch-validation results:**

| Metric | Result |
|---|---:|
| Precision | 80.67% |
| Recall | 84.32% |
| F1 score | 82.45% |
| Training loss | 0.2296 |
| Classification threshold | 0.1154 |
| Best epoch | 8 |
| Training stopped | Epoch 13 |

These results were calculated on a fixed, sampled validation subset containing 287 positive and 500 negative examples.

The results are promising, but they should not be interpreted as full-video or end-to-end impact detection performance. A larger validation evaluation is needed to measure false positives under a more realistic class distribution.

### Model Comparison

| Model | Task | Key Finding |
|---|---|---|
| YOLO11n | Helmet detection | Strong spatial helmet localization |
| Two-class YOLO11n | Single-frame impact detection | Impact recognition remained difficult |
| 3D ResNet-18 | Nine-frame impact classification | Promising temporal classification performance |

Because these experiments use different prediction targets and evaluation settings, their metrics are not directly interchangeable.

The central research hypothesis is that temporal visual information provides useful evidence for impact classification beyond what is available in an individual frame.

## Dataset

This project uses the NFL Impact Detection dataset, originally provided through a Kaggle computer vision competition.

The dataset includes:

- Annotated NFL sideline and endzone videos.
- Helmet bounding boxes.
- Labeled helmet-impact events.
- Player tracking information.
- Image-level helmet annotations.

### Helmet Detection Dataset

| Split | Images | Groups | Helmet Boxes |
|---|---:|---:|---:|
| Train | 7,938 | 508 | 154,383 |
| Validation | 978 | 64 | 19,026 |
| Test | 1,031 | 63 | 20,327 |

*Groups represent the game/play-level groupings used in the prepared image dataset.*

Splits were created using seed 42, with related frames and camera views assigned together to reduce data leakage.

### Temporal Impact Dataset

| Split | Positive Clips | Negative Clips |
|---|---:|---:|
| Train | 1,379 | 6,895 |
| Validation | 287 | 149,570 |
| Test | 222 | 114,244 |

Training negatives are sampled to control class imbalance. The complete validation and test splits retain their much larger negative populations for more representative evaluation.

Definitive impact annotations require:

- `impact == 1`
- `confidence > 1`
- `visibility > 0`

Non-definitive impact annotations are excluded from positive training targets.

Game-based splitting prevents the same games from appearing in multiple splits.
