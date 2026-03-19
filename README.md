# LiveDeepfakeDetection

**Live Detection of Synthetic Faces in Video Conferencing Using Lightweight CNN Models**

> Rohan Mishra, Siddharth Kumar Singh, Manak Yadav · Amity University, Uttar Pradesh  
> Supervisor: Prof. Abhishek Singhal

---

## Overview

This project implements a real-time deepfake detection system for live video conferencing, based on a custom **Lightweight Convolutional Neural Network (CNN)** that uses depthwise-separable convolutions.

| Metric | Value |
|--------|-------|
| Classification accuracy | **92.3 %** |
| Target FPS (Raspberry Pi 4) | **28–35 FPS** |
| Model size (int8 quantised) | **~3.2 MB** |
| Model size reduction vs Xception | **97 %** |

The system processes live webcam video, detects faces, runs them through the CNN, and raises alerts using **temporal consistency verification** — a 5-frame sliding-window consensus mechanism that dramatically reduces false positives.

---

## Architecture

```
Input (224×224×3)
  └─▶ Conv1 (32 filters, 3×3, stride 2)
        └─▶ DSC Block 1 (64  filters, stride 2)
              └─▶ DSC Block 2 (128 filters, stride 2)
                    └─▶ DSC Block 3 (128 filters, stride 1)
                          └─▶ DSC Block 4 (256 filters, stride 2)
                                └─▶ DSC Block 5 (256 filters, stride 1)
                                      └─▶ DSC Block 6 (512 filters, stride 2)
                                            └─▶ GlobalAveragePooling
                                                  └─▶ Dense(128) → Dropout(0.5)
                                                        └─▶ Dense(2, softmax)
                                                              [P(Real), P(Fake)]
```

Each **Depthwise-Separable Convolution (DSC)** block:

```
DepthwiseConv2D(3×3) → BatchNorm(momentum=0.99) → ReLU
→ Conv2D(1×1)        → BatchNorm               → ReLU
```

---

## Features

- **Real-time face detection** — OpenCV DNN (MobileNet-SSD) with Haar-cascade fallback
- **Face preprocessing** — 10 % margin crop → 224×224 resize → ImageNet normalisation
- **Training augmentation** — random flip, ±15° rotation, brightness/contrast/saturation jitter, Gaussian blur, random erasing
- **Temporal consistency verification** — 5-frame sliding window with adaptive consensus (θ=0.65) and consistency thresholds (0.75)
- **Quantisation & pruning** — post-training int8 TFLite export; magnitude-based filter pruning
- **Privacy-preserving** — all processing is local; no frames leave the device
- **Tkinter GUI** — live webcam feed with detection overlays, confidence bars, alert log

---

## Directory Structure

```
LiveDeepfakeDetection/
├── src/
│   ├── model/
│   │   └── lightweight_cnn.py      # CNN architecture + TFLite export
│   ├── preprocessing/
│   │   └── face_processor.py       # Face detection + preprocessing
│   ├── detection/
│   │   └── detector.py             # Temporal consistency + detection logic
│   └── training/
│       └── trainer.py              # Training pipeline (multi-dataset)
├── gui/
│   └── app.py                      # Tkinter GUI
├── tests/
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_detector.py
├── main.py                         # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Rohan-m2004/LiveDeepfakeDetection.git
cd LiveDeepfakeDetection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Launch the GUI (demo mode — no model required)

```bash
python main.py
```

Click **▶ Start Camera** to open the webcam. The system will detect faces and display bounding boxes. Without a trained model, predictions are 50 % (uniform uncertainty).

### 2. Load a pre-trained model

```bash
python main.py --model models/deepfake_detector.keras
# or TFLite (edge/Raspberry Pi deployment)
python main.py --model models/deepfake_detector.tflite
```

### 3. Train a model

Organise your dataset:

```
data/
├── train/
│   ├── real/   ← original face frames (e.g. from FaceForensics++)
│   └── fake/   ← manipulated face frames
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Recommended datasets (as used in the paper):
- **Train**: FaceForensics++ + Celeb-DF
- **Validation**: DFDC
- **Test (held-out)**: DeeperForensics

```bash
python main.py --train --data data/ --output-dir models/ --epochs 100
```

### 4. Headless inference on a video file

```bash
python main.py --video path/to/meeting_recording.mp4 --model models/deepfake_detector.keras
```

---

## GUI Guide

| Element | Description |
|---------|-------------|
| **Camera feed** | Live annotated webcam feed. Green box = REAL, Red box = FAKE. |
| **P(Fake) / P(Real)** | Per-frame softmax probabilities from the CNN. |
| **Consensus** | Mean P(Fake) over the last 5 frames (temporal window). |
| **Consistency** | Temporal stability score — low variance = high consistency. |
| **Alert banner** | Red banner when both consensus > 0.65 and consistency > 0.75. |
| **Alert log** | Timestamped history of raised alerts. |

---

## Model Compression

After training, the model is compressed via:

1. **Magnitude-based structured pruning** — filters with the lowest L1-norm importance are zeroed (configurable prune ratio, default 30 %).
2. **Post-training int8 quantisation** — weights and activations converted to 8-bit integers via TFLite.

| Format | Size | Relative |
|--------|------|----------|
| Float32 Keras | ~12.8 MB | baseline |
| Pruned float32 | ~6–9 MB | −30–50 % |
| Pruned + int8 TFLite | **~3.2 MB** | **−75 %** |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Hardware Targets

| Device | Backend | FPS | Power |
|--------|---------|-----|-------|
| Desktop i7 + GTX 1080 Ti | GPU (TF) | 120+ | – |
| NVIDIA Jetson Nano | GPU (TFLite) | 45–60 | ~5 W |
| Raspberry Pi 4 (4 GB) | CPU (TFLite int8) | **28–35** | 4.2 W |

---

## References

1. Rössler et al., "FaceForensics++", ICCV 2019.
2. Sandler et al., "MobileNetV2", CVPR 2018.
3. Afchar et al., "MesoNet", IEEE WIFS 2018.
4. Li et al., "Celeb-DF", CVPR 2020.
5. Dolhansky et al., "The DeepFake Detection Challenge (DFDC)", arXiv 2020.
6. Li et al., "DeeperForensics-1.0", CVPR 2020.

---

## License

This project is for academic research purposes.  
© 2024 Rohan Mishra, Amity University, Uttar Pradesh.