# Indian Sign Language Detection

Real-time recognition of **Indian Sign Language (ISL)** from a standard webcam.
The system recognises **static signs** (the A–Z alphabet and finger-counted numbers 0–5) and
**dynamic signs** (61 words and phrases such as *Hello*, *Thank you* and *Good Morning*),
combining MediaPipe hand tracking with deep-learning models in a single OpenCV desktop application.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-1.1-0097A7)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv&logoColor=white)
![Tests](https://img.shields.io/badge/tests-76%20passing-2EA44F)

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Retraining the Models](#retraining-the-models)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

---

## Features

- **Static sign recognition** – live A–Z alphabet recognition, supporting both one- and two-handed signs.
- **Number mode** – counts raised fingers (0–5) on one hand.
- **Dynamic sign recognition** – classifies the last ~4 seconds of video into one of 61 ISL words and phrases.
- **Sign shortcuts** – hold a sign for one second to open an app or website (e.g. **W** → WhatsApp, **Y** → YouTube, **L** → Google).
- **Guided self-tests** – step-by-step A–Z and 0–5 webcam tests that save results to CSV.
- **Runs offline on CPU** – no GPU, web server or internet connection required.

---

## How It Works

### Static signs (A–Z)

```
Webcam frame ─► MediaPipe hand landmarks ─┬─► Geometric rule classifier ─┐
                                          └─► Hand crop ─► Custom CNN ───┴─► Fusion + smoothing ─► Letter
```

- **Hand tracking:** MediaPipe Gesture Recognizer (`gesture_recognizer.task`) detects up to two hands.
- **CNN:** a custom convolutional network (3× Conv2D → Dense 128 → Dense 26, softmax), exported to
  TensorFlow Lite (`model/custom_cnn.tflite`). Input: 224×224 RGB.
- **Rules:** landmark geometry provides an independent prediction for a subset of letters when a single hand is visible.
- **Smoothing:** predictions are averaged over a short window to keep the output stable.

### Dynamic signs (words and phrases)

```
Last ~4 s of video ─► 16 uniformly sampled frames ─► MobileNetV2 (frozen, per frame) ─► 2-layer Bi-GRU ─► Softmax (61 signs)
```

- **Trained model (default):** `models/dynamic/dynamic_model.keras` — ImageNet MobileNetV2 features feed a
  bidirectional GRU (128 → 64 units). Inference runs on a background thread once per second.
- **One-shot model (alternative):** `models/dynamic_oneshot/gallery.npz` — matches the live clip against one
  reference video per sign using MobileNetV2 features and dynamic time warping. No training required.

---

## Results

| Model | Evaluation | Result |
| :--- | :--- | :--- |
| Dynamic — MobileNetV2 + Bi-GRU | Held-out test set, 140 clips, 61 classes | **99.3 % accuracy** (139 / 140), macro-F1 **0.978** |
| Dynamic — one-shot (DTW) | 240 clips, 61 classes | 24.2 % top-1 (chance: 1.7 %) |

Dynamic training data: 1,043 clips split at recording level into 760 train / 143 validation / 140 test.
The best checkpoint was reached at epoch 49 (validation accuracy 97.2 %). Full metrics, per-class
recall, confusion matrix and predictions are in [`evaluation_results/dynamic_trained/`](evaluation_results/dynamic_trained/).

> **Note:** the test recordings are from the same signers and environment as the training data, so the
> 99.3 % figure reflects within-session performance. Expect lower accuracy for new signers, lighting and
> backgrounds. See [Limitations](#limitations).

---

## Getting Started

### Prerequisites

- Windows 10/11, macOS or Linux (verified on **Windows 11**)
- **Python 3.13** (verified with 3.13.5)
- A webcam

### Installation

```bash
git clone https://github.com/saisharatkaruturi/Indian-Sign-Language-Detection.git
cd Indian-Sign-Language-Detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source .venv/bin/activate         # macOS / Linux

python -m pip install -r requirements.txt
```

> The first launch can take 1–3 minutes while TensorFlow and MediaPipe initialise. Later launches start in a few seconds.

---

## Usage

Run all commands from the repository root.

```bash
python src/main.py                       # interactive menu
python src/main.py --mode static         # live A–Z recognition
python src/main.py --mode dynamic        # live word/phrase recognition
python src/main.py --mode camera_test    # check that the webcam works
python src/main.py --mode letter_test    # guided A–Z test (results saved to evaluation_results/live_tests/)
python src/main.py --mode number_test    # guided 0–5 finger-count test
```

### Keyboard controls

| Key | Action |
| :---: | :--- |
| `Q` | Quit |
| `M` | Toggle mirrored view |
| `C` | Clear the recognised sequence |
| `N` | Toggle number mode (0–5) |

### Sign shortcuts

Hold one of these letters for one second (5-second cooldown):

| Sign | Opens |
| :---: | :--- |
| **W** | WhatsApp Web |
| **Y** | YouTube |
| **L** | Google |

---

## Configuration

All runtime settings live in [`config.yaml`](config.yaml).

| Section | Key settings |
| :--- | :--- |
| `camera` | `index`, resolution, FPS |
| `dynamic` | `live_method` (`trained` or `oneshot`), `sequence_length` (16), `live_window_seconds` |
| `inference` | `confidence_threshold`, smoothing window |
| `actions` | enable/disable sign shortcuts and edit their targets (only `http(s)://` and `whatsapp:` URLs are accepted) |

---

## Testing

```bash
python -m pytest -v            # full suite: unit, model and integration tests
python integration_tests.py    # preprocessing-contract tests only
```

The suite covers preprocessing, both inference engines, the shipped models, number mode, sign shortcuts and
the live-test recorder. Tests that need hardware or data that is not present are skipped automatically.

---

## Retraining the Models

The dynamic training dataset (`Video_Dataset/`) is not included in this repository because of its size.
Place it in the repository root, then run:

```bash
python src/training/train_dynamic.py           # train MobileNetV2 + Bi-GRU on Video_Dataset/
python src/training/build_oneshot_gallery.py   # rebuild the one-shot gallery from Sample Videos/
python src/training/evaluate_oneshot.py        # evaluate the one-shot matcher
```

Training writes the model to `models/dynamic/` and the evaluation report to `evaluation_results/dynamic_trained/`.

---

## Project Structure

```
Indian-Sign-Language-Detection/
├── src/
│   ├── main.py                    # Application entry point and mode menu
│   ├── camera/                    # Webcam access and camera test
│   ├── dataset/                   # Dataset loading, validation and leakage-free splitting
│   ├── inference/                 # Static/dynamic engines, real-time pipeline, shortcuts, live tests
│   ├── models/                    # CNN, GRU and LSTM model definitions
│   ├── preprocessing/             # Image, video and frame-sequence preprocessing
│   ├── training/                  # Training, gallery building and evaluation scripts
│   └── utils/                     # Configuration, logging and class mappings
├── model/
│   └── custom_cnn.tflite          # Static A–Z CNN (TensorFlow Lite)
├── models/
│   ├── static/                    # Static class mapping (A–Z)
│   ├── dynamic/                   # Trained Bi-GRU model + class mapping (61 signs)
│   └── dynamic_oneshot/           # One-shot reference gallery
├── Sample Videos/                 # One reference clip per dynamic sign (61 videos)
├── evaluation_results/            # Saved metrics and live-test results
├── data_pipeline/                 # Reusable preprocessing contracts
├── integration/                   # Mock recognisers used by the contract tests
├── tests/                         # Pytest suite
├── gesture_recognizer.task        # MediaPipe hand-tracking model
├── dynamic_video_splits.csv       # Recording-level train/val/test manifest
├── config.yaml                    # Runtime configuration
├── integration_tests.py           # Data-contract tests
├── pytest.ini
└── requirements.txt
```

---

## Limitations

- **Signer independence is not yet measured.** The dynamic test set shares signers and recording sessions
  with the training set; performance on new users will be lower than the reported 99.3 %.
- **Partial dynamic dataset.** The model was trained on 1,043 of the 2,723 available clips, with 1–8 test clips per sign.
- **One-shot matching is weak** (24.2 % top-1) and is intended only as a fallback for signs the trained model does not know.
- **Numbers 6–9** (two-handed in ISL) are not supported; number mode covers 0–5 only.
- **Rule-based confidences** for the geometric classifier are fixed values, not calibrated probabilities.
- **Static model accuracy** has not been re-measured on a held-out set in this repository, because the static image dataset is not included.

---

## Troubleshooting

| Problem | Solution |
| :--- | :--- |
| `ModuleNotFoundError: mediapipe` | Activate the virtual environment and reinstall `requirements.txt`. |
| `cv2` import errors | Install only `opencv-contrib-python` (pinned in `requirements.txt`); uninstall `opencv-python` if present. |
| Camera does not open | Close other applications using the webcam, or change `camera.index` in `config.yaml`. |
| `Static CNN not loaded (...)` | Check that `model/custom_cnn.tflite` and `models/static/class_mapping.json` exist. |
| `tf.lite.Interpreter is deprecated` warning | Harmless. Install `ai-edge-litert` to use the newer interpreter automatically. |
| Slow first start | Expected on a fresh environment (TensorFlow/MediaPipe initialisation and antivirus scanning). |

---

## Author

Developed by [@saisharatkaruturi](https://github.com/saisharatkaruturi).

If you find this project useful, consider giving it a ⭐.
