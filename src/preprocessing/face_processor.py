"""
Face detection and preprocessing pipeline.

Responsibilities
----------------
* Detect face bounding boxes in BGR frames (OpenCV convention).
* Crop, align and normalise each face region to a 224×224 RGB tensor
  ready for CNN inference.
* Apply training-time augmentation when requested.

Face detection falls back gracefully:
    1. OpenCV DNN (SSD + MobileNet) — fastest, most accurate.
    2. OpenCV Haar-cascade            — always available, no extra downloads.
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACE_SIZE: int = 224  # target H × W fed to the CNN
FACE_MARGIN: float = 0.10  # 10 % bounding-box expansion (as in the paper)

# ImageNet mean / std used for per-channel normalisation
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Haar-cascade shipped with every OpenCV installation
_HAAR_PATH = str(
    Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
)

# Optional: lightweight OpenCV DNN face-detector (downloaded on first use)
_DNN_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_DNN_PROTO = str(_DNN_MODEL_DIR / "deploy.prototxt")
_DNN_WEIGHTS = str(_DNN_MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel")

_DNN_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_DNN_WEIGHTS_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_dnn_detector() -> Optional[cv2.dnn_Net]:
    """Try to load the OpenCV DNN face detector.  Returns None on failure."""
    _DNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not os.path.exists(_DNN_PROTO) or not os.path.exists(_DNN_WEIGHTS):
            # Attempt to download the tiny model files (~2 MB total)
            urllib.request.urlretrieve(_DNN_PROTO_URL, _DNN_PROTO)
            urllib.request.urlretrieve(_DNN_WEIGHTS_URL, _DNN_WEIGHTS)
        net = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_WEIGHTS)
        return net
    except Exception:
        return None


def _haar_detector() -> cv2.CascadeClassifier:
    """Return an OpenCV Haar-cascade face classifier."""
    if not os.path.exists(_HAAR_PATH):
        raise FileNotFoundError(
            f"Haar cascade not found at {_HAAR_PATH}.  "
            "Please verify your OpenCV installation."
        )
    clf = cv2.CascadeClassifier(_HAAR_PATH)
    return clf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FaceDetector:
    """Detect face bounding boxes in BGR images.

    The detector tries to use the OpenCV DNN backend first (more accurate).
    If that is unavailable it falls back to a Haar-cascade detector.

    Args:
        confidence_threshold: Minimum DNN confidence score to accept a
                              detection (ignored for Haar backend).
        prefer_dnn:           Try DNN detector first (default: True).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        prefer_dnn: bool = True,
    ) -> None:
        self._conf_thresh = confidence_threshold
        self._dnn_net: Optional[cv2.dnn_Net] = None
        self._haar_clf: Optional[cv2.CascadeClassifier] = None

        if prefer_dnn:
            self._dnn_net = _load_dnn_detector()

        if self._dnn_net is None:
            self._haar_clf = _haar_detector()

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return "dnn" if self._dnn_net is not None else "haar"

    # ------------------------------------------------------------------
    def detect(self, bgr_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in *bgr_frame*.

        Args:
            bgr_frame: BGR image as returned by ``cv2.VideoCapture.read()``.

        Returns:
            List of ``(x, y, w, h)`` bounding boxes in pixel coordinates.
        """
        if self._dnn_net is not None:
            return self._detect_dnn(bgr_frame)
        return self._detect_haar(bgr_frame)

    # ------------------------------------------------------------------
    def _detect_dnn(self, bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        boxes: List[Tuple[int, int, int, int]] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self._conf_thresh:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                boxes.append((x1, y1, bw, bh))
        return boxes

    # ------------------------------------------------------------------
    def _detect_haar(self, bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


# ---------------------------------------------------------------------------

class FacePreprocessor:
    """Crop, resize and normalise face regions for CNN input.

    Args:
        face_size:  Target spatial resolution (square).  Default: 224.
        margin:     Fractional margin to expand the bounding box.
        augment:    Apply random augmentation (training only).
    """

    def __init__(
        self,
        face_size: int = FACE_SIZE,
        margin: float = FACE_MARGIN,
        augment: bool = False,
    ) -> None:
        self._face_size = face_size
        self._margin = margin
        self._augment = augment

    # ------------------------------------------------------------------
    def process(
        self,
        bgr_frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Extract and preprocess a face patch.

        Args:
            bgr_frame: Full BGR frame from the camera.
            bbox:      ``(x, y, w, h)`` bounding box in pixel coords.

        Returns:
            Float32 array of shape ``(224, 224, 3)`` normalised to
            ImageNet statistics, or ``None`` if the crop is invalid.
        """
        patch = self._crop(bgr_frame, bbox)
        if patch is None:
            return None

        # BGR → RGB
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        # Resize to target resolution
        patch = cv2.resize(patch, (self._face_size, self._face_size))

        # Scale to [0, 1]
        patch = patch.astype(np.float32) / 255.0

        if self._augment:
            patch = self._apply_augmentation(patch)

        # Per-channel ImageNet normalisation
        patch = (patch - _IMAGENET_MEAN) / _IMAGENET_STD

        return patch

    # ------------------------------------------------------------------
    def _crop(
        self,
        bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Return an expanded crop of the face bounding box."""
        frame_h, frame_w = bgr.shape[:2]
        x, y, w, h = bbox

        # Expand bounding box by margin
        pad_x = int(w * self._margin)
        pad_y = int(h * self._margin)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return bgr[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a float32 RGB image in [0, 1].

        Augmentation strategy (as described in the paper):
            * Random horizontal flip (p=0.5)
            * Random rotation  ±15°
            * Colour jitter: brightness ±0.3, contrast ±0.3, saturation ±0.2
            * Gaussian blur   σ ∈ [0.1, 2.0]  (p=0.3)
            * Random erasing  (p=0.2)
        """
        rng = np.random.default_rng()

        # Horizontal flip
        if rng.random() < 0.5:
            img = img[:, ::-1, :]

        # Rotation ±15°
        angle = rng.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

        # Brightness jitter
        brightness = rng.uniform(-0.3, 0.3)
        img = np.clip(img + brightness, 0.0, 1.0)

        # Contrast jitter
        contrast = rng.uniform(0.7, 1.3)
        mean = img.mean()
        img = np.clip((img - mean) * contrast + mean, 0.0, 1.0)

        # Saturation jitter (in HSV space)
        img_uint8 = (img * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        # Gaussian blur
        if rng.random() < 0.3:
            sigma = rng.uniform(0.1, 2.0)
            k = int(2 * round(3 * sigma) + 1)
            img = cv2.GaussianBlur(img, (k | 1, k | 1), sigma)

        # Random erasing
        if rng.random() < 0.2:
            h, w = img.shape[:2]
            er_h = int(rng.uniform(0.05, 0.2) * h)
            er_w = int(rng.uniform(0.05, 0.2) * w)
            er_y = int(rng.uniform(0, h - er_h))
            er_x = int(rng.uniform(0, w - er_w))
            img[er_y : er_y + er_h, er_x : er_x + er_w] = rng.random(
                (er_h, er_w, 3)
            ).astype(np.float32)

        return img.astype(np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def denormalize(img: np.ndarray) -> np.ndarray:
        """Reverse ImageNet normalisation for visualisation.

        Args:
            img: Float32 array (H, W, 3) in ImageNet normalised space.

        Returns:
            uint8 RGB array in [0, 255].
        """
        out = img * _IMAGENET_STD + _IMAGENET_MEAN
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return out
