"""
Core deepfake detection logic.

Implements:
    * Frame-level inference via the CNN model.
    * Temporal consistency verification over a sliding window of N frames.
    * Adaptive confidence thresholding.
    * Two-stage alert generation (per the paper).

Detection pipeline
------------------

    raw frame
        └─▶ FaceDetector  ──▶  list[(x,y,w,h)]
                                     │
                              FacePreprocessor
                                     │
                              Model.predict()          ← P(Real), P(Fake)
                                     │
                            TemporalAnalyzer           ← sliding-window consensus
                                     │
                            AdaptiveThreshold          ← two-stage alert
                                     │
                              DetectionResult
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple, Union

import numpy as np

from src.preprocessing.face_processor import FaceDetector, FacePreprocessor

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ModelType = Union[object]  # Keras model or TFLiteInferenceModel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FaceDetectionResult:
    """Per-face detection result for a single frame.

    Attributes:
        bbox:               (x, y, w, h) in pixel coordinates.
        p_fake:             Probability of the face being fake [0, 1].
        p_real:             Probability of the face being real [0, 1].
        label:              ``"FAKE"`` or ``"REAL"``.
        frame_confidence:   Raw model confidence for the predicted class.
    """
    bbox: Tuple[int, int, int, int]
    p_fake: float
    p_real: float
    label: str
    frame_confidence: float


@dataclass
class TemporalResult:
    """Temporal consensus result for a face track over N frames.

    Attributes:
        consensus_p_fake:   Mean P(Fake) over the sliding window.
        consistency_score:  1 − std / max_std (higher = more consistent).
        alert:              True when both thresholds are exceeded.
        window_size:        Number of frames in the current window.
    """
    consensus_p_fake: float
    consistency_score: float
    alert: bool
    window_size: int


@dataclass
class DetectionResult:
    """Combined per-frame result returned to the caller.

    Attributes:
        timestamp:          Unix timestamp when this result was produced.
        faces:              Per-face frame-level results.
        temporal:           Temporal result (None until window is filled).
        fps:                Current processing rate (frames per second).
        alert:              Convenience alias — True if any face is alerted.
    """
    timestamp: float
    faces: List[FaceDetectionResult]
    temporal: Optional[TemporalResult]
    fps: float
    alert: bool = field(init=False)

    def __post_init__(self) -> None:
        self.alert = self.temporal is not None and self.temporal.alert


# ---------------------------------------------------------------------------
# Temporal Analyser
# ---------------------------------------------------------------------------

class TemporalAnalyzer:
    """Aggregate per-frame predictions over a sliding window.

    Implements the two-stage filtering described in the paper:
        1. Frame-level confidence gate  (``frame_conf_threshold``).
        2. Sliding-window consensus + consistency check.

    Args:
        window_size:            Number of frames in the sliding window (N=5).
        consensus_threshold:    Minimum mean P(Fake) to raise an alert (θ=0.65).
        consistency_threshold:  Minimum consistency score required (0.75).
        frame_conf_threshold:   Only include predictions with
                                P(Fake) > this value (0.65).
        max_std:                Normalisation constant for std → consistency.
    """

    def __init__(
        self,
        window_size: int = 5,
        consensus_threshold: float = 0.65,
        consistency_threshold: float = 0.75,
        frame_conf_threshold: float = 0.65,
        max_std: float = 0.5,
    ) -> None:
        self._window_size = window_size
        self._consensus_threshold = consensus_threshold
        self._consistency_threshold = consistency_threshold
        self._frame_conf_threshold = frame_conf_threshold
        self._max_std = max_std
        self._window: Deque[float] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    def update(self, p_fake: float) -> TemporalResult:
        """Add a new P(Fake) value and return the current temporal result.

        Args:
            p_fake: Probability of fake for the most recent frame.

        Returns:
            :class:`TemporalResult` with current consensus and alert status.
        """
        if p_fake >= self._frame_conf_threshold:
            self._window.append(p_fake)

        if len(self._window) == 0:
            return TemporalResult(
                consensus_p_fake=0.0,
                consistency_score=1.0,
                alert=False,
                window_size=0,
            )

        arr = np.array(self._window, dtype=np.float32)
        consensus = float(arr.mean())
        std = float(arr.std())
        consistency = 1.0 - min(std / self._max_std, 1.0)

        alert = (
            len(self._window) >= self._window_size
            and consensus >= self._consensus_threshold
            and consistency >= self._consistency_threshold
        )

        return TemporalResult(
            consensus_p_fake=consensus,
            consistency_score=consistency,
            alert=alert,
            window_size=len(self._window),
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the sliding window."""
        self._window.clear()


# ---------------------------------------------------------------------------
# Main Detector
# ---------------------------------------------------------------------------

class DeepfakeDetector:
    """End-to-end deepfake detection pipeline.

    Orchestrates face detection, preprocessing, model inference, and
    temporal consistency verification for live video frames.

    Args:
        model:              Keras or TFLite model with a ``predict()`` method.
        confidence_threshold:   DNN face-detection confidence.
        prefer_dnn_faces:   Use OpenCV DNN face detector when available.
        temporal_window:    Sliding-window size (N).
        consensus_threshold: Temporal consensus threshold (θ).
        consistency_threshold: Temporal consistency threshold.
        frame_conf_threshold:  Minimum per-frame P(Fake) to feed temporal window.
    """

    def __init__(
        self,
        model: Optional[ModelType] = None,
        confidence_threshold: float = 0.5,
        prefer_dnn_faces: bool = True,
        temporal_window: int = 5,
        consensus_threshold: float = 0.65,
        consistency_threshold: float = 0.75,
        frame_conf_threshold: float = 0.65,
    ) -> None:
        self._model = model

        self._face_detector = FaceDetector(
            confidence_threshold=confidence_threshold,
            prefer_dnn=prefer_dnn_faces,
        )
        self._preprocessor = FacePreprocessor(augment=False)

        self._temporal = TemporalAnalyzer(
            window_size=temporal_window,
            consensus_threshold=consensus_threshold,
            consistency_threshold=consistency_threshold,
            frame_conf_threshold=frame_conf_threshold,
        )

        self._last_time: float = time.perf_counter()
        self._fps: float = 0.0

    # ------------------------------------------------------------------
    @property
    def model(self) -> Optional[ModelType]:
        return self._model

    @model.setter
    def model(self, m: ModelType) -> None:
        self._model = m
        self._temporal.reset()

    # ------------------------------------------------------------------
    def process_frame(self, bgr_frame: np.ndarray) -> DetectionResult:
        """Process one BGR video frame end-to-end.

        Args:
            bgr_frame: Raw camera frame (BGR, uint8).

        Returns:
            :class:`DetectionResult` with all per-face and temporal data.
        """
        now = time.perf_counter()
        elapsed = now - self._last_time
        self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._last_time = now

        timestamp = time.time()
        face_results: List[FaceDetectionResult] = []
        temporal_result: Optional[TemporalResult] = None

        # --- face detection -------------------------------------------
        bboxes = self._face_detector.detect(bgr_frame)

        for bbox in bboxes:
            patch = self._preprocessor.process(bgr_frame, bbox)
            if patch is None:
                continue

            if self._model is not None:
                probs = self._model.predict(np.expand_dims(patch, 0))[0]
                p_real = float(probs[0])
                p_fake = float(probs[1])
            else:
                # No model loaded — return uniform uncertainty
                p_real = 0.5
                p_fake = 0.5

            label = "FAKE" if p_fake > 0.5 else "REAL"
            confidence = max(p_real, p_fake)

            face_results.append(
                FaceDetectionResult(
                    bbox=bbox,
                    p_fake=p_fake,
                    p_real=p_real,
                    label=label,
                    frame_confidence=confidence,
                )
            )

            # Use the most prominent face for temporal tracking
            temporal_result = self._temporal.update(p_fake)

        return DetectionResult(
            timestamp=timestamp,
            faces=face_results,
            temporal=temporal_result,
            fps=self._fps,
        )

    # ------------------------------------------------------------------
    def reset_temporal(self) -> None:
        """Reset the temporal sliding window (call on scene cuts)."""
        self._temporal.reset()
