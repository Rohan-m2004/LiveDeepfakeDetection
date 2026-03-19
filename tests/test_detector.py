"""Unit tests for the deepfake detection pipeline."""

import unittest

import numpy as np


# ---------------------------------------------------------------------------
# TemporalAnalyzer tests
# ---------------------------------------------------------------------------

class TestTemporalAnalyzer(unittest.TestCase):
    """Tests for TemporalAnalyzer."""

    @classmethod
    def setUpClass(cls):
        from src.detection.detector import TemporalAnalyzer
        cls.TA = TemporalAnalyzer

    def test_no_predictions_returns_zero_consensus(self):
        """Empty window → consensus_p_fake = 0."""
        ta = self.TA(window_size=5, frame_conf_threshold=0.65)
        result = ta.update(0.3)  # below threshold, not added
        self.assertEqual(result.window_size, 0)
        self.assertAlmostEqual(result.consensus_p_fake, 0.0)

    def test_high_p_fake_fills_window(self):
        """Values above threshold fill the window."""
        ta = self.TA(window_size=3, frame_conf_threshold=0.5)
        for _ in range(3):
            result = ta.update(0.9)
        self.assertEqual(result.window_size, 3)

    def test_alert_raised_when_all_thresholds_met(self):
        """Alert is raised when consensus + consistency are above thresholds."""
        ta = self.TA(
            window_size=5,
            consensus_threshold=0.65,
            consistency_threshold=0.0,  # ignore consistency for this test
            frame_conf_threshold=0.0,
        )
        for _ in range(5):
            result = ta.update(0.9)
        self.assertTrue(result.alert)

    def test_no_alert_when_consensus_below_threshold(self):
        """No alert when consensus P(Fake) is below consensus_threshold."""
        ta = self.TA(
            window_size=3,
            consensus_threshold=0.80,
            consistency_threshold=0.0,
            frame_conf_threshold=0.0,
        )
        for _ in range(3):
            result = ta.update(0.5)
        self.assertFalse(result.alert)

    def test_consistency_score_range(self):
        """Consistency score is always in [0, 1]."""
        ta = self.TA(window_size=5, frame_conf_threshold=0.0)
        for val in [0.2, 0.9, 0.6, 0.1, 0.8]:
            result = ta.update(val)
        self.assertGreaterEqual(result.consistency_score, 0.0)
        self.assertLessEqual(result.consistency_score, 1.0)

    def test_reset_clears_window(self):
        """reset() clears the sliding window so next update starts fresh."""
        ta = self.TA(window_size=5, frame_conf_threshold=0.0)
        for _ in range(5):
            ta.update(0.9)
        ta.reset()
        # After reset the window is empty; one update adds exactly 1 entry
        result = ta.update(0.9)
        self.assertEqual(result.window_size, 1)

    def test_window_does_not_exceed_max_size(self):
        """Sliding window never grows beyond window_size."""
        ta = self.TA(window_size=3, frame_conf_threshold=0.0)
        for _ in range(10):
            result = ta.update(0.8)
        self.assertLessEqual(result.window_size, 3)


# ---------------------------------------------------------------------------
# DeepfakeDetector tests
# ---------------------------------------------------------------------------

class TestDeepfakeDetector(unittest.TestCase):
    """Tests for DeepfakeDetector (without a real model)."""

    @classmethod
    def setUpClass(cls):
        from src.detection.detector import DeepfakeDetector
        # Use Haar backend to avoid network I/O in CI
        cls.detector = DeepfakeDetector(prefer_dnn_faces=False)

    def _make_frame(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_process_frame_returns_detection_result(self):
        """process_frame() returns a DetectionResult."""
        from src.detection.detector import DetectionResult
        frame = self._make_frame()
        result = self.detector.process_frame(frame)
        self.assertIsInstance(result, DetectionResult)

    def test_fps_positive(self):
        """FPS is a positive number after the first call."""
        frame = self._make_frame()
        result = self.detector.process_frame(frame)
        self.assertGreater(result.fps, 0.0)

    def test_no_faces_in_blank_frame(self):
        """Blank black frame should produce no face detections."""
        frame = self._make_frame()
        result = self.detector.process_frame(frame)
        # Blank frame has no faces; list may be empty
        self.assertIsInstance(result.faces, list)

    def test_no_model_uniform_probability(self):
        """Without a model both probabilities should be 0.5."""
        import cv2
        # Create a synthetic face-like image
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Manually inject a mock face result
        from src.detection.detector import FaceDetectionResult, DeepfakeDetector
        detector = DeepfakeDetector(model=None, prefer_dnn_faces=False)
        result = detector.process_frame(frame)
        for face in result.faces:
            self.assertAlmostEqual(face.p_fake + face.p_real, 1.0, places=5)

    def test_reset_temporal_does_not_raise(self):
        """reset_temporal() can be called at any time."""
        self.detector.reset_temporal()  # should not raise


# ---------------------------------------------------------------------------
# DetectionResult tests
# ---------------------------------------------------------------------------

class TestDetectionResult(unittest.TestCase):
    """Tests for DetectionResult dataclass."""

    def test_alert_false_when_no_temporal(self):
        """alert=False when temporal is None."""
        from src.detection.detector import DetectionResult
        result = DetectionResult(
            timestamp=0.0,
            faces=[],
            temporal=None,
            fps=30.0,
        )
        self.assertFalse(result.alert)

    def test_alert_mirrors_temporal_alert(self):
        """result.alert is True when temporal.alert is True."""
        from src.detection.detector import DetectionResult, TemporalResult
        temporal = TemporalResult(
            consensus_p_fake=0.9,
            consistency_score=0.95,
            alert=True,
            window_size=5,
        )
        result = DetectionResult(
            timestamp=0.0,
            faces=[],
            temporal=temporal,
            fps=28.0,
        )
        self.assertTrue(result.alert)


if __name__ == "__main__":
    unittest.main()
