"""Unit tests for the face-detection and preprocessing pipeline."""

import unittest

import cv2
import numpy as np


class TestFaceDetector(unittest.TestCase):
    """Tests for FaceDetector."""

    @classmethod
    def setUpClass(cls):
        from src.preprocessing.face_processor import FaceDetector
        # Force Haar backend to avoid network I/O in CI
        cls.detector = FaceDetector(prefer_dnn=False)

    def _make_blank_frame(self, h: int = 480, w: int = 640) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_detect_returns_list(self):
        """detect() always returns a list."""
        frame = self._make_blank_frame()
        result = self.detector.detect(frame)
        self.assertIsInstance(result, list)

    def test_detect_bbox_format(self):
        """Each detected bbox is a 4-tuple of ints."""
        # Synthesise a rough face-like patch (white rectangle on grey)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bboxes = self.detector.detect(frame)
        for bbox in bboxes:
            self.assertEqual(len(bbox), 4)
            self.assertTrue(all(isinstance(v, int) for v in bbox))

    def test_haar_backend(self):
        """Backend name is 'haar' when prefer_dnn=False."""
        self.assertEqual(self.detector.backend, "haar")


class TestFacePreprocessor(unittest.TestCase):
    """Tests for FacePreprocessor."""

    @classmethod
    def setUpClass(cls):
        from src.preprocessing.face_processor import FacePreprocessor
        cls.preprocessor = FacePreprocessor(augment=False)
        cls.augment_preprocessor = FacePreprocessor(augment=True)

    def _make_frame(self, h: int = 480, w: int = 640) -> np.ndarray:
        return (np.random.rand(h, w, 3) * 255).astype(np.uint8)

    def test_process_returns_correct_shape(self):
        """process() returns (224, 224, 3) float32 array."""
        frame = self._make_frame()
        bbox = (50, 50, 200, 200)
        result = self.preprocessor.process(frame, bbox)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.float32)

    def test_process_invalid_bbox_returns_none(self):
        """process() returns None for an out-of-bounds bbox."""
        frame = self._make_frame(100, 100)
        bbox = (200, 200, 300, 300)  # completely outside frame
        result = self.preprocessor.process(frame, bbox)
        self.assertIsNone(result)

    def test_augment_returns_same_shape(self):
        """Augmented output has the same shape as non-augmented."""
        frame = self._make_frame()
        bbox = (50, 50, 200, 200)
        result = self.augment_preprocessor.process(frame, bbox)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (224, 224, 3))

    def test_denormalize_round_trip(self):
        """denormalize(process(frame)) produces a uint8 array in [0,255]."""
        from src.preprocessing.face_processor import FacePreprocessor
        prep = FacePreprocessor(augment=False)
        frame = self._make_frame()
        bbox = (50, 50, 200, 200)
        normalized = prep.process(frame, bbox)
        self.assertIsNotNone(normalized)
        recovered = prep.denormalize(normalized)
        self.assertEqual(recovered.dtype, np.uint8)
        self.assertTrue((recovered >= 0).all() and (recovered <= 255).all())

    def test_custom_face_size(self):
        """FacePreprocessor respects custom face_size."""
        from src.preprocessing.face_processor import FacePreprocessor
        prep = FacePreprocessor(face_size=112, augment=False)
        frame = self._make_frame()
        bbox = (50, 50, 200, 200)
        result = prep.process(frame, bbox)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (112, 112, 3))


if __name__ == "__main__":
    unittest.main()
