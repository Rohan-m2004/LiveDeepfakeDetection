"""Unit tests for the Lightweight CNN model architecture."""

import unittest

import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


class TestBuildLightweightCNN(unittest.TestCase):
    """Tests for build_lightweight_cnn factory function."""

    @classmethod
    def setUpClass(cls):
        from src.model.lightweight_cnn import build_lightweight_cnn
        cls.build = staticmethod(build_lightweight_cnn)

    def test_model_output_shape(self):
        """Model produces (batch, 2) output for binary classification."""
        model = self.build()
        dummy = np.zeros((2, 224, 224, 3), dtype=np.float32)
        preds = model.predict(dummy, verbose=0)
        self.assertEqual(preds.shape, (2, 2))

    def test_output_sums_to_one(self):
        """Softmax outputs must sum to ~1 for each sample."""
        model = self.build()
        dummy = np.random.rand(4, 224, 224, 3).astype(np.float32)
        preds = model.predict(dummy, verbose=0)
        for row in preds:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=5)

    def test_output_in_zero_one(self):
        """All probability outputs are in [0, 1]."""
        model = self.build()
        dummy = np.random.rand(3, 224, 224, 3).astype(np.float32)
        preds = model.predict(dummy, verbose=0)
        self.assertTrue((preds >= 0).all() and (preds <= 1).all())

    def test_model_name(self):
        """Model has the expected name."""
        model = self.build()
        self.assertIn("Lightweight", model.name)

    def test_has_dsc_layers(self):
        """Model contains 6 depthwise and 6 pointwise convolution layers."""
        from tensorflow.keras import layers as klayers
        model = self.build()
        dw_layers = [l for l in model.layers if isinstance(l, klayers.DepthwiseConv2D)]
        pw_layers = [
            l for l in model.layers
            if isinstance(l, klayers.Conv2D) and l.kernel_size == (1, 1)
        ]
        self.assertEqual(len(dw_layers), 6, "Expected 6 DepthwiseConv2D layers")
        self.assertEqual(len(pw_layers), 6, "Expected 6 pointwise Conv2D layers")

    def test_custom_input_shape(self):
        """Model can be built with a different input resolution."""
        model = self.build(input_shape=(112, 112, 3))
        dummy = np.zeros((1, 112, 112, 3), dtype=np.float32)
        preds = model.predict(dummy, verbose=0)
        self.assertEqual(preds.shape, (1, 2))

    def test_parameter_count_reasonable(self):
        """Model should have fewer than 5 million parameters."""
        model = self.build()
        n_params = model.count_params()
        self.assertLess(n_params, 5_000_000, f"Got {n_params} params")

    def test_single_sample_inference(self):
        """Single-sample inference returns a (1, 2) array."""
        model = self.build()
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        preds = model.predict(dummy, verbose=0)
        self.assertEqual(preds.shape, (1, 2))


class TestTFLiteInferenceModel(unittest.TestCase):
    """Tests for TFLiteInferenceModel wrapper."""

    def test_predict_shape(self):
        """TFLite model predict returns correct output shape."""
        import tempfile
        from src.model.lightweight_cnn import (
            TFLiteInferenceModel,
            build_lightweight_cnn,
            convert_to_tflite,
        )

        model = build_lightweight_cnn()
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            tmp_path = f.name

        try:
            convert_to_tflite(model, tmp_path, quantize=False)
            tfl = TFLiteInferenceModel(tmp_path)
            dummy = np.zeros((3, 224, 224, 3), dtype=np.float32)
            preds = tfl.predict(dummy)
            self.assertEqual(preds.shape, (3, 2))
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
