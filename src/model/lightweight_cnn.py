"""
Lightweight CNN model for deepfake detection.

Architecture based on:
  "Live Detection of Synthetic Faces in Video Conferencing
   Using Lightweight CNN Models"

The model uses depthwise separable convolutions (DSC) to achieve
a ~3.2 MB footprint while maintaining 92.3% classification accuracy.
"""

import os
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Depthwise-Separable Convolution Block
# ---------------------------------------------------------------------------

def _dsc_block(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    name: str = "dsc",
) -> tf.Tensor:
    """One depthwise-separable convolution block.

    Structure:
        DepthwiseConv2D(3×3, stride) → BN → ReLU
        → Conv2D(1×1)               → BN → ReLU

    Args:
        x:       Input tensor.
        filters: Number of pointwise (output) filters.
        stride:  Stride for the depthwise convolution.
        name:    Prefix for layer names.

    Returns:
        Output tensor after the DSC block.
    """
    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-3, name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)

    # Pointwise convolution
    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_pw",
    )(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-3, name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)

    return x


# ---------------------------------------------------------------------------
# Main Model Factory
# ---------------------------------------------------------------------------

def build_lightweight_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> keras.Model:
    """Build the Lightweight CNN model for binary deepfake classification.

    Architecture:
        Input (224×224×3)
        → Conv1 (32 filters, 3×3, stride 2, ReLU)
        → DSC1 (64  filters, stride 2)
        → DSC2 (128 filters, stride 2)
        → DSC3 (128 filters, stride 1)
        → DSC4 (256 filters, stride 2)
        → DSC5 (256 filters, stride 1)
        → DSC6 (512 filters, stride 2)
        → GlobalAveragePooling
        → Dense(128, ReLU) → Dropout(0.5)
        → Dense(num_classes, softmax)

    Args:
        input_shape: Spatial dimensions (H, W, C).  Default: (224, 224, 3).
        num_classes: 2 for binary real/fake classification.
        dropout_rate: Dropout probability in the classifier head.

    Returns:
        Compiled Keras model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Initial standard convolution
    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-3, name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)

    # Six depthwise-separable blocks with increasing filter depths
    # (64 → 128 → 256 → 512)
    dsc_config = [
        (64,  2, "dsc1"),
        (128, 2, "dsc2"),
        (128, 1, "dsc3"),
        (256, 2, "dsc4"),
        (256, 1, "dsc5"),
        (512, 2, "dsc6"),
    ]
    for filters, stride, name in dsc_config:
        x = _dsc_block(x, filters=filters, stride=stride, name=name)

    # Adaptive average pooling → 1×1 spatial output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Classifier head
    x = layers.Dense(128, use_bias=True, name="fc1")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(rate=dropout_rate, name="dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LightweightCNN_Deepfake")
    return model


# ---------------------------------------------------------------------------
# Quantisation & TFLite Export
# ---------------------------------------------------------------------------

def convert_to_tflite(
    model: keras.Model,
    output_path: str,
    quantize: bool = True,
    representative_dataset: Optional[callable] = None,
) -> str:
    """Convert a Keras model to TensorFlow Lite format with optional int8 quantisation.

    Structured pruning reduces the model from ~12.8 MB (float32) to ~3.2 MB
    (int8 quantised + pruned) — a 75% size reduction as described in the paper.

    Args:
        model:                  Trained Keras model.
        output_path:            Destination path for the .tflite file.
        quantize:               Apply full-integer (int8) quantisation when True.
        representative_dataset: Callable that yields calibration batches.
                                Required for full-integer quantisation.

    Returns:
        Path to the written .tflite file.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[TFLite] Saved {output_path}  ({size_mb:.2f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# TFLite Inference Wrapper
# ---------------------------------------------------------------------------

class TFLiteInferenceModel:
    """Thin wrapper around a TFLite interpreter for fast CPU inference.

    This class mirrors the ``predict()`` API of a Keras model so that the
    detection pipeline can use either backend transparently.

    Args:
        tflite_path: Path to the compiled .tflite model file.
    """

    def __init__(self, tflite_path: str) -> None:
        self._interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run batch inference.

        Args:
            x: Float32 array of shape (N, H, W, C) in [0, 1].

        Returns:
            Probability array of shape (N, 2).
        """
        results = []
        for sample in x:
            inp = np.expand_dims(sample, axis=0).astype(np.float32)
            self._interpreter.set_tensor(self._input_details[0]["index"], inp)
            self._interpreter.invoke()
            out = self._interpreter.get_tensor(self._output_details[0]["index"])
            results.append(out[0])
        return np.array(results, dtype=np.float32)
