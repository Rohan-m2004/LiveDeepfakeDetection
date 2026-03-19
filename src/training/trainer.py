"""
Training pipeline for the Lightweight CNN deepfake detector.

Implements:
    * Data loading from a directory tree (real/ and fake/ sub-folders).
    * Binary cross-entropy loss with label smoothing.
    * Adam optimiser with step-decay learning-rate schedule.
    * Structured filter pruning via magnitude-based importance scoring.
    * Post-training quantisation via TFLite conversion.
    * Multi-dataset training protocol (FF++ + Celeb-DF train,
      DFDC validation, DeeperForensics held-out test).

Directory layout expected by the data loader::

    data_root/
        train/
            real/   ← original frames
            fake/   ← manipulated frames
        val/
            real/
            fake/
        test/
            real/
            fake/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.model.lightweight_cnn import build_lightweight_cnn, convert_to_tflite
from src.preprocessing.face_processor import FACE_SIZE, FacePreprocessor

# ---------------------------------------------------------------------------
# Constants (hyper-parameters from the paper)
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 32
EPOCHS: int = 100
INITIAL_LR: float = 1e-3
LR_DECAY_EPOCHS: int = 30
LR_DECAY_FACTOR: float = 0.1
LABEL_SMOOTHING: float = 0.1
L2_LAMBDA: float = 1e-4
EARLY_STOP_PATIENCE: int = 15
DROPOUT_RATE: float = 0.5
NUM_CLASSES: int = 2


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

class StepDecaySchedule(keras.optimizers.schedules.LearningRateSchedule):
    """LR(t) = LR₀ × 0.1^⌊t / decay_steps⌋ (Equation 5 in the paper).

    Args:
        initial_lr:    LR₀.
        decay_steps:   Number of epochs per decay step.
        decay_factor:  Multiplicative reduction factor.
    """

    def __init__(
        self,
        initial_lr: float = INITIAL_LR,
        decay_steps: int = LR_DECAY_EPOCHS,
        decay_factor: float = LR_DECAY_FACTOR,
    ) -> None:
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step_float = tf.cast(step, tf.float32)
        n = tf.math.floor(step_float / float(self.decay_steps))
        lr = self.initial_lr * (self.decay_factor ** n)
        return lr

    def get_config(self) -> dict:
        return {
            "initial_lr": self.initial_lr,
            "decay_steps": self.decay_steps,
            "decay_factor": self.decay_factor,
        }


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_image_dataset(
    directory: str,
    image_size: Tuple[int, int] = (FACE_SIZE, FACE_SIZE),
    batch_size: int = BATCH_SIZE,
    augment: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Load a directory of images into a ``tf.data.Dataset``.

    Expects sub-directories named ``real`` and ``fake`` under *directory*.
    Classes are sorted alphabetically → fake=0, real=1.  The model output
    index 0 is P(Real) and index 1 is P(Fake), so we remap: real→[1,0],
    fake→[0,1].

    Args:
        directory:  Root directory with ``real/`` and ``fake/`` sub-dirs.
        image_size: ``(height, width)`` to resize all images to.
        batch_size: Mini-batch size.
        augment:    Apply random augmentation layers.
        seed:       Random seed for reproducibility.

    Returns:
        ``tf.data.Dataset`` yielding ``(images, one_hot_labels)`` batches.
    """
    ds = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=["fake", "real"],  # alphabetical: 0=fake, 1=real
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )

    # Normalise to [0,1] and one-hot encode
    imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def _preprocess(images: tf.Tensor, labels: tf.Tensor):
        images = tf.cast(images, tf.float32) / 255.0
        images = (images - imagenet_mean) / imagenet_std
        # one-hot: fake=[1,0], real=[0,1]  → swap so index-1 is P(Fake)
        # labels: 0=fake → [1,0], 1=real → [0,1]
        one_hot = tf.one_hot(labels, depth=NUM_CLASSES)
        # Swap columns so [P(Real), P(Fake)]
        one_hot = tf.stack([one_hot[:, 1], one_hot[:, 0]], axis=1)
        return images, one_hot

    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=15 / 360),
            layers.RandomBrightness(factor=0.3),
            layers.RandomContrast(factor=0.3),
        ],
        name="augmentation",
    )

    def _augment(images: tf.Tensor, labels: tf.Tensor):
        images = augmentation(images, training=True)
        return images, labels

    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Pruning helper
# ---------------------------------------------------------------------------

def prune_model(model: keras.Model, prune_ratio: float = 0.3) -> keras.Model:
    """Apply structured (filter-level) magnitude-based pruning.

    The paper describes zeroing filters whose L1-norm importance score
    falls below an adaptive threshold.  This implementation zeroes the
    weights of the lowest-importance filters (by channel L1 norm) in each
    ``Conv2D`` and ``DepthwiseConv2D`` layer.

    Args:
        model:       Trained Keras model.
        prune_ratio: Fraction of filters to prune per layer.

    Returns:
        The same model with pruned weights set to zero (in-place).

    Note:
        For production pruning (removing filters structurally), use the
        TensorFlow Model Optimization Toolkit
        (``tensorflow_model_optimization``).
    """
    for layer in model.layers:
        if not isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]  # shape: (kH, kW, C_in, C_out) or (kH, kW, C_in, depth_mult)
        # Importance score: sum of absolute values per output filter
        importance = np.sum(np.abs(kernel), axis=(0, 1, 2))
        threshold = np.percentile(importance, prune_ratio * 100)
        mask = importance > threshold
        kernel[..., ~mask] = 0.0
        weights[0] = kernel
        layer.set_weights(weights)

    return model


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DeepfakeTrainer:
    """High-level trainer wrapping the full training pipeline.

    Args:
        data_root:    Root directory with ``train/``, ``val/``, ``test/``
                      sub-directories.
        output_dir:   Directory where model checkpoints are saved.
        epochs:       Maximum training epochs (default: 100).
        batch_size:   Mini-batch size (default: 32).
        prune_ratio:  Fraction of filters to prune after training (0 = off).
        quantize:     Convert to int8 TFLite after training.
    """

    def __init__(
        self,
        data_root: str,
        output_dir: str = "models",
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        prune_ratio: float = 0.3,
        quantize: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.prune_ratio = prune_ratio
        self.quantize = quantize

        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None

    # ------------------------------------------------------------------
    def _build_datasets(
        self,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        train_dir = str(self.data_root / "train")
        val_dir = str(self.data_root / "val")
        test_dir = str(self.data_root / "test")

        train_ds = _load_image_dataset(
            train_dir, batch_size=self.batch_size, augment=True
        )
        val_ds = _load_image_dataset(
            val_dir, batch_size=self.batch_size, augment=False
        )
        test_ds = _load_image_dataset(
            test_dir, batch_size=self.batch_size, augment=False
        )
        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------
    def train(self) -> keras.Model:
        """Run the full training pipeline.

        Returns:
            Trained (and optionally pruned + quantised) Keras model.
        """
        print("[Trainer] Building model …")
        self.model = build_lightweight_cnn(
            input_shape=(FACE_SIZE, FACE_SIZE, 3),
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
        )
        self.model.summary()

        print("[Trainer] Loading datasets …")
        train_ds, val_ds, test_ds = self._build_datasets()

        # Compile
        lr_schedule = StepDecaySchedule(
            initial_lr=INITIAL_LR,
            decay_steps=LR_DECAY_EPOCHS,
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        loss_fn = keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        )
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

        # Callbacks
        ckpt_path = str(self.output_dir / "best_model.keras")
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.output_dir / "logs"),
                histogram_freq=1,
            ),
        ]

        print("[Trainer] Starting training …")
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
        )

        # --- Evaluate on held-out test set ----------------------------
        print("[Trainer] Evaluating on test set …")
        results = self.model.evaluate(test_ds, verbose=1)
        print(f"[Trainer] Test results: {dict(zip(self.model.metrics_names, results))}")

        # --- Optional pruning -----------------------------------------
        if self.prune_ratio > 0:
            print(f"[Trainer] Pruning {self.prune_ratio*100:.0f}% of filters …")
            self.model = prune_model(self.model, self.prune_ratio)
            # Fine-tune for a few epochs after pruning
            self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=max(5, self.epochs // 10),
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor="val_accuracy",
                        patience=5,
                        restore_best_weights=True,
                    )
                ],
            )

        # --- Save Keras model -----------------------------------------
        final_path = str(self.output_dir / "deepfake_detector.keras")
        self.model.save(final_path)
        print(f"[Trainer] Saved Keras model → {final_path}")

        # --- Optional TFLite quantisation -----------------------------
        if self.quantize:
            tflite_path = str(self.output_dir / "deepfake_detector.tflite")

            def representative_data_gen():
                for images, _ in train_ds.take(200):
                    yield [images]

            convert_to_tflite(
                self.model,
                tflite_path,
                quantize=True,
                representative_dataset=representative_data_gen,
            )

        return self.model
