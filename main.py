"""
Entry point for the LiveDeepfakeDetection application.

Usage::

    # ── Quick demo (no datasets needed) ──────────────────────────────────
    # Automatically creates a randomly initialised model and opens the GUI.
    python main.py --demo

    # ── GUI ──────────────────────────────────────────────────────────────
    # Launch GUI (no model pre-loaded — shows 50/50 uncertainty)
    python main.py

    # Launch GUI with a pre-trained model
    python main.py --model models/deepfake_detector.keras

    # Launch GUI with a TFLite model (edge deployment)
    python main.py --model models/deepfake_detector.tflite

    # ── Training ─────────────────────────────────────────────────────────
    # Train a new model on real data
    python main.py --train --data data/

    # End-to-end pipeline smoke-test using synthetic data
    python scripts/generate_synthetic_data.py
    python main.py --train --data demo_data/ --epochs 5

    # ── Headless inference ────────────────────────────────────────────────
    python main.py --video path/to/video.mp4 --model models/deepfake_detector.keras
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="LiveDeepfakeDetection",
        description="Live detection of synthetic faces in video conferencing "
                    "using a Lightweight CNN model.",
    )
    parser.add_argument(
        "--model", "-m",
        metavar="PATH",
        help="Path to a saved Keras (.keras / .h5) or TFLite (.tflite) model.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Quick-start demo mode: automatically create a randomly initialised "
            "model (models/demo_model.keras) if one does not already exist, then "
            "launch the GUI with that model.  No training data required."
        ),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model (requires --data).",
    )
    parser.add_argument(
        "--data", "-d",
        metavar="DIR",
        help="Root directory containing train/, val/, test/ sub-directories "
             "(required when --train is used).",
    )
    parser.add_argument(
        "--video", "-v",
        metavar="PATH",
        help="Run headless inference on a video file instead of opening the GUI.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        metavar="DIR",
        help="Output directory for trained models (default: models/).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip TFLite quantisation after training.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Suppress the GUI even when not training (useful in CI).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """Create a demo model (if needed) and launch the GUI with it."""
    demo_path = os.path.join("models", "demo_model.keras")

    if not os.path.isfile(demo_path):
        print("Demo model not found — creating one now …")
        from scripts.create_demo_model import create_demo_model
        create_demo_model(demo_path)
    else:
        print(f"Using existing demo model: {demo_path}")

    from gui.app import launch
    launch(model_path=demo_path)


# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------

def _run_training(args: argparse.Namespace) -> None:
    if not args.data:
        print("ERROR: --data is required when --train is set.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.data):
        print(f"ERROR: Data directory not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    from src.training.trainer import DeepfakeTrainer

    trainer = DeepfakeTrainer(
        data_root=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        quantize=not args.no_quantize,
    )
    trainer.train()


# ---------------------------------------------------------------------------
# Headless video inference
# ---------------------------------------------------------------------------

def _run_video_inference(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required for headless video inference.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.video):
        print(f"ERROR: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    import cv2
    import tensorflow as tf

    from src.detection.detector import DeepfakeDetector

    # Load model
    if args.model.endswith(".tflite"):
        from src.model.lightweight_cnn import TFLiteInferenceModel
        model = TFLiteInferenceModel(args.model)
    else:
        model = tf.keras.models.load_model(args.model)

    detector = DeepfakeDetector(model=model)
    cap = cv2.VideoCapture(args.video)
    frame_count = 0
    alert_count = 0

    print(f"Processing video: {args.video}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.process_frame(frame)
        frame_count += 1
        if result.alert:
            alert_count += 1
            ts = result.timestamp
            print(
                f"  Frame {frame_count:06d} — ALERT  "
                f"P(fake)={result.faces[0].p_fake:.3f}"
            )

    cap.release()
    print(f"\nDone.  Processed {frame_count} frames, raised {alert_count} alerts.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.demo:
        _run_demo()
        return

    if args.train:
        _run_training(args)
        return

    if args.video:
        _run_video_inference(args)
        return

    if args.headless:
        print("Headless mode: no GUI launched.")
        return

    # Default: launch GUI
    from gui.app import launch
    launch(model_path=args.model)


if __name__ == "__main__":
    main()
