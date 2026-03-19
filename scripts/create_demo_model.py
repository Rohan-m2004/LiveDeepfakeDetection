"""
Create and save a freshly initialised (random-weight) demo model.

This lets you immediately explore the GUI and the entire detection pipeline
without needing any training data.  Because the weights are random, confidence
scores will be meaningless, but every component of the system (face detection,
preprocessing, temporal analyser, GUI overlays, alert log) will function
exactly as in production.

Usage::

    python scripts/create_demo_model.py
    # Writes  models/demo_model.keras

    # Then launch the GUI with this model:
    python main.py --model models/demo_model.keras
    # --- or use the shorthand ---
    python main.py --demo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Silence TF startup noise
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make sure the repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.lightweight_cnn import build_lightweight_cnn  # noqa: E402


def create_demo_model(output_path: str = "models/demo_model.keras") -> str:
    """Build, compile, and save a randomly initialised model.

    Args:
        output_path: Where to write the Keras model file.

    Returns:
        Absolute path of the saved model.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Building Lightweight CNN …")
    model = build_lightweight_cnn()

    # Compile so the saved model carries optimizer / loss metadata (optional
    # but matches what the training pipeline produces).
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    model.save(str(out))
    size_mb = out.stat().st_size / (1024 * 1024)

    print(f"✓  Demo model saved → {out.resolve()}  ({size_mb:.1f} MB)")
    print()
    print("Launch the GUI with this model:")
    print(f"    python main.py --model {out}")
    print("  — or simply —")
    print("    python main.py --demo")
    return str(out.resolve())


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a randomly initialised demo model for the GUI."
    )
    parser.add_argument(
        "--output",
        default="models/demo_model.keras",
        metavar="PATH",
        help="Output path (default: models/demo_model.keras).",
    )
    args = parser.parse_args()
    create_demo_model(args.output)


if __name__ == "__main__":
    main()
