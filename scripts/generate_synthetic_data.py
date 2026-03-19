"""
Generate a small synthetic dataset for an end-to-end pipeline smoke-test.

This script creates a tiny (but structurally correct) dataset of fake face
images so you can run a full training cycle in minutes — no real video data
required.

How the images are created
--------------------------
* **REAL** samples — random Gaussian noise with a slight blue-green tint,
  simulating natural skin-tone variation.
* **FAKE** samples — the same Gaussian noise base with a strong red-magenta
  tint + a high-frequency grid pattern overlaid, representing GAN compression
  artefacts that the model learns to flag.

A simple CNN trained on this synthetic data reaches near-100% accuracy within
a few epochs, proving the full pipeline is wired correctly.

Usage::

    # Create data in the default location (./demo_data/)
    python scripts/generate_synthetic_data.py

    # Train on it
    python main.py --train --data demo_data/ --epochs 5 --output-dir models/

    # Load the result into the GUI
    python main.py --model models/deepfake_detector.keras

Options::

    --output-dir DIR   Root directory for the dataset   [default: demo_data]
    --samples-per-class N  Images per split × class     [default: 200]
    --image-size N     Square pixel size of each image  [default: 224]
    --seed N           Random seed for reproducibility  [default: 42]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# Make sure the repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------

def _make_real_image(rng: np.random.Generator, size: int = 224) -> np.ndarray:
    """Return a uint8 (H, W, 3) image that looks 'natural'.

    A randomised skin-tone base (warm mid-tones) with soft Gaussian noise
    and a slight radial gradient to simulate face lighting.
    """
    # Base skin-tone colour (varies per sample)
    r_base = rng.integers(140, 220)
    g_base = rng.integers(100, 175)
    b_base = rng.integers(80, 150)

    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r_base
    img[:, :, 1] = g_base
    img[:, :, 2] = b_base

    # Soft radial lighting gradient (brighter in centre)
    cx, cy = size / 2, size / 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    gradient = 1.0 - 0.4 * (dist / (size * 0.7))
    img *= gradient[:, :, np.newaxis]

    # Gaussian noise (natural skin texture)
    noise = rng.normal(0, 12, (size, size, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    return img.astype(np.uint8)


def _make_fake_image(rng: np.random.Generator, size: int = 224) -> np.ndarray:
    """Return a uint8 (H, W, 3) image that looks 'synthetic'.

    Unnatural colour saturation (red-magenta tint), a regular high-frequency
    grid artefact, and banding in the brightness channel — all hallmarks of
    GAN-generated faces.
    """
    # Unnatural colour: dominant red channel, suppressed green/blue
    r_base = rng.integers(180, 255)
    g_base = rng.integers(40, 100)
    b_base = rng.integers(100, 180)

    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r_base
    img[:, :, 1] = g_base
    img[:, :, 2] = b_base

    # High-frequency grid pattern (GAN checkerboard artefact)
    grid_period = rng.integers(4, 10)
    grid = ((np.arange(size) % grid_period) < (grid_period // 2)).astype(np.float32)
    grid_2d = np.outer(grid, grid) * 30.0
    img[:, :, 0] = np.clip(img[:, :, 0] + grid_2d, 0, 255)

    # Horizontal banding (JPEG/compression ringing)
    band_period = rng.integers(8, 20)
    banding = np.sin(np.arange(size) * (2 * np.pi / band_period)) * 15
    img[:, :, 1] = np.clip(img[:, :, 1] + banding[:, np.newaxis], 0, 255)

    # Low-magnitude Gaussian noise
    noise = rng.normal(0, 6, (size, size, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: str = "demo_data",
    samples_per_class: int = 200,
    image_size: int = 224,
    seed: int = 42,
) -> Path:
    """Create the full train / val / test directory tree.

    Layout::

        output_dir/
            train/
                real/  (samples_per_class images)
                fake/  (samples_per_class images)
            val/
                real/  (samples_per_class // 3 images)
                fake/  (samples_per_class // 3 images)
            test/
                real/  (samples_per_class // 3 images)
                fake/  (samples_per_class // 3 images)

    Args:
        output_dir:         Root directory to create the dataset in.
        samples_per_class:  Number of images per split × class.
        image_size:         Pixel dimensions (square) for each image.
        seed:               Random seed.

    Returns:
        Path to the created ``output_dir``.
    """
    rng = np.random.default_rng(seed)
    root = Path(output_dir)

    split_sizes = {
        "train": samples_per_class,
        "val": max(1, samples_per_class // 3),
        "test": max(1, samples_per_class // 3),
    }

    total = sum(v * 2 for v in split_sizes.values())
    print(f"Generating synthetic dataset → {root.resolve()}")
    print(f"  {total} images total  ({image_size}×{image_size} px)")

    done = 0
    for split, n in split_sizes.items():
        for label, fn in [("real", _make_real_image), ("fake", _make_fake_image)]:
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img_arr = fn(rng, image_size)
                img = Image.fromarray(img_arr, mode="RGB")
                img.save(folder / f"{label}_{i:04d}.jpg", quality=95)
                done += 1
            print(f"  ✓  {split}/{label}  ({n} images)")

    print(f"\n✓  Dataset ready:  {root.resolve()}")
    print()
    print("Next steps:")
    print(f"    python main.py --train --data {output_dir}/ --epochs 5 --output-dir models/")
    print("    python main.py --model models/deepfake_detector.keras")
    return root


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic deepfake detection dataset for demos."
    )
    parser.add_argument(
        "--output-dir",
        default="demo_data",
        metavar="DIR",
        help="Root directory for the dataset (default: demo_data).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=200,
        metavar="N",
        help="Images per split × class (default: 200).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        metavar="N",
        help="Square image size in pixels (default: 224).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
