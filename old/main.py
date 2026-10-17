"""Convert any image into a Kyrgyz ornamental pattern.

Pipeline:
    1. Load image and pad to square.
    2. Posterize to a small colour palette (K-means).
    3. Build a binary 45-45-90 triangle tree over the posterized image.
    4. Render the ornament mask into each leaf triangle using its dominant colour.
    5. Save the result (and optional debug images).

Usage:
    python main.py <input_image> [output_image]
"""

import os
import sys

import cv2
import numpy as np

from config import (
    BACKGROUND_COLOR,
    HOMOGENEITY_THRESHOLD,
    MAX_DEPTH,
    MIN_DEPTH,
    NUM_COLORS,
    ORNAMENT_FILENAME,
)
from posterize import posterize
from renderer import draw_debug_boundaries, load_ornament_mask, render
from triangle_tree import build_tree, collect_leaves, make_initial_triangles


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_and_square(path: str) -> np.ndarray:
    """Load an image and centre-pad it to a square with white borders."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    h, w = image.shape[:2]
    size = max(h, w)

    squared = np.full((size, size, 3), 255, dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    squared[y_off : y_off + h, x_off : x_off + w] = image
    return squared


def _save_side_by_side(name: str, base: str, image: np.ndarray) -> None:
    root, ext = os.path.splitext(base)
    path = f"{root}_{name}{ext}"
    cv2.imwrite(path, image)
    print(f"  {name:.<20s} {path}")


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_image> [output_image]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.png"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ornament_path = os.path.join(script_dir, ORNAMENT_FILENAME)

    # 1. Load -------------------------------------------------------
    print("[1/4] Loading image ...")
    image = _load_and_square(input_path)
    size = image.shape[0]
    print(f"       Square size: {size}x{size}")

    # 2. Posterize --------------------------------------------------
    print(f"[2/4] Posterizing to {NUM_COLORS} colours ...")
    posterized, palette = posterize(image, NUM_COLORS)
    print(f"       Palette (BGR): {palette.tolist()}")

    # 3. Build tree -------------------------------------------------
    print(f"[3/4] Building triangle tree  (depth {MIN_DEPTH}-{MAX_DEPTH}, threshold={HOMOGENEITY_THRESHOLD}) ...")
    roots = make_initial_triangles(size)

    all_leaves: list = []
    for root_tri in roots:
        build_tree(posterized, root_tri, MAX_DEPTH, HOMOGENEITY_THRESHOLD, MIN_DEPTH)
        all_leaves.extend(collect_leaves(root_tri))

    print(f"       {len(all_leaves)} leaf triangles")

    # 4. Render ornament --------------------------------------------
    print("[4/4] Rendering ornamental pattern ...")
    ornament_mask, ornament_mask_inv, src_tri = load_ornament_mask(ornament_path)

    output = np.full((size, size, 3), BACKGROUND_COLOR, dtype=np.uint8)
    render(output, all_leaves, ornament_mask, ornament_mask_inv, src_tri)

    # 5. Save -------------------------------------------------------
    cv2.imwrite(output_path, output)
    print(f"\nSaved:")
    print(f"  {'result':.<20s} {output_path}")

    _save_side_by_side("posterized", output_path, posterized)

    debug = posterized.copy()
    draw_debug_boundaries(debug, all_leaves)
    _save_side_by_side("triangles", output_path, debug)

    print("\nDone.")


if __name__ == "__main__":
    main()
