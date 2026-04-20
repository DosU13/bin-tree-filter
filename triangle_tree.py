"""Binary triangle tree — recursive 45-45-90 subdivision of an image.

Geometry refresher:
    A 45-45-90 right isosceles triangle has a 90-degree angle and two 45-degree
    angles.  Splitting from the right-angle vertex to the hypotenuse midpoint
    produces two smaller 45-45-90 triangles whose right angles sit at that midpoint.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Triangle:
    """A 45-45-90 triangle defined by three vertices in (x, y) pixel coords.

    Attributes:
        right_angle: Vertex at the 90-degree angle.
        hyp_a:       First endpoint of the hypotenuse (45-degree angle).
        hyp_b:       Second endpoint of the hypotenuse (45-degree angle).
        depth:       Current recursion depth.
        dominant_color: BGR tuple of the most frequent color (set for leaves).
        children:    Two child triangles if subdivided, empty list if leaf.
    """
    right_angle: tuple[float, float]
    hyp_a: tuple[float, float]
    hyp_b: tuple[float, float]
    depth: int = 0
    dominant_color: tuple[int, ...] | None = None
    children: list[Triangle] = field(default_factory=list)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return [self.right_angle, self.hyp_a, self.hyp_b]

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Subdivision
# ---------------------------------------------------------------------------

def subdivide(tri: Triangle) -> tuple[Triangle, Triangle]:
    """Split a 45-45-90 triangle into two children.

    The cut goes from *right_angle* to the midpoint of the hypotenuse.
    Both children have their right angle at that midpoint.
    """
    mid = (
        (tri.hyp_a[0] + tri.hyp_b[0]) / 2,
        (tri.hyp_a[1] + tri.hyp_b[1]) / 2,
    )
    child1 = Triangle(
        right_angle=mid,
        hyp_a=tri.right_angle,
        hyp_b=tri.hyp_a,
        depth=tri.depth + 1,
    )
    child2 = Triangle(
        right_angle=mid,
        hyp_a=tri.right_angle,
        hyp_b=tri.hyp_b,
        depth=tri.depth + 1,
    )
    return child1, child2


# ---------------------------------------------------------------------------
# Color analysis
# ---------------------------------------------------------------------------

def _dominant_color(image: np.ndarray, tri: Triangle) -> tuple[tuple[int, ...], float]:
    """Find the most frequent color and its ratio inside a triangle.

    Returns:
        color: BGR tuple of the most common color.
        ratio: Fraction of pixels with that color (0.0 – 1.0).
    """
    h, w = image.shape[:2]

    # Rasterise the triangle into a mask
    pts = np.array(tri.vertices, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return (0, 0, 0), 1.0

    # Encode each BGR pixel as a single int for fast counting
    colors = image[ys, xs]
    encoded = (
        colors[:, 0].astype(np.int64) * 65536
        + colors[:, 1].astype(np.int64) * 256
        + colors[:, 2].astype(np.int64)
    )
    unique_codes, counts = np.unique(encoded, return_counts=True)

    best_idx = np.argmax(counts)
    best_code = int(unique_codes[best_idx])
    ratio = counts[best_idx] / len(ys)

    b = best_code >> 16
    g = (best_code >> 8) & 0xFF
    r = best_code & 0xFF
    return (b, g, r), ratio


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_tree(
    image: np.ndarray,
    tri: Triangle,
    max_depth: int,
    threshold: float,
    min_depth: int = 0,
) -> Triangle:
    """Recursively subdivide *tri* until the region is uniform enough.

    Stops when:
        - depth >= min_depth AND dominant color covers >= *threshold*, **or**
        - *max_depth* is reached.
    """
    color, ratio = _dominant_color(image, tri)

    if tri.depth >= max_depth:
        tri.dominant_color = color
        return tri

    if tri.depth >= min_depth and ratio >= threshold:
        tri.dominant_color = color
        return tri

    c1, c2 = subdivide(tri)
    tri.children = [
        build_tree(image, c1, max_depth, threshold, min_depth),
        build_tree(image, c2, max_depth, threshold, min_depth),
    ]
    return tri


def collect_leaves(tri: Triangle) -> list[Triangle]:
    """Return every leaf node in the tree rooted at *tri*."""
    if tri.is_leaf:
        return [tri]
    leaves: list[Triangle] = []
    for child in tri.children:
        leaves.extend(collect_leaves(child))
    return leaves


# ---------------------------------------------------------------------------
# Initial split
# ---------------------------------------------------------------------------

def make_initial_triangles(size: int) -> list[Triangle]:
    """Split a square image along its diagonal into two 45-45-90 triangles.

    Upper-right half:  right angle at top-right    (size, 0)
    Lower-left half:   right angle at bottom-left  (0, size)
    """
    upper_right = Triangle(
        right_angle=(size, 0),
        hyp_a=(0, 0),
        hyp_b=(size, size),
        depth=0,
    )
    lower_left = Triangle(
        right_angle=(0, size),
        hyp_a=(size, size),
        hyp_b=(0, 0),
        depth=0,
    )
    return [upper_right, lower_left]
