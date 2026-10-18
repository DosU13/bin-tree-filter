"""2D transforms between the coordinate spaces of a scene.

Spaces:
    world      scene canvas, in world pixels (y down)
    cell       stage cells: (world - stage origin) / cell_size
    normalised an image's own (u, v) in 0..1, u right, v down (whole image)
    source     a camera frame's continuous pixel coords (pixel i spans [i, i+1))

All transforms are 3x3 matrices applied to (x, y, 1); rectangles are affine,
4-corner warps are homographies.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.engine import apply_matrix

UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])   # TL, TR, BR, BL


def rect_matrix(x: float, y: float, w: float, h: float) -> np.ndarray:
    """Normalised (u, v) -> world for an axis-aligned rectangle."""
    return np.array([[w, 0.0, x], [0.0, h, y], [0.0, 0.0, 1.0]])


def quad_matrix(corners) -> np.ndarray:
    """Normalised (u, v) -> world for a 4-corner quad given TL, TR, BR, BL."""
    dst = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(UNIT_SQUARE, dst).astype(np.float64)


def translate_scale(tx: float, ty: float, sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])


def invert(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(m, dtype=np.float64))


def apply(m: np.ndarray, points) -> np.ndarray:
    """Apply a 3x3 matrix to (N, 2) points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return apply_matrix(np.asarray(m, dtype=np.float64), pts)


def orient_quad(quad) -> np.ndarray:
    """Return the quad with positive (clockwise on a y-down screen) winding."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2).copy()
    x, y = q[:, 0], q[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        q = q[::-1].copy()
    return q


def quad_distance(points, quad) -> np.ndarray:
    """Signed distance of points to a convex quad's edges: positive inside, negative outside."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = orient_quad(quad)
    dist = np.full(len(pts), np.inf)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        e = b - a
        n = np.array([-e[1], e[0]]) / max(np.hypot(*e), 1e-12)
        dist = np.minimum(dist, (pts - a) @ n)
    return dist


def source_to_display(width: int, height: int, rotation: int = 0, mirror: bool = False) -> np.ndarray:
    """Source continuous pixel coords -> normalised display coords.

    The display image is the source mirrored horizontally (if *mirror*) and then
    rotated clockwise by *rotation* degrees (0, 90, 180, 270).
    """
    w, h = float(width), float(height)
    m = np.array([[1 / w, 0, 0], [0, 1 / h, 0], [0, 0, 1.0]])         # -> normalised source
    if mirror:
        m = np.array([[-1.0, 0, 1], [0, 1, 0], [0, 0, 1]]) @ m
    rot = rotation % 360
    if rot == 90:      # (u, v) -> (1 - v, u)
        r = np.array([[0.0, -1, 1], [1, 0, 0], [0, 0, 1]])
    elif rot == 180:   # (u, v) -> (1 - u, 1 - v)
        r = np.array([[-1.0, 0, 1], [0, -1, 1], [0, 0, 1]])
    elif rot == 270:   # (u, v) -> (v, 1 - u)
        r = np.array([[0.0, 1, 0], [-1, 0, 1], [0, 0, 1]])
    else:
        r = np.eye(3)
    return r @ m


def display_size(width: int, height: int, rotation: int = 0) -> tuple[int, int]:
    """Size of the display image after rotation."""
    return (height, width) if rotation % 180 == 90 else (width, height)
