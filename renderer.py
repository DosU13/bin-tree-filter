"""Render the ornamental pattern onto every leaf triangle."""

import cv2
import numpy as np

from triangle_tree import Triangle


# ---------------------------------------------------------------------------
# Ornament mask
# ---------------------------------------------------------------------------

def load_ornament_mask(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the ornament image and extract triangular binary masks.

    Cuts the upper-right 45-45-90 triangle from the square ornament.
    Returns both the normal mask (ink = 255) and an inverted mask
    (background = 255, ink = 0) — both clipped to the triangle so that
    warping either one keeps border pixels at 0.

    Returns:
        mask:         Binary image (H, W) — 255 where ornament ink exists.
        mask_inv:     Binary image (H, W) — inverted pattern inside the triangle.
        src_triangle: Float32 array (3, 2) of the canonical triangle vertices
                      ordered as [right_angle, hyp_a, hyp_b].
    """
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape

    # Ornament is black-on-white → invert so the pattern becomes 255
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Canonical triangle: upper-right half of the square
    #   right angle at (w-1, 0)   — top-right corner
    #   hyp_a      at (0, 0)      — top-left corner
    #   hyp_b      at (w-1, h-1)  — bottom-right corner
    tri_pts = np.array([[w - 1, 0], [0, 0], [w - 1, h - 1]], dtype=np.int32)

    triangle_region = np.zeros_like(binary)
    cv2.fillPoly(triangle_region, [tri_pts], 255)

    mask = cv2.bitwise_and(binary, triangle_region)
    mask_inv = cv2.bitwise_and(cv2.bitwise_not(binary), triangle_region)

    src_triangle = np.array(
        [[w - 1, 0], [0, 0], [w - 1, h - 1]], dtype=np.float32
    )
    return mask, mask_inv, src_triangle


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(
    output: np.ndarray,
    leaves: list[Triangle],
    ornament_mask: np.ndarray,
    ornament_mask_inv: np.ndarray,
    src_triangle: np.ndarray,
) -> np.ndarray:
    """Paint each leaf triangle with the ornament pattern in its dominant color.

    For every leaf the ornament mask is affine-warped to fit the triangle,
    then pixels where the warped mask is active are set to the leaf's color.
    Even-depth triangles use the inverted mask so adjacent edges interlock.

    Args:
        output:            Destination image (H, W, 3), modified in-place.
        leaves:            Leaf Triangle nodes (each with .dominant_color set).
        ornament_mask:     Normal binary mask from :func:`load_ornament_mask`.
        ornament_mask_inv: Inverted binary mask (pre-clipped to triangle).
        src_triangle:      Canonical triangle vertices matching the masks.

    Returns:
        The same *output* array, for convenience.
    """
    h, w = output.shape[:2]

    for leaf in leaves:
        if leaf.dominant_color is None:
            continue

        # Pick normal or inverted mask based on depth parity
        mask = ornament_mask_inv if leaf.depth % 2 == 0 else ornament_mask

        # Map canonical ornament triangle → leaf triangle
        dst_pts = np.array(
            [leaf.right_angle, leaf.hyp_a, leaf.hyp_b], dtype=np.float32
        )
        M = cv2.getAffineTransform(src_triangle, dst_pts)

        warped = cv2.warpAffine(
            mask, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Clip to the leaf polygon (warp can bleed a few pixels outside)
        poly = np.array(leaf.vertices, dtype=np.int32).reshape((-1, 1, 2))
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 255)

        combined = cv2.bitwise_and(warped, poly_mask)

        output[combined > 0] = leaf.dominant_color

    return output


def draw_debug_boundaries(
    image: np.ndarray,
    leaves: list[Triangle],
    color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw triangle outlines on *image* for visual debugging."""
    for leaf in leaves:
        pts = np.array(leaf.vertices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    return image
