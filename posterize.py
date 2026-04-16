"""Image posterization via K-means color quantization."""

import cv2
import numpy as np


def posterize(image: np.ndarray, num_colors: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an image to a limited color palette using K-means clustering.

    Args:
        image: Input BGR image (H, W, 3).
        num_colors: Number of colors in the output palette.

    Returns:
        posterized: Image with reduced colors (H, W, 3).
        palette: Array of shape (num_colors, 3) with the BGR palette colors.
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    posterized = centers[labels.flatten()].reshape(h, w, 3)

    return posterized, centers
