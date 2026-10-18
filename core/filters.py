"""Input image filters, depth-band labels and colour helpers."""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Depth-mode colour presets (BGR)
# ---------------------------------------------------------------------------

FG_PRESETS = [
    (0,   0,   160),   # dark red
    (120, 30,  0),     # dark navy
    (0,   100, 10),    # dark green
    (30,  0,   130),   # dark maroon
    (20,  20,  20),    # near-black
]

BG_PRESETS = [
    (245, 235, 210),   # warm cream
    (255, 245, 220),   # light gold
    (255, 255, 255),   # white
    (220, 235, 245),   # cool white
]


# ---------------------------------------------------------------------------
# Colour conversion
# ---------------------------------------------------------------------------

def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (b, g, r)."""
    v = value.lstrip("#")
    r, g, b = (int(v[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def bgr_to_hex(bgr) -> str:
    """(b, g, r) -> '#rrggbb'."""
    b, g, r = (int(round(float(c))) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Image calibration filters
# ---------------------------------------------------------------------------

def apply_blur(image: np.ndarray, blur_size: int) -> np.ndarray:
    """Gaussian blur. blur_size=0 means no blur; each step adds 2px to the kernel."""
    if blur_size <= 0:
        return image
    k = blur_size * 2 + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp-mask sharpening. amount=0 means no sharpening."""
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def apply_filters(image: np.ndarray, brightness: int, contrast: float, saturation: float) -> np.ndarray:
    """Apply brightness, contrast, and saturation adjustments."""
    if brightness == 0 and contrast == 1.0 and saturation == 1.0:
        return image
    result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    if saturation != 1.0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


# ---------------------------------------------------------------------------
# Geometry / labels
# ---------------------------------------------------------------------------

def centre_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0 : y0 + s, x0 : x0 + s]


def depth_labels(gray: np.ndarray, lo: int, hi: int, size: int | tuple[int, int]) -> np.ndarray:
    """Band mask at full resolution, area-downsampled, then majority-thresholded.

    Args:
        size: Output side length, or (width, height).
    """
    w, h = (size, size) if isinstance(size, int) else size
    band = ((gray >= lo) & (gray <= hi)).astype(np.float32)
    small = band if band.shape[:2] == (h, w) else cv2.resize(band, (w, h), interpolation=cv2.INTER_AREA)
    return (small > 0.5).astype(np.int64)
